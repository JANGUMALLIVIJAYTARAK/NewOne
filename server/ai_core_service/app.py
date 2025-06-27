# FusedChatbot/server/ai_core_service/app.py
import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import redis
import hashlib
import json

try:
    from . import config, file_parser, faiss_handler, llm_handler, llm_router
    from .tools import web_search
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
    import config, file_parser, faiss_handler, llm_handler, llm_router
    from tools import web_search

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    redis_client = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=int(os.getenv('REDIS_PORT', 6379)), db=0, decode_responses=True)
    redis_client.ping()
    logger.info("Successfully connected to Redis. Caching is enabled.")
except redis.exceptions.ConnectionError as e:
    logger.error(f"Could not connect to Redis: {e}. Caching will be DISABLED.")
    redis_client = None

def create_error_response(message, status_code=500):
    logger.error(f"API Error Response ({status_code}): {message}")
    return jsonify({"error": message, "status": "error"}), status_code

@app.route('/health', methods=['GET'])
def health_check():
    logger.info("\n--- Received request at /health ---")
    status_details = { 
        "status": "error", "embedding_model_type": config.EMBEDDING_TYPE, 
        "embedding_model_name": config.EMBEDDING_MODEL_NAME, "embedding_dimension": None, 
        "default_index_loaded": False, "gemini_sdk_installed": True, 
        "ollama_available": llm_handler.ollama_available, "groq_sdk_installed": True, "message": "" 
    }
    http_status_code = 503
    try:
        model = faiss_handler.get_embedding_model()
        if model is None: raise RuntimeError("Embedding model could not be initialized.")
        status_details["embedding_dimension"] = faiss_handler.get_embedding_dimension(model)
        status_details["default_index_loaded"] = config.DEFAULT_INDEX_USER_ID in faiss_handler.loaded_indices
        status_details["status"] = "ok"; status_details["message"] = "AI Core service is running."
        http_status_code = 200
    except Exception as e:
        logger.error(f"--- Health Check Critical Error ---", exc_info=True)
        status_details["message"] = f"Health check failed critically: {str(e)}"
    return jsonify(status_details), http_status_code

@app.route('/add_document', methods=['POST'])
def add_document():
    if not request.is_json: return create_error_response("Request must be JSON", 400)
    data = request.get_json()
    user_id, file_path, original_name = data.get('user_id'), data.get('file_path'), data.get('original_name')
    if not all([user_id, file_path, original_name]): return create_error_response("Missing required fields", 400)
    if not os.path.exists(file_path): return create_error_response(f"File not found: {file_path}", 404)
    try:
        text = file_parser.parse_file(file_path)
        if not text or not text.strip(): return jsonify({"message": f"No text content in '{original_name}'.", "status": "skipped"}), 200
        docs = file_parser.chunk_text(text, original_name, user_id)
        faiss_handler.add_documents_to_index(user_id, docs)
        return jsonify({"message": f"'{original_name}' processed successfully.", "chunks_added": len(docs), "status": "added"}), 200
    except Exception as e:
        return create_error_response(f"Failed to process '{original_name}': {e}", 500)

@app.route('/analyze_document', methods=['POST'])
def analyze_document_route():
    if not request.is_json: return create_error_response("Request must be JSON", 400)
    data = request.get_json()
    if not all(field in data for field in ['file_path_for_analysis', 'analysis_type', 'llm_provider']):
        return create_error_response("Missing required fields", 400)
    try:
        document_text = file_parser.parse_file(data['file_path_for_analysis'])
        if not document_text or not document_text.strip():
            return create_error_response("Could not parse text from the document.", 400)
        
        analysis_result, thinking_content = llm_handler.perform_document_analysis(
            document_text=document_text, analysis_type=data['analysis_type'], llm_provider=data['llm_provider'],
            api_keys=data.get('api_keys', {}), model_name=data.get('llm_model_name'), ollama_host=data.get('ollama_host')
        )
        return jsonify({"analysis_result": analysis_result, "thinking_content": thinking_content, "status": "success"}), 200
    except (ValueError, ConnectionError) as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        return create_error_response(f"Failed to perform analysis: {str(e)}", 500)


@app.route('/generate_chat_response', methods=['POST'])
def generate_chat_response_route():
    logger.info("\n--- Received request at /generate_chat_response ---")
    data = request.get_json()
    
    user_id = data.get('user_id')
    current_user_query = data.get('query')
    if not user_id or not current_user_query:
        return create_error_response("Missing user_id or query in request", 400)

    routing_decision = llm_router.route_query(current_user_query, data.get('llm_provider', config.DEFAULT_LLM_PROVIDER), data.get('llm_model_name'))
    final_provider = routing_decision['provider']
    final_model = routing_decision['model']
    logger.info(f"Router decision: Provider='{final_provider}', Model='{final_model}'")

    handler_kwargs = {
        'api_keys': data.get('api_keys', {}), 'model_name': final_model,
        'chat_history': data.get('chat_history', []), 'system_prompt': data.get('system_prompt'),
        'ollama_host': data.get('ollama_host')
    }

    context_text_for_llm = "No relevant context was found from any source."
    rag_references_for_client = []
    context_source = "None" 

    perform_rag = data.get('perform_rag', True)
    if perform_rag:
        # --- Multi-Query RAG Search ---
        logger.info("Performing multi-query RAG search on local documents...")
        queries_to_search = [current_user_query]
        if data.get('enable_multi_query', True):
            try:
                sub_queries = llm_handler.generate_sub_queries(
                    original_query=current_user_query,
                    llm_provider=final_provider,
                    **handler_kwargs
                )
                if sub_queries: queries_to_search.extend(sub_queries)
            except Exception as e:
                logger.error(f"Error during sub-query generation: {e}", exc_info=True)

        unique_chunks = set()
        docs_for_context = []
        for q in queries_to_search:
            results = faiss_handler.query_index(user_id, q, k=config.DEFAULT_RAG_K_PER_SUBQUERY_CONFIG, active_file=data.get('active_file'))
            for doc, score in results:
                if doc.page_content not in unique_chunks:
                    unique_chunks.add(doc.page_content)
                    docs_for_context.append((doc, score))
        
        # --- Relevance Check and Web Search Fallback ---
        is_relevant = False
        if docs_for_context:
            temp_context = "\n\n".join([doc.page_content for doc, score in docs_for_context])
            is_relevant = llm_handler.check_context_relevance(current_user_query, temp_context, **handler_kwargs)

        if docs_for_context and is_relevant:
            logger.info(f"Found {len(docs_for_context)} RELEVANT document chunks.")
            context_parts = [f"[{i+1}] Source: {d.metadata.get('documentName')}\n{d.page_content}" for i, (d, s) in enumerate(docs_for_context)]
            context_text_for_llm = "\n\n---\n\n".join(context_parts)
            rag_references_for_client = [{"documentName": d.metadata.get("documentName"), "score": float(s)} for d, s in docs_for_context]
            context_source = "Local Documents"
        else:
            if docs_for_context:
                logger.info("Local documents found, but they were NOT RELEVANT to the query. Falling back to web search.")
            else:
                logger.info("No relevant local documents found. Falling back to web search.")
            
            try:
                web_context = web_search.perform_search(current_user_query)
                if web_context:
                    context_text_for_llm = web_context
                    context_source = "Web Search"
            except Exception as e:
                logger.error(f"Web search failed: {e}", exc_info=True)
    
    try:
        logger.info(f"Calling final LLM provider: {final_provider} with context from: {context_source}")
        
        final_answer, thinking_content = llm_handler.generate_response(
            llm_provider=final_provider,
            query=current_user_query,
            context_text=context_text_for_llm,
            **handler_kwargs
        )
        
        return jsonify({
            "llm_response": final_answer, "references": rag_references_for_client,
            "thinking_content": thinking_content, "status": "success",
            "provider_used": final_provider, "model_used": final_model,
            "context_source": context_source
        }), 200

    except (ConnectionError, ValueError) as e:
        return create_error_response(str(e), 502)
    except Exception as e:
        logger.error(f"Unhandled exception in generate_chat_response: {e}", exc_info=True)
        return create_error_response(f"Failed to generate chat response: {str(e)}", 500)


if __name__ == '__main__':
    try:
        faiss_handler.ensure_faiss_dir()
        faiss_handler.get_embedding_model()
        faiss_handler.load_or_create_index(config.DEFAULT_INDEX_USER_ID)
    except Exception as e:
        logger.critical(f"CRITICAL STARTUP FAILURE: {e}", exc_info=True)
        sys.exit(1)
        
    port = int(os.getenv("AI_CORE_SERVICE_PORT", 9000))
    host = '0.0.0.0'
    logger.info(f"--- Starting AI Core Service (Flask App) on http://{host}:{port} ---")
    logger.info(f"Gemini SDK Installed: {bool(llm_handler.genai)}")
    logger.info(f"Groq SDK Installed: {bool(llm_handler.Groq)}")
    logger.info(f"Ollama Available: {llm_handler.ollama_available}")
    logger.info(f"Redis Connected: {redis_client is not None}")
    logger.info("---------------------------------------------")
    app.run(host=host, port=port, debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true')