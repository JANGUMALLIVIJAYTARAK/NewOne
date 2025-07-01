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
    from ai_core_service.kg_service import KnowledgeGraphService
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
    import config, file_parser, faiss_handler, llm_handler, llm_router
    from tools import web_search
    from ai_core_service.kg_service import KnowledgeGraphService

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

# Initialize KG service (customize connection params as needed)
kg_service = KnowledgeGraphService()

def create_error_response(message, status_code=500):
    logger.error(f"API Error Response ({status_code}): {message}")
    return jsonify({"error": message, "status": "error"}), status_code

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'}), 200

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


@app.route('/generate_report', methods=['POST'])
def generate_report_route():
    """
    Endpoint to generate a full report on a given topic.
    This is a multi-step process:
    1. Perform a web search to gather up-to-date context.
    2. Use an LLM to synthesize the context into a structured Markdown report.
    """
    data = request.get_json()
    logger.info(f"--- DEBUG: Full payload received by Python: {data} ---")
    
    if not data or 'topic' not in data:
        return create_error_response("Request body must be JSON and contain a 'topic' field.", 400)
    
    topic = data['topic']
    logger.info(f"Received request to generate report for topic: '{topic}'")

    try:
        # 1. Gather rich context using the existing web search tool
        logger.info(f"Performing web search for report context...")
        # ✅ THE FIX: The call to perform_search is now correct
        context_text = web_search.perform_search(topic, max_results=10)

        # ✅ THE FIX: This logic is now correctly placed within the try block
        if not context_text or not context_text.strip():
            logger.warning(f"Web search returned no results for topic: '{topic}'")
            return create_error_response(f"Could not find sufficient information on the topic '{topic}' to generate a report.", 404)

        logger.info(f"Web search successful. Synthesizing report from {len(context_text)} characters of context.")

        # 2. Call the new handler function to generate the report from the context
        report_markdown = llm_handler.generate_report_from_text(
            topic=topic,
            context_text=context_text,
            api_keys=data.get('api_keys', {}),
            llm_provider=data.get('llm_provider', 'gemini'),
            model_name=data.get('llm_model_name')
        )
        
        if not report_markdown:
            logger.error("Report generation returned empty content.")
            raise ValueError("The AI model failed to generate the report structure.")

        logger.info(f"Successfully generated Markdown report for topic: '{topic}'")
        return jsonify({"status": "success", "report_markdown": report_markdown})

    except Exception as e:
        # This single catch block will now correctly handle all errors
        logger.error(f"An unexpected error occurred during report generation for topic '{topic}': {e}", exc_info=True)
        return create_error_response(f"An internal error occurred: {e}", 500)


@app.route('/generate_chat_response', methods=['POST'])
def generate_chat_response_route():
    logger.info("\n--- Received request at /generate_chat_response ---")
    data = request.get_json()
    
    user_id = data.get('user_id')
    current_user_query = data.get('query')
    if not user_id or not current_user_query:
        return create_error_response("Missing user_id or query in request", 400)

    # KG Integration, Routing, and Handler setup (No changes needed here)
    kg_entities = kg_service.extract_entities_and_relations(current_user_query)
    kg_facts = kg_service.query_kg(current_user_query)
    kg_context = f"\n[KG Entities]: {kg_entities}\n[KG Facts]: {kg_facts}\n"
    routing_decision = llm_router.route_query(current_user_query, data.get('llm_provider', config.DEFAULT_LLM_PROVIDER), data.get('llm_model_name'))
    final_provider = routing_decision['provider']
    final_model = routing_decision['model']
    logger.info(f"Router decision: Provider='{final_provider}', Model='{final_model}'")
    handler_kwargs = {
        'api_keys': data.get('api_keys', {}), 'model_name': final_model,
        'chat_history': data.get('chat_history', []), 'system_prompt': data.get('system_prompt'),
        'ollama_host': data.get('ollama_host')
    }

    # Initialize variables
    context_text_for_llm = "" # Start with empty context
    rag_references_for_client = []
    context_source = "None"
    final_answer = ""
    thinking_content = None

    # This is the main RAG and response generation logic block
    perform_rag = data.get('perform_rag', True)
    if perform_rag:
        # --- Multi-Query RAG Search (No changes here) ---
        logger.info("Performing multi-query RAG search on local documents...")
        queries_to_search = [current_user_query]
        if data.get('enable_multi_query', True):
            try:
                sub_queries = llm_handler.generate_sub_queries(original_query=current_user_query, llm_provider=final_provider, **handler_kwargs)
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
        
        # --- Relevance Check and Web Search Fallback (No changes here) ---
        is_relevant = False
        if docs_for_context:
            temp_context = "\n\n".join([doc.page_content for doc, score in docs_for_context])
            is_relevant = llm_handler.check_context_relevance(current_user_query, temp_context, **handler_kwargs)

        if docs_for_context and is_relevant:
            logger.info(f"Found {len(docs_for_context)} RELEVANT document chunks. Using RAG response function.")
            context_parts = [f"[{i+1}] Source: {d.metadata.get('documentName')}\n{d.page_content}" for i, (d, s) in enumerate(docs_for_context)]
            context_text_for_llm = "\n\n---\n\n".join(context_parts)
            rag_references_for_client = [{"documentName": d.metadata.get("documentName"), "score": float(s)} for d, s in docs_for_context]
            context_source = "Local Documents"
            
            # ✅ USE THE RAG-SPECIFIC RESPONSE FUNCTION
            final_answer, thinking_content = llm_handler.generate_response(
                llm_provider=final_provider, query=current_user_query, context_text=context_text_for_llm, **handler_kwargs
            )

        else: # This block handles cases where no relevant docs were found
            if docs_for_context:
                logger.info("Local documents found, but they were NOT RELEVANT to the query.")
            else:
                logger.info("No relevant local documents found.")
            
            # Special case for "no content found" message
            if data.get('active_file') and not docs_for_context:
                logger.info("Active file was selected, but no content found. Returning specific message.")
                final_answer = "[No content found for the selected PDF. Please check if the file was indexed correctly or try re-uploading.]"
                context_source = "PDF (not found)"
            else:
                # Fallback to web search OR direct chat
                logger.info("Attempting web search fallback...")
                try:
                    web_context = web_search.perform_search(current_user_query)
                    if web_context:
                        logger.info("Web search found relevant context. Using RAG response function.")
                        context_text_for_llm = web_context
                        context_source = "Web Search"
                        # ✅ USE THE RAG-SPECIFIC RESPONSE FUNCTION
                        final_answer, thinking_content = llm_handler.generate_response(
                            llm_provider=final_provider, query=current_user_query, context_text=context_text_for_llm, **handler_kwargs
                        )
                    else:
                        # If web search also fails, use the new direct chat function
                        logger.info("Web search found no context. Using direct conversational function.")
                        # ✅ USE THE NEW CONVERSATIONAL RESPONSE FUNCTION
                        final_answer, thinking_content = llm_handler.generate_chat_response(
                            llm_provider=final_provider, query=current_user_query, **handler_kwargs
                        )
                except Exception as e:
                    logger.error(f"Web search failed: {e}", exc_info=True)
                    # If web search throws an error, fall back to direct chat
                    logger.info("Web search failed. Using direct conversational function as final fallback.")
                    # ✅ USE THE NEW CONVERSATIONAL RESPONSE FUNCTION
                    final_answer, thinking_content = llm_handler.generate_chat_response(
                        llm_provider=final_provider, query=current_user_query, **handler_kwargs
                    )

    else: # This block runs if RAG is disabled entirely
        logger.info("RAG is disabled. Using direct conversational function.")
        # ✅ USE THE NEW CONVERSATIONAL RESPONSE FUNCTION
        final_answer, thinking_content = llm_handler.generate_chat_response(
            llm_provider=final_provider, query=current_user_query, **handler_kwargs
        )

    # --- Final Response to Client (No changes here) ---
    return jsonify({
        "llm_response": final_answer, "references": rag_references_for_client,
        "thinking_content": thinking_content, "status": "success",
        "provider_used": final_provider, "model_used": final_model,
        "context_source": context_source,
        "kg_entities": kg_entities,
        "kg_facts": kg_facts
    }), 200


@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    message = data['message']
    user_history_summary = data.get('user_history_summary', '')
    user_ollama_host = data.get('user_ollama_host')  # Optionally sent from frontend
    api_keys = data.get('api_keys', {})  # Optionally sent from frontend or loaded from settings
    model_name = data.get('model_name')

    # --- KG Integration: Extract entities/relations from user query and context ---
    kg_entities = kg_service.extract_entities_and_relations(message)
    kg_facts = kg_service.query_kg(message)
    kg_context = f"\n[KG Entities]: {kg_entities}\n[KG Facts]: {kg_facts}\n"

    system_prompt = f"""You are a helpful assistant. Here is some background about the user:\n{user_history_summary}\n{kg_context}\nRespond to their message: {message}\n"""

    try:
        response, provider_used = get_llm_response_with_fallback(
            prompt=message,
            is_chat=True,
            chat_history=[],  # Optionally pass chat history if available
            system_prompt=system_prompt,
            user_ollama_host=user_ollama_host,
            api_keys=api_keys,
            model_name=model_name
        )
        return jsonify({'response': response, 'provider': provider_used, 'kg_entities': kg_entities, 'kg_facts': kg_facts})
    except Exception as e:
        return create_error_response(f"All LLM providers failed: {e}")

@app.route('/extract_topics_from_file', methods=['POST'])
def extract_topics_from_file():
    data = request.get_json()
    file_path = data.get('file_path')
    chapter_prefix = data.get('chapter_prefix', '5.')  # Optional: for specific chapters

    if not file_path or not os.path.exists(file_path):
        return create_error_response("Valid file_path required", 400)

    try:
        text = file_parser.parse_file(file_path)
        if not text:
            return create_error_response("Unable to extract text from file.", 400)
        
        topics = file_parser.extract_headings(text, chapter_prefix=chapter_prefix)
        return jsonify({"topics": topics, "status": "success"}), 200
    except Exception as e:
        return create_error_response(f"Failed to extract topics: {e}", 500)

@app.route('/extract_headings', methods=['POST'])
def extract_headings_route():
    data = request.get_json()
    file_path = data.get('file_path')
    print(f"DEBUG: Received /extract_headings for file_path: {file_path}")
    if not file_path or not os.path.exists(file_path):
        print(f"ERROR: File not found for extraction: {file_path}")
        return jsonify({'status': 'error', 'error': 'File not found'}), 404
    try:
        text = file_parser.parse_file(file_path)
        print(f"DEBUG: Extracted text length: {len(text) if text else 0}")
        if not text:
            print(f"ERROR: Could not extract text from file: {file_path}")
            return jsonify({'status': 'error', 'error': 'Could not extract text from file'}), 400
        import re
        match = re.search(r'Chapter(\d+)', os.path.basename(file_path), re.IGNORECASE)
        chapter_prefix = f"{match.group(1)}." if match else ''
        headings = file_parser.extract_headings(text, chapter_prefix=chapter_prefix)
        print(f"DEBUG: Extracted headings: {headings}")
        return jsonify({'status': 'success', 'headings': headings}), 200
    except Exception as e:
        print(f"ERROR: Exception in /extract_headings: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500

@app.route('/extract_text', methods=['POST'])
def extract_text_route():
    data = request.get_json()
    file_path = data.get('file_path')
    print(f"DEBUG: Received /extract_text for file_path: {file_path}")
    if not file_path or not os.path.exists(file_path):
        print(f"ERROR: File not found for extraction: {file_path}")
        return jsonify({'status': 'error', 'error': 'File not found', 'file_path': file_path}), 404
    try:
        text = file_parser.parse_file(file_path)
        print(f"DEBUG: Extracted text length: {len(text) if text else 0}")
        if not text:
            print(f"ERROR: Could not extract text from file: {file_path}")
            # Try to debug PDF parsing specifically
            import traceback
            try:
                from . import file_parser as fp
                _, ext = os.path.splitext(file_path)
                if ext.lower() == '.pdf':
                    print("DEBUG: Attempting direct PDF parse for error details...")
                    pdf_text = fp.parse_pdf(file_path)
                    print(f"DEBUG: Direct PDF parse result: {pdf_text is not None}, length: {len(pdf_text) if pdf_text else 0}")
            except Exception as pdf_debug_err:
                print(f"DEBUG: Exception during direct PDF parse: {pdf_debug_err}")
                traceback.print_exc()
            return jsonify({'status': 'error', 'error': 'Could not extract text from file', 'file_path': file_path}), 400
        return jsonify({'status': 'success', 'text': text}), 200
    except Exception as e:
        import traceback
        print(f"ERROR: Exception in /extract_text: {e}")
        traceback.print_exc()
        return jsonify({'status': 'error', 'error': str(e), 'file_path': file_path}), 500

@app.route('/kg/extract', methods=['POST'])
def kg_extract():
    data = request.json
    text = data.get('text', '')
    if not text:
        return create_error_response('No text provided for KG extraction.', 400)
    result = kg_service.extract_entities_and_relations(text)
    return jsonify(result)

@app.route('/kg/query', methods=['POST'])
def kg_query():
    data = request.json
    query = data.get('query', '')
    if not query:
        return create_error_response('No query provided for KG search.', 400)
    result = kg_service.query_kg(query)
    return jsonify(result)

def get_llm_response_with_fallback(prompt, is_chat=True, chat_history=None, system_prompt=None, user_ollama_host=None, api_keys=None, model_name=None):
    """
    Try LLM providers in order: user Ollama, default Ollama, Gemini, Groq.
    Returns (response, provider_used).
    """
    from ai_core_service import llm_handler, config
    api_keys = api_keys or {}
    chat_history = chat_history or []
    # 1. Try user-specified Ollama
    if user_ollama_host:
        try:
            handler = llm_handler.get_handler(
                provider_name="ollama",
                api_keys=api_keys,
                model_name=model_name,
                ollama_host=user_ollama_host,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            response = handler.generate_response(prompt, is_chat=is_chat)
            return response, "ollama:user"
        except Exception as e:
            logger.warning(f"User Ollama failed: {e}")
    # 2. Try default Ollama
    try:
        handler = llm_handler.get_handler(
            provider_name="ollama",
            api_keys=api_keys,
            model_name=model_name,
            chat_history=chat_history,
            system_prompt=system_prompt
        )
        response = handler.generate_response(prompt, is_chat=is_chat)
        return response, "ollama:default"
    except Exception as e:
        logger.warning(f"Default Ollama failed: {e}")
    # 3. Try Gemini
    if api_keys.get('gemini'):
        try:
            handler = llm_handler.get_handler(
                provider_name="gemini",
                api_keys=api_keys,
                model_name=model_name,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            response = handler.generate_response(prompt, is_chat=is_chat)
            return response, "gemini"
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
    # 4. Try Groq
    if api_keys.get('grok'):
        try:
            handler = llm_handler.get_handler(
                provider_name="groq",
                api_keys=api_keys,
                model_name=model_name,
                chat_history=chat_history,
                system_prompt=system_prompt
            )
            response = handler.generate_response(prompt, is_chat=is_chat)
            return response, "groq"
        except Exception as e:
            logger.warning(f"Groq failed: {e}")
    raise RuntimeError("All LLM providers failed. Check configuration and connectivity.")

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