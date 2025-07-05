# FusedChatbot/server/ai_core_service/llm_handler.py

import os
import logging
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Any

# --- SDK Imports ---
# Use httpx to catch specific network errors
import httpx

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from groq import Groq as GroqClient
    from langchain_groq import ChatGroq
except ImportError:
    GroqClient = None

try:
    import ollama
    from langchain_ollama.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    ollama, ChatOllama, HumanMessage, SystemMessage, AIMessage = None, None, None, None, None

# --- Local Imports ---
try:
    from . import config as service_config
except ImportError:
    import config as service_config

logger = logging.getLogger(__name__)

# --- Prompt Templates (No changes needed here) ---
# ... (keep all your existing _SYNTHESIS_PROMPT_TEMPLATE, _ANALYSIS_PROMPT_TEMPLATES, etc.) ...
_SYNTHESIS_PROMPT_TEMPLATE = """You are a helpful AI assistant. Your behavior depends entirely on whether 'CONTEXT' is provided.
**RULE 1: ANSWER FROM CONTEXT**
If the 'CONTEXT' section below is NOT empty, you MUST base your answer *only* on the information within that context.
- Your response MUST begin with a "**Chain of Thought:**" section explaining which parts of the context you used.
- Following the Chain of Thought, provide the final answer under an "**Answer:**" section.
**RULE 2: ANSWER FROM GENERAL KNOWLEDGE**
If the 'CONTEXT' section below IS empty, you MUST act as a general knowledge assistant.
- Answer the user's 'QUERY' directly and conversationally.
- Do NOT mention context.
- Do NOT include a "Chain of Thought" or "Answer" section.
---
**CONTEXT:**
{context_text}
---
**QUERY:**
{query}
---
EXECUTE NOW based on the rules.
"""
_REPORT_GENERATION_PROMPT_TEMPLATE = """You are a professional research analyst and technical writer. Your sole task is to generate a comprehensive, well-structured report on a given topic. You must base your report *exclusively* on the provided context from web search results.

**CRITICAL RULES:**
1.  **Strictly Use Context:** You MUST base your entire report on the information found in the "SEARCH RESULTS CONTEXT" section below. Do not use any external or prior knowledge.
2.  **Markdown Formatting:** The entire output MUST be in valid, clean Markdown format. Use headings (e.g., `#`, `##`, `###`), bold text, bullet points, and numbered lists to create a readable and professional document.
3.  **Report Structure:** The report must follow this exact structure, section by section:
    - A main title: `# Report: {topic}`
    - `## 1. Executive Summary`: A brief, high-level paragraph summarizing the most critical aspects of the topic and the key conclusions of the report.
    - `## 2. Key Findings`: A bulleted list that concisely presents the most important points, data, or facts discovered in the context (aim for 3-5 distinct bullet points).
    - `## 3. Detailed Analysis`: A more in-depth section expanding on the key findings. This should be the longest part of the report. Use subheadings (e.g., `### Sub-Topic 1`, `### Sub-Topic 2`) for clarity and to organize different facets of the analysis.
    - `## 4. Conclusion`: A concluding paragraph that summarizes the analysis and provides a final, overarching takeaway.
    - `## 5. Sources Used`: A numbered list of the sources from the context that were used to build the report. You MUST cite which information came from which source in the analysis section using footnotes like `[1]`, `[2]`, etc.

---
**SEARCH RESULTS CONTEXT:**
{context_text}
---
**TOPIC TO REPORT ON:**
{topic}
---
GENERATE THE MARKDOWN REPORT NOW.
"""
# ... (and all your other prompt templates)


def _parse_thinking_and_answer(full_llm_response: str) -> tuple[str, str | None]:
    # ... (This function remains unchanged)
    response_text = full_llm_response.strip()
    cot_start_tag = "**Chain of Thought:**"
    answer_start_tag = "**Answer:**"
    cot_start_index = response_text.find(cot_start_tag)
    if cot_start_index != -1:
        answer_start_index = response_text.find(answer_start_tag, cot_start_index)
        if answer_start_index != -1:
            thinking_content = response_text[cot_start_index + len(cot_start_tag):answer_start_index].strip()
            answer = response_text[answer_start_index + len(answer_start_tag):].strip()
            return answer, thinking_content
    return response_text, None


# --- Abstract Base Class (Unchanged) ---
class BaseLLMHandler(ABC):
    # ... (This class remains unchanged)
    def __init__(self, api_keys: dict, model_name: str = None, **kwargs):
        self.api_keys = api_keys
        self.model_name = model_name
        self.kwargs = kwargs
        self._validate_sdk()
        self._configure_client()

    @abstractmethod
    def _validate_sdk(self): pass
    @abstractmethod
    def _configure_client(self): pass
    @abstractmethod
    def invoke(self, messages: List[Dict[str, Any]]) -> str: pass


# --- Provider-Specific Handlers with Refactored invoke methods ---

# CORRECTED version
class GeminiHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not genai: raise ImportError("Gemini SDK (google-generativeai) is not installed.")
    def _configure_client(self):
        gemini_key = self.api_keys.get('gemini') or os.getenv('ADMIN_GEMINI_API_KEY')
        if not gemini_key: raise ValueError("Gemini API key is missing.")
        genai.configure(api_key=gemini_key)

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        system_instruction = messages[0]['content'] if messages and messages[0]['role'] == 'system' else None
        
        # --- START OF FIX ---
        # Translate our standard message format to Gemini's required format.
        gemini_formatted_messages = []
        for msg in messages:
            # Skip the system message as it's handled separately
            if msg['role'] == 'system':
                continue
            
            # Gemini uses 'model' for the assistant role
            role = 'model' if msg['role'] in ['assistant', 'model'] else 'user'
            
            gemini_formatted_messages.append({
                'role': role,
                'parts': [{'text': msg['content']}] # Gemini expects content inside a 'parts' array
            })
        # --- END OF FIX ---

        model = genai.GenerativeModel(
            self.model_name or "gemini-1.5-flash",
            system_instruction=system_instruction
        )
        
        # Pass the correctly formatted messages to the API
        response = model.generate_content(gemini_formatted_messages)
        return response.text

class GroqHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not GroqClient: raise ImportError("Groq SDK is not installed.")
    def _configure_client(self):
        groq_key = self.api_keys.get('groq') or os.getenv('ADMIN_GROQ_API_KEY')
        if not groq_key: raise ValueError("Groq API key is missing.")
        self.client = ChatGroq(model_name=self.model_name or "llama3-8b-8192", groq_api_key=groq_key)

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        # Translate to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg['role'] == 'system': lc_messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user': lc_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant' or msg['role'] == 'model': lc_messages.append(AIMessage(content=msg['content']))
        
        response = self.client.invoke(lc_messages)
        return response.content

class OllamaHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatOllama: raise ImportError("Ollama SDK (langchain-ollama) is not installed.")
    def _configure_client(self):
        host = self.kwargs.get('ollama_host') or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ChatOllama(base_url=host, model=self.model_name or "llama3")

    def invoke(self, messages: List[Dict[str, Any]]) -> str:
        # Translate to LangChain messages (same as Groq)
        lc_messages = []
        for msg in messages:
            if msg['role'] == 'system': lc_messages.append(SystemMessage(content=msg['content']))
            elif msg['role'] == 'user': lc_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant' or msg['role'] == 'model': lc_messages.append(AIMessage(content=msg['content']))
        
        response = self.client.invoke(lc_messages)
        return response.content

PROVIDER_MAP = {"gemini": GeminiHandler, "groq": GroqHandler, "ollama": OllamaHandler}

def get_handler(provider_name: str, **kwargs) -> BaseLLMHandler:
    handler_class = PROVIDER_MAP.get(provider_name)
    if not handler_class: raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return handler_class(**kwargs)

def _build_message_list(prompt: str, is_chat: bool, **kwargs) -> List[Dict[str, Any]]:
    """Helper to construct the standard message list."""
    messages = []
    if is_chat and (system_prompt := kwargs.get('system_prompt')):
        messages.append({"role": "system", "content": system_prompt})
    
    if is_chat and (history := kwargs.get('chat_history', [])):
        for msg in history:
            role = 'assistant' if msg.get('role') == 'model' else 'user'
            content = " ".join([part.get('text', "") for part in msg.get('parts', []) if part.get('text')])
            if content:
                messages.append({'role': role, 'content': content})
    
    messages.append({"role": "user", "content": prompt})
    return messages


# --- CORE FUNCTION WITH FALLBACK LOGIC ---

# def _execute_with_fallback(prompt_template: str, prompt_data: Dict, provider_preference: str, **kwargs) -> str:
#     """
#     Executes an LLM call with a defined provider fallback order.
    
#     Args:
#         prompt_template (str): The f-string template for the prompt.
#         prompt_data (Dict): The data to format into the template.
#         provider_preference (str): The user's preferred provider (e.g., 'ollama').
#         **kwargs: Other arguments for the handlers.

#     Returns:
#         The string response from the successful LLM call.
        
#     Raises:
#         RuntimeError: If all providers in the fallback chain fail.
#     """
#     # 1. Define the full, ordered chain of providers.
#     fallback_chain = ['ollama', 'gemini', 'groq']
    
#     # 2. Create the effective order, prioritizing the user's preference.
#     if provider_preference in fallback_chain:
#         fallback_chain.remove(provider_preference)
#     provider_order = [provider_preference] + fallback_chain
#     provider_order = list(dict.fromkeys(provider_order)) # Remove duplicates
    
#     logger.info(f"Executing LLM call with provider fallback order: {provider_order}")

#     final_prompt = prompt_template.format(**prompt_data)
#     messages = _build_message_list(final_prompt, is_chat=kwargs.get('is_chat', True), **kwargs)

#     last_error = None
#     # 3. Loop through the providers until one succeeds.
#     for provider in provider_order:
#         try:
#             logger.info(f"--> Attempting with provider: [{provider.upper()}]")
#             handler = get_handler(provider_name=provider, **kwargs)
#             response_content = handler.invoke(messages)

#             if response_content and isinstance(response_content, str):
#                 logger.info(f"<-- Success with provider: [{provider.upper()}]")
#                 return response_content # Success!

#         except (httpx.ConnectTimeout, httpx.ConnectError, ConnectionRefusedError, ollama.ResponseError if ollama else Exception) as e:
#             logger.warning(f"--> Provider [{provider.upper()}] failed with a connection error: {e}. Trying next provider.")
#             last_error = e
#         except Exception as e:
#             logger.error(f"--> Provider [{provider.upper()}] failed unexpectedly: {e}", exc_info=True)
#             last_error = e
#             # Continue to the next provider

#     # 4. If the loop finishes, all providers have failed.
#     logger.critical("All LLM providers in the fallback chain failed.")
#     raise RuntimeError(f"All providers failed. Last error: {last_error}")
# FusedChatbot/server/ai_core_service/llm_handler.py

# ... (all imports and other functions are the same) ...

# CORRECTED VERSION of the executor function
import hashlib
import json

try:
    import redis
    redis_client = redis.StrictRedis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    CACHE_ENABLED = True
except ImportError:
    redis_client = None
    CACHE_ENABLED = False
except Exception as e:
    logger.error(f"Redis client initialization failed: {e}")
    redis_client = None
    CACHE_ENABLED = False

def _execute_with_fallback(prompt_template: str, prompt_data: Dict, provider_preference: str, **kwargs) -> str:
    """
    Executes an LLM call with a defined provider fallback order and Redis caching.
    """
    # --- 1. Clean up and separate arguments ---
    # Pop 'is_chat' from kwargs to avoid the TypeError. Default to True if not present.
    is_chat = kwargs.pop('is_chat', True)
    
    # Now, `kwargs` no longer contains 'is_chat', so we can safely use **kwargs later.
    
    # 2. Prepare the prompt and messages
    final_prompt = prompt_template.format(**prompt_data)
    # Pass the separated `is_chat` argument directly.
    messages = _build_message_list(final_prompt, is_chat=is_chat, **kwargs)

    # 3. Caching logic (this remains the same)
    if CACHE_ENABLED and redis_client:
        cache_payload = {
            "messages": messages, "model": kwargs.get('model_name', 'default'),
            "provider_preference": provider_preference
        }
        cache_key = f"llm_cache:{hashlib.sha256(json.dumps(cache_payload, sort_keys=True).encode()).hexdigest()}"
        try:
            cached_response = redis_client.get(cache_key)
            if cached_response:
                logger.info(f"CACHE HIT for key: {cache_key}")
                # redis returns bytes, decode to string
                return cached_response.decode() if isinstance(cached_response, bytes) else cached_response
        except Exception as e:
            logger.error(f"Redis GET failed: {e}. Proceeding without cache.")

    logger.info(f"CACHE MISS. Executing LLM call...")

    # 4. Fallback logic (this remains the same)
    provider_order = ['ollama', 'gemini', 'groq']
    if provider_preference in provider_order:
        provider_order.remove(provider_preference)
    provider_order = [provider_preference] + list(dict.fromkeys(provider_order))
    
    logger.info(f"Effective provider fallback order: {provider_order}")

    last_error = None
    for provider in provider_order:
        try:
            logger.info(f"--> Attempting with provider: [{provider.upper()}]")
            # Pass the clean kwargs dictionary without the conflicting key
            handler = get_handler(provider_name=provider, **kwargs)
            response_content = handler.invoke(messages)

            if response_content and isinstance(response_content, str):
                logger.info(f"<-- Success with provider: [{provider.upper()}]")
                if CACHE_ENABLED:
                    try:
                        redis_client.set(cache_key, response_content, ex=86400)
                        logger.info(f"CACHE SET for key: {cache_key}")
                    except Exception as e:
                        logger.error(f"Redis SET failed: {e}")
                return response_content

        except (httpx.ConnectTimeout, httpx.ConnectError, ConnectionRefusedError, ollama.ResponseError if ollama else Exception) as e:
            logger.warning(f"--> Provider [{provider.upper()}] failed with a connection error: {e}. Trying next provider.")
            last_error = e
        except Exception as e:
            logger.error(f"--> Provider [{provider.upper()}] failed unexpectedly: {e}", exc_info=True)
            last_error = e

    logger.critical("All LLM providers in the fallback chain failed.")
    raise RuntimeError(f"All providers failed. Last error: {last_error}")

# ... (the rest of your file remains the same) ...

# --- Public-Facing Functions (Now using the fallback executor) ---

def generate_report_from_text(topic: str, context_text: str, **kwargs) -> str:
    """Uses the fallback executor to synthesize a report."""
    logger.info(f"Synthesizing Markdown report for topic: '{topic}'")
    provider_preference = kwargs.get('llm_provider', 'ollama')
    
    return _execute_with_fallback(
        prompt_template=_REPORT_GENERATION_PROMPT_TEMPLATE,
        prompt_data={'topic': topic, 'context_text': context_text},
        provider_preference=provider_preference,
        is_chat=False,
        **kwargs
    )

def generate_response(llm_provider: str, query: str, context_text: str, **kwargs) -> tuple[str, str | None]:
    """Generates a RAG-based response using the fallback executor."""
    logger.info(f"Generating RAG response with preferred provider: {llm_provider}.")
    raw_response = _execute_with_fallback(
        prompt_template=_SYNTHESIS_PROMPT_TEMPLATE,
        prompt_data={'query': query, 'context_text': context_text},
        provider_preference=llm_provider,
        is_chat=True,
        **kwargs
    )
    return _parse_thinking_and_answer(raw_response)
    
def generate_chat_response(llm_provider: str, query: str, **kwargs) -> tuple[str, str | None]:
    """Generates a direct chat response using the fallback executor."""
    logger.info(f"Generating conversational response with preferred provider: {llm_provider}.")
    # For direct chat, the prompt template is just the query itself.
    raw_response = _execute_with_fallback(
        prompt_template="{query}",
        prompt_data={'query': query},
        provider_preference=llm_provider,
        is_chat=True,
        **kwargs
    )
    return raw_response, None

# ... (keep your other utility functions like check_context_relevance, etc. They can also be updated to use the fallback if needed)

