# FusedChatbot/server/ai_core_service/llm_handler.py
import os
import logging
from abc import ABC, abstractmethod

# --- SDK Imports ---
try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig
except ImportError:
    genai = None
try:
    from groq import Groq
except ImportError:
    Groq = None
try:
    from langchain_ollama import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
except ImportError:
    ChatOllama, HumanMessage, SystemMessage, AIMessage = None, None, None, None

# --- Local Imports ---
try:
    from . import config as service_config
except ImportError:
    import config

logger = logging.getLogger(__name__)

ollama_available = bool(ChatOllama and HumanMessage)


# --- Prompt Templates (Full versions for stability) ---
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

_ANALYSIS_PROMPT_TEMPLATES = {
    "faq": """You are a data processing machine. Your only function is to extract questions and answers from the provided text.
**CRITICAL RULES:**
1.  **FORMAT:** Your output MUST strictly follow the `Q: [Question]\nA: [Answer]` format for each item.
2.  **NO PREAMBLE:** Your entire response MUST begin directly with `Q:`. Do not output any other text.
3.  **DATA SOURCE:** Base all questions and answers ONLY on the provided document text.
4.  **QUANTITY:** Generate approximately {num_items} questions.
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
EXECUTE NOW.""",
    "topics": """You are a document analysis specialist. Your task is to identify the main topics from the provided text and give a brief explanation for each. From the context below, identify the top {num_items} most important topics. For each topic, provide a single-sentence explanation.
Context:
---
{doc_text_for_llm}
---
Format the output as a numbered list. Example:
1. **Topic Name:** A brief, one-sentence explanation.
""",
    "mindmap": """You are an expert text-to-Mermaid-syntax converter. Your only job is to create a valid Mermaid.js mind map from the provided text. Your entire response MUST begin with the word `mindmap` and contain PURE Mermaid syntax.
--- START DOCUMENT TEXT ---
{doc_text_for_llm}
--- END DOCUMENT TEXT ---
EXECUTE NOW. CREATE THE MERMAID MIND MAP."""
}

_SUB_QUERY_TEMPLATE = """You are an AI assistant skilled at query decomposition. Your task is to break down a complex user question into {num_queries} simpler, self-contained sub-questions that can be answered independently by a search engine.
**CRITICAL RULES:**
1.  **ONLY OUTPUT THE QUESTIONS:** Do not include any preamble, numbering, or explanation.
2.  **ONE QUESTION PER LINE:** Each of the sub-questions must be on a new line.

**ORIGINAL USER QUERY:**
"{original_query}"

**SUB-QUESTIONS (One per line):**
"""

_RELEVANCE_CHECK_PROMPT_TEMPLATE = """You are a relevance-checking AI. Your task is to determine if the provided 'CONTEXT' is useful for answering the 'USER QUERY'.
Respond with only 'Yes' or 'No'. Do not provide any other explanation.

USER QUERY: "{query}"

CONTEXT:
---
{context}
---

Is the context relevant to the query? Answer Yes or No.
"""

def _parse_thinking_and_answer(full_llm_response: str) -> tuple[str, str | None]:
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
        else:
            return response_text, None
    return response_text, None

class BaseLLMHandler(ABC):
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
    def generate_response(self, prompt: str, is_chat: bool = True) -> str: pass
    
    def analyze_document(self, document_text: str, analysis_type: str) -> str:
        prompt_template = _ANALYSIS_PROMPT_TEMPLATES.get(analysis_type)
        if not prompt_template: raise ValueError(f"Invalid analysis type: {analysis_type}")
        doc_text_for_llm = document_text[:service_config.ANALYSIS_MAX_CONTEXT_LENGTH]
        num_items = min(5 + (len(doc_text_for_llm) // 4000), 20)
        final_prompt = prompt_template.format(doc_text_for_llm=doc_text_for_llm, num_items=num_items)
        return self.generate_response(final_prompt, is_chat=False)

class GeminiHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not genai: raise ConnectionError("Gemini SDK (google.generativeai) not installed.")
    def _configure_client(self):
        gemini_key = self.api_keys.get('gemini')
        if not gemini_key: raise ValueError("Gemini API key not found.")
        genai.configure(api_key=gemini_key)
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        system_instruction = self.kwargs.get('system_prompt') if is_chat else None
        client = genai.GenerativeModel(self.model_name or os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash"),
            generation_config=GenerationConfig(temperature=0.7),
            system_instruction=system_instruction)
        if is_chat:
            history = self.kwargs.get('chat_history', [])
            history_for_api = [{'role': 'user' if msg.get('role') == 'user' else 'model', 'parts': [msg.get('parts', [{}])[0].get('text', "")]} for msg in history if msg.get('parts', [{}])[0].get('text')]
            chat_session = client.start_chat(history=history_for_api)
            response = chat_session.send_message(prompt)
        else:
            response = client.generate_content(prompt)
        return response.text

class GroqHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not Groq: raise ConnectionError("Groq SDK not installed.")
    def _configure_client(self):
        grok_key = self.api_keys.get('grok')
        if not grok_key: raise ValueError("Groq API key not found.")
        self.client = Groq(api_key=grok_key)
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        if is_chat:
            if system_prompt := self.kwargs.get('system_prompt'):
                messages.append({"role": "system", "content": system_prompt})
            history = self.kwargs.get('chat_history', [])
            messages.extend([{'role': 'assistant' if msg.get('role') == 'model' else 'user', 'content': msg.get('parts', [{}])[0].get('text', "")} for msg in history])
        messages.append({"role": "user", "content": prompt})
        completion = self.client.chat.completions.create(messages=messages, model=self.model_name or os.getenv("DEFAULT_GROQ_LLAMA3_MODEL", "llama3-8b-8192"))
        return completion.choices[0].message.content

class OllamaHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatOllama: raise ConnectionError("Ollama SDK (langchain_ollama) not installed.")
    def _configure_client(self):
        host = self.kwargs.get('ollama_host') or self.api_keys.get("ollama_host") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = ChatOllama(base_url=host, model=self.model_name or os.getenv("DEFAULT_OLLAMA_MODEL", "llama3"))
    def generate_response(self, prompt: str, is_chat: bool = True) -> str:
        messages = []
        if is_chat:
            if system_prompt := self.kwargs.get('system_prompt'):
                messages.append(SystemMessage(content=system_prompt))
            history = self.kwargs.get('chat_history', [])
            messages.extend([AIMessage(content=msg.get('parts', [{}])[0].get('text', "")) if msg.get('role') == 'model' else HumanMessage(content=msg.get('parts', [{}])[0].get('text', "")) for msg in history])
        messages.append(HumanMessage(content=prompt))
        response = self.client.invoke(messages)
        return response.content

PROVIDER_MAP = {"gemini": GeminiHandler, "groq": GroqHandler, "ollama": OllamaHandler}

def get_handler(provider_name: str, **kwargs) -> BaseLLMHandler:
    handler_class = next((handler for key, handler in PROVIDER_MAP.items() if provider_name.startswith(key)), None)
    if not handler_class: raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return handler_class(**kwargs)

# ==================================================================
#  DEFINITIVE FIX: Correct the relevance check function
# ==================================================================
def check_context_relevance(query: str, context: str, **kwargs) -> bool:
    """
    Uses a fast LLM to check if the retrieved context is relevant to the user's query.
    """
    logger.info("Performing relevance check on retrieved context...")
    try:
        # Use a hardcoded, fast provider for the relevance check to prevent errors.
        # We pass the api_keys from the original call.
        handler = get_handler(
            provider_name="groq", 
            api_keys=kwargs.get('api_keys', {}),
            model_name="llama3-8b-8192" # Use a specific, fast model
        )
        
        prompt = _RELEVANCE_CHECK_PROMPT_TEMPLATE.format(query=query, context=context)
        
        # This is a utility task, not a chat. Set is_chat=False
        raw_response = handler.generate_response(prompt, is_chat=False)
        
        decision = raw_response.strip().lower()
        logger.info(f"Relevance check decision: '{decision}'")
        
        # Check for 'yes' at the beginning of the string for robustness
        return decision.startswith('yes')
    except Exception as e:
        logger.error(f"Context relevance check failed: {e}. Defaulting to 'relevant'.")
        # Default to true to avoid breaking the chain if the check fails.
        return True
# ==================================================================

def generate_sub_queries(original_query: str, llm_provider: str, num_queries: int = 3, **kwargs) -> list[str]:
    """Generates sub-queries for multi-query RAG."""
    logger.info(f"Generating sub-queries for: '{original_query[:50]}...'")
    try:
        utility_kwargs = kwargs.copy()
        utility_kwargs.pop('chat_history', None)
        utility_kwargs.pop('system_prompt', None)

        handler = get_handler(provider_name=llm_provider, **utility_kwargs)
        prompt = _SUB_QUERY_TEMPLATE.format(original_query=original_query, num_queries=num_queries)
        raw_response = handler.generate_response(prompt, is_chat=False)
        sub_queries = [q.strip() for q in raw_response.strip().split('\n') if q.strip()]
        logger.info(f"Generated {len(sub_queries)} sub-queries.")
        return sub_queries[:num_queries]
    except Exception as e:
        logger.error(f"Failed to generate sub-queries: {e}", exc_info=True)
        return []

def generate_response(llm_provider: str, query: str, context_text: str, **kwargs) -> tuple[str, str | None]:
    logger.info(f"Generating CHAT response with provider: {llm_provider}.")
    final_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context_text=context_text)
    handler = get_handler(provider_name=llm_provider, **kwargs)
    raw_response = handler.generate_response(final_prompt, is_chat=True)
    return _parse_thinking_and_answer(raw_response)

def perform_document_analysis(document_text: str, analysis_type: str, llm_provider: str, **kwargs) -> tuple[str | None, str | None]:
    logger.info(f"Performing '{analysis_type}' analysis with {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    analysis_result = handler.analyze_document(document_text, analysis_type)
    return analysis_result, None