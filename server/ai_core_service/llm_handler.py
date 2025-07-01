# server/ai_core_service/llm_handler.py (Refined with fallback logic + full functionality)
import os
import logging
from abc import ABC, abstractmethod

# SDK imports
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
try:
    import ollama
    ollama_available = True
except ImportError:
    ollama_available = False

# Local imports
try:
    from . import config as service_config
except ImportError:
    import config as service_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Prompt templates
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
EXECUTE NOW based on the rules."""

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

# Utility: Parse LLM output into answer + reasoning
def _parse_thinking_and_answer(full_llm_response: str):
    response_text = full_llm_response.strip()
    cot_start_tag = "**Chain of Thought:**"
    answer_start_tag = "**Answer:**"
    cot_index = response_text.find(cot_start_tag)
    answer_index = response_text.find(answer_start_tag)
    if cot_index != -1 and answer_index != -1:
        thinking = response_text[cot_index + len(cot_start_tag):answer_index].strip()
        answer = response_text[answer_index + len(answer_start_tag):].strip()
        return answer, thinking
    return response_text, None

# Base Handler
class BaseLLMHandler(ABC):
    def __init__(self, api_keys, model_name=None, **kwargs):
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
    def generate_response(self, prompt, is_chat=True): pass

    def analyze_document(self, document_text: str, analysis_type: str) -> str:
        prompt_template = _ANALYSIS_PROMPT_TEMPLATES.get(analysis_type)
        if not prompt_template:
            raise ValueError(f"Invalid analysis type: {analysis_type}")
        doc_text_for_llm = document_text[:service_config.ANALYSIS_MAX_CONTEXT_LENGTH]
        num_items = min(5 + (len(doc_text_for_llm) // 4000), 20)
        final_prompt = prompt_template.format(doc_text_for_llm=doc_text_for_llm, num_items=num_items)
        return self.generate_response(final_prompt, is_chat=False)

# Provider Handlers
class GeminiHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not genai:
            raise ConnectionError("Gemini SDK missing.")
    def _configure_client(self):
        genai.configure(api_key=self.api_keys.get('gemini'))
    def generate_response(self, prompt, is_chat=True):
        model = genai.GenerativeModel(self.model_name or "gemini-1.5-flash")
        return model.generate_content(prompt).text

class GroqHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not Groq:
            raise ConnectionError("Groq SDK missing.")
    def _configure_client(self):
        self.client = Groq(api_key=self.api_keys.get('groq'))
    def generate_response(self, prompt, is_chat=True):
        messages = [{"role": "user", "content": prompt}]
        return self.client.chat.completions.create(messages=messages, model=self.model_name).choices[0].message.content

class OllamaHandler(BaseLLMHandler):
    def _validate_sdk(self):
        if not ChatOllama:
            raise ConnectionError("Ollama SDK missing.")
    def _configure_client(self):
        base_url = self.api_keys.get("ollama_host", "http://localhost:11434")
        self.client = ChatOllama(base_url=base_url, model=self.model_name)
    def generate_response(self, prompt, is_chat=True):
        messages = [HumanMessage(content=prompt)]
        return self.client.invoke(messages).content

# Provider map
PROVIDER_MAP = {
    "gemini": GeminiHandler,
    "groq": GroqHandler,
    "ollama": OllamaHandler
}

def get_handler(provider_name, **kwargs):
    handler_cls = PROVIDER_MAP.get(provider_name)
    if not handler_cls:
        raise ValueError(f"Unsupported LLM provider: {provider_name}")
    return handler_cls(**kwargs)

# Generate with fallback across providers
def generate_with_fallback(query, context_text, api_keys, history=[], system_prompt=None):
    providers = [
        ("ollama", "llama3"),
        ("groq", "llama3-8b-8192"),
        ("gemini", "gemini-1.5-flash")
    ]
    final_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context_text=context_text)

    for provider, model in providers:
        try:
            handler = get_handler(
                provider_name=provider,
                api_keys=api_keys,
                model_name=model,
                system_prompt=system_prompt,
                chat_history=history
            )
            raw_response = handler.generate_response(final_prompt, is_chat=True)
            answer, thinking = _parse_thinking_and_answer(raw_response)
            return answer, thinking, provider
        except Exception as e:
            logger.warning(f"[Fallback] {provider.upper()} failed: {e}")

    return "Sorry, all AI services are currently down. Try again later.", None, "none"

# Relevance check utility
def check_context_relevance(query: str, context: str, **kwargs) -> bool:
    logger.info("Performing relevance check on retrieved context...")
    try:
        handler = get_handler(
            provider_name="groq",
            api_keys=kwargs.get('api_keys', {}),
            model_name="llama3-8b-8192"
        )
        prompt = _RELEVANCE_CHECK_PROMPT_TEMPLATE.format(query=query, context=context)
        raw_response = handler.generate_response(prompt, is_chat=False)
        return raw_response.strip().lower().startswith('yes')
    except Exception as e:
        logger.error(f"Context relevance check failed: {e}. Defaulting to 'relevant'.")
        return True

# Sub-query decomposition
def generate_sub_queries(original_query: str, llm_provider: str, num_queries: int = 3, **kwargs) -> list[str]:
    logger.info(f"Generating sub-queries for: '{original_query[:50]}...'")
    try:
        utility_kwargs = kwargs.copy()
        utility_kwargs.pop('chat_history', None)
        utility_kwargs.pop('system_prompt', None)

        handler = get_handler(provider_name=llm_provider, **utility_kwargs)
        prompt = _SUB_QUERY_TEMPLATE.format(original_query=original_query, num_queries=num_queries)
        raw_response = handler.generate_response(prompt, is_chat=False)
        return [q.strip() for q in raw_response.strip().split('\n') if q.strip()][:num_queries]
    except Exception as e:
        logger.error(f"Failed to generate sub-queries: {e}", exc_info=True)
        return []

# Normal chat generation using specific provider
def generate_response(llm_provider: str, query: str, context_text: str, **kwargs) -> tuple[str, str | None]:
    logger.info(f"Generating CHAT response with provider: {llm_provider}.")
    final_prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(query=query, context_text=context_text)
    handler = get_handler(provider_name=llm_provider, **kwargs)
    raw_response = handler.generate_response(final_prompt, is_chat=True)
    return _parse_thinking_and_answer(raw_response)

# Document analysis wrapper
def perform_document_analysis(document_text: str, analysis_type: str, llm_provider: str, **kwargs) -> tuple[str | None, str | None]:
    logger.info(f"Performing '{analysis_type}' analysis with {llm_provider}.")
    handler = get_handler(provider_name=llm_provider, **kwargs)
    analysis_result = handler.analyze_document(document_text, analysis_type)
    return analysis_result, None
