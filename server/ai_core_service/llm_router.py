# FusedChatbot/server/ai_core_service/llm_router.py

import logging

logger = logging.getLogger(__name__)

# --- Corrected Router Configuration ---

# Define our specialized models (these are Ollama models)
DEEPSEEK_CODER_MODEL = "deepseek-coder"
QWEN_TECHNICAL_MODEL = "qwen:7b" # Corrected example name for Ollama
LLAMA3_CHAT_MODEL = "llama3-8b-8192" # This is a Groq model

# ==================================================================
#  DEFINITIVE FIX: Correctly map intents to the OLLAMA provider
# ==================================================================
INTENT_TO_MODEL_MAP = {
    "coding_assistance": {
        "provider": "ollama", # CORRECT: This should be Ollama
        "model": DEEPSEEK_CODER_MODEL
    },
    "technical_explanation": {
        "provider": "ollama", # CORRECT: This should be Ollama
        "model": QWEN_TECHNICAL_MODEL
    },
    "general_chat": {
        "provider": "groq",   # Groq is great for fast, general chat
        "model": LLAMA3_CHAT_MODEL
    },
    # Default fallback if no other intent is matched
    "default": {
        "provider": "groq",
        "model": LLAMA3_CHAT_MODEL
    }
}
# ==================================================================

# Map keywords to intents. The first match in the list wins.
KEYWORD_TO_INTENT_MAP = [
    # Coding-related keywords
    ({"code", "python", "javascript", "script", "function", "debug", "error"}, "coding_assistance"),
    # Technical explanation keywords
    ({"explain", "what is", "how does", "technical", "architecture", "database"}, "technical_explanation"),
    # General conversation starters (less likely to be hit if there's a real question)
    ({"hi", "hello", "how are you"}, "general_chat"),
]


def route_query(query: str, default_provider: str, default_model: str) -> dict:
    """
    Analyzes a query and routes it to the best LLM provider and model.
    """
    logger.info(f"Routing query: '{query[:50]}...'")
    query_lower = query.lower()
    query_words = set(query_lower.split())

    for keywords, intent in KEYWORD_TO_INTENT_MAP:
        if not keywords.isdisjoint(query_words):
            logger.info(f"Intent matched: '{intent}'. Routing to configured model.")
            return INTENT_TO_MODEL_MAP[intent]

    logger.info("No specific intent matched. Using default provider from UI or system default.")
    if default_provider:
        return {
            "provider": default_provider,
            "model": default_model or INTENT_TO_MODEL_MAP["default"]["model"]
        }
    
    return INTENT_TO_MODEL_MAP["default"]