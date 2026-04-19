"""
core/llm_factory.py — Centralized LLM factory for the Agentic OS.
Handles switching between local Ollama and cloud-based Gemini API.
"""

import os
from typing import Any, Optional
from dotenv import load_dotenv

from core.logging_config import get_logger

# Load environment variables from .env if present
load_dotenv()

logger = get_logger(__name__)

_llm_cache: dict[str, Any] = {}

def get_llm(
    model_override: Optional[str] = None,
    temperature: float = 0.0,
) -> Any:
    """
    Returns a configured LangChain ChatModel based on environment variables.
    
    Environment Variables:
        LLM_PROVIDER: "ollama" (default) or "gemini"
        OLLAMA_DEFAULT_MODEL: default model if provider is ollama
        OLLAMA_BASE_URL: default ollama endpoint
        GEMINI_DEFAULT_MODEL: default model if provider is gemini (e.g. gemini-2.5-flash)
        GOOGLE_API_KEY: required if provider is gemini
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").lower()
    
    if provider == "gemini":
        # If model_override looks like a local model (contains :), ignore it and use Gemini default
        if model_override and ":" in model_override:
            model = os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3-flash-preview")
        else:
            model = model_override or os.getenv("GEMINI_DEFAULT_MODEL", "gemini-3-flash-preview")
        
        cache_key = f"gemini:{model}:{temperature}"
        
        if cache_key not in _llm_cache:
            if not os.getenv("GOOGLE_API_KEY"):
                logger.warning("GOOGLE_API_KEY is not set, Gemini instantiation may fail.")
                
            from langchain_google_genai import ChatGoogleGenerativeAI
            _llm_cache[cache_key] = ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
            )
        return _llm_cache[cache_key]
        
    else:
        # Default to Ollama
        model = model_override or os.getenv("OLLAMA_DEFAULT_MODEL", "llama3.2:3b")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        cache_key = f"ollama:{model}:{base_url}:{temperature}"
        
        if cache_key not in _llm_cache:
            from langchain_ollama import ChatOllama
            _llm_cache[cache_key] = ChatOllama(
                model=model,
                temperature=temperature,
                base_url=base_url,
            )
        return _llm_cache[cache_key]
