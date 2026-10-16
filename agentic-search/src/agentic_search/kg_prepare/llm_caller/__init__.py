# -*- coding: utf-8 -*-
"""
LLM Caller Module

Provides a singleton LLMCaller class that creates OpenAI-compatible
LLM handlers based on model configuration.

All LLM calls go through the standard OpenAI Chat Completions API
(``client.chat.completions.create``), which is compatible with:
- OpenAI (GPT-4o, o1, ...)
- DeepSeek API
- Any OpenAI-compatible endpoint (vLLM, SGLang, Ollama, LiteLLM, ...)
"""

from ..config.model_config import model_config
from .direct_llm_handler import DirectLLMHandler


# Handler registry – only the OpenAI-compatible handler is supported.
_HANDLERS = {
    "DirectLLMHandler": DirectLLMHandler,
}


class LLMCaller:
    """
    Singleton class for accessing LLM handlers.

    Usage::

        caller = LLMCaller()
        handler = caller["direct_dsr1"]
        content, reasoning = handler.call_llm(messages)
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMCaller, cls).__new__(cls)
        return cls._instance

    def __getitem__(self, model_name: str):
        """Get a handler instance for the specified model."""
        if model_name not in model_config:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models: {list(model_config.keys())}"
            )

        config = model_config[model_name]
        handler_name = config.get("handler", "DirectLLMHandler")

        if handler_name not in _HANDLERS:
            raise ValueError(
                f"Unknown handler: {handler_name}. "
                f"Available handlers: {list(_HANDLERS.keys())}"
            )

        return _HANDLERS[handler_name](config)

    def list_models(self):
        """List all available model names."""
        return list(model_config.keys())


__all__ = ["LLMCaller", "DirectLLMHandler"]
