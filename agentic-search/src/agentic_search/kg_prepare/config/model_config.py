# -*- coding: utf-8 -*-
"""
LLM Model Configuration for KG Subgraph Extraction

All models use the standard OpenAI Chat Completions API via ``openai.OpenAI``.
This is compatible with any OpenAI-compatible provider:
  - OpenAI (https://api.openai.com/v1)
  - DeepSeek (https://api.deepseek.com)
  - Together AI, Fireworks, Groq, etc.
  - Self-hosted vLLM / SGLang / Ollama endpoints

Configure your API credentials via environment variables:
  export LLM_API_KEY="sk-..."
  export LLM_BASE_URL="https://api.openai.com/v1"   # optional
  export LLM_MODEL="gpt-4o"                          # optional
"""

import os

# ---------------------------------------------------------------------------
# Default configuration – uses environment variables so that no secrets are
# hard-coded.  Override LLM_BASE_URL for non-OpenAI providers.
# ---------------------------------------------------------------------------
model_config = dict(
    # ── Default entry (used by --llm-caller direct_dsr1) ──────────────────
    direct_dsr1={
        "payload": {
            "model": os.getenv("LLM_MODEL", "deepseek-reasoner"),
            "messages": None,           # filled at call time
            "stream": False,
            "max_tokens": 1024,
            "temperature": 0.6,
        },
        "api_key": os.getenv("LLM_API_KEY", "YOUR_API_KEY"),
        "base_url": os.getenv("LLM_BASE_URL", "https://api.deepseek.com"),
        "handler": "DirectLLMHandler",  # uses openai.OpenAI under the hood
    },
)

# ---------------------------------------------------------------------------
# To add more model entries, append to ``model_config`` above or create a
# local override file.  Examples:
#
#   # OpenAI GPT-4o
#   model_config["openai_gpt4o"] = {
#       "payload": {"model": "gpt-4o", "messages": None, "stream": False,
#                   "max_tokens": 1024, "temperature": 0.6},
#       "api_key": "sk-...",
#       "base_url": "https://api.openai.com/v1",
#       "handler": "DirectLLMHandler",
#   }
#
#   # Local vLLM / SGLang server
#   model_config["local_llm"] = {
#       "payload": {"model": "Qwen/Qwen2.5-72B-Instruct", "messages": None,
#                   "stream": False, "max_tokens": 1024, "temperature": 0.6},
#       "api_key": "dummy",
#       "base_url": "http://localhost:8000/v1",
#       "handler": "DirectLLMHandler",
#   }
# ---------------------------------------------------------------------------
