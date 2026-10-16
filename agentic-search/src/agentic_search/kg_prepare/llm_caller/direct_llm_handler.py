# -*- coding: utf-8 -*-
"""
OpenAI-compatible LLM Handler

Calls any LLM that exposes the OpenAI Chat Completions API:
  - OpenAI, DeepSeek, Together AI, Fireworks, Groq, ...
  - Self-hosted vLLM, SGLang, Ollama, LiteLLM, ...

Uses the official ``openai`` Python SDK (``pip install openai``).

Features:
  - Multiple API keys for round-robin load balancing
  - Thread-safe (deep-copies payload per request)
  - Automatic retry with exponential back-off
  - Configurable base URL and timeout
"""

import copy
import random
import time
from typing import List, Dict, Any, Tuple, Optional

from openai import OpenAI


class DirectLLMHandler:
    """
    OpenAI-compatible LLM handler.

    Wraps ``openai.OpenAI.chat.completions.create()``.
    """

    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Configuration dict with keys:
                - api_key (str | list[str]): API key(s).
                - payload (dict): Base payload passed to
                  ``chat.completions.create()`` (model, temperature, …).
                - base_url (str, optional): API base URL.
                  Defaults to ``https://api.openai.com/v1``.
                - timeout (int, optional): Request timeout in seconds
                  (default 120).
        """
        api_key_config = config.get("api_key", "")
        if isinstance(api_key_config, list):
            self.api_keys = api_key_config
        else:
            self.api_keys = [api_key_config]

        self._base_payload = copy.deepcopy(config.get("payload", {}))
        self.timeout = config.get("timeout", 120)
        self.base_url = config.get("base_url") or self.DEFAULT_BASE_URL

    # ------------------------------------------------------------------
    def _get_client(self) -> OpenAI:
        """Return a new ``OpenAI`` client with a randomly chosen API key."""
        return OpenAI(
            api_key=random.choice(self.api_keys),
            base_url=self.base_url,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------
    def call_llm(
        self,
        messages: List[Dict[str, str]],
        max_retry: int = 3,
        **kwargs,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Call the Chat Completions API.

        Args:
            messages: ``[{"role": "user", "content": "..."}]``
            max_retry: Number of retries on transient failure.
            **kwargs: Extra keys merged into the request payload.

        Returns:
            ``(content, reasoning_content)`` — *reasoning_content* is
            ``None`` for models that do not expose a separate reasoning
            field (most models).

        Raises:
            RuntimeError: After exhausting all retries.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, max_retry + 1):
            # Thread-safe: fresh copy per request
            payload = copy.deepcopy(self._base_payload)
            payload["messages"] = messages
            payload.update(kwargs)

            client = self._get_client()
            try:
                completion = client.chat.completions.create(**payload)

                if not completion or not getattr(completion, "choices", None):
                    raise ValueError("LLM returned empty response")

                msg = completion.choices[0].message
                if msg is None:
                    raise ValueError("LLM response missing message")

                content = msg.content
                # Some reasoning models (e.g. DeepSeek-R1) return a
                # separate ``reasoning_content`` field.
                reasoning_content = getattr(msg, "reasoning_content", None)
                return content, reasoning_content

            except Exception as err:
                last_error = err
                print(
                    f"DirectLLMHandler error "
                    f"(attempt {attempt}/{max_retry}): {err}"
                )
                if attempt < max_retry:
                    time.sleep(2 ** (attempt - 1))  # exponential back-off

        raise RuntimeError(
            f"DirectLLMHandler failed after {max_retry} retries"
        ) from last_error
