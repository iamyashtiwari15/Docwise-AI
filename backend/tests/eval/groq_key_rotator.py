"""
groq_key_rotator.py — Transparent Groq API key rotation
=========================================================
Reads GROQ_API_KEY_1 … GROQ_API_KEY_4 from the environment.
On a 429 rate-limit error it silently switches to the next available key
and retries the call — no sleep, no broken eval run.

Usage:
    from tests.eval.groq_key_rotator import RotatingGroqClient
    llm = RotatingGroqClient(model="llama-3.3-70b-versatile", temperature=0)
    result = llm.invoke("your prompt")   # same interface as ChatGroq
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger("groq_key_rotator")


def _load_keys() -> List[str]:
    """
    Collect all non-empty GROQ_API_KEY_1 … GROQ_API_KEY_4 from env.
    Falls back to GROQ_API_KEY if none of the numbered keys are set.
    """
    keys: List[str] = []
    for i in range(1, 5):
        k = os.getenv(f"GROQ_API_KEY_{i}", "").strip()
        if k:
            keys.append(k)

    if not keys:
        fallback = os.getenv("GROQ_API_KEY", "").strip()
        if fallback:
            keys.append(fallback)

    if not keys:
        raise ValueError(
            "No Groq API keys found. Set GROQ_API_KEY_1 … GROQ_API_KEY_4 "
            "(or GROQ_API_KEY) in your .env file."
        )

    logger.info("[ROTATOR] Loaded %d Groq API key(s)", len(keys))
    return keys


class RotatingGroqClient:
    """
    Drop-in replacement for ChatGroq that rotates API keys on 429.

    Supports the same .invoke(prompt) interface used in eval_runner.
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", temperature: float = 0):
        from langchain_groq import ChatGroq

        self._model = model
        self._temperature = temperature
        self._keys = _load_keys()
        self._index = 0          # current key index
        self._exhausted: set = set()   # indices of keys that are daily-exhausted
        self._clients: dict = {}  # key index → ChatGroq instance (lazy)

    def _get_client(self, index: int):
        from langchain_groq import ChatGroq
        if index not in self._clients:
            self._clients[index] = ChatGroq(
                model=self._model,
                temperature=self._temperature,
                api_key=self._keys[index],
            )
        return self._clients[index]

    def _rotate(self) -> bool:
        """
        Advance to the next non-exhausted key.
        Returns True if a new key is available, False if all are exhausted.
        """
        self._exhausted.add(self._index)
        for i in range(len(self._keys)):
            candidate = (self._index + 1 + i) % len(self._keys)
            if candidate not in self._exhausted:
                self._index = candidate
                logger.warning(
                    "[ROTATOR] Key #%d exhausted — switching to key #%d",
                    len(self._exhausted),
                    self._index + 1,
                )
                return True
        return False

    def invoke(self, prompt: str):
        """
        Invoke the LLM. On a 429 rate-limit error, rotate to the next key
        and retry automatically. Raises if all keys are exhausted.
        """
        while True:
            client = self._get_client(self._index)
            try:
                return client.invoke(prompt)
            except Exception as exc:
                err = str(exc)
                if "429" in err:
                    logger.warning(
                        "[ROTATOR] 429 on key #%d: %s",
                        self._index + 1,
                        err[:120],
                    )
                    if not self._rotate():
                        raise RuntimeError(
                            "All Groq API keys are rate-limited or exhausted. "
                            "Add more keys (GROQ_API_KEY_1 … GROQ_API_KEY_4) in .env "
                            "or wait for the daily limit to reset."
                        ) from exc
                    # Retry immediately with the new key
                else:
                    raise

    @property
    def active_key_index(self) -> int:
        return self._index + 1   # 1-based for display

    @property
    def keys_remaining(self) -> int:
        return len(self._keys) - len(self._exhausted)
