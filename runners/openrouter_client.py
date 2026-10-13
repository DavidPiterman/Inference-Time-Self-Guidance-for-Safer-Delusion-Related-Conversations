"""OpenRouter access to the fixed outcome judge."""

import os
import time
import logging
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

logger = logging.getLogger("openrouter_client")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# The permanent judge standard: the original paper's judge (Au Yeung et al.,
# 2025, §3.3). Single source of truth is config.JUDGE_MODEL.
from config import JUDGE_MODEL as STANDARD_JUDGE_MODEL


class OpenRouterClient:
    """Minimal OpenRouter chat client with retry/backoff; judge use only."""

    MAX_RETRIES = 5
    BACKOFF_BASE = 10  # seconds, doubles per retry on rate limits

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or OPENROUTER_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY required. Add it to .env "
                "(get a key at https://openrouter.ai/keys)."
            )
        self.total_calls = 0

    def judge(self, prompt: str, model: str = STANDARD_JUDGE_MODEL) -> str:
        """Single-turn judge call at temperature 0.0 (mirrors GeminiClient.judge)."""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.MAX_RETRIES):
            try:
                resp = requests.post(
                    API_URL, json=payload, headers=headers, timeout=120
                )
                if resp.status_code == 429:
                    wait = self.BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"Rate limited; waiting {wait}s (attempt {attempt+1})"
                    )
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                if text is None or not str(text).strip():
                    raise ValueError("Empty judge response")
                self.total_calls += 1
                time.sleep(0.5)  # gentle pacing
                return text
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"Judge call failed ({e}); retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"OpenRouter judge failed after {self.MAX_RETRIES} attempts: {e}"
                )
