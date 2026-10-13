"""OpenAI requests for the fixed outcome judge, with retry handling."""

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

logger = logging.getLogger("openai_client")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIClient:
    """Minimal OpenAI chat client with retry/backoff; judge use only."""

    MAX_RETRIES = 5
    BACKOFF_BASE = 10  # seconds, doubles per retry on rate limits

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or OPENAI_API_KEY
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY required. Add it to .env "
                "(get a key at https://platform.openai.com/api-keys)."
            )
        self.total_calls = 0

    def judge(self, prompt: str, model: str = "gpt-4o-mini") -> str:
        """Single-turn judge call at temperature 0.0 (mirrors GeminiClient.judge)."""
        # OpenAI ids have no vendor prefix; tolerate an 'openai/' prefix if passed.
        if model.startswith("openai/"):
            model = model.split("/", 1)[1]
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
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
                time.sleep(0.3)  # gentle pacing
                return text
            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 * (attempt + 1)
                    logger.warning(f"Judge call failed ({e}); retrying in {wait}s")
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"OpenAI judge failed after {self.MAX_RETRIES} attempts: {e}"
                )
