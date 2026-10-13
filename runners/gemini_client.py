"""Gemini requests for target replies and guidance generation.

Includes retries, response timing and request logging."""

import time
import logging
from pathlib import Path
from typing import Optional
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, API_DELAY, TARGET_TEMPERATURE

# Keep a local audit of calls without mixing it into the experimental results.
LOG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "api_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("gemini_client")


class GeminiClient:
    """Client for Google AI Studio (Gemini) API with rate limit protection."""

    MAX_RETRIES = 5
    RATE_LIMIT_BACKOFF_BASE = 10  # seconds, doubles each retry

    def __init__(self, api_key: Optional[str] = None):
        key = api_key or GOOGLE_API_KEY
        if not key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        self.client = genai.Client(api_key=key)
        self.total_calls = 0
        self.total_errors = 0
        self.total_rate_limits = 0
        self.latencies = []

    def chat(
        self,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        temperature: float = TARGET_TEMPERATURE,
    ) -> str:
        """Send a chat request with retry logic and rate limit handling.

        Args:
            model: Gemini model name (e.g., 'gemini-2.5-flash')
            messages: List of dicts with 'role' ('user'/'model') and 'content' keys
            system_prompt: Optional system instruction
            temperature: Sampling temperature

        Returns:
            Response text string

        Raises:
            RuntimeError: After all retries exhausted
        """
        # Translate the shared runner format into Gemini's role names.
        contents = []
        for msg in messages:
            role = msg["role"]
            if role == "assistant":
                role = "model"
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part(text=msg["content"])],
                )
            )

        config = types.GenerateContentConfig(
            temperature=temperature,
        )
        if system_prompt:
            config.system_instruction = system_prompt

        # Retry transient failures; permanent failure remains visible to the runner.
        for attempt in range(self.MAX_RETRIES):
            start_time = time.time()
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                # An empty or blocked response is not a valid experimental reply.
                # Retry it instead of allowing ``None`` into the saved transcript.
                if response.text is None:
                    raise ValueError(
                        "Empty response (no text part; possibly blocked candidate)"
                    )

                latency = time.time() - start_time
                self.total_calls += 1
                self.latencies.append(latency)

                # Log every 10 calls
                if self.total_calls % 10 == 0:
                    avg_latency = sum(self.latencies[-10:]) / min(
                        10, len(self.latencies)
                    )
                    logger.info(
                        f"API stats: calls={self.total_calls}, "
                        f"avg_latency={avg_latency:.1f}s, "
                        f"errors={self.total_errors}, "
                        f"rate_limits={self.total_rate_limits}"
                    )

                time.sleep(API_DELAY)
                return response.text

            except Exception as e:
                latency = time.time() - start_time
                error_str = str(e).lower()

                # Check if rate limited (429)
                if (
                    "429" in str(e)
                    or "rate" in error_str
                    or "quota" in error_str
                    or "resource" in error_str
                ):
                    self.total_rate_limits += 1
                    backoff = self.RATE_LIMIT_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"RATE LIMITED (attempt {attempt+1}/{self.MAX_RETRIES}). "
                        f"Waiting {backoff}s before retry. Error: {e}"
                    )
                    time.sleep(backoff)
                    continue

                # Other errors
                self.total_errors += 1
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 * (attempt + 1)
                    logger.warning(
                        f"API error (attempt {attempt+1}/{self.MAX_RETRIES}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                    continue

                logger.error(f"API failed after {self.MAX_RETRIES} attempts: {e}")
                raise RuntimeError(
                    f"API request failed after {self.MAX_RETRIES} attempts: {e}"
                )

        # A fully exhausted retry loop must fail explicitly rather than return None.
        raise RuntimeError(
            f"API request failed after {self.MAX_RETRIES} attempts "
            f"(all rate-limited / quota exhausted)"
        )

    def judge(self, prompt: str, model: str) -> str:
        """Send a single-turn judge prompt.

        Args:
            prompt: The evaluation prompt
            model: Model to use for judging

        Returns:
            Judge response text
        """
        return self.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )

    def get_stats(self) -> dict:
        """Get API usage statistics."""
        return {
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "total_rate_limits": self.total_rate_limits,
            "avg_latency": (
                sum(self.latencies) / len(self.latencies) if self.latencies else 0
            ),
            "max_latency": max(self.latencies) if self.latencies else 0,
            "min_latency": min(self.latencies) if self.latencies else 0,
        }
