"""Target-model clients for Gemini, GPT-5 and DeepSeek.

chat() generates a reply; judge() is the interface used for guidance steps.
GPT-5 sampling fields are omitted. DeepSeek thinking is disabled."""

import os
import time
import logging
from pathlib import Path
from typing import Optional

import requests

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from config import TARGET_MODEL, TARGET_TEMPERATURE, API_DELAY

logger = logging.getLogger("target_client")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"

DEEPSEEK_THINKING_DISABLED = {"type": "disabled"}
DEEPSEEK_MAX_TOKENS = 8192

# Allow room for internal reasoning and the visible answer.
MAX_COMPLETION_TOKENS = 16000


def _is_openai_reasoning(model: str) -> bool:
    """GPT-5 family and the o-series are reasoning models. `gpt-5-chat-latest`
    is the NON-reasoning chat variant, so exclude any id containing 'chat'."""
    m = model.lower()
    if m.startswith("openai/"):
        m = m.split("/", 1)[1]
    if "chat" in m:
        return False
    return m.startswith("gpt-5") or m.startswith(("o1", "o3", "o4"))


class OpenAITargetClient:
    """OpenAI chat client for the TARGET/PIPELINE role (mirrors GeminiClient's
    surface). Handles reasoning models transparently."""

    MAX_RETRIES = 5
    RATE_LIMIT_BACKOFF_BASE = 10  # seconds, doubles each retry

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = OPENAI_API_URL,
        provider: str = "openai",
    ):
        self.provider = provider  # "openai" | "deepseek" (both OpenAI-compatible)
        self.api_url = base_url
        self.api_key = api_key or (
            DEEPSEEK_API_KEY if provider == "deepseek" else OPENAI_API_KEY
        )
        if not self.api_key:
            envvar = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
            raise ValueError(
                f"{envvar} required for a {provider} target. Add it to .env."
            )
        self.total_calls = 0
        self.total_errors = 0
        self.total_rate_limits = 0
        self.latencies: list[float] = []

    # -- low-level completion with retry/backoff -------------------------------
    def _complete(
        self, model: str, messages: list[dict], temperature: Optional[float]
    ) -> str:
        if "/" in model:  # tolerate a 'openai/'/'deepseek/' vendor prefix
            model = model.split("/", 1)[1]

        payload: dict = {"model": model, "messages": messages}
        if self.provider == "deepseek":
            # DeepSeek V4 is a thinking model line; force thinking OFF so the
            # target is a NON-reasoning model, comparable to the paper's
            # deepseek-chat-v3.1. Uses max_tokens (not max_completion_tokens).
            payload["thinking"] = DEEPSEEK_THINKING_DISABLED
            payload["max_tokens"] = DEEPSEEK_MAX_TOKENS
            if temperature is not None:
                payload["temperature"] = temperature
        elif _is_openai_reasoning(model):
            # OpenAI reasoning models: large completion budget, DEFAULT temp only.
            payload["max_completion_tokens"] = MAX_COMPLETION_TOKENS
        else:
            if temperature is not None:
                payload["temperature"] = temperature

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # DeepSeek normally returns within seconds; a 600-second socket wait can
        # freeze a case for nearly an hour across retries. Keep the long timeout
        # only for OpenAI reasoning targets, where it is genuinely required.
        request_timeout = 120 if self.provider == "deepseek" else 600

        for attempt in range(self.MAX_RETRIES):
            start = time.time()
            try:
                # 600s (not 300): GPT-5 deep reasoning inside Treatment C's
                # rollouts can legitimately exceed 5 min on some turns and was
                # crashing the run on repeated read-timeouts (2026-07-12).
                resp = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=request_timeout
                )
                if resp.status_code == 429:
                    self.total_rate_limits += 1
                    backoff = self.RATE_LIMIT_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"RATE LIMITED (attempt {attempt+1}/"
                        f"{self.MAX_RETRIES}). Waiting {backoff}s."
                    )
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"]
                # Empty content = reasoning drained the budget (shouldn't happen
                # with our large cap) or a blocked/empty candidate. Treat as
                # transient so the retry loop gets another attempt (mirrors
                # GeminiClient's empty-response handling).
                if text is None or not str(text).strip():
                    finish = data["choices"][0].get("finish_reason")
                    raise ValueError(f"Empty response (finish_reason={finish})")
                latency = time.time() - start
                self.total_calls += 1
                self.latencies.append(latency)
                if self.total_calls % 10 == 0:
                    avg = sum(self.latencies[-10:]) / min(10, len(self.latencies))
                    logger.info(
                        f"API stats: calls={self.total_calls}, "
                        f"avg_latency={avg:.1f}s, errors={self.total_errors}, "
                        f"rate_limits={self.total_rate_limits}"
                    )
                time.sleep(API_DELAY)
                return text
            except Exception as e:
                err = str(e).lower()
                if "429" in str(e) or "rate" in err or "quota" in err:
                    self.total_rate_limits += 1
                    backoff = self.RATE_LIMIT_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        f"Rate/quota (attempt {attempt+1}); waiting {backoff}s: {e}"
                    )
                    time.sleep(backoff)
                    continue
                self.total_errors += 1
                if attempt < self.MAX_RETRIES - 1:
                    wait = 2 * (attempt + 1)
                    logger.warning(
                        f"OpenAI target error (attempt {attempt+1}/"
                        f"{self.MAX_RETRIES}): {e}. Retrying in {wait}s."
                    )
                    time.sleep(wait)
                    continue
                logger.error(f"OpenAI target failed after {self.MAX_RETRIES}: {e}")
                raise RuntimeError(
                    f"OpenAI target request failed after {self.MAX_RETRIES} attempts: {e}"
                )

        raise RuntimeError(
            f"OpenAI target request failed after {self.MAX_RETRIES} attempts "
            f"(all rate-limited / quota exhausted)"
        )

    # -- GeminiClient-compatible surface --------------------------------------
    def chat(
        self,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        temperature: float = TARGET_TEMPERATURE,
    ) -> str:
        """Multi-turn target response. `messages` use 'user'/'assistant' roles
        (OpenAI-native; no remap needed). A system_prompt becomes a leading
        system message (byte-identical Baseline still passes system_prompt=None)."""
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            role = "assistant" if m["role"] in ("assistant", "model") else "user"
            msgs.append({"role": role, "content": m["content"]})
        return self._complete(model, msgs, temperature=temperature)

    def judge(self, prompt: str, model: str) -> str:
        """Single-turn call used by the treatment pipeline stages (model =
        PIPELINE_MODEL == TARGET_MODEL). temperature=0.0 is requested but is
        dropped automatically for reasoning models."""
        return self._complete(
            model, [{"role": "user", "content": prompt}], temperature=0.0
        )

    def get_stats(self) -> dict:
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


def make_target_client(model: Optional[str] = None):
    """Return the right client for the TARGET/PIPELINE role, routed by model id.
    Gemini ids -> GeminiClient (Google); OpenAI ids (gpt-*/o-series, with or
    without an 'openai/' prefix) -> OpenAITargetClient."""
    model = model or TARGET_MODEL
    m = model.lower()
    if m.startswith(("gemini", "google/")):
        from gemini_client import GeminiClient

        return GeminiClient()
    if m.startswith("deepseek"):
        # DeepSeek's API is OpenAI-compatible; reuse the same client pointed at
        # DeepSeek's endpoint, with thinking forced off (see _complete).
        return OpenAITargetClient(base_url=DEEPSEEK_API_URL, provider="deepseek")
    if m.startswith(("gpt", "openai/", "o1", "o3", "o4", "chatgpt")):
        return OpenAITargetClient()
    raise ValueError(
        f"Unsupported target model: {model}. Use a Gemini, GPT-5 or DeepSeek launcher."
    )
