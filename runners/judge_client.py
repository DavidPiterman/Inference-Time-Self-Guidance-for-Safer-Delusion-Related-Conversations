"""Connect to the fixed outcome judge through OpenAI or OpenRouter."""

import os

from config import JUDGE_MODEL as STANDARD_JUDGE_MODEL  # re-export for callers


def make_judge_client():
    """Return the external judge client matching the available API key."""
    if os.getenv("OPENAI_API_KEY"):
        from openai_client import OpenAIClient

        return OpenAIClient()
    if os.getenv("OPENROUTER_API_KEY"):
        from openrouter_client import OpenRouterClient

        return OpenRouterClient()
    raise ValueError(
        "No judge API key found. Add OPENAI_API_KEY (direct) or "
        "OPENROUTER_API_KEY (reseller) to .env — the standardized judge "
        f"({STANDARD_JUDGE_MODEL}) cannot run without one."
    )
