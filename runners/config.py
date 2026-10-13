"""Shared settings used by every runner.

Model roles (target, guidance model and fixed judge), response temperature,
outcome scoring windows, and where scenarios are read and raw runs are written."""

import os
from pathlib import Path


# 1. Repository paths
EXPERIMENT_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = EXPERIMENT_DIR.parent

try:
    from dotenv import load_dotenv

    load_dotenv(REPOSITORY_DIR / ".env")
except ImportError:
    pass

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")


# 2. Target, guidance model and outcome judge
TARGET_MODEL = os.getenv("TARGET_MODEL", "deepseek-v4-flash")

# Keep the same judge across models and conditions.
JUDGE_MODEL = "gpt-4o-mini-2024-07-18"

# The target model also generates the guidance.
PIPELINE_MODEL = TARGET_MODEL


# 3. Input and output files
TEST_CASES_PATH = Path(
    os.getenv("TEST_CASES_PATH", REPOSITORY_DIR / "data" / "test_cases.json")
)
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", REPOSITORY_DIR / "outputs" / "raw_runs"))


# 4. Outcome scoring windows
DCS_START_TURN = 4
HES_START_TURN = 7
SIS_START_TURN = 7

API_DELAY = 1.0

# Target temperature for Gemini and DeepSeek. GPT-5 omits this field.
TARGET_TEMPERATURE = 1.0
