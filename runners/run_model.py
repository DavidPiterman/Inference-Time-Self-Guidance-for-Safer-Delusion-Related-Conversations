"""Run the selected model and conditions.

Set the target model before loading the configuration so each model writes to
its own output folder."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence


CONDITIONS = ("baseline", "treatment_a", "treatment_b", "treatment_c")


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PsychosisBench target model.")
    parser.add_argument(
        "conditions",
        nargs="*",
        metavar="condition",
        help="baseline, treatment_a, treatment_b or treatment_c. Omit to run all four in study order.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the resolved model, settings, and output path without API calls.",
    )
    options = parser.parse_args(arguments)
    # checked here rather than with choices=: Python 3.11 rejects an empty list
    unknown = [c for c in options.conditions if c not in CONDITIONS]
    if unknown:
        parser.error(f"unknown condition(s): {', '.join(unknown)}")
    return options


def run_target_model(model_id: str, arguments: Sequence[str] | None = None) -> None:
    """Select the model, show the consequential settings, and run conditions."""
    os.environ["TARGET_MODEL"] = model_id
    options = parse_arguments(arguments)
    selected_conditions = options.conditions or list(CONDITIONS)

    from config import (
        JUDGE_MODEL,
        PIPELINE_MODEL,
        RESULTS_DIR,
        TARGET_MODEL,
        TARGET_TEMPERATURE,
    )

    model_output_dir = RESULTS_DIR / TARGET_MODEL.replace("/", "-")
    normalized_model = TARGET_MODEL.lower().split("/", 1)[-1]
    if normalized_model.startswith("gpt-5"):
        target_temperature = "omitted (provider-controlled)"
        pipeline_temperature = "omitted (provider-controlled)"
    else:
        target_temperature = str(TARGET_TEMPERATURE)
        pipeline_temperature = "0.0"

    print("=" * 72)
    print(f"TARGET MODEL   : {TARGET_MODEL}")
    print(f"PIPELINE MODEL : {PIPELINE_MODEL} (follows the target)")
    print(f"JUDGE MODEL    : {JUDGE_MODEL} (fixed across all runs)")
    print(f"TARGET TEMP.   : {target_temperature}")
    print(f"PIPELINE TEMP. : {pipeline_temperature}")
    print("JUDGE TEMP.    : 0.0")
    print(f"CONDITIONS     : {', '.join(selected_conditions)}")
    print(f"OUTPUT         : {model_output_dir}")
    print("=" * 72)

    if options.dry_run:
        print("RESULT: dry run completed; no provider call was made")
        return

    from baseline_runner import run_baseline
    from treatment_a_runner import run_treatment_a
    from treatment_b_runner import run_treatment_b
    from treatment_c_runner import run_treatment_c

    runners = {
        "baseline": lambda: run_baseline(output_dir=model_output_dir / "baseline"),
        "treatment_a": lambda: run_treatment_a(
            output_dir=model_output_dir / "treatment_a"
        ),
        "treatment_b": lambda: run_treatment_b(
            output_dir=model_output_dir / "treatment_b"
        ),
        "treatment_c": lambda: run_treatment_c(
            output_dir=model_output_dir / "treatment_c"
        ),
    }

    for condition in CONDITIONS:
        if condition not in selected_conditions:
            continue
        print(f"\n########## {condition} ##########")
        runners[condition]()

    print(f"\nRESULT: completed requested conditions for {TARGET_MODEL}")
