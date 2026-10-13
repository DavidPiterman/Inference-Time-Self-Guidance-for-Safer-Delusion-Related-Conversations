"""Baseline

Read the scripted messages, generate one reply per turn, and score each reply.
The target receives the conversation history without additional guidance."""

import json
import time
from pathlib import Path
from datetime import datetime

from gemini_client import GeminiClient
from target_client import make_target_client
from judge import score_turn
from config import (
    TARGET_MODEL,
    TEST_CASES_PATH,
    RESULTS_DIR,
    DCS_START_TURN,
    HES_START_TURN,
    SIS_START_TURN,
)


def load_test_cases(path: Path = TEST_CASES_PATH) -> list[dict]:
    """Load test cases from psychosis-bench JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def run_scenario(
    client: GeminiClient,
    case: dict,
    model: str = TARGET_MODEL,
    system_prompt: str | None = None,
    verbose: bool = True,
) -> dict:
    """Run a single 12-turn scenario and score each turn.

    Args:
        client: Gemini API client
        case: Test case dict with 'name', 'prompts', 'harm_type', etc.
        model: Model to use
        system_prompt: Optional system prompt to prepend (used by treatment conditions)
        verbose: Print progress

    Returns:
        Dict with scenario results including all turns and scores.
    """
    case_name = case["name"]
    prompts = case["prompts"]
    harm_type = case.get("harm_type", "Unknown")

    if verbose:
        print(f"\n{'='*60}")
        print(f"Running: {case_name}")
        print(
            f"Theme: {case.get('theme', 'N/A')} | Condition: {case.get('condition', 'N/A')}"
        )
        print(f"Harm type: {harm_type}")
        print(f"{'='*60}")

    # Keep the live model history separate from the transcript used for scoring.
    messages = []
    turns = []

    for i, prompt in enumerate(prompts):
        turn_number = i + 1

        if verbose:
            print(f"\n--- Turn {turn_number} ---")
            print(f"User: {prompt[:100]}...")

        # Give every model the same scripted user turn.
        messages.append({"role": "user", "content": prompt})

        # Baseline adds no treatment guidance.
        try:
            response = client.chat(
                model=model,
                messages=messages,
                system_prompt=system_prompt,
            )
        except Exception as e:
            response = f"ERROR: {e}"
            if verbose:
                print(f"ERROR: {e}")

        # The reply becomes part of the history seen on the next turn.
        messages.append({"role": "assistant", "content": response})

        if verbose:
            print(f"Bot: {response[:150]}...")

        # Save an inspectable copy before applying the fixed score windows.
        turn_data = {
            "number": turn_number,
            "user_message": prompt,
            "bot_response": response,
        }
        turns.append(turn_data)

        # Score this turn
        scores = score_turn(
            client=client,
            turns=turns,
            harm_type=harm_type,
            turn_number=turn_number,
            dcs_start=DCS_START_TURN,
            hes_start=HES_START_TURN,
            sis_start=SIS_START_TURN,
        )
        turn_data["scores"] = {k: v["score"] for k, v in scores.items()}
        turn_data["score_explanations"] = {
            k: v["explanation"] for k, v in scores.items()
        }

        if verbose:
            score_str = " | ".join(
                f"{k.upper()}={v['score']}"
                for k, v in scores.items()
                if v["score"] >= 0
            )
            if score_str:
                print(f"Scores: {score_str}")

    # Compute summary statistics
    valid_dcs = [t["scores"]["dcs"] for t in turns if t["scores"]["dcs"] >= 0]
    valid_hes = [t["scores"]["hes"] for t in turns if t["scores"]["hes"] >= 0]
    valid_sis = [t["scores"]["sis"] for t in turns if t["scores"]["sis"] >= 0]

    summary = {
        "avg_dcs": sum(valid_dcs) / len(valid_dcs) if valid_dcs else 0,
        "avg_hes": sum(valid_hes) / len(valid_hes) if valid_hes else 0,
        "total_sis": sum(valid_sis),
        "num_turns": len(turns),
    }

    if verbose:
        print(f"\n--- Summary for {case_name} ---")
        print(f"Avg DCS: {summary['avg_dcs']:.2f}")
        print(f"Avg HES: {summary['avg_hes']:.2f}")
        print(f"Total SIS: {summary['total_sis']}")

    return {
        "case": {
            "name": case_name,
            "theme": case.get("theme", ""),
            "condition": case.get("condition", ""),
            "harm_type": harm_type,
        },
        "model": model,
        "turns": turns,
        "summary": summary,
    }


def run_baseline(
    cases: list[dict] | None = None,
    model: str = TARGET_MODEL,
    verbose: bool = True,
    output_dir: Path | None = None,
) -> list[dict]:
    """Run baseline (control condition) on all scenarios.

    Args:
        cases: Test cases to run (loads default if None)
        model: Model to test
        verbose: Print progress
        output_dir: Where to save results

    Returns:
        List of scenario results
    """
    client = make_target_client(model)

    if cases is None:
        cases = load_test_cases()

    output_dir = output_dir or RESULTS_DIR / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for i, case in enumerate(cases):
        # Skip already completed cases (resume support)
        result_file = output_dir / f"case_{case['id']}.json"
        if result_file.exists():
            print(f"\n[{i+1}/{len(cases)}] SKIPPING {case['name']} (already done)")
            try:
                with open(result_file, encoding="utf-8") as f:
                    all_results.append(json.load(f))
            except UnicodeDecodeError:
                with open(result_file, encoding="cp1252") as f:
                    all_results.append(json.load(f))
            continue

        print(f"\n[{i+1}/{len(cases)}] Processing {case['name']}...")
        result = run_scenario(client, case, model=model, verbose=verbose)
        all_results.append(result)

        # Save incrementally
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    # Save complete results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"baseline_results_{timestamp}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model": model,
                "timestamp": timestamp,
                "condition": "baseline",
                "num_cases": len(all_results),
                "results": all_results,
                "overall_summary": compute_overall_summary(all_results),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n{'='*60}")
    print("BASELINE COMPLETE")
    print(f"Results saved to: {summary_file}")
    overall = compute_overall_summary(all_results)
    print(f"Overall Avg DCS: {overall['avg_dcs']:.2f}")
    print(f"Overall Avg HES: {overall['avg_hes']:.2f}")
    print(f"Overall Avg SIS: {overall['avg_sis_per_scenario']:.2f}")
    print(f"{'='*60}")

    return all_results


def compute_overall_summary(results: list[dict]) -> dict:
    """Compute aggregate statistics across all scenarios."""
    all_dcs, all_hes, all_sis = [], [], []
    for r in results:
        for t in r["turns"]:
            if t["scores"]["dcs"] >= 0:
                all_dcs.append(t["scores"]["dcs"])
            if t["scores"]["hes"] >= 0:
                all_hes.append(t["scores"]["hes"])
            if t["scores"]["sis"] >= 0:
                all_sis.append(t["scores"]["sis"])

    sis_per_scenario = [r["summary"]["total_sis"] for r in results]

    return {
        "avg_dcs": sum(all_dcs) / len(all_dcs) if all_dcs else 0,
        "avg_hes": sum(all_hes) / len(all_hes) if all_hes else 0,
        "avg_sis_per_scenario": (
            sum(sis_per_scenario) / len(sis_per_scenario) if sis_per_scenario else 0
        ),
        "total_scored_turns_dcs": len(all_dcs),
        "total_scored_turns_hes": len(all_hes),
        "total_scored_turns_sis": len(all_sis),
    }


if __name__ == "__main__":
    run_baseline(verbose=True)
