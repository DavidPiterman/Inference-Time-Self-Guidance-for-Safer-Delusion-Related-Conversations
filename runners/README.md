# Generate new conversations

The Python files contain the model clients, condition implementations and prompts for generating new conversations. To analyze the recorded study conversations, use the [analysis notebook](../analysis/PsychosisBench_Analysis.ipynb) with the two supplied benchmark CSVs.

## 1. Install and add your keys

From the repository root:

```bash
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env` in the repository root and fill in the keys for your run:

| Target model | Required keys |
|---|---|
| Gemini 2.5 Pro | `GOOGLE_API_KEY`, plus `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for scoring |
| GPT-5 | `OPENAI_API_KEY` for responses and scoring |
| DeepSeek V4 Flash | `DEEPSEEK_API_KEY`, plus `OPENAI_API_KEY` or `OPENROUTER_API_KEY` for scoring |

The outcome judge is `gpt-4o-mini-2024-07-18` for every model and condition. If both judge keys are supplied, OpenAI is used. Credentials are read from `.env`, which is ignored by Git. Live runs use API credits.

## 2. Choose a model

Set `TEST_CASES_PATH` in `.env` to a scenario JSON file. Its top-level `cases` array contains objects with `id`, `name` and `prompts` fields; `prompts` is the ordered list of scripted user messages. The scenario JSON is not included in this repository.

Run one of these commands from the repository root:

```bash
python runners/run_gemini_2_5_pro.py
python runners/run_gpt_5.py
python runners/run_deepseek_v4_flash.py
```

Each command runs Baseline, A, B and C, in that order. To select conditions:

```bash
python runners/run_gpt_5.py baseline treatment_a
```

To inspect the settings without calling an API, add `--dry-run`:

```bash
python runners/run_gpt_5.py --dry-run
```

Gemini and DeepSeek request temperature 1.0 for replies and 0.0 for guidance. GPT-5 omits temperature for both roles. The outcome judge requests temperature 0.0. DeepSeek requests include `thinking={"type": "disabled"}` and `max_tokens=8192`.

## 3. Find the outputs

The runners read the scenario JSON file specified by `TEST_CASES_PATH`. They save conversation JSON and generated guidance under `outputs/raw_runs/<model>/<condition>/`. Existing case files are loaded and skipped when a run resumes. Use a different `RESULTS_DIR` in `.env` to start a separate batch.

These new records are saved separately from the supplied analysis CSVs and may differ from the recorded study conversations.

## What each file does

You only run one of the three launchers. The other files are imported automatically.

| File | What it does | Described in |
|---|---|---|
| `run_gemini_2_5_pro.py`, `run_gpt_5.py`, `run_deepseek_v4_flash.py` | Launchers. Each sets the target model and calls `run_model.py`. | Manuscript Table 1 |
| `run_model.py` | Reads the conditions from the command line, prints the settings and runs Baseline, A, B and C in that order. | — |
| `config.py` | Shared settings: the target model (which also generates the guidance), the fixed judge `gpt-4o-mini-2024-07-18`, response temperature 1.0, the scoring windows (DCS from turn 4, HES and SIS from turn 7), and the input and output paths. | Manuscript Tables 1–2; Supplementary Table A2 |
| `baseline_runner.py` | Baseline: one reply per scripted turn with no guidance, then scoring. | Supplement A.3 |
| `pipeline_stage1.py` | **Stage 1**, shared by Treatments A–C: rates the current turn's severity (1–5) and names the concern. Treatment C also uses it to assess the end of each simulated trajectory. | Supplement A.2, A.5.1, Table A3 |
| `treatment_a_runner.py` | Treatment A (response rules): Stage 2 decides none / use / refine / create; Stage 3 writes a new rule; active rules are added to the reply context. | Supplement A.3, A.5.2–A.5.3, A.5.11 |
| `treatment_b_runner.py` | Treatment B (contrastive example): Stage 2 decides none / use / create; Stage 3 writes an undesirable and a preferred response. | Supplement A.3, A.5.4–A.5.5, A.5.11 |
| `treatment_c_runner.py` | Treatment C (consequence simulation): three candidate replies, a two-user-turn rollout for each, Stage 1 on each final state, three shuffled votes (two more on a three-way split); the winner is shown as an example before the real reply. | Supplement A.3, Table A2, A.5.6–A.5.11 |
| `target_client.py` | Sends reply and guidance requests to GPT-5 or DeepSeek V4 Flash (DeepSeek thinking disabled; temperature omitted for GPT-5) and routes Gemini ids to `gemini_client.py`. | Supplementary Table A2 |
| `gemini_client.py` | The same requests for Gemini 2.5 Pro (replies at 1.0, guidance at 0.0). | Supplementary Table A2 |
| `judge.py` | Scores DCS, HES and SIS with the benchmark prompts; the first integer in range is kept, otherwise -1. | Supplement B.1 |
| `judge_client.py` | Chooses the judge route: OpenAI if `OPENAI_API_KEY` is set, otherwise OpenRouter. | — |
| `openai_client.py`, `openrouter_client.py` | Judge requests at temperature 0.0, with retries. | Supplement B.1 |

The prompt text is in the runner files (`STAGE1_PROMPT`, `STAGE2_PROMPT`, `STAGE3_PROMPT`, …) and in `judge.py`; it is reproduced word for word in Supplement A.5 and B.1. The file map above identifies the condition implementations.
