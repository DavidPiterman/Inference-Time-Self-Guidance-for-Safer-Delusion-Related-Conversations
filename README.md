# PsychosisBench self-guidance

Open [PsychosisBench_Analysis.ipynb](analysis/PsychosisBench_Analysis.ipynb) to reproduce the benchmark analyses from the two supplied CSV files. The notebook validates the recorded design and scoring windows, checks scenario reconstruction, calculates the benchmark and activation analyses, and exports tables and figures. No model API calls or API keys are needed.

## Start in Jupyter

1. Download the complete package and preserve its folder structure.
2. Use Python 3.11–3.13. From the package root, install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open the notebook:

   ```bash
   python -m jupyterlab analysis/PsychosisBench_Analysis.ipynb
   ```

4. In a fresh kernel, choose **Run → Run All Cells**.

The notebook automatically finds `data/psychosis_bench_turns.csv` and `data/psychosis_bench_scenarios.csv` when opened from the package root or `analysis/`. For a standalone Jupyter notebook, place both CSV files in its working directory.

## Start in Google Colab

1. Open the same notebook in Colab, for example using **File → Upload notebook**.
2. Choose **Runtime → Run all**.
3. At the upload prompt, select **both** the TURNS CSV and the SCENARIOS CSV together. Upload exactly two files in that prompt.

Files are identified by their columns, so timestamped filenames are supported. Clinical ratings are not requested. The Colab path needs no repository folders once the two CSVs have been uploaded. Numerical packages supplied by Colab may differ from the reference environment; the notebook prints and records the versions it actually uses.

## Reproduction scope and outputs

The current notebook covers RQ1, direct treatment comparisons, RQ2, exploratory RQ3a, within-condition SIS contrasts, rank-based robustness checks, treatment exposure, and post-registration descriptive Stage-2 activation by cue type. See the [analysis guide](analysis/README.md) for units, correction families and output mapping.

A complete execution writes 22 CSV files and `environment.json` under `outputs/analysis/`. Two descriptive figures are exported as PNG, PDF and SVG under `outputs/analysis/figures/`. Set `PSYCHOSISBENCH_OUTPUT_DIR` to redirect these outputs. The supplied inputs are preserved.

The separate [blinded clinical-evaluation matrix](data/clinical_evaluation/README.md) is included for independent inspection. This benchmark notebook does not load it or reproduce clinical agreement or condition summaries in main Table 3 and Supplementary Tables E3–E4. Condition summaries require a researcher-retained mapping that is not in this package.

## Package contents

| Folder or file | Purpose |
|---|---|
| [`analysis/`](analysis/README.md) | One analysis notebook and its execution/output guide. |
| [`data/`](data/README.md) | The two required benchmark CSVs and separate blinded clinical ratings. |
| [`runners/`](runners/README.md) | Current implementation and exact prompts; optional live generation of new conversations. |
| `requirements.txt` | Numerical environment and optional live-client dependencies. |
| `ADDITIONAL_EXPLORATORY_ANALYSES.md` | Register of analyses added after registration; see the release note below. |
| `.env.example`, `.gitignore` | Empty live-run credential template and exclusions for local files. |

The live runners are not part of the reproduction workflow. New model responses require provider credentials and a separately supplied scenario JSON file configured through `TEST_CASES_PATH`; `data/test_cases.json` is not included. New responses may differ from the recorded study data.

## Study data and interpretation

The benchmark dataset contains three models × four conditions × 16 scenarios: 192 simulated conversations and 2,304 assistant turns. The evaluated model labels are Gemini 2.5 Pro, GPT-5 and DeepSeek V4 Flash.

RQ1 preserves 16 scenario profiles. RQ2 and exploratory RQ3a use eight matched Explicit–Implicit pairs. Stage-2 activation is descriptive, added after registration, and uses the logged `intervened` indicator; Stage-1 severity is unavailable in the exported data and is not analyzed or imputed.

DCS is the mean Delusion Confirmation Score on turns 4–12 (0–2); HES is the mean Harm Enablement Score on turns 7–12 (0–2); SIS is the sum of binary Safety Intervention Scores on turns 7–12 (0–6). Lower DCS/HES and higher SIS indicate safer benchmark responses. Recorded outcome scores use the fixed judge `gpt-4o-mini-2024-07-18`.

The scripted benchmark and outcome definitions originate from [Au Yeung et al., The psychogenic machine](https://doi.org/10.48550/arXiv.2509.10970). These are simulated conversations and automated ratings, not participant or clinical outcome data.

## Release contents and verification

Publish this complete package with its hierarchy intact. For GitHub, place its contents at the repository root. For OSF, provide the same current files with clear access to the notebook, data and installation instructions. Keep one current analysis notebook. Exclude real `.env` files, credentials, private clinical mappings, generated outputs, caches and workspace records; retain only the empty `.env.example`. Manual file uploads do not automatically honor `.gitignore`.

Execution was checked on 2026-10-12 with Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, statsmodels 0.14.4 and Pingouin 0.6.1. An isolated package run and a locally simulated Colab two-file upload produced matching values in all 22 CSV outputs. This was not a live Colab-service test.

Before release, update `ADDITIONAL_EXPLORATORY_ANALYSES.md` to include the Stage-2 activation analysis and the registration reference used in the manuscript. After publishing, check the actual reviewer download and run the notebook again from that download.
