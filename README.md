# Inference-Time Self-Guidance for Conversational Safety in Delusion-Related Interactions

**Running title:** Inference-Time Self-Guidance for Safer Delusion-Related Conversations

Open [PsychosisBench_Analysis.ipynb](analysis/PsychosisBench_Analysis.ipynb) to reproduce the benchmark analyses from the two supplied CSV files. The notebook validates the recorded design and scoring windows, checks scenario reconstruction, and displays the analyses, tables and figures. No model API calls or API keys are needed.

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

Files are identified by their columns, so timestamped filenames are supported. Clinical ratings are not requested. The Colab path needs no repository folders once the two CSVs have been uploaded. The notebook displays the package versions it uses.

## Results

Run all cells to display the benchmark analyses, tables and two figures inside the notebook. Results remain available as dataframes; the notebook does not create separate result files. See the [analysis guide](analysis/README.md) for execution instructions.

The separate [blinded clinical-evaluation matrix](data/clinical_evaluation/README.md) is included for independent inspection. This benchmark notebook does not load it or reproduce clinical agreement or condition summaries in main Table 3 and Supplementary Tables E3–E4. Condition summaries require a researcher-retained mapping that is not in this package.

## Package contents

| Folder or file | Purpose |
|---|---|
| [`analysis/`](analysis/README.md) | One analysis notebook and its execution guide. |
| [`data/`](data/README.md) | The two required benchmark CSVs and separate blinded clinical ratings. |
| [`runners/`](runners/README.md) | Current implementation and exact prompts; optional live generation of new conversations. |
| `requirements.txt` | Numerical environment and optional live-client dependencies. |
| `.env.example`, `.gitignore` | Empty live-run credential template and exclusions for local files. |

The live runners are not part of the reproduction workflow. New model responses require provider credentials and a separately supplied scenario JSON file configured through `TEST_CASES_PATH`; `data/test_cases.json` is not included. New responses may differ from the recorded study data.

## Study data and interpretation

The benchmark dataset contains three models × four conditions × 16 scenarios: 192 simulated conversations and 2,304 assistant turns. The evaluated model labels are Gemini 2.5 Pro, GPT-5 and DeepSeek V4 Flash.

DCS is the mean Delusion Confirmation Score on turns 4–12 (0–2); HES is the mean Harm Enablement Score on turns 7–12 (0–2); SIS is the sum of binary Safety Intervention Scores on turns 7–12 (0–6). Lower DCS/HES and higher SIS indicate safer benchmark responses. Recorded outcome scores use the fixed judge `gpt-4o-mini-2024-07-18`.

The scripted benchmark and outcome definitions originate from [Au Yeung et al., The psychogenic machine](https://doi.org/10.48550/arXiv.2509.10970). These are simulated conversations and automated ratings, not participant or clinical outcome data.

