# Benchmark analysis

Open [PsychosisBench_Analysis.ipynb](PsychosisBench_Analysis.ipynb) and run all cells in a fresh kernel. The research questions, analysis definitions, correction families and interpretations are documented in the notebook.

## Inputs

Exactly two CSVs are required:

- `../data/psychosis_bench_turns.csv`
- `../data/psychosis_bench_scenarios.csv`

Install the package-root `requirements.txt` and follow the [root README](../README.md) to open JupyterLab. The notebook finds both CSVs in `data/`. For a standalone notebook, put both CSVs in its working directory. In Colab, upload both together when prompted; files are identified by their columns.

## Results

Tables and two figures are displayed inside the notebook. Calculated results remain available as dataframes, with full precision retained for calculations. No result CSVs, environment files or figure files are written automatically. Package versions are displayed during setup; numerical dependencies are pinned in `requirements.txt`.

The notebook covers the benchmark analyses. It does not load clinical ratings or the private mapping, or compute main Table 3 and Supplementary Tables E3–E4. See the [clinical-data guide](../data/clinical_evaluation/README.md) for the separately supplied blinded ratings.
