# Benchmark analysis

Open [PsychosisBench_Analysis.ipynb](PsychosisBench_Analysis.ipynb) and run all cells in a fresh kernel. The research questions, analysis definitions, correction families and interpretations are documented in the notebook.

## Inputs

Exactly two CSVs are required:

- `../data/psychosis_bench_turns.csv`
- `../data/psychosis_bench_scenarios.csv`

Follow the [root README](../README.md) for installation and Jupyter or Colab instructions. The notebook identifies the two input files by their columns and checks the scenario scores against values reconstructed from the turn data.

## Results

Tables and two figures are displayed inside the notebook. Calculated results remain available as dataframes, with full precision retained for calculations. No separate result files are written automatically. Package versions are displayed during setup; numerical dependencies are pinned in `requirements.txt`.

The [clinical-ratings CSV](../data/clinical_evaluation/README.md) is a separate dataset and is not an input to this notebook.
