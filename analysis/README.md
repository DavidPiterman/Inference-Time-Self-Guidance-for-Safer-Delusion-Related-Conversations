# Benchmark analysis guide

**Open [PsychosisBench_Analysis.ipynb](PsychosisBench_Analysis.ipynb) and run all cells in a fresh kernel.** This is the sole benchmark-analysis entry point. It requires exactly two CSV inputs: TURNS and SCENARIOS.

Install the root `requirements.txt`, then open the notebook in JupyterLab. See the [root README](../README.md) for the exact commands. From the package root or `analysis/`, the notebook finds both CSVs in `data/`. For standalone Jupyter use, place both files in the notebook's working directory. In Colab, upload both CSVs together in the same prompt. The files are identified by their columns, so timestamped filenames are supported.

Results are written to `outputs/analysis/` and its `figures/` subfolder. Rank-based robustness analyses (Wilcoxon signed-rank and Friedman tests) are reported for RQ1, RQ2, exploratory RQ3a and the within-condition cue contrasts in Supplement D.6 to D.9 (Tables D10–D17). The direct A–B, A–C and B–C comparisons are reported with paired t tests only (Supplement D4), so the notebook contains no rank-based counterpart for them.

The current notebook does not load clinical ratings or a private mapping, and does not reproduce main Table 3 or Supplementary Tables E3–E4. Those clinical analyses are separate from this two-CSV benchmark workflow; see the [clinical-data guide](../data/clinical_evaluation/README.md).

## Questions and correction families

| Analysis | Unit | Test and multiplicity |
|---|---|---|
| RQ1 pooled | 16 scenario profiles, after averaging the three models | Repeated-measures ANOVA over four conditions; use GG when sphericity is violated. Directional treatment-vs-Baseline paired t tests; BH-FDR within each outcome's three contrasts. |
| RQ1 model-specific | 16 profiles per model | Model-specific ANOVAs; directional planned t contrasts in one 27-test BH-FDR family. |
| Direct treatment comparisons | 16 profiles per model | Secondary/exploratory, two-sided paired t tests; one 27-test BH-FDR family, unadjusted 95% t intervals. |
| RQ2 Baseline cue comparison | Eight matched explicit–implicit pairs | Directional paired t tests; three pooled outcomes form one family; nine model-specific tests form another. |
| Exploratory RQ3a gap reduction | Eight matched pairs | Directional paired t tests of Baseline gap minus treatment gap; pooled three-test families within outcomes; separate model-specific 27-test family. |
| Within-condition cue comparison, added after registration | Eight matched pairs, within each condition separately | Two-sided paired t tests of Explicit versus Implicit SIS; one BH-FDR family over Treatments A, B and C; Baseline stays in the RQ2 family and is shown two-sided for context. |

No comparison averages the two cue variants before RQ1. The three model scores are averaged equally; they are fixed evaluated systems, not a random sample of models. Registration status differs by analysis; the additional sections are identified below. The implementation described here reflects the current manuscript analysis.

## Signs and interpretation

- RQ1 `mean_improvement`: Baseline − Treatment for DCS/HES, Treatment − Baseline for SIS. Positive means safer. The t statistic uses native Treatment − Baseline, so t is negative for safer DCS/HES; `cohens_dz` uses safety improvement.
- RQ2 `mean_improvement`: Implicit − Explicit for DCS/HES, Explicit − Implicit for SIS. Positive is an explicit-cue safety advantage. Native t uses Explicit − Implicit.
- RQ3 `gap_reduction`: signed Baseline gap − signed treatment gap. Positive denotes a reduction in the signed gap, which is not necessarily a reduction in absolute gap magnitude.
- Direct comparisons: positive `mean_safety_difference`, t and `cohens_dz` favor treatment 1. The 95% confidence intervals are not adjusted for multiplicity.
- `pFDR` is the BH-adjusted p value; `significant_fdr` means `pFDR < .05`. A positive estimate alone does not establish significance. Nonsignificance does not establish no effect or equivalence.
- Within-condition cue comparison `explicit_advantage`: Explicit − Implicit SIS inside one condition. It answers whether a cue-type difference is present in that condition. It does not test whether the difference changed between conditions; that remains `gap_reduction`. A contrast that is significant in one condition and nonsignificant in another is not evidence that the two conditions differ.

## Generated output mapping

Files are created under `outputs/analysis/`; all numeric CSV values retain full precision.

| File | Manuscript use |
|---|---|
| `rq1_pooled_anova_*.csv` | Supplement D1; retains GG and sphericity fields and selects `p_used`. |
| `rq1_pooled_contrasts.csv` | Main Table 4, including Baseline/Treatment M and SD. |
| `rq1_model_anova.csv` | Supplement D2. |
| `rq1_model_contrasts.csv` | Supplement D3: all 27 tests, Baseline/Treatment M and SD, safety improvement. |
| `treatment_pair_contrasts.csv` | Supplement D4. |
| `rq2_pooled_cue_contrasts.csv`, `rq2_model_cue_contrasts.csv` | Supplement D5 and D6. |
| `rq3_pooled_gap_reduction.csv`, `rq3_model_gap_reduction.csv` | Supplement D7 and D8. |
| `cue_within_condition_results.csv` | Supplement D9; the `pFDR` column is empty for Baseline, which belongs to the registered RQ2 family. |
| `treatment_exposure_overall.csv`, `treatment_exposure_by_model.csv` | Supplement Table B1: guidance-active turns and decision counts. |
| `psychosis_bench_scenarios.csv` | Reconstructed scores, compared to the supplied checkpoint before inference. |
| `*_descriptives.csv`, `descriptive_stats.csv` | Supporting descriptive summaries. |
| `cue_activation_overall.csv` | Supplement B2: activation rates by treatment, cue type and turn window. |
| `cue_activation_by_model.csv` | Activation-rate summaries by evaluated model. |
| `cue_activation_pairwise.csv` | Matched Explicit–Implicit activation differences by scenario pair. |
| `cue_activation_pair_summary.csv` | Summary across the eight matched pairs, including greater/tied/lower activation counts. |
| `environment.json` | Runtime package versions. |

To redirect outputs, set `PSYCHOSISBENCH_OUTPUT_DIR` before running the notebook. The workflow reconstructs and validates the scenario dataset, summarizes exposure and activation, runs hypothesis and robustness tests, then exports tables and figures. It produces 22 CSV files and `environment.json`; the two figures each have PNG, PDF and SVG exports. Rank-based results are displayed in notebook cells; they are not separate CSV exports in this version.

Figure 1 shows condition means by model. Figure 2 shows model-averaged signed cue gaps. They are descriptive summaries; the corresponding tests are in the CSV files.

## Analyses added after registration

Section 1.7 descriptively compares Stage-2 activation by Explicit/Implicit cue in Treatments A–C, across turns 1–12 and separately turns 7–12. Activation is `intervened == 1`; it includes creation, reuse or refinement. Comparisons use the eight matched pairs with equal averaging across models; differences are Explicit minus Implicit in percentage points. No inferential tests are performed. This uses TURNS only after its reconstruction has been validated against SCENARIOS. Stage-1 severity was not exported and is not analyzed or imputed. The four `cue_activation_*.csv` outputs support Supplementary Table B2.

Section 2.8 of the notebook tests the Explicit-versus-Implicit Safety Intervention Score contrast separately within Baseline and within each treatment. It was added after the registration, it is labelled as such in the notebook, and it does not replace the registered exploratory RQ3a gap-reduction analysis in Section 2.6. Only SIS is tested, because it is the only outcome with a significant Baseline cue-type disparity, so no new DCS or HES post hoc family is created. See `../ADDITIONAL_EXPLORATORY_ANALYSES.md` for the within-condition analysis boundary. Before release, that register also needs the Section 1.7 activation analysis and the current manuscript registration reference.

## Package versions for the reported tables

The reported tables were produced with Python 3.13.5, NumPy 2.1.3, pandas 2.2.3, SciPy 1.15.3, statsmodels 0.14.4 and Pingouin 0.6.1. The notebook prints the versions of the environment it runs in and writes them to `environment.json`. Rank-based p values can differ between SciPy releases, so compare that file against these versions before treating a rerun as a reproduction.

## Publishing and reader checks

Include this notebook and README, both CSV inputs in `../data/`, and the package-root README and `requirements.txt` in the reader download. Publish the complete package to retain implementation and data documentation. Do not distribute this folder as a standalone claim to reproduce clinical tables.

On 2026-10-12 the current notebook passed isolated package execution and a locally simulated Colab two-file upload using the reference numerical environment. All 22 CSV outputs matched between the two routes. The older flat OSF download is not this checked release; update the reviewer-facing files before closing that reproduction issue.
