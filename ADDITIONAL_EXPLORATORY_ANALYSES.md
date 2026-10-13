# Additional exploratory analyses added after registration

The registered analysis plan is the OSF registration for *Model-Generated Safeguards at
Inference Time: A Multi-Model Evaluation in Delusion-Related Conversations*
(https://osf.io/d5q46/). Analyses listed here were added after that registration. They are
labelled as such in the manuscript, in the Supplementary Appendix, and in the analysis notebook,
and they do not replace or modify any registered analysis.

## 1. Within-condition Explicit-versus-Implicit Safety Intervention Score comparisons

**Analysis.** For the Safety Intervention Score (SIS), Explicit and Implicit scenarios were
compared separately within Baseline and within each of Treatments A, B and C.

**Motivation.** The registered RQ2 analysis established an Explicit–Implicit SIS disparity at
Baseline, and the registered exploratory RQ3a analysis tested whether that disparity was smaller
under each treatment. Neither asks whether a cue-type difference is still detectable once a
treatment is active. These contrasts answer that question. SIS was selected because it is the
only outcome with a statistically significant Baseline cue-type disparity; DCS and HES were not
tested, so no new family of DCS/HES post hoc comparisons was created.

**Registration status.** Not preregistered. The registration specifies RQ2 as the Baseline
Explicit-versus-Implicit comparison and exploratory RQ3a as the reduction of the Explicit
advantage under treatment; it permits characterising cue-type effects "within individual models
and, where appropriate, jointly across models", which concerns model-level characterisation of
those registered contrasts and does not cover a within-treatment cue-type test. These analyses
are therefore reported as additional and post-registration.

**Statistical specification.**

- Unit of analysis: the eight matched Explicit–Implicit scenario pairs, identical to RQ2 and RQ3a.
- Averaging: each cue score averaged across the three target models within scenario pair, cue
  type and condition, identical to the pooled RQ2/RQ3a procedure.
- Explicit advantage: SIS_Explicit − SIS_Implicit, so positive values favour Explicit.
- Tests: two-sided paired *t* tests, one per condition.
- Effect size: Cohen's *dz* = mean paired difference / SD of the paired differences (ddof = 1).
- Interval: unadjusted 95% *t* confidence interval for the paired mean difference.
- Multiplicity: one Benjamini–Hochberg family across the three treatment tests (A, B, C). The
  Baseline comparison belongs to the registered RQ2 family and is not re-corrected; it is
  reported here two-sided for context only.

**Inferential boundary.** These tests characterise whether a cue-type disparity is detectable
*within* a condition. They are not a test of whether the disparity *changed* across conditions.
That question is answered by the registered exploratory RQ3a gap-reduction contrasts, which
compare the Baseline gap with each treatment gap directly. A contrast that is significant in one
condition and nonsignificant in another is not evidence that the two conditions differ. A
nonsignificant within-condition contrast does not establish equivalence or a negligible residual
gap; no equivalence or non-inferiority margin is defined in this project, and none was introduced.

**Code.** `analysis/PsychosisBench_Analysis.ipynb`, Section 2.8 ("Added after registration — Explicit vs Implicit SIS within each condition"),
with its Wilcoxon counterpart in Section 3.6. It reuses `rq3_mean`, the pooled cue × condition grid built for RQ3a, and writes
`cue_within_condition_results.csv` to the notebook's output directory. Running the notebook with
Run All reproduces it together with all registered results.

**Reported in.** Manuscript Statistical analysis and Results (exploratory RQ3a subsection);
Supplementary Appendix C.2 and D.5 (Table D9).
