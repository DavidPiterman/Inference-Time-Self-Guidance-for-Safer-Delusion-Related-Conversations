# Study data

The analysis uses the two benchmark CSV files below. Clinical ratings are supplied separately in `clinical_evaluation/`.

| File or folder | Role |
|---|---|
| `psychosis_bench_turns.csv` | Required benchmark input: 2,304 assistant turns from 192 conversations, with text, scores and treatment activity. |
| `psychosis_bench_scenarios.csv` | Scores for 192 conversations. The notebook checks these against scores recalculated from the turn data. |
| [`clinical_evaluation/`](clinical_evaluation/README.md) | Blinded clinical-evaluation ratings: two raters × 12 selected conversations × 4 items. |

The CSVs use UTF-8 and open in spreadsheet software. Empty score fields before the applicable scoring window mean not scored, not zero. Baseline treatment-specific fields are structurally missing. Conversation text and scores for all four conditions are present in the turn CSV.

## Turn-data dictionary

| Columns | Meaning |
|---|---|
| `model` | `gemini-2.5-pro`, `gpt-5`, or `deepseek-v4-flash`. |
| `condition`, `condition_label`, `is_baseline` | Baseline or treatment_a/b/c; display label; Baseline indicator. |
| `case_id`, `case_number`, `case_name` | Stable scenario identifier and benchmark labels. |
| `scenario_pair`, `pair_name` | Identifier/name linking the explicit and implicit variants. |
| `scenario_type`, `is_implicit` | Explicit/Implicit and its binary indicator. |
| `theme_raw`, `theme_group`, `harm_type` | Source scenario theme, grouped theme and harm-type labels. |
| `turn`, `phase` | Assistant turn 1–12 and benchmark phase. |
| `in_dcs_window`, `in_hes_window`, `in_sis_window` | Indicators for outcome-specific scoring windows. |
| `dcs`, `hes`, `sis` | Per-turn automated scores: DCS/HES 0,1,2; SIS 0,1; empty outside the scoring window. |
| `stage2_decision` | Logged treatment decision: none_needed, use, refine, or create_needed, as applicable to condition. |
| `intervened` | Whether guidance was active on this turn (0/1); not a unique artifact count. |
| `stage3_failed` | Logged guidance-generation failure flag; structurally missing at Baseline. |
| `user_message`, `bot_response` | Scripted user text and generated assistant response. |
| `dcs_explanation`, `hes_explanation`, `sis_explanation` | Automated judge explanations. |

The unique turn key is `model × condition × case_id × turn`. The same scenario content is reused across conditions and models; each conversation is separately generated.

## Scenario-data dictionary

Scenario rows retain the identifying and descriptive fields above and add:

| Columns | Meaning |
|---|---|
| `DCS_mean`, `HES_mean`, `SIS_sum` | The three analysis outcomes: mean over turns 4–12, mean over turns 7–12, and sum over turns 7–12. |
| `DCS_sum`, `HES_sum` | Sums of the corresponding turn scores. |
| `n_dcs_turns`, `n_hes_turns`, `n_sis_turns` | Scored-turn counts: 9, 6, 6. |
| `intervention_rate`, `n_turns_intervened` | Proportion/count of active guidance across all 12 turns. |
| `n_stage2_create_needed`, `n_stage3_failed` | Counts of create-needed decisions and logged failures. |
| `anchor_selection_risk` | Recorded index: `1 + 0.5 × DCS_mean + 0.5 × HES_mean − SIS_sum / n_sis_turns`. Not used as a benchmark outcome. |
