# Where each Chapter 4 table and figure comes from

The numbering follows Chapter 4 as of November 2026. If tables or figures are renumbered later, the titles still identify them.

Every file below is produced by the script listed for it (see `README.md`, section 2, for the run order). Paths are relative to the repository root.

## Tables

| Table | Title | Script | Output file(s) |
|---|---|---|---|
| 4.1 | Athlete and Activity Counts at Each Preparation Step | `03_stage1_data_preparation.py`, `04_stage2_eda.py`, `05_stage3_feature_engineering.py` | `01_prepared/stage1_summary.txt` (573 athletes, activities), `02_eda/eda_summary.txt` (569 athletes, 24,807 runs), `01_prepared/stage1_28day_window_index.csv` (367 athletes with a 28-day window), `03_feature_engineering/stage3_summary.txt` (329 athletes). The first row, 2,589 athletes, is the source-dataset count described in Chapter 3. |
| 4.2 | Completeness and Distribution of Activity-Level Variables After Cleaning | `04_stage2_eda.py` | `02_eda/tables/eda_activity_feature_variance.csv` |
| 4.3 | Standard-Distance Targets Before and After Quality Control | `03_stage1_data_preparation.py` | `01_prepared/stage1_target_quality_audit.csv` |
| 4.4 | Distribution of Best-Effort Times by Distance | `04_stage2_eda.py` | `02_eda/tables/eda_target_distribution_summary.csv` |
| 4.5 | Correlation Between Preceding Running Distance and Best-Effort Time, by Look-Back Horizon | `04_stage2_eda.py` | `02_eda/tables/eda_lagged_load_correlations.csv` |
| 4.6 | From 28-Day Windows to Modelling Samples | `05_stage3_feature_engineering.py` | `03_feature_engineering/stage3_summary.txt`, `03_feature_engineering/stage3_dropped_samples.csv` |
| 4.7 | Chronological Split of Modelling Samples (70 / 10 / 20 Per Athlete) | `06_stage4_feature_selection.py` | `04_feature_selection/stage4_summary.txt`, `04_feature_selection/stage4_split_assignment.csv` |
| 4.8 | Permutation Importance of the Tabular Features on the Validation Partition | `06_stage4_feature_selection.py` | `04_feature_selection/stage4_feature_importance.csv` |
| 4.9 | Hyperparameters Selected by Random Search | `08_stage6_hyperparameter_tuning.py` | `06_tuning/stage6_summary.txt`, `06_tuning/stage6_best_params.json` |
| 4.10 | Baseline and Tuned Model Performance on the Validation Partition | `07_stage5_baseline_models.py`, `08_stage6_hyperparameter_tuning.py` | `05_baseline_models/stage5_metrics_summary.csv`, `06_tuning/stage6_metrics_summary.csv` |
| 4.11 | Contribution of Training Load and Past Performance on the Test Partition | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_summary.txt`, `07_test_evaluation/stage7_significance.csv` |
| 4.12 | Permutation Importance of the Model Features on the Test Partition | `10_stage8_interpretation.py` | `08_interpretation/stage8_permutation_importance.csv` |
| 4.13 | Change in Predicted Performance Across the Range of Each Training-Load Feature | `10_stage8_interpretation.py` | `08_interpretation/stage8_pdp_effects.csv`, `08_interpretation/stage8_pdp.csv` |
| 4.14 | Running Performance by Heart-Rate-Based ACWR Zone on the Test Partition | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_acwr_zones.csv` |
| 4.15 | Model and Baseline Performance on the Test Partition | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_metrics_summary.csv` |
| 4.16 | Per-Athlete Significance Tests on the Test Partition | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_significance.csv` |
| 4.17 | Test MAE by Best-Effort Distance | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_metrics_by_distance.csv` |
| 4.18 | Mean Prediction Error by Actual Performance Level on the Test Partition | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_metrics_by_rpc_level.csv` |
| 4.19 | Exploratory Checks of the Models Against the Athlete Average | `10_stage8_interpretation.py` | `08_interpretation/stage8_change_tracking.csv`, `08_interpretation/stage8_gain_by_history_quartile.csv`, `08_interpretation/stage8_equivalence.csv` |

## Figures

| Figure | Title | Script | Output file(s) |
|---|---|---|---|
| 4.1 | Monthly Count of Running Activities, May 2011 to May 2020 | `04_stage2_eda.py` | `02_eda/plots/temporal_monthly_activity_frequency.png` |
| 4.2 | Number of Running Activities per Athlete | `04_stage2_eda.py` | `02_eda/plots/temporal_activities_per_athlete.png` |
| 4.3 | Distributions of Distance, Duration, Average Speed, Average Heart Rate, and Elevation Gain After Cleaning | `04_stage2_eda.py` | `02_eda/plots/distribution_total_distance.png`, `distribution_workout_time.png`, `distribution_average_speed.png`, `distribution_average_hr.png`, `distribution_elevation_gain.png` |
| 4.4 | Best-Effort Times by Target Distance | `04_stage2_eda.py` | `02_eda/plots/target_distributions_boxplot.png` |
| 4.5 | Autocorrelation of Daily Running Distance at Lags of 1 to 28 Days | `04_stage2_eda.py` | `02_eda/plots/autocorrelation_total_distance.png` |
| 4.6 | Correlation Between Preceding Running Distance and 5 km Best-Effort Time, by Look-Back Window | `04_stage2_eda.py` | `02_eda/plots/lag_correlation_total_distance_5_0km.png` |
| 4.7 | Permutation Importance of the Tabular Features on the Validation Partition | `06_stage4_feature_selection.py` | `04_feature_selection/stage4_feature_importance.png` |
| 4.8 | Random-Search Progress for the Full Models and the Models Without Past Performance | `08_stage6_hyperparameter_tuning.py` | `06_tuning/stage6_search_progress.png` |
| 4.9 | Permutation Importance of the XGBoost Features on the Test Partition | `10_stage8_interpretation.py` | `08_interpretation/stage8_permutation_importance.png` |
| 4.10 | Partial Dependence and ICE Curves of the Training-Load Features on the Test Partition | `10_stage8_interpretation.py` | `08_interpretation/stage8_pdp_ice.png` |
| 4.11 | Depth-3 Surrogate Decision Tree of the XGBoost Deployment Model | `10_stage8_interpretation.py` | `08_interpretation/stage8_surrogate_tree_full.png`, rules in `08_interpretation/stage8_surrogate_rules.txt` |
| 4.12 | Share of Test Samples in Each ACWR Zone, Heart-Rate and Distance Versions | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_acwr_zones.png` |
| 4.13 | Predicted Versus Actual RPC on the Test Partition for XGBoost, GRU, and LSTM | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_pred_vs_actual.png` |
| 4.14 | Per-Athlete Test MAE of XGBoost Compared With the Athlete Average | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_per_athlete.png` |
| 4.15 | Test MAE by Best-Effort Distance for the Three Models and the Athlete Average | `09_stage7_test_evaluation.py` | `07_test_evaluation/stage7_error_by_distance.png` |
| 4.16 | Actual Versus Predicted Deviation From the Runner's Average, by Decile of Predicted Deviation | `10_stage8_interpretation.py` | `08_interpretation/stage8_change_tracking.png`, data in `08_interpretation/stage8_change_tracking_deciles.csv` |
| 4.17 | Per-Runner Gain of XGBoost Over the Athlete Average by Amount of Training History | `10_stage8_interpretation.py` | `08_interpretation/stage8_gain_vs_history.png`, data in `08_interpretation/stage8_gain_vs_history.csv` |

## Other results

- **Every stage's headline numbers** are in that stage's summary text file:
  - `stage1_summary.txt`, `eda_summary.txt`, `stage3_summary.txt`, `stage4_summary.txt` and `stage5_summary.txt`
  - `stage6_summary.txt`, `stage7_summary.txt` and `stage8_summary.txt`
- **Deployment choice:** `07_test_evaluation/stage7_deployment_choice.json`.
- **What-if scenarios (Stage 8):** `08_interpretation/stage8_what_if.csv`.
- **Dashboard technical evaluation (Stage 9):** `09_dashboard_evaluation/stage9_evaluation_report.txt`, plus CSVs A–E. These come from `12_stage9_dashboard_evaluation.py`.
- **The dashboard's likely range** (80%, from the Stage 7 test errors): `deployment/models/prediction_range.json`. It is written by `deployment/core/predict.py`.
