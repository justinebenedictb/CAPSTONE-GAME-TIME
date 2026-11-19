# CAPSTONE GAME TIME: running performance from training load

**Study:** *Assessing Goal Attainment and Training Load Management for Recreational Runners in the Philippines Using Strava*
**Authors:** Justine Benedict L. Bautista and Nathan Lee P. Tagama (3DSA, College of Science, University of Santo Tomas)
**Adviser:** Asst. Prof. Arturo J. Patungan Jr., LPT, Ph.D.

**Live dashboard (RunState):** https://runstate-ust.streamlit.app/

This repository holds the full pipeline for the study: the data, every script, every output, the trained models, and the RunState dashboard.

- **Model task.** The models estimate a runner's **current best-effort level** from the **28 days of training before a given day**. The estimate is expressed as a Relative Performance Coefficient (RPC) and converted to 5K, 10K, 15K and half-marathon finish-time equivalents.
- **Training-load monitoring.** Training load is monitored with the Acute:Chronic Workload Ratio (ACWR). It is used as a descriptive workload alert, not as an injury prediction.

> The output is a best-effort estimate, not a race prediction. The targets are the best efforts inside any recorded run, because the data contain no race labels.

---

## 1. Repository layout

| Folder / file | What it holds |
|---|---|
| `manifest file/run_activities_manifest.csv` | List of the GoldenCheetah OpenData run activities used (the starting point of the pipeline). |
| `00_raw/granular/` | Second-by-second activity files, one folder per athlete (produced by scripts 01–02). |
| `scripts/` | All pipeline scripts, numbered in run order (section 2), plus `capstone_data.py` (shared helpers). |
| `01_prepared/` | Stage 1 outputs: activity index, best-effort targets, quality-control audit, daily timeline, 28-day windows. |
| `02_eda/` | Stage 2 exploratory analysis: `eda_summary.txt`, `tables/`, `plots/`. |
| `03_feature_engineering/` | Stage 3 outputs: run features (HR TRIMP, grade-adjusted speed), daily features (ATL, CTL, ACWR), modelling samples, feature dictionary. |
| `04_feature_selection/` | Stage 4 outputs: per-athlete chronological split, correlation filter, permutation importance, selected features, preprocessing parameters. |
| `05_baseline_models/` | Stage 5 outputs: baseline XGBoost, GRU and LSTM models (`models/`), validation metrics. |
| `06_tuning/` | Stage 6 outputs: random-search trials, best parameters, tuned models (`models/`). |
| `07_test_evaluation/` | Stage 7 outputs: one-shot test-set results, significance tests, ACWR zones, deployment choice. |
| `08_interpretation/` | Stage 8 outputs: permutation importance, PDP/ICE, what-if scenarios, surrogate trees, and exploratory checks. |
| `09_dashboard_evaluation/` | Stage 9 technical evaluation of the dashboard (section 4). |
| `deployment/` | The RunState dashboard: `app.py`, `core/` modules, copies of the deployed models, demo data, settings. |
| `logs/` | Download and audit logs for the source data. |
| `requirements.txt` | Pinned packages for the full pipeline. `deployment/requirements.txt` lists the dashboard's packages only. |
| `REPRODUCE.md` | Which script and output file each table and figure in Chapter 4 comes from. |

The empty folders `data/`, `eda/`, `evaluation/`, `features/`, `interpretation/`, `models/` and `raw/` are early placeholders and hold nothing. All results are in the numbered folders.

---

## 2. Reproducing the pipeline

**Environment used:** Windows 10, Python 3.11.5, CPU only (no GPU).

```
py -m pip install -r requirements.txt
```

Run the scripts from the project folder, in order. Each one reads the previous stage's outputs and writes to its own numbered folder.

| # | Script | Stage | Output folder | Notes |
|---|---|---|---|---|
| 01 | `scripts/01_audit_osf_dataset.py` | Source audit | `logs/` | Matches the manifest to the OSF archive (project `6hfpz`). Needs internet. |
| 02 | `scripts/02_download_osf_dataset.py` | Download | `00_raw/granular/` | Downloads and extracts the activity files. The extracted files are already in this repository, so steps 01–02 can be skipped. |
| 03 | `scripts/03_stage1_data_preparation.py` | 1 · Data preparation | `01_prepared/` | Moving time, best-effort targets, quality control, 28-day windows. `03_stage1_data_preparation_pre_qc.py` is the earlier version, kept for the record and not part of the final pipeline. |
| 04 | `scripts/04_stage2_eda.py` | 2 · EDA | `02_eda/` | |
| 05 | `scripts/05_stage3_feature_engineering.py` | 3 · Feature engineering | `03_feature_engineering/` | Option `--limit-athletes N` for a quick test. |
| 06 | `scripts/06_stage4_feature_selection.py` | 4 · Split + feature selection | `04_feature_selection/` | Split is per-athlete chronological 70/10/20. Preparation is fitted on the training part only. |
| 07 | `scripts/07_stage5_baseline_models.py` | 5 · Baseline models | `05_baseline_models/` | `--quick` for a smoke test. |
| 08 | `scripts/08_stage6_hyperparameter_tuning.py` | 6 · Tuning | `06_tuning/` | 40 random-search trials per model and version; about 7 hours on a laptop CPU. Resumable. `--quick` for a smoke test. |
| 09 | `scripts/09_stage7_test_evaluation.py` | 7 · Test evaluation | `07_test_evaluation/` | One-shot test with a pre-specified protocol. `--quick` for a smoke test. |
| 10 | `scripts/10_stage8_interpretation.py` | 8 · Interpretation | `08_interpretation/` | `--quick` for a smoke test. |
| 11 | `scripts/11_stage9_setup.py` | 9 · Dashboard setup | `deployment/` | Copies the deployed models into `deployment/models/` with SHA-256 checksums. |
| 12 | `scripts/12_stage9_dashboard_evaluation.py` | 9 · Dashboard evaluation | `09_dashboard_evaluation/` | About 15 minutes; `--quick` for 10 test runners. |

**Seeds:**
- Stage 4 importance uses seeds 11, 22, 33, 44 and 55.
- Models in Stages 5–7 use seeds 11, 22 and 33. The reported prediction of each model is the mean of its three seed models.
- In the Stage 6 random search, every trial runs with seed 11, and the top trials are re-checked with seeds 22 and 33.

**The `--quick` options:**
- Stages 5–8 write them to `_quick/` sub-folders, which git ignores.
- Script 12 adds `_quick` to its report name.
- Either way, a quick run never overwrites the real results.

**Deployed model:** tuned XGBoost "full" (mean of seeds 11/22/33). It was chosen by the rule fixed before testing: XGBoost unless a recurrent model is significantly better, and none was. The choice is recorded in `07_test_evaluation/stage7_deployment_choice.json`.

---

## 3. The RunState dashboard

**Online:** https://runstate-ust.streamlit.app/. It runs on Streamlit Community Cloud's free tier and may need a minute to wake up the first time.

**On your own computer:**

```
cd deployment
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

**What it does:**
- **Input.** A runner uploads their own Strava bulk-export ZIP, or an activity list (`.csv` or `.xlsx`). The Strava API is not used. Uploads are processed in memory and nothing is stored.
- **Processing.** The dashboard runs the same Stages 1–3 on that one runner and applies the deployed model.
- **"You now".** The current best-effort estimate for 5K, 10K, 15K and half marathon, with an 80% likely range taken from the Stage 7 test errors. Also a weekly fitness trend and a "last 90 days" summary.
- **Workload.** A distance-based ACWR alert, with heart-rate ACWR when heart-rate data exist.
- **"How the estimate responds".** A panel showing the model's sensitivity to training volume. It is not training advice and not a causal effect.
- **"This run".** Per-run analysis: summary, km splits, best efforts, and a route map linked to a pace, heart-rate and elevation chart.
- **Summary-only data.** With an activity list only, the dashboard shows training load and the ACWR alert, but no estimate.
- **Route privacy.** GPS is used only to draw the route map. It is never stored and never used as a model input. For uploads, "hide 500 m around start and finish" is preselected. The demo data are shown in full.
- **Map.** Map tiles come from OpenStreetMap (© OpenStreetMap contributors), drawn with Leaflet.

---

## 4. Dashboard technical evaluation (Stage 9)

`scripts/12_stage9_dashboard_evaluation.py` runs the dashboard's own code without the web page. Results are in `09_dashboard_evaluation/`, from the run of 2026-11-18 on the team laptop:

- **Accuracy:** the dashboard reproduces all **3,090 of 3,090** Stage 7 test predictions (largest difference 5.6e-17 RPC). Test MAE is 0.0380 RPC, the same as Stage 7, and the likely range covers 80.0% of test best efforts.
- **Edge cases:** 8 of 8 behave as designed: no heart rate, short history, corrupt or missing files, ZIP without `activities.csv`, wrong table, treadmill runs, activity list only, and stale data.
- **ACWR alert:** zone boundaries (0.8 / 1.3 / 1.5) and the "fewer than 4 run days in 28" rule are correct.
- **Privacy:** with hiding on, no route point sent to the browser lies within 500 m of a start or finish (closest 500.1 m). Coordinate columns in activity lists are dropped on load.

---

## 5. Data sources and licences

| Data | Used for | Licence | Where it is kept |
|---|---|---|---|
| **GoldenCheetah OpenData** (Liversedge, 2024), accessed through Kosmidis & Passfield (2023) | Model development and testing (May 2011 – May 2020) | CC0 1.0 (public domain) | `manifest file/`, `00_raw/granular/` |
| **Strava export of Justin Ho Junting**, from [justinhojt/strava-viz](https://github.com/justinhojt/strava-viz), commit `c27e5dc` | Dashboard demo 1 (full mode) | MIT License, "Copyright (c) 2026 Justin Ho Junting" | `deployment/demo_data/justinhojt_strava_export.zip`, used unmodified. The original `LICENSE` and a `SOURCE.txt` are kept inside the ZIP. |
| **Kaggle "Strava Running Activity Data"** (ajitjadhav1) | Dashboard demo 2 (summary-only mode) | CC0 1.0 | `deployment/demo_data/kaggle/`, with a CC0 note |

**No runners were recruited and no personal data were requested or collected.** The demonstration data were already published by their owners under open licences.

Citations:
- Ho, J. J. (2026). *strava-viz* [Computer software and data]. GitHub. https://github.com/justinhojt/strava-viz (MIT License)
- Jadhav, A. (n.d.). *Strava Running Activity Data* [Data set]. Kaggle. (CC0 1.0)

---

## 6. Version for the defense

The tag **`v1.0-defense`** marks the exact version of the code, outputs and dashboard presented at the capstone defense.
