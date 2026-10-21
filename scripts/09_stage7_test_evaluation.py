"""
09_stage7_test_evaluation.py

CAPSTONE
Stage 7 - Test-Set Evaluation (one shot)

The test split (20% of each athlete's samples, latest in time) is loaded here
for the first time. Nothing is fitted on it and nothing is re-tuned after it:
every transformation comes from the training split (Stage 4 parameters,
Stage 5/6 sequence preparation) and every model is loaded exactly as saved.

Decisions fixed BEFORE the test results were seen (team, 2026-10-21):
  - Models: tuned models as trained in Stage 6 (training split only), plus the
    Stage 5 baseline models; each model = mean of its 3 seeds.
  - Main naive comparison: athlete average from training + validation
    (everything known about the runner before the test period).
  - Significance: per-athlete paired comparison, bootstrap 95% CI of the mean
    difference + Wilcoxon signed-rank, Holm correction within each question.
  - Deployment model: XGBoost, unless GRU or LSTM is significantly better than
    XGBoost on test (Holm-adjusted p < 0.05).

Outputs to 07_test_evaluation/.

Usage:
    py 09_stage7_test_evaluation.py           (full, ~2-5 min)
    py 09_stage7_test_evaluation.py --quick   (25 athletes, writes to 07_test_evaluation/_quick)
Requires: pandas, numpy, scipy, xgboost, tensorflow, matplotlib
"""

from pathlib import Path
import argparse
import importlib
import json
import sys
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capstone_data as cd

s5 = importlib.import_module("07_stage5_baseline_models")  # metric + plot helpers


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = cd.ROOT / "07_test_evaluation"
MODEL_DIRS = {"baseline": cd.ROOT / "05_baseline_models", "tuned": cd.ROOT / "06_tuning"}
PREP_PREFIX = {"baseline": "stage5", "tuned": "stage6"}
LOG_DIR = cd.ROOT / "logs"

MODELS = ["xgb", "gru", "lstm"]
VARIANTS = ["full", "no_past_rpc"]
SEEDS = [11, 22, 33]
N_BOOT = 10_000
ALPHA = 0.05
MAIN_NAIVE = "naive_athlete_mean_trainval"

NAIVE_LABELS = {
    "naive_athlete_mean_trainval": "Athlete average (training + validation) [main]",
    "naive_athlete_mean_train": "Athlete average (training only)",
    "naive_running_mean": "Athlete running average (all earlier results)",
    "naive_last_rpc": "Last result",
    "naive_global_mean": "Overall average (training)",
}
ACWR_ZONES = [(-np.inf, 0.8, "under-training (<0.8)"), (0.8, 1.3, "sweet spot (0.8-1.3)"),
              (1.3, 1.5, "monitoring (1.3-1.5)"), (1.5, np.inf, "danger (>=1.5)")]


# ============================================================
# 1. TEST INPUTS
# ============================================================

def build_inputs(quick):
    print("=" * 70)
    print("STEP 1 - TEST INPUTS (ALL PREPARATION FROM TRAINING)")
    print("=" * 70)
    samples = cd.add_personal_level(cd.load_samples())
    if quick:
        keep = samples.loc[samples["split"] == "test", "id"].drop_duplicates().head(25)
        samples = samples[samples["id"].isin(keep)]
    params = cd.load_params()
    raw_test = samples[samples["split"] == "test"].reset_index(drop=True)   # unprepared (ACWR analysis)
    tab = cd.prepare_tabular(samples, params)
    train = tab[tab["split"] == "train"].reset_index(drop=True)
    val = tab[tab["split"] == "val"].reset_index(drop=True)
    test = tab[tab["split"] == "test"].reset_index(drop=True)
    assert (raw_test["sample_id"].to_numpy() == test["sample_id"].to_numpy()).all()

    # time-order check: every test target is later than all of that athlete's train/val targets
    last_known = pd.concat([train, val]).groupby("id")["target_date"].max()
    assert (test["target_date"] > test["id"].map(last_known)).all(), "test date not after train/val"
    assert not set(test["sample_id"]) & (set(train["sample_id"]) | set(val["sample_id"]))

    tab_full = cd.load_feature_list("tabular")
    seq_full = cd.load_feature_list("sequence") + cd.PRIOR_SEQUENCE
    features = {
        "xgb": {"full": tab_full, "no_past_rpc": [f for f in tab_full if f not in cd.PRIOR_TABULAR]},
        "seq": {"full": seq_full, "no_past_rpc": [f for f in seq_full if f not in cd.PRIOR_SEQUENCE]},
    }
    daily = cd.load_daily(seq_full)
    X_raw = cd.build_windows(daily, test, seq_full)

    seq = {}
    for stage in MODEL_DIRS:
        for variant, feats in features["seq"].items():
            prep = json.loads((MODEL_DIRS[stage] / f"{PREP_PREFIX[stage]}_sequence_prep_{variant}.json")
                              .read_text(encoding="utf-8"))
            assert prep["features"] == feats, f"{stage} {variant}: feature order differs from saved prep"
            cols = [seq_full.index(f) for f in feats]
            seq[(stage, variant)] = cd.apply_sequence_prep(X_raw[:, :, cols], feats, prep)
            assert np.isfinite(seq[(stage, variant)]).all()

    print(f"Test samples: {len(test):,} ({test['id'].nunique()} athletes) | "
          f"best-effort distances: " + ", ".join(f"{k:g} km {v}" for k, v in
                                                test['target_distance_km'].value_counts().sort_index().items()))
    return train, val, test, raw_test, features, seq


# ============================================================
# 2. PREDICTIONS
# ============================================================

def naive_predictions(train, val, test):
    trainval = pd.concat([train, val])
    global_mean = train[cd.TARGET].mean()
    return {
        "naive_athlete_mean_trainval": test["id"].map(trainval.groupby("id")[cd.TARGET].mean())
                                                 .fillna(global_mean).to_numpy(),
        "naive_athlete_mean_train": test["id"].map(train.groupby("id")[cd.TARGET].mean())
                                              .fillna(global_mean).to_numpy(),
        "naive_running_mean": test["prior_mean_rpc"].to_numpy(),      # prepared (training-median fill)
        "naive_last_rpc": test["last_rpc"].to_numpy(),
        "naive_global_mean": np.full(len(test), global_mean),
    }


def model_predictions(test, features, seq):
    from xgboost import XGBRegressor
    from tensorflow import keras

    per_seed = {}
    for stage, folder in MODEL_DIRS.items():
        for model in MODELS:
            for variant in VARIANTS:
                for seed in SEEDS:
                    name = f"{model}_{variant}_seed{seed}"
                    if model == "xgb":
                        m = XGBRegressor()
                        m.load_model(str(folder / "models" / f"{name}.json"))
                        pred = m.predict(test[features["xgb"][variant]])
                    else:
                        m = keras.models.load_model(str(folder / "models" / f"{name}.keras"), compile=False)
                        pred = m.predict(seq[(stage, variant)], verbose=0).ravel()
                    per_seed[(stage, model, variant, seed)] = np.asarray(pred, dtype=np.float64)
                print(f"  loaded {stage:<8} {model:<4} {variant}")
    return per_seed


# ============================================================
# 3. METRICS
# ============================================================

def athlete_errors(test, pred):
    return (pd.DataFrame({"id": test["id"], "ae": np.abs(pred - test[cd.TARGET].to_numpy())})
            .groupby("id")["ae"].mean())


def bias(test, pred):
    w = cd.athlete_weights(test)
    return float(np.sum(w * (pred - test[cd.TARGET].to_numpy())) / w.sum())


def by_rpc_level(test, preds):
    level = pd.qcut(test[cd.TARGET], 5, labels=["1 slowest", "2", "3", "4", "5 fastest"])
    rows = []
    for name, pred in preds.items():
        df = pd.DataFrame({"level": level, "err": pred - test[cd.TARGET].to_numpy()})
        for lv, g in df.groupby("level", observed=True):
            rows.append({"prediction": name, "rpc_fifth": lv, "n": len(g),
                         "mean_error": g["err"].mean(), "mae": g["err"].abs().mean()})
    return pd.DataFrame(rows)


# ============================================================
# 4. SIGNIFICANCE
# ============================================================

def paired_test(test, pred_a, pred_b, rng, n_boot):
    """Athlete-level comparison of A vs B. diff < 0 means A has lower error."""
    from scipy.stats import wilcoxon
    d = (athlete_errors(test, pred_a) - athlete_errors(test, pred_b)).to_numpy()
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[idx].mean(axis=1)
    p_boot = min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))
    p_wil = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
    return {"mean_diff": d.mean(), "ci_low": np.percentile(boot, 2.5), "ci_high": np.percentile(boot, 97.5),
            "p_bootstrap": p_boot, "p_wilcoxon": p_wil, "athletes_a_better": int((d < 0).sum()),
            "athletes": len(d)}


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def significance(test, preds, rng, n_boot):
    families = {
        "Q1 models vs athlete average (train+val) [main]":
            [(f"tuned_{m}_full", MAIN_NAIVE) for m in MODELS] + [("tuned_average_full", MAIN_NAIVE)],
        "Q1b models vs athlete average (training only)":
            [(f"tuned_{m}_full", "naive_athlete_mean_train") for m in MODELS]
            + [("tuned_average_full", "naive_athlete_mean_train")],
        "Q1c models vs athlete running average (all earlier results)":
            [(f"tuned_{m}_full", "naive_running_mean") for m in MODELS]
            + [("tuned_average_full", "naive_running_mean")],
        "Q2 training load only vs simple guesses":
            [(f"tuned_{m}_no_past_rpc", b) for m in MODELS for b in ("naive_global_mean", "naive_last_rpc")],
        "Q3 ablation: full vs no past performance":
            [(f"tuned_{m}_full", f"tuned_{m}_no_past_rpc") for m in MODELS],
        "Q4 tuned vs Stage 5 baseline":
            [(f"tuned_{m}_{v}", f"baseline_{m}_{v}") for m in MODELS for v in VARIANTS],
        "Q5 model vs model (tuned, full)":
            [("tuned_gru_full", "tuned_xgb_full"), ("tuned_lstm_full", "tuned_xgb_full"),
             ("tuned_lstm_full", "tuned_gru_full")],
    }
    rows = []
    for family, pairs in families.items():
        part = [{"question": family, "a": a, "b": b, **paired_test(test, preds[a], preds[b], rng, n_boot)}
                for a, b in pairs]
        for key in ("p_bootstrap", "p_wilcoxon"):
            adj = holm([r[key] for r in part])
            for r, v in zip(part, adj):
                r[f"{key}_holm"] = v
        rows += part
    df = pd.DataFrame(rows)
    df["a_significantly_better"] = (df["mean_diff"] < 0) & (df["p_bootstrap_holm"] < ALPHA)
    df["a_significantly_worse"] = (df["mean_diff"] > 0) & (df["p_bootstrap_holm"] < ALPHA)
    return df


def deployment_choice(sig):
    q5 = sig[sig["question"].str.startswith("Q5")]
    winners = q5[(q5["b"] == "tuned_xgb_full") & q5["a_significantly_better"]]
    if winners.empty:
        return "xgb", "No RNN is significantly better than XGBoost on test -> XGBoost (simplest model)."
    best = winners.sort_values("mean_diff").iloc[0]
    model = best["a"].split("_")[1]
    return model, (f"{model.upper()} is significantly better than XGBoost on test "
                   f"(mean difference {best['mean_diff']:.4f}, Holm p {best['p_bootstrap_holm']:.3f}).")


# ============================================================
# 5. ACWR (RQ2, descriptive)
# ============================================================

def acwr_analysis(raw_test, test, preds, deploy_key):
    from scipy.stats import spearmanr
    relative = test[cd.TARGET].to_numpy() - preds[MAIN_NAIVE]      # vs the runner's usual level
    ae = np.abs(preds[deploy_key] - test[cd.TARGET].to_numpy())
    rows, corr = [], []
    for col, label in (("acwr", "HR-load ACWR (TRIMP)"), ("acwr_km", "distance ACWR (km)")):
        if col not in raw_test.columns:
            continue
        value = raw_test[col].to_numpy(dtype=float)
        zone = np.full(len(value), "not enough runs (undefined)", dtype=object)
        for lo, hi, name in ACWR_ZONES:
            zone[(value >= lo) & (value < hi)] = name
        for name in [z[2] for z in ACWR_ZONES] + ["not enough runs (undefined)"]:
            m = zone == name
            rows.append({"acwr_type": label, "zone": name, "n": int(m.sum()),
                         "share_pct": 100 * m.mean(), "athletes": int(test.loc[m, "id"].nunique()),
                         "mean_rpc": float(test.loc[m, cd.TARGET].mean()) if m.any() else np.nan,
                         "mean_rpc_vs_own_average": float(relative[m].mean()) if m.any() else np.nan,
                         "deployment_model_mae": float(ae[m].mean()) if m.any() else np.nan})
        ok = np.isfinite(value)
        r1 = spearmanr(value[ok], test.loc[ok, cd.TARGET])
        r2 = spearmanr(value[ok], relative[ok])
        corr.append({"acwr_type": label, "n": int(ok.sum()),
                     "spearman_vs_rpc": r1.statistic, "p_vs_rpc": r1.pvalue,
                     "spearman_vs_rpc_relative_to_own_average": r2.statistic, "p_vs_relative": r2.pvalue})
    return pd.DataFrame(rows), pd.DataFrame(corr)


# ============================================================
# PLOTS
# ============================================================

def _style(ax):
    ax.grid(color=s5.GRID, lw=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_distance(dist, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    show = {"tuned_xgb_full": ("XGBoost", s5.COLORS["xgb"]), "tuned_gru_full": ("GRU", s5.COLORS["gru"]),
            "tuned_lstm_full": ("LSTM", s5.COLORS["lstm"]), MAIN_NAIVE: ("Athlete average", "#9a9a95")}
    d = dist[dist["prediction"].isin(show) & (dist["n"] >= 20)]
    kms = sorted(d["distance_km"].unique())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    width = 0.8 / len(show)
    for i, (name, (label, color)) in enumerate(show.items()):
        vals = [d[(d["prediction"] == name) & (d["distance_km"] == k)]["rpc_mae"].mean() for k in kms]
        ax.bar(np.arange(len(kms)) + (i - (len(show) - 1) / 2) * width, vals, width, color=color, label=label)
    ns = d.groupby("distance_km")["n"].first()
    ax.set_xticks(range(len(kms)), [f"{k:g} km\n(n={ns[k]})" for k in kms], fontsize=8)
    ax.set_ylabel("Test MAE (RPC)", color=s5.MUTED, fontsize=8)
    ax.set_title("Test error by best-effort distance (marathon omitted, n < 20)", fontsize=9,
                 color=s5.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_athletes(test, preds, deploy_key, label, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = athlete_errors(test, preds[deploy_key])
    b = athlete_errors(test, preds[MAIN_NAIVE])
    better = a < b
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.scatter(b[better], a[better], s=12, alpha=0.6, color=s5.COLORS["xgb"], linewidths=0,
               label=f"model better ({better.sum()})")
    ax.scatter(b[~better], a[~better], s=12, alpha=0.6, color="#b0b0ab", linewidths=0,
               label=f"average better ({(~better).sum()})")
    hi = float(max(a.max(), b.max())) * 1.05
    ax.plot([0, hi], [0, hi], color=s5.MUTED, lw=1)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Athlete average (train+val): athlete's test MAE", color=s5.MUTED, fontsize=8)
    ax.set_ylabel(f"{label}: athlete's test MAE", color=s5.MUTED, fontsize=8)
    ax.set_title("Per-athlete test error (below the line = model better)", fontsize=9, color=s5.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_acwr(zones, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if zones.empty:
        return
    types = zones["acwr_type"].unique()
    fig, axes = plt.subplots(1, len(types), figsize=(5 * len(types), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, types):
        z = zones[zones["acwr_type"] == t]
        ax.barh(z["zone"], z["share_pct"], color=["#2a78d6", "#1baf7a", "#e8a33d", "#d64545", "#b0b0ab"][:len(z)])
        for y, (s, r) in enumerate(zip(z["share_pct"], z["mean_rpc_vs_own_average"])):
            ax.text(s + 0.5, y, f"{s:.1f}%  (RPC vs own avg {r:+.3f})", va="center", fontsize=7, color=s5.TEXT)
        ax.set_title(t, fontsize=9, color=s5.TEXT, loc="left")
        ax.set_xlabel("Share of test samples (%)", color=s5.MUTED, fontsize=8)
        ax.set_xlim(0, z["share_pct"].max() * 1.6)
        ax.invert_yaxis()
        _style(ax)
    fig.suptitle("ACWR zones in the test period", fontsize=10, color=s5.TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main(quick):
    global OUTPUT_DIR
    if quick:
        OUTPUT_DIR = OUTPUT_DIR / "_quick"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    n_boot = 1000 if quick else N_BOOT
    rng = np.random.default_rng(2026)

    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 7 TEST-SET EVALUATION" + (" (QUICK TEST)" if quick else ""))
    print("=" * 70)
    print()
    try:
        train, val, test, raw_test, features, seq = build_inputs(quick)

        print()
        print("=" * 70)
        print("STEP 2 - PREDICTIONS (NO TRAINING)")
        print("=" * 70)
        per_seed = model_predictions(test, features, seq)
        preds = dict(naive_predictions(train, val, test))
        for stage in MODEL_DIRS:
            for model in MODELS:
                for variant in VARIANTS:
                    preds[f"{stage}_{model}_{variant}"] = np.mean(
                        [per_seed[(stage, model, variant, s)] for s in SEEDS], axis=0)
        preds["tuned_average_full"] = np.mean([preds[f"tuned_{m}_full"] for m in MODELS], axis=0)
        for name, pred in preds.items():
            assert np.isfinite(pred).all(), f"non-finite predictions: {name}"

        print()
        print("=" * 70)
        print("STEP 3 - ACCURACY")
        print("=" * 70)
        rows = []
        for (stage, model, variant, seed), pred in per_seed.items():
            rows.append({**s5.evaluate(f"{stage}_{model}_{variant}_seed{seed}", model, variant, seed, test, pred),
                         "stage": stage})
        runs = pd.DataFrame(rows)
        runs.to_csv(OUTPUT_DIR / "stage7_metrics_runs.csv", index=False)

        seed_stats = runs.groupby(["stage", "model", "variant"])["mae"].agg(["mean", "std"])
        summary = []
        for name, pred in preds.items():
            m = s5.evaluate(name, name, "", 0, test, pred)
            row = {"prediction": name, "label": NAIVE_LABELS.get(name, name), **{k: m[k] for k in (
                "mae", "rmse", "r2", "mape", "mae_unweighted", "athlete_mae_median", "athlete_mae_q1",
                "athlete_mae_q3", "finish_time_mae_s", "finish_time_mape")}, "bias": bias(test, pred)}
            parts = name.split("_")
            if parts[0] in MODEL_DIRS and parts[1] in MODELS:
                key = (parts[0], parts[1], "_".join(parts[2:]))
                row["mae_per_seed_mean"], row["mae_per_seed_std"] = seed_stats.loc[key, "mean"], seed_stats.loc[key, "std"]
            summary.append(row)
        summary = pd.DataFrame(summary).sort_values("mae")
        summary.to_csv(OUTPUT_DIR / "stage7_metrics_summary.csv", index=False)
        dist = s5.by_distance(test, preds)
        dist.to_csv(OUTPUT_DIR / "stage7_metrics_by_distance.csv", index=False)
        levels = by_rpc_level(test, {k: preds[k] for k in
                                     ["tuned_xgb_full", "tuned_gru_full", "tuned_lstm_full",
                                      "tuned_average_full", MAIN_NAIVE]})
        levels.to_csv(OUTPUT_DIR / "stage7_metrics_by_rpc_level.csv", index=False)
        for _, r in summary.iterrows():
            print(f"  {r['prediction']:<30} MAE {r['mae']:.4f}  R2 {r['r2']:.3f}  "
                  f"finish-time {r['finish_time_mae_s']:.0f}s ({r['finish_time_mape']:.1f}%)")

        print()
        print("=" * 70)
        print("STEP 4 - SIGNIFICANCE (PER-ATHLETE, BOOTSTRAP + WILCOXON, HOLM)")
        print("=" * 70)
        sig = significance(test, preds, rng, n_boot)
        sig.to_csv(OUTPUT_DIR / "stage7_significance.csv", index=False)
        deploy, deploy_reason = deployment_choice(sig)
        deploy_key = f"tuned_{deploy}_full"
        print(f"  Deployment model: {deploy.upper()} - {deploy_reason}")

        print()
        print("=" * 70)
        print("STEP 5 - ACWR ZONES (DESCRIPTIVE)")
        print("=" * 70)
        zones, acwr_corr = acwr_analysis(raw_test, test, preds, deploy_key)
        zones.to_csv(OUTPUT_DIR / "stage7_acwr_zones.csv", index=False)
        acwr_corr.to_csv(OUTPUT_DIR / "stage7_acwr_correlation.csv", index=False)

        pred_df = test[["sample_id", "id", "target_date", "target_distance_km", "target_seconds",
                        cd.TARGET]].copy()
        for name, pred in preds.items():
            pred_df[name] = pred
        pred_df.to_csv(OUTPUT_DIR / "stage7_test_predictions.csv", index=False)

        s5.plot_pred_vs_actual(test, {f"{m}_full": preds[f"tuned_{m}_full"] for m in MODELS},
                               OUTPUT_DIR / "stage7_pred_vs_actual.png")
        plot_distance(dist, OUTPUT_DIR / "stage7_error_by_distance.png")
        plot_athletes(test, preds, deploy_key, s5.MODEL_LABELS[deploy], OUTPUT_DIR / "stage7_per_athlete.png")
        plot_acwr(zones, OUTPUT_DIR / "stage7_acwr_zones.png")
        (OUTPUT_DIR / "stage7_deployment_choice.json").write_text(json.dumps(
            {"model": deploy, "reason": deploy_reason,
             "files": [f"06_tuning/models/{deploy}_full_seed{s}" + (".json" if deploy == "xgb" else ".keras")
                       for s in SEEDS]}, indent=2), encoding="utf-8")
        print("All checks passed.")

        # ---------------- summary text ----------------
        S = summary.set_index("prediction")

        def mae(k):
            return S.loc[k, "mae"]

        def red(a, b):
            return 100 * (mae(b) - mae(a)) / mae(b)

        def sig_line(r):
            verdict = ("significantly lower error" if r["a_significantly_better"] else
                       "significantly higher error" if r["a_significantly_worse"] else "no significant difference")
            return (f"  {r['a']:<26} vs {r['b']:<28} diff {r['mean_diff']:+.4f} "
                    f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  p_boot(Holm) {r['p_bootstrap_holm']:.3f}  "
                    f"p_Wilcoxon(Holm) {r['p_wilcoxon_holm']:.3f}  "
                    f"better for {r['athletes_a_better']}/{r['athletes']} athletes -> {verdict}")

        lines = [
            "CAPSTONE - STAGE 7 TEST-SET EVALUATION SUMMARY" + (" (QUICK TEST)" if quick else ""),
            "=" * 60,
            "",
            "DATA",
            "-" * 60,
            f"Test: {len(test):,} samples, {test['id'].nunique()} athletes (first use of the test split; "
            "nothing fitted on it).",
            "Models: tuned (Stage 6) and baseline (Stage 5), each = mean of seeds " + str(SEEDS) + ".",
            "Metrics athlete-weighted (each athlete counts equally) unless stated.",
            "",
            "RESULTS (test)",
            "-" * 60,
            f"{'Prediction':<32}{'MAE':<9}{'RMSE':<9}{'R2':<8}{'MAPE %':<9}{'Finish time':<18}{'Bias':<9}"
            f"{'Per-seed MAE'}",
        ]
        for _, r in summary.iterrows():
            ps = (f"{r['mae_per_seed_mean']:.4f} ± {r['mae_per_seed_std']:.4f}"
                  if pd.notna(r.get("mae_per_seed_mean")) else "")
            lines.append(f"{r['prediction']:<32}{r['mae']:<9.4f}{r['rmse']:<9.4f}{r['r2']:<8.3f}{r['mape']:<9.2f}"
                         f"{r['finish_time_mae_s']:.0f} s ({r['finish_time_mape']:.1f}%)   "
                         f"{r['bias']:+.4f}  {ps}")
        lines += ["", "Naive baselines: " + "; ".join(f"{k} = {v}" for k, v in NAIVE_LABELS.items()), ""]

        lines += ["ERROR REDUCTION (test; (baseline MAE - model MAE) / baseline MAE)", "-" * 60]
        for m in MODELS:
            lines.append(f"{s5.MODEL_LABELS[m]:<8} full vs athlete average (train+val) {red(f'tuned_{m}_full', MAIN_NAIVE):+.1f}% | "
                         f"no past perf. vs overall average {red(f'tuned_{m}_no_past_rpc', 'naive_global_mean'):+.1f}% | "
                         f"adding past perf. {red(f'tuned_{m}_full', f'tuned_{m}_no_past_rpc'):+.1f}% | "
                         f"tuned vs Stage 5 (full) {red(f'tuned_{m}_full', f'baseline_{m}_full'):+.1f}%")
        lines.append(f"Average of 3 tuned full models vs athlete average (train+val): "
                     f"{red('tuned_average_full', MAIN_NAIVE):+.1f}%")

        lines += ["", "SIGNIFICANCE (per athlete; diff < 0 = first prediction has lower error; "
                  "95% bootstrap CI; Holm-adjusted p within each question)", "-" * 60]
        for q, g in sig.groupby("question", sort=False):
            lines.append(q)
            lines += [sig_line(r) for _, r in g.iterrows()]
        lines += ["", "DEPLOYMENT MODEL (rule fixed before the test run)", "-" * 60,
                  f"{deploy.upper()}: {deploy_reason}"]

        lines += ["", "MAE BY BEST-EFFORT DISTANCE (test)", "-" * 60]
        piv = dist.pivot_table(index="prediction", columns="distance_km", values="rpc_mae")
        n_by = dist.groupby("distance_km")["n"].first()
        lines.append("n per distance: " + ", ".join(f"{k:g} km {int(v)}" for k, v in n_by.items()))
        lines.append(piv.loc[[k for k in ["tuned_xgb_full", "tuned_gru_full", "tuned_lstm_full",
                                          "tuned_average_full", MAIN_NAIVE, "naive_last_rpc",
                                          "naive_global_mean"] if k in piv.index]].round(4).to_string())

        lines += ["", "REGRESSION TO THE MEAN: mean error (pred - actual) by actual-RPC fifth (test)", "-" * 60,
                  levels.pivot_table(index="prediction", columns="rpc_fifth", values="mean_error",
                                     observed=True).round(4).to_string()]

        lines += ["", "ACWR ZONES IN THE TEST PERIOD (descriptive; no injury labels in the data)", "-" * 60,
                  zones.round(4).to_string(index=False), "", acwr_corr.round(4).to_string(index=False)]

        q1 = sig[sig["question"].str.startswith("Q1 ")]
        q2 = sig[sig["question"].str.startswith("Q2")]
        q5 = sig[sig["question"].str.startswith("Q5")]
        lines += [
            "", "VERDICTS (plain language)", "-" * 60,
            f"RQ1 - training load alone: {int(q2['a_significantly_better'].sum())} of {len(q2)} comparisons "
            "(no-past-performance models vs overall average / last result) significantly better.",
            f"RQ1 - full models vs runner's own average (train+val): {int(q1['a_significantly_better'].sum())} of "
            f"{len(q1)} significantly better, {int(q1['a_significantly_worse'].sum())} significantly worse.",
            f"Model comparison: {int((q5['a_significantly_better'] | q5['a_significantly_worse']).sum())} of "
            f"{len(q5)} pairwise differences significant.",
            f"Deployment: {deploy.upper()}.",
            "",
            "STATUS", "-" * 60,
            "Stage 7 test evaluation completed (one shot - do not re-tune on these results). "
            "Next: Stage 8 interpretation.",
        ]
        (OUTPUT_DIR / "stage7_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines[lines.index("VERDICTS (plain language)"):]))
        print(f"\nOutputs written to:\n{OUTPUT_DIR}")

    except Exception as exc:
        (LOG_DIR / "stage7_test_error.log").write_text(
            "STAGE 7 ERROR\n" + "=" * 60 + "\n\n" + str(exc) + "\n\n" + traceback.format_exc(),
            encoding="utf-8")
        print("\nERROR - see logs/stage7_test_error.log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test: 25 test athletes, 1,000 bootstraps")
    main(parser.parse_args().quick)
