"""
10_stage8_interpretation.py

CAPSTONE
Stage 8 - Model Interpretation (+ exploratory checks added after Stage 7)

Model interpreted: the deployment model chosen in Stage 7 (XGBoost, tuned,
full; prediction = mean of its 3 seeds). Everything is computed on the TEST
split, except the surrogate trees, which learn the model's behaviour on the
training split and are checked on test. No model is re-trained or re-tuned.

Part A - planned interpretation (manuscript 3.3, Stage 8):
  A1  Permutation importance on test (single features + feature groups)
  A2  PDP / ICE for the training-load features (full model and the
      training-load-only model)
  A3  What-if scenarios (train more / less, acute spike, detraining)
  A4  Surrogate decision trees (fidelity + readable rules)

Part B - EXPLORATORY checks added after the Stage 7 results (team decision
2026-10-22; report them as exploratory, not pre-specified):
  B1  Change tracking: when the model predicts a runner will be above/below
      their usual level, are they? (all three tuned models + their average)
  B2  Gain vs runner history: does the model's advantage over the runner's
      average grow with the amount of training history?
  B3  Equivalence with the runner's average (bootstrap TOST, margin 0.002 RPC)

Usage:
    py 10_stage8_interpretation.py           (full, ~3-10 min)
    py 10_stage8_interpretation.py --quick   (25 test athletes -> 08_interpretation/_quick)
Requires: pandas, numpy, scipy, scikit-learn, xgboost, matplotlib
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

s5 = importlib.import_module("07_stage5_baseline_models")  # colours / labels


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = cd.ROOT / "08_interpretation"
TUNED_DIR = cd.ROOT / "06_tuning"
TEST_DIR = cd.ROOT / "07_test_evaluation"
LOG_DIR = cd.ROOT / "logs"

SEEDS = [11, 22, 33]
AVG = "naive_athlete_mean_trainval"            # the runner's average (Stage 7 main comparison)
TRACK_MODELS = {"tuned_xgb_full": "XGBoost", "tuned_gru_full": "GRU",
                "tuned_lstm_full": "LSTM", "tuned_average_full": "Average of 3"}
LOAD_FEATURES = ["ctl", "atl", "acwr", "mean_ga_speed_28", "longest_run_km_28", "mean_hr_pct_28"]
FEATURE_LABELS = {
    "prior_mean_rpc": "Personal average RPC", "last_rpc": "Last RPC", "best_rpc_90": "Best RPC (90 d)",
    "mean_ga_speed_28": "Mean grade-adj. speed (28 d)", "mean_hr_pct_28": "Mean %HRmax (28 d)",
    "ctl": "CTL (chronic load)", "atl": "ATL (acute load)", "acwr": "ACWR",
    "longest_run_km_28": "Longest run (28 d)", "mean_ga_speed_28_missing": "Speed missing flag",
}
EQUIV_MARGIN = 0.002        # RPC; chosen after Stage 7 (about 6 s on a typical 5K) - exploratory
MIN_DEV = 0.005             # |predicted deviation| needed to count a "call" in the direction check
TYPICAL_KM = 5.0


def label(f):
    return FEATURE_LABELS.get(f, f)


# ============================================================
# DATA AND MODEL
# ============================================================

def load_data(quick):
    samples = cd.add_personal_level(cd.load_samples())
    if quick:
        keep = samples.loc[samples["split"] == "test", "id"].drop_duplicates().head(25)
        samples = samples[samples["id"].isin(keep)]
    params = cd.load_params()
    tab = cd.prepare_tabular(samples, params)
    train = tab[tab["split"] == "train"].reset_index(drop=True)
    test = tab[tab["split"] == "test"].reset_index(drop=True)

    preds = pd.read_csv(TEST_DIR / "stage7_test_predictions.csv")
    preds = test[["sample_id"]].merge(preds, on="sample_id", how="left", validate="one_to_one")
    assert preds[cd.TARGET].notna().all(), "test samples missing from stage7_test_predictions.csv"
    assert np.allclose(preds[cd.TARGET].to_numpy(), test[cd.TARGET].to_numpy())

    deploy = json.loads((TEST_DIR / "stage7_deployment_choice.json").read_text(encoding="utf-8"))
    assert deploy["model"] == "xgb", "this script interprets the XGBoost deployment model"
    feats_full = cd.load_feature_list("tabular")
    feats = {"full": feats_full, "no_past_rpc": [f for f in feats_full if f not in cd.PRIOR_TABULAR]}
    return samples, train, test, preds, feats


class SeedEnsemble:
    """Mean of the 3 seed models - exactly the Stage 7 prediction."""

    def __init__(self, variant, feats):
        from xgboost import XGBRegressor
        self.feats = feats
        self.models = []
        for seed in SEEDS:
            m = XGBRegressor()
            m.load_model(str(TUNED_DIR / "models" / f"xgb_{variant}_seed{seed}.json"))
            self.models.append(m)

    def predict(self, X):
        return np.mean([m.predict(X[self.feats]) for m in self.models], axis=0)


def wmae(y, pred, w):
    return float(np.sum(w * np.abs(pred - y)) / w.sum())


def five_k_seconds(rpc_from, rpc_to):
    """Change in 5K finish time (s) when RPC moves from rpc_from to rpc_to (negative = faster)."""
    return cd.time_from_rpc(np.asarray(rpc_to), TYPICAL_KM) - cd.time_from_rpc(np.asarray(rpc_from), TYPICAL_KM)


# ============================================================
# A1 PERMUTATION IMPORTANCE
# ============================================================

def permutation_importance(model, test, feats, repeats, rng):
    y = test[cd.TARGET].to_numpy()
    w = cd.athlete_weights(test)
    base = wmae(y, model.predict(test), w)
    groups = {f: [f] for f in feats}
    groups["GROUP: past performance"] = [f for f in feats if f in cd.PRIOR_TABULAR]
    groups["GROUP: training load"] = [f for f in feats if f not in cd.PRIOR_TABULAR]
    rows = []
    for name, cols in groups.items():
        incs = []
        for _ in range(repeats):
            X = test.copy()
            order = rng.permutation(len(X))
            X[cols] = X[cols].to_numpy()[order]          # shuffle the group jointly
            incs.append(wmae(y, model.predict(X), w) - base)
        rows.append({"feature": name, "label": label(name), "group": name.startswith("GROUP"),
                     "mae_increase_mean": np.mean(incs), "mae_increase_sd": np.std(incs),
                     "pct_of_base": 100 * np.mean(incs) / base})
    df = pd.DataFrame(rows).sort_values(["group", "mae_increase_mean"], ascending=[True, False])
    return df, base


# ============================================================
# A2 PDP / ICE
# ============================================================

def pdp_ice(models, test, features, n_grid, n_ice, rng):
    rows, ice_rows = [], []
    ice_idx = rng.choice(len(test), size=min(n_ice, len(test)), replace=False)
    for f in features:
        lo, hi = np.percentile(test[f], [5, 95])
        grid = np.linspace(lo, hi, n_grid)
        for variant, model in models.items():
            if f not in model.feats:
                continue
            curves = []
            for g in grid:
                X = test.copy()
                X[f] = g
                curves.append(model.predict(X))
            curves = np.array(curves)                       # (grid, samples)
            pdp = curves.mean(axis=1)
            centred = curves[:, ice_idx] - curves[0, ice_idx]
            for i, g in enumerate(grid):
                rows.append({"feature": f, "variant": variant, "value": g, "pdp": pdp[i],
                             "ice_centred_p10": np.percentile(centred[i], 10),
                             "ice_centred_p50": np.percentile(centred[i], 50),
                             "ice_centred_p90": np.percentile(centred[i], 90)})
            if variant == "full":
                for j, s in enumerate(ice_idx):
                    ice_rows.append(pd.DataFrame({"feature": f, "sample": int(s), "value": grid,
                                                  "ice_centred": centred[:, j]}))
    pdp_df = pd.DataFrame(rows)
    effects = []
    from scipy.stats import spearmanr
    for (f, v), g in pdp_df.groupby(["feature", "variant"], sort=False):
        mid = float(np.median(g["pdp"]))
        rng_rpc = g["pdp"].iloc[-1] - g["pdp"].iloc[0]            # 5th -> 95th percentile
        effects.append({"feature": f, "label": label(f), "variant": v,
                        "value_p5": g["value"].iloc[0], "value_p95": g["value"].iloc[-1],
                        "pdp_change_p5_to_p95": rng_rpc,
                        "pdp_max_minus_min": g["pdp"].max() - g["pdp"].min(),
                        "five_k_seconds_change": float(five_k_seconds(mid, mid + rng_rpc)),
                        "direction_spearman": spearmanr(g["value"], g["pdp"]).statistic})
    ice_df = pd.concat(ice_rows, ignore_index=True) if ice_rows else pd.DataFrame()
    return pdp_df, ice_df, pd.DataFrame(effects)


# ============================================================
# A3 WHAT-IF SCENARIOS
# ============================================================

def what_if(model, test):
    base = model.predict(test)
    scen = {
        "Train 20% more (ATL, CTL, longest run x1.2; ACWR unchanged)":
            {"atl": 1.2, "ctl": 1.2, "longest_run_km_28": 1.2},
        "Train 20% less (ATL, CTL, longest run x0.8; ACWR unchanged)":
            {"atl": 0.8, "ctl": 0.8, "longest_run_km_28": 0.8},
        "Detraining (ATL, CTL, longest run x0.5; ACWR unchanged)":
            {"atl": 0.5, "ctl": 0.5, "longest_run_km_28": 0.5},
        "Acute spike (ATL x1.5 -> ACWR x1.5; CTL unchanged)":
            {"atl": 1.5, "acwr": 1.5},
        "Faster training runs (grade-adjusted speed x1.05)":
            {"mean_ga_speed_28": 1.05},
    }
    rows = []
    for name, changes in scen.items():
        X = test.copy()
        for col, factor in changes.items():
            if col in model.feats:
                X[col] = X[col] * factor
        d = model.predict(X) - base
        secs = five_k_seconds(base, base + d)
        rows.append({"scenario": name, "mean_rpc_change": d.mean(), "median_rpc_change": np.median(d),
                     "share_faster_pct": 100 * (d > 0).mean(), "share_unchanged_pct": 100 * (np.abs(d) < 1e-6).mean(),
                     "mean_5k_seconds_change": secs.mean(), "median_5k_seconds_change": np.median(secs)})
    return pd.DataFrame(rows)


# ============================================================
# A4 SURROGATE TREES
# ============================================================

def surrogates(models, train, test):
    from sklearn.tree import DecisionTreeRegressor, export_text
    rows, texts, trees = [], {}, {}
    w_train, w_test = cd.athlete_weights(train), cd.athlete_weights(test)
    y_test = test[cd.TARGET].to_numpy()
    for variant, model in models.items():
        target_train = model.predict(train)
        target_test = model.predict(test)
        for depth in (2, 3, 4):
            tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=100, random_state=0)
            tree.fit(train[model.feats], target_train, sample_weight=w_train)
            s = tree.predict(test[model.feats])
            ss_res = np.sum(w_test * (s - target_test) ** 2)
            ss_tot = np.sum(w_test * (target_test - np.average(target_test, weights=w_test)) ** 2)
            rows.append({"variant": variant, "depth": depth, "leaves": tree.get_n_leaves(),
                         "fidelity_r2_vs_xgboost_test": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                         "surrogate_mae_vs_actual_test": wmae(y_test, s, w_test),
                         "xgboost_mae_vs_actual_test": wmae(y_test, target_test, w_test)})
            if depth == 3:
                texts[variant] = export_text(tree, feature_names=list(model.feats), decimals=3)
                trees[variant] = tree
    return pd.DataFrame(rows), texts, trees


# ============================================================
# B1 CHANGE TRACKING (exploratory)
# ============================================================

def cluster_boot(ids, stat_fn, n_boot, rng):
    """Bootstrap over athletes: stat_fn(index array) -> value."""
    groups = pd.Series(np.arange(len(ids))).groupby(ids).apply(np.array).tolist()
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        vals.append(stat_fn(idx))
    return np.percentile(vals, [2.5, 97.5])


def change_tracking(preds, n_boot, rng):
    from scipy.stats import spearmanr
    ids = preds["id"].to_numpy()
    act = (preds[cd.TARGET] - preds[AVG]).to_numpy()
    act_w = act - pd.Series(act).groupby(ids).transform("mean").to_numpy()   # within-runner
    rows, bins = [], []
    for col, name in TRACK_MODELS.items():
        pdv = (preds[col] - preds[AVG]).to_numpy()
        pdv_w = pdv - pd.Series(pdv).groupby(ids).transform("mean").to_numpy()

        def r(idx, a=act, p=pdv):
            return np.corrcoef(p[idx], a[idx])[0, 1]

        def r_within(idx):
            return np.corrcoef(pdv_w[idx], act_w[idx])[0, 1]

        def slope(idx):
            return np.polyfit(pdv[idx], act[idx], 1)[0]

        calls = np.abs(pdv) >= MIN_DEV

        def hit(idx):
            idx = idx[calls[idx]]
            return np.mean(np.sign(pdv[idx]) == np.sign(act[idx]))

        all_idx = np.arange(len(act))
        rows.append({
            "model": name,
            "pearson_r": r(all_idx), "pearson_ci": cluster_boot(ids, r, n_boot, rng),
            "spearman_r": spearmanr(pdv, act).statistic,
            "within_runner_r": r_within(all_idx), "within_runner_ci": cluster_boot(ids, r_within, n_boot, rng),
            "slope": slope(all_idx), "slope_ci": cluster_boot(ids, slope, n_boot, rng),
            "calls_n": int(calls.sum()), "calls_pct": 100 * calls.mean(),
            "direction_hit_rate": hit(all_idx), "direction_hit_ci": cluster_boot(ids, hit, n_boot, rng),
        })
        q = pd.qcut(pdv, 10, labels=False, duplicates="drop")
        for b, g in pd.DataFrame({"q": q, "p": pdv, "a": act}).groupby("q"):
            bins.append({"model": name, "decile": int(b) + 1, "mean_predicted_deviation": g["p"].mean(),
                         "mean_actual_deviation": g["a"].mean(), "n": len(g),
                         "se": g["a"].std(ddof=1) / np.sqrt(len(g))})
    out = pd.DataFrame(rows)
    for c in ("pearson_ci", "within_runner_ci", "slope_ci", "direction_hit_ci"):
        out[c.replace("_ci", "_ci_low")] = out[c].map(lambda v: v[0])
        out[c.replace("_ci", "_ci_high")] = out[c].map(lambda v: v[1])
        out = out.drop(columns=c)
    return out, pd.DataFrame(bins)


# ============================================================
# B2 GAIN vs HISTORY (exploratory)
# ============================================================

def gain_vs_history(preds, samples, n_boot, rng):
    from scipy.stats import spearmanr
    runs = pd.read_csv(cd.FEATURE_DIR / "stage3_run_features.csv", usecols=["id", "calendar_date"],
                       parse_dates=["calendar_date"])
    first_test = preds.groupby("id")["target_date"].min().pipe(pd.to_datetime)
    runs = runs[runs["id"].isin(first_test.index)]
    prior_runs = (runs["calendar_date"] < runs["id"].map(first_test)).groupby(runs["id"]).sum()
    prior_samples = samples[samples["split"].isin(["train", "val"])].groupby("id").size()

    err = preds.assign(**{c: (preds[c] - preds[cd.TARGET]).abs() for c in ["tuned_xgb_full", "tuned_average_full", AVG]})
    per = err.groupby("id").agg(mae_xgb=("tuned_xgb_full", "mean"), mae_avg3=("tuned_average_full", "mean"),
                                mae_runner_avg=(AVG, "mean"), test_samples=(cd.TARGET, "size"))
    per["prior_runs"] = per.index.map(prior_runs).fillna(0).astype(int)
    per["prior_samples"] = per.index.map(prior_samples).fillna(0).astype(int)
    for m in ("xgb", "avg3"):
        per[f"gain_{m}_pct"] = 100 * (per["mae_runner_avg"] - per[f"mae_{m}"]) / per["mae_runner_avg"]
        per[f"gain_{m}_abs"] = per["mae_runner_avg"] - per[f"mae_{m}"]

    corr = []
    for m in ("xgb", "avg3"):
        for h in ("prior_runs", "prior_samples", "test_samples"):
            x, y = per[h].to_numpy(), per[f"gain_{m}_abs"].to_numpy()
            r = spearmanr(x, y)
            boots = [spearmanr(x[i], y[i]).statistic for i in
                     (rng.integers(0, len(x), len(x)) for _ in range(n_boot))]
            corr.append({"model": "XGBoost" if m == "xgb" else "Average of 3", "history": h,
                         "spearman_r": r.statistic, "p": r.pvalue,
                         "ci_low": np.nanpercentile(boots, 2.5), "ci_high": np.nanpercentile(boots, 97.5)})
    per["history_quartile"] = pd.qcut(per["prior_runs"].rank(method="first"), 4,
                                      labels=["Q1 least", "Q2", "Q3", "Q4 most"])
    groups = []
    for qn, g in per.groupby("history_quartile", observed=True):
        for m in ("xgb", "avg3"):
            v = g[f"gain_{m}_abs"].to_numpy()
            boots = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n_boot)]
            groups.append({"model": "XGBoost" if m == "xgb" else "Average of 3", "quartile": qn,
                           "athletes": len(g), "prior_runs_range": f"{g['prior_runs'].min()}-{g['prior_runs'].max()}",
                           "mean_gain_abs": v.mean(), "ci_low": np.percentile(boots, 2.5),
                           "ci_high": np.percentile(boots, 97.5),
                           "mean_gain_pct": g[f"gain_{m}_pct"].mean(),
                           "share_model_better_pct": 100 * (v > 0).mean()})
    return per.reset_index(), pd.DataFrame(corr), pd.DataFrame(groups)


# ============================================================
# B3 EQUIVALENCE (exploratory)
# ============================================================

def equivalence(preds, n_boot, rng):
    a_avg = (preds[AVG] - preds[cd.TARGET]).abs().groupby(preds["id"]).mean()
    rows = []
    for col, name in TRACK_MODELS.items():
        d = ((preds[col] - preds[cd.TARGET]).abs().groupby(preds["id"]).mean() - a_avg).to_numpy()
        boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
        lo, hi = np.percentile(boot, [5, 95])                       # 90% CI = TOST at alpha 0.05
        rows.append({"model": name, "mean_diff": d.mean(), "ci90_low": lo, "ci90_high": hi,
                     "margin": EQUIV_MARGIN, "equivalent_within_margin": bool(lo > -EQUIV_MARGIN and hi < EQUIV_MARGIN),
                     "smallest_margin_showing_equivalence": max(abs(lo), abs(hi))})
    mid = float(np.median(preds[cd.TARGET]))
    secs = float(abs(five_k_seconds(mid, mid + EQUIV_MARGIN)))
    return pd.DataFrame(rows), secs


# ============================================================
# PLOTS
# ============================================================

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _style(ax):
    ax.grid(color=s5.GRID, lw=0.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)


def plot_importance(imp, path):
    plt = _plt()
    d = imp[~imp["group"]].sort_values("mae_increase_mean")
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["#8a6fd1" if f in cd.PRIOR_TABULAR else s5.COLORS["xgb"] for f in d["feature"]]
    ax.barh(d["label"], d["mae_increase_mean"], xerr=d["mae_increase_sd"], color=colors, ecolor=s5.MUTED)
    ax.set_xlabel("Increase in test MAE when shuffled (RPC)", color=s5.MUTED, fontsize=8)
    ax.set_title("Permutation importance on test - XGBoost (purple = past performance, green = training)",
                 fontsize=8.5, color=s5.TEXT, loc="left")
    ax.tick_params(labelsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pdp(pdp_df, ice_df, path):
    plt = _plt()
    feats = list(dict.fromkeys(pdp_df["feature"]))
    cols = 3
    rows = int(np.ceil(len(feats) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 2.9 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feats):
        full = pdp_df[(pdp_df["feature"] == f) & (pdp_df["variant"] == "full")]
        if not ice_df.empty:
            for _, g in ice_df[ice_df["feature"] == f].groupby("sample"):
                ax.plot(g["value"], full["pdp"].iloc[0] + g["ice_centred"].to_numpy(),
                        color=s5.COLORS["xgb"], alpha=0.06, lw=0.8)
        ax.plot(full["value"], full["pdp"], color=s5.COLORS["xgb"], lw=2.2, label="full model (PDP)")
        nop = pdp_df[(pdp_df["feature"] == f) & (pdp_df["variant"] == "no_past_rpc")]
        if len(nop):
            ax2 = ax.twinx()
            ax2.plot(nop["value"], nop["pdp"], color="#8a6fd1", lw=1.6, ls="--", label="training-load-only model")
            ax2.tick_params(labelsize=7, colors="#8a6fd1")
            ax2.spines["top"].set_visible(False)
        ax.set_title(label(f), fontsize=9, color=s5.TEXT, loc="left")
        ax.tick_params(labelsize=7)
        _style(ax)
    for ax in axes[len(feats):]:
        ax.axis("off")
    axes[0].set_ylabel("Predicted RPC", color=s5.MUTED, fontsize=8)
    fig.suptitle("Partial dependence (thick) and ICE (thin) on test; dashed = training-load-only model (right axis)",
                 fontsize=9, color=s5.TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tracking(bins, path):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    palette = {"XGBoost": s5.COLORS["xgb"], "GRU": s5.COLORS["gru"], "LSTM": s5.COLORS["lstm"], "Average of 3": "#333333"}
    for name, g in bins.groupby("model", sort=False):
        ax.errorbar(g["mean_predicted_deviation"], g["mean_actual_deviation"], yerr=1.96 * g["se"],
                    marker="o", ms=3.5, lw=1.2, capsize=2, color=palette.get(name, "#555"), label=name)
    lim = float(np.nanmax(np.abs(bins[["mean_predicted_deviation", "mean_actual_deviation"]].to_numpy()))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], color=s5.MUTED, lw=1, ls=":")
    ax.axhline(0, color=s5.MUTED, lw=0.6)
    ax.axvline(0, color=s5.MUTED, lw=0.6)
    ax.set_xlabel("Predicted deviation from runner's average (RPC)", color=s5.MUTED, fontsize=8)
    ax.set_ylabel("Actual deviation from runner's average (RPC)", color=s5.MUTED, fontsize=8)
    ax.set_title("Change tracking on test (deciles of predicted deviation, 95% CI)", fontsize=8.5,
                 color=s5.TEXT, loc="left")
    ax.legend(frameon=False, fontsize=7)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_history(per, groups, path):
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    ax = axes[0]
    ax.scatter(per["prior_runs"], per["gain_xgb_abs"], s=10, alpha=0.5, color=s5.COLORS["xgb"], linewidths=0)
    ax.axhline(0, color=s5.MUTED, lw=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Runs logged before the test period (log scale)", color=s5.MUTED, fontsize=8)
    ax.set_ylabel("Gain vs runner's average (RPC MAE; >0 = model better)", color=s5.MUTED, fontsize=8)
    ax.set_title("Per-runner gain - XGBoost", fontsize=9, color=s5.TEXT, loc="left")
    _style(ax)
    ax = axes[1]
    g = groups[groups["model"] == "XGBoost"]
    x = np.arange(len(g))
    ax.bar(x, g["mean_gain_abs"], color=s5.COLORS["xgb"])
    ax.errorbar(x, g["mean_gain_abs"], yerr=[g["mean_gain_abs"] - g["ci_low"], g["ci_high"] - g["mean_gain_abs"]],
                fmt="none", ecolor=s5.TEXT, capsize=3)
    ax.axhline(0, color=s5.MUTED, lw=0.8)
    ax.set_xticks(x, [f"{q}\n({r} runs)" for q, r in zip(g["quartile"], g["prior_runs_range"])], fontsize=7)
    ax.set_title("Mean gain by history quartile (95% CI)", fontsize=9, color=s5.TEXT, loc="left")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_tree(tree, feats, path):
    plt = _plt()
    from sklearn.tree import plot_tree as sk_plot_tree
    fig, ax = plt.subplots(figsize=(14, 6))
    sk_plot_tree(tree, feature_names=[label(f) for f in feats], filled=True, rounded=True,
                 precision=3, fontsize=7, ax=ax, impurity=False)
    ax.set_title("Surrogate tree (depth 3) imitating the XGBoost deployment model", fontsize=10,
                 color=s5.TEXT, loc="left")
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
    rng = np.random.default_rng(2026)
    repeats, n_boot = (3, 200) if quick else (20, 2000)

    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 8 INTERPRETATION" + (" (QUICK TEST)" if quick else ""))
    print("=" * 70)
    try:
        samples, train, test, preds, feats = load_data(quick)
        models = {v: SeedEnsemble(v, feats[v]) for v in ("full", "no_past_rpc")}
        check = models["full"].predict(test)
        assert np.allclose(check, preds["tuned_xgb_full"].to_numpy(), atol=1e-5), \
            "XGBoost predictions differ from Stage 7"
        print(f"Test: {len(test):,} samples, {test['id'].nunique()} athletes | "
              "XGBoost predictions match Stage 7 exactly.")

        print("\nA1 permutation importance ...")
        imp, base_mae = permutation_importance(models["full"], test, feats["full"], repeats, rng)
        imp_nop, base_nop = permutation_importance(models["no_past_rpc"], test, feats["no_past_rpc"], repeats, rng)
        imp.assign(model="full").pipe(lambda d: pd.concat([d, imp_nop.assign(model="no_past_rpc")])) \
           .to_csv(OUTPUT_DIR / "stage8_permutation_importance.csv", index=False)

        print("A2 PDP / ICE ...")
        load_feats = [f for f in LOAD_FEATURES if f in feats["full"]]
        pdp_df, ice_df, effects = pdp_ice(models, test, load_feats, 10 if quick else 20, 30 if quick else 150, rng)
        pdp_df.to_csv(OUTPUT_DIR / "stage8_pdp.csv", index=False)
        effects.to_csv(OUTPUT_DIR / "stage8_pdp_effects.csv", index=False)

        print("A3 what-if scenarios ...")
        whatif = what_if(models["full"], test)
        whatif.to_csv(OUTPUT_DIR / "stage8_what_if.csv", index=False)

        print("A4 surrogate trees ...")
        sur, sur_text, sur_trees = surrogates(models, train, test)
        sur.to_csv(OUTPUT_DIR / "stage8_surrogate_fidelity.csv", index=False)
        (OUTPUT_DIR / "stage8_surrogate_rules.txt").write_text(
            "\n\n".join(f"=== {v} model, depth 3 (leaf value = predicted RPC) ===\n{t}" for v, t in sur_text.items()),
            encoding="utf-8")

        print("B1 change tracking (exploratory) ...")
        track, track_bins = change_tracking(preds, n_boot, rng)
        track.to_csv(OUTPUT_DIR / "stage8_change_tracking.csv", index=False)
        track_bins.to_csv(OUTPUT_DIR / "stage8_change_tracking_deciles.csv", index=False)

        print("B2 gain vs runner history (exploratory) ...")
        preds_dates = preds.assign(target_date=test["target_date"].to_numpy())
        per, hist_corr, hist_groups = gain_vs_history(preds_dates, samples, n_boot, rng)
        per.to_csv(OUTPUT_DIR / "stage8_gain_per_runner.csv", index=False)
        hist_corr.to_csv(OUTPUT_DIR / "stage8_gain_vs_history.csv", index=False)
        hist_groups.to_csv(OUTPUT_DIR / "stage8_gain_by_history_quartile.csv", index=False)

        print("B3 equivalence (exploratory) ...")
        equiv, margin_secs = equivalence(preds, n_boot * 5, rng)
        equiv.to_csv(OUTPUT_DIR / "stage8_equivalence.csv", index=False)

        print("Plots ...")
        plot_importance(imp, OUTPUT_DIR / "stage8_permutation_importance.png")
        plot_pdp(pdp_df, ice_df, OUTPUT_DIR / "stage8_pdp_ice.png")
        plot_tracking(track_bins, OUTPUT_DIR / "stage8_change_tracking.png")
        plot_history(per, hist_groups, OUTPUT_DIR / "stage8_gain_vs_history.png")
        plot_tree(sur_trees["full"], feats["full"], OUTPUT_DIR / "stage8_surrogate_tree_full.png")
        plot_tree(sur_trees["no_past_rpc"], feats["no_past_rpc"], OUTPUT_DIR / "stage8_surrogate_tree_training_only.png")

        # ---------------- summary ----------------
        def fmt_imp(d):
            return [f"  {r['label']:<34} {r['mae_increase_mean']:+.5f} ± {r['mae_increase_sd']:.5f}  "
                    f"({r['pct_of_base']:+.1f}% of base MAE)" for _, r in d.iterrows()]

        lines = [
            "CAPSTONE - STAGE 8 INTERPRETATION SUMMARY" + (" (QUICK TEST)" if quick else ""),
            "=" * 60, "",
            f"Model: XGBoost deployment model (tuned; mean of seeds {SEEDS}); test split "
            f"({len(test):,} samples, {test['id'].nunique()} athletes). Base test MAE {base_mae:.4f} "
            f"(training-load-only model {base_nop:.4f}).",
            "",
            "PART A - PLANNED INTERPRETATION", "=" * 60, "",
            f"A1 PERMUTATION IMPORTANCE (test; {repeats} shuffles; MAE increase)", "-" * 60,
            "Full model:", *fmt_imp(imp), "Training-load-only model:", *fmt_imp(imp_nop), "",
            "A2 PARTIAL DEPENDENCE (5th -> 95th percentile of each feature on test)", "-" * 60,
        ]
        for _, r in effects.iterrows():
            lines.append(f"  {r['label']:<30} {r['variant']:<12} {r['value_p5']:.3g} -> {r['value_p95']:.3g}: "
                         f"RPC {r['pdp_change_p5_to_p95']:+.4f} (5K {r['five_k_seconds_change']:+.0f} s), "
                         f"direction rho {r['direction_spearman']:+.2f}")
        lines += ["  (rho > 0: higher value -> faster prediction; 5K seconds negative = faster)",
                  "  Caution: ATL, CTL and ACWR are linked (ACWR = ATL/CTL); single-feature PDPs vary one while "
                  "holding the others fixed.", "",
                  "A3 WHAT-IF SCENARIOS (all test samples; full model)", "-" * 60]
        for _, r in whatif.iterrows():
            lines.append(f"  {r['scenario']:<66} RPC {r['mean_rpc_change']:+.4f} | 5K {r['mean_5k_seconds_change']:+.1f} s "
                         f"(median {r['median_5k_seconds_change']:+.1f}) | faster for {r['share_faster_pct']:.0f}%")
        lines += ["", "A4 SURROGATE TREES (fit on training-set XGBoost predictions; checked on test)", "-" * 60,
                  sur.round(4).to_string(index=False), "  Rules (depth 3): stage8_surrogate_rules.txt", "",
                  "PART B - EXPLORATORY CHECKS (added after Stage 7; not pre-specified)", "=" * 60, "",
                  "B1 CHANGE TRACKING: predicted vs actual deviation from the runner's average (train+val)", "-" * 60]
        for _, r in track.iterrows():
            lines.append(f"  {r['model']:<13} r {r['pearson_r']:+.3f} [{r['pearson_ci_low']:+.3f}, {r['pearson_ci_high']:+.3f}] | "
                         f"within-runner r {r['within_runner_r']:+.3f} [{r['within_runner_ci_low']:+.3f}, "
                         f"{r['within_runner_ci_high']:+.3f}] | slope {r['slope']:.2f} [{r['slope_ci_low']:.2f}, "
                         f"{r['slope_ci_high']:.2f}] | direction right {100 * r['direction_hit_rate']:.0f}% "
                         f"[{100 * r['direction_hit_ci_low']:.0f}, {100 * r['direction_hit_ci_high']:.0f}] of "
                         f"{r['calls_n']} calls (|pred. dev| >= {MIN_DEV})")
        lines += ["  (athlete-cluster bootstrap 95% CIs; r > 0 and direction > 50% = the model's movements are real;",
                  "   slope 1 = right size, < 1 = actual changes smaller than predicted)", "",
                  "B2 GAIN vs RUNNER HISTORY (gain = runner-average MAE - model MAE, per runner)", "-" * 60,
                  hist_corr.round(4).to_string(index=False), "",
                  hist_groups.round(4).to_string(index=False), "",
                  f"B3 EQUIVALENCE WITH THE RUNNER'S AVERAGE (90% bootstrap CI within ±{EQUIV_MARGIN} RPC "
                  f"≈ ±{margin_secs:.0f} s on a typical 5K; margin chosen after Stage 7)", "-" * 60,
                  equiv.round(4).to_string(index=False), "",
                  "STATUS", "-" * 60, "Stage 8 interpretation completed. Next: Stage 9 deployment."]
        (OUTPUT_DIR / "stage8_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"\nOutputs written to:\n{OUTPUT_DIR}")

    except Exception as exc:
        (LOG_DIR / "stage8_interpretation_error.log").write_text(
            "STAGE 8 ERROR\n" + "=" * 60 + "\n\n" + str(exc) + "\n\n" + traceback.format_exc(), encoding="utf-8")
        print("\nERROR - see logs/stage8_interpretation_error.log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="smoke test: 25 test athletes")
    main(parser.parse_args().quick)
