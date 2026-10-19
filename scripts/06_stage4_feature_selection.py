"""
06_stage4_feature_selection.py

CAPSTONE
Stage 4 - Splitting and Feature Selection

Inputs (from 05_stage3_feature_engineering.py):
    03_feature_engineering/stage3_samples.csv
    03_feature_engineering/stage3_daily_features.csv

Pipeline:
1. Train / validation / test split (FIRST, so selection never sees val/test)
     - per athlete, chronological 70 / 10 / 20 by target date
     - no purge gap: the split is forward-only in time, so no training
       label or feature uses information from validation/test days
     - athletes with < 10 samples go entirely to training
2. Add the long-run personal level (mean + count of all earlier RPCs) and
   remove columns that must never be model inputs (IDs, dates, target info)
3. Preparation fitted on TRAINING data only
     - cap long gaps at the training 99th percentile
     - ACWR missing -> 1.0 (neutral) + existing acwr_valid flag
     - other missing values -> training median + "<name>_missing" flag
4. Screening: drop constant features and features missing in > 50% of training
5. Filter method: Spearman correlation matrix; |r| >= 0.85 = redundant
     - keep ATL / CTL / ACWR (core study variables) over volume proxies
     - otherwise keep the feature more correlated with RPC
     - VIF table for the remaining features
6. Tree-based importance: XGBoost on training (athlete-weighted), 5 seeds
     - gain importance
     - permutation importance on the validation set (decision metric)
7. Selection: keep features whose permutation importance is clearly above 0
   (mean - 1 SD > 0 across seeds); ATL, CTL, ACWR kept by design.
   Confirm on validation: use the selected set only if its MAE is within 1%
   of the full (post-redundancy) set.
8. RNN daily sequence features: correlation-based redundancy only
9. Checks + outputs to 04_feature_selection/

Requires: pandas, numpy, scikit-learn, xgboost, matplotlib

Usage:
    python 06_stage4_feature_selection.py
"""

from pathlib import Path
import json
import traceback

import numpy as np
import pandas as pd

import capstone_data as cd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Users\ACER\Desktop\CAPSTONE GAME TIME")

FEATURE_DIR = ROOT / "03_feature_engineering"
OUTPUT_DIR = ROOT / "04_feature_selection"
LOG_DIR = ROOT / "logs"

WINDOW_DAYS = 28

TRAIN_FRAC, VAL_FRAC = 0.70, 0.10        # test = remaining 20%
MIN_SAMPLES_TO_SPLIT = 10                # fewer -> training only

CORR_THRESHOLD = 0.85
MAX_MISSING_FRAC = 0.50
CAP_QUANTILE = 0.99
CAP_COLUMNS = ["days_since_last_run", "days_since_last_target"]
ACWR_COLUMNS = ["acwr", "acwr_km"]       # missing -> 1.0 (acwr_valid marks it)
ACWR_ZONE_COLUMNS = ["acwr_zone", "acwr_km_zone"]  # missing -> 1 (Sweet Spot code)

# Core study variables (research questions + alert framework): never dropped
PROTECTED = ["atl", "ctl", "acwr"]

SEEDS = [11, 22, 33, 44, 55]
PERMUTATION_REPEATS = 5
MAX_MAE_INCREASE = 0.01                  # selected set may be at most 1% worse

# Fixed, untuned importance model (tuning happens in a later stage)
XGB_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    importance_type="gain",
    n_jobs=-1,
)

TARGET = "rpc"

# Never model inputs: identifiers, dates, and information about the target itself
NON_FEATURES = [
    "id", "sample_id", "target_date", "window_start", "window_end",
    "rpc", "target_seconds", "target_distance_km",
]

# Daily sequence features from Stage 3; earlier = preferred when redundant
SEQUENCE_PREFERENCE = [
    "is_activity_day", "trimp", "atl", "ctl", "acwr", "acwr_valid",
    "effort_km", "ga_speed_kmh", "hr_pct_max", "acwr_km", "run_count",
    "distance_km", "moving_time_min", "days_since_previous_activity",
]
SEQUENCE_PROTECTED = ["is_activity_day", "trimp", "atl", "ctl", "acwr", "acwr_valid"]

# Colours: reference palette sequential blue + diverging blue/gray/red
BLUE = "#2a78d6"
DIVERGING = ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a12e2d"]
TEXT = "#0b0b0b"
MUTED = "#6b6b68"


# ============================================================
# 1. SPLIT
# ============================================================

def split_samples(samples):
    print("=" * 70)
    print("STEP 1 - TRAIN / VALIDATION / TEST SPLIT")
    print("=" * 70)

    s = samples.sort_values(["id", "target_date"]).reset_index(drop=True)
    s["split"] = "train"

    for athlete, idx in s.groupby("id").groups.items():
        idx = list(idx)
        n = len(idx)
        if n < MIN_SAMPLES_TO_SPLIT:
            continue

        n_train = int(np.floor(TRAIN_FRAC * n))
        n_val = max(1, int(np.floor(VAL_FRAC * n)))
        s.loc[idx[n_train:n_train + n_val], "split"] = "val"
        s.loc[idx[n_train + n_val:], "split"] = "test"

    counts = s["split"].value_counts()
    athletes = s.groupby("split")["id"].nunique()
    for part in ["train", "val", "test"]:
        print(f"  {part:<12} samples {int(counts.get(part, 0)):>6,}   athletes {int(athletes.get(part, 0)):>4,}")

    return s


# ============================================================
# 2-3. CANDIDATES AND PREPARATION (TRAIN-FITTED)
# ============================================================

def fit_preparation(train):
    params = {"caps": {}, "fill": {}, "missing_flags": []}

    for col in CAP_COLUMNS:
        params["caps"][col] = float(train[col].quantile(CAP_QUANTILE))

    for col in ACWR_COLUMNS:
        params["fill"][col] = 1.0
    for col in ACWR_ZONE_COLUMNS:
        params["fill"][col] = 1.0

    candidates = [c for c in train.columns if c not in NON_FEATURES + ["split"]]
    for col in candidates:
        if col in params["fill"] or not train[col].isna().any():
            continue
        params["fill"][col] = float(train[col].median())
        params["missing_flags"].append(col)

    return params


def apply_preparation(df, params):
    out = df.copy()
    for col in params["missing_flags"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)
    for col, cap in params["caps"].items():
        out[col] = out[col].clip(upper=cap)
    for col, value in params["fill"].items():
        out[col] = out[col].fillna(value)
    return out


def feature_columns(df):
    return [c for c in df.columns if c not in NON_FEATURES + ["split"]]


# ============================================================
# HELPERS
# ============================================================

def athlete_weights(df):
    """Each athlete contributes equally: weight = 1 / athlete's sample count."""
    return (1.0 / df.groupby("id")["id"].transform("size")).to_numpy()


def weighted_mae(y, pred, w):
    return float(np.sum(w * np.abs(y - pred)) / np.sum(w))


def vif_table(X):
    """Variance inflation factor for each column (OLS on the others)."""
    Z = (X - X.mean()) / X.std(ddof=0)
    rows = []
    for col in Z.columns:
        others = Z.drop(columns=col).to_numpy()
        A = np.column_stack([np.ones(len(Z)), others])
        coef, *_ = np.linalg.lstsq(A, Z[col].to_numpy(), rcond=None)
        resid = Z[col].to_numpy() - A @ coef
        r2 = 1 - resid.var() / Z[col].to_numpy().var()
        rows.append({"feature": col, "vif": np.inf if r2 >= 1 else 1 / (1 - r2)})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def drop_redundant(corr, keep_priority, protected):
    """
    Walk feature pairs from most to least correlated; for each pair above the
    threshold where both are still kept, drop one.
    keep_priority: feature -> score, higher = keep.
    """
    kept = list(corr.columns)
    decisions = []
    pairs = (
        corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        .stack().rename("r").reset_index()
    )
    pairs = pairs.reindex(pairs["r"].abs().sort_values(ascending=False).index)

    for a, b, r in pairs.itertuples(index=False):
        if abs(r) < CORR_THRESHOLD:
            break
        if a not in kept or b not in kept:
            continue
        if a in protected and b in protected:
            decisions.append({"feature_a": a, "feature_b": b, "spearman_r": r,
                              "dropped": "", "kept": f"{a}, {b}",
                              "reason": "both are core study variables (kept by design)"})
            continue
        if a in protected or b in protected:
            keep, drop = (a, b) if a in protected else (b, a)
            reason = f"{keep} is a core study variable"
        else:
            keep, drop = (a, b) if keep_priority[a] >= keep_priority[b] else (b, a)
            reason = f"{keep} ranks higher (see priority rule)"
        kept.remove(drop)
        decisions.append({"feature_a": a, "feature_b": b, "spearman_r": r,
                          "dropped": drop, "kept": keep, "reason": reason})

    return kept, pd.DataFrame(decisions,
                              columns=["feature_a", "feature_b", "spearman_r",
                                       "dropped", "kept", "reason"])


def plot_heatmap(corr, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    n = len(corr)
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2))
    im = ax.imshow(corr.to_numpy(), cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=90, fontsize=8, color=TEXT)
    ax.set_yticks(range(n), corr.index, fontsize=8, color=TEXT)
    for i in range(n):
        for j in range(n):
            v = corr.iat[i, j]
            if i != j and abs(v) >= CORR_THRESHOLD:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="#ffffff" if abs(v) > 0.6 else TEXT)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman r", color=MUTED)
    ax.set_title(f"{title}\n(values shown where |r| >= {CORR_THRESHOLD})",
                 fontsize=10, color=TEXT, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_importance(imp, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    imp = imp.sort_values("perm_mean")
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.5))
    y = np.arange(len(imp))
    ax.barh(y, imp["perm_mean"], xerr=imp["perm_std"], color=BLUE, height=0.6,
            error_kw=dict(ecolor=MUTED, lw=1, capsize=2))
    ax.set_yticks(y, imp["feature"], fontsize=8, color=TEXT)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("Increase in validation MAE when shuffled (RPC units)", color=MUTED, fontsize=8)
    ax.set_title("Permutation importance (mean ± SD over 5 seeds)", fontsize=10,
                 color=TEXT, loc="left")
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# 6. TREE-BASED IMPORTANCE
# ============================================================

def fit_xgb(X, y, w, seed):
    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=seed, **XGB_PARAMS)
    model.fit(X, y, sample_weight=w)
    return model


def tree_importance(train, val, features):
    from sklearn.inspection import permutation_importance

    Xt, yt, wt = train[features], train[TARGET].to_numpy(), athlete_weights(train)
    Xv, yv, wv = val[features], val[TARGET].to_numpy(), athlete_weights(val)

    gains, perms, maes = [], [], []
    for seed in SEEDS:
        print(f"  seed {seed}: fitting XGBoost on {len(features)} features")
        model = fit_xgb(Xt, yt, wt, seed)
        maes.append(weighted_mae(yv, model.predict(Xv), wv))
        gains.append(model.feature_importances_)
        perm = permutation_importance(
            model, Xv, yv, sample_weight=wv, scoring="neg_mean_absolute_error",
            n_repeats=PERMUTATION_REPEATS, random_state=seed, n_jobs=1,
        )
        perms.append(perm.importances_mean)

    gains, perms = np.array(gains), np.array(perms)
    imp = pd.DataFrame({
        "feature": features,
        "gain_mean": gains.mean(axis=0),
        "perm_mean": perms.mean(axis=0),
        "perm_std": perms.std(axis=0),
    })
    imp["perm_rank"] = imp["perm_mean"].rank(ascending=False).astype(int)
    return imp.sort_values("perm_rank"), float(np.mean(maes))


def validation_mae(train, val, features):
    Xt, yt, wt = train[features], train[TARGET].to_numpy(), athlete_weights(train)
    Xv, yv, wv = val[features], val[TARGET].to_numpy(), athlete_weights(val)
    return float(np.mean([
        weighted_mae(yv, fit_xgb(Xt, yt, wt, seed).predict(Xv), wv) for seed in SEEDS
    ]))


# ============================================================
# 8. SEQUENCE FEATURES
# ============================================================

def select_sequence_features(daily, split):
    """Correlation redundancy on daily features, training period only."""
    last_train = split[split["split"] == "train"].groupby("id")["target_date"].max()
    d = daily[daily["id"].isin(last_train.index)].copy()
    d = d[d["calendar_date"] < d["id"].map(last_train)]
    # Activity days only: shared zeros on rest days would inflate every correlation
    d = d[d["is_activity_day"] == 1]

    seq = [c for c in SEQUENCE_PREFERENCE if c in d.columns]
    corr = d[seq].corr(method="spearman")
    priority = {f: len(seq) - i for i, f in enumerate(seq)}
    # is_activity_day is constant on activity days: keep, exclude from the check
    check = [c for c in seq if d[c].nunique() > 1]
    kept, decisions = drop_redundant(corr.loc[check, check], priority, SEQUENCE_PROTECTED)
    kept = [c for c in seq if c in kept or c not in check]
    return kept, corr, decisions, len(d)


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 4 SPLITTING AND FEATURE SELECTION")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        samples = pd.read_csv(FEATURE_DIR / "stage3_samples.csv",
                              parse_dates=["target_date", "window_start", "window_end"])
        # Long-run personal level (mean of all earlier RPCs) - added after Stage 5
        # baselines showed the athlete's average beats every model without it.
        samples = cd.add_personal_level(samples)
        daily = pd.read_csv(FEATURE_DIR / "stage3_daily_features.csv",
                            parse_dates=["calendar_date"])

        # ---- 1. split ----
        split = split_samples(samples)
        split[["sample_id", "id", "target_date", "split"]].to_csv(
            OUTPUT_DIR / "stage4_split_assignment.csv", index=False)

        train_raw = split[split["split"] == "train"]
        val_raw = split[split["split"] == "val"]

        # ---- 2-3. preparation fitted on training only ----
        print()
        print("=" * 70)
        print("STEP 2-3 - CANDIDATE FEATURES AND PREPARATION (TRAIN-FITTED)")
        print("=" * 70)
        params = fit_preparation(train_raw)
        (OUTPUT_DIR / "stage4_preprocessing_params.json").write_text(
            json.dumps(params, indent=2), encoding="utf-8")
        train = apply_preparation(train_raw, params)
        val = apply_preparation(val_raw, params)
        candidates = feature_columns(train)
        print(f"Candidate tabular features: {len(candidates)}")
        print(f"Caps: {params['caps']}")
        print(f"Missing-value flags added: {params['missing_flags']}")

        # ---- 4. screening ----
        raw_missing = train_raw[[c for c in candidates if c in train_raw]].isna().mean()
        screened_out = [c for c in candidates
                        if train[c].nunique() <= 1 or raw_missing.get(c, 0) > MAX_MISSING_FRAC]
        candidates = [c for c in candidates if c not in screened_out]
        print(f"Screened out (constant or > {MAX_MISSING_FRAC:.0%} missing): {screened_out or 'none'}")

        # ---- 5. correlation filter ----
        print()
        print("=" * 70)
        print("STEP 5 - CORRELATION FILTER (SPEARMAN, TRAINING DATA)")
        print("=" * 70)
        corr = train[candidates].corr(method="spearman")
        corr.to_csv(OUTPUT_DIR / "stage4_correlation_tabular.csv")
        plot_heatmap(corr, OUTPUT_DIR / "stage4_correlation_tabular.png",
                     "Tabular features: Spearman correlation (training set)")

        target_corr = train[candidates].corrwith(train[TARGET], method="spearman").abs().fillna(0)
        after_corr, decisions = drop_redundant(corr, target_corr.to_dict(), PROTECTED)
        decisions.to_csv(OUTPUT_DIR / "stage4_redundancy_decisions.csv", index=False)
        print(f"Redundant pairs (|r| >= {CORR_THRESHOLD}): {len(decisions)}")
        for r in decisions.itertuples(index=False):
            print(f"  {r.feature_a} ~ {r.feature_b} (r={r.spearman_r:.2f}) -> drop {r.dropped or 'none'}")

        vif = vif_table(train[after_corr])
        vif.to_csv(OUTPUT_DIR / "stage4_vif.csv", index=False)

        # ---- 6. importance ----
        print()
        print("=" * 70)
        print("STEP 6 - TREE-BASED IMPORTANCE (XGBOOST, ATHLETE-WEIGHTED)")
        print("=" * 70)
        imp, full_mae = tree_importance(train, val, after_corr)
        imp["target_spearman_abs"] = imp["feature"].map(target_corr)

        # ---- 7. selection ----
        imp["clearly_useful"] = (imp["perm_mean"] - imp["perm_std"]) > 0
        imp["kept_by_design"] = imp["feature"].isin(PROTECTED)
        imp["selected"] = imp["clearly_useful"] | imp["kept_by_design"]
        selected = imp.loc[imp["selected"], "feature"].tolist()

        print()
        print("Validating the selected set ...")
        selected_mae = validation_mae(train, val, selected)
        use_selected = selected_mae <= full_mae * (1 + MAX_MAE_INCREASE)
        final_tabular = selected if use_selected else after_corr
        imp["final"] = imp["feature"].isin(final_tabular)
        imp.to_csv(OUTPUT_DIR / "stage4_feature_importance.csv", index=False)
        plot_importance(imp, OUTPUT_DIR / "stage4_feature_importance.png")
        (OUTPUT_DIR / "stage4_selected_features_tabular.txt").write_text(
            "\n".join(final_tabular), encoding="utf-8")

        # ---- 8. sequence features ----
        print()
        print("=" * 70)
        print("STEP 8 - RNN DAILY SEQUENCE FEATURES (REDUNDANCY ONLY)")
        print("=" * 70)
        final_sequence, seq_corr, seq_decisions, n_days = select_sequence_features(daily, split)
        seq_corr.to_csv(OUTPUT_DIR / "stage4_correlation_sequence.csv")
        plot_heatmap(seq_corr, OUTPUT_DIR / "stage4_correlation_sequence.png",
                     "Daily sequence features: Spearman correlation (training-period run days)")
        seq_decisions.to_csv(OUTPUT_DIR / "stage4_sequence_redundancy_decisions.csv", index=False)
        (OUTPUT_DIR / "stage4_selected_features_sequence.txt").write_text(
            "\n".join(final_sequence), encoding="utf-8")
        print(f"Sequence features kept ({len(final_sequence)}): {', '.join(final_sequence)}")

        # ---- 9. checks ----
        print()
        print("=" * 70)
        print("STEP 9 - CHECKS")
        print("=" * 70)
        leaked = set(NON_FEATURES) & set(final_tabular + final_sequence)
        assert not leaked, f"non-feature selected: {leaked}"
        assert split["sample_id"].is_unique
        assert set(split["split"]) <= {"train", "val", "test"}
        for athlete, g in split.groupby("id"):
            order = g.sort_values("target_date")["split"].map({"train": 0, "val": 1, "test": 2})
            assert order.is_monotonic_increasing, f"split out of time order for {athlete}"
        assert set(PROTECTED) <= set(final_tabular)
        print("All checks passed.")

        # ---- summary ----
        counts = split["split"].value_counts()
        athletes = split.groupby("split")["id"].nunique()
        dist_mix = (split
                    .groupby("split")["target_distance_km"].value_counts().unstack(fill_value=0))
        top_vif = vif.head(5)
        lines = [
            "CAPSTONE - STAGE 4 SPLITTING AND FEATURE SELECTION SUMMARY",
            "=" * 60,
            "",
            "SPLIT (per athlete, chronological, by target date)",
            "-" * 60,
            f"Rule: {TRAIN_FRAC:.0%} train / {VAL_FRAC:.0%} validation / {1 - TRAIN_FRAC - VAL_FRAC:.0%} test; "
            f"athletes with < {MIN_SAMPLES_TO_SPLIT} samples -> training only; "
            "every validation/test target date is later than all of that athlete's training targets",
        ]
        for part in ["train", "val", "test"]:
            lines.append(f"{part:<12} samples {int(counts.get(part, 0)):>6,}   athletes {int(athletes.get(part, 0)):>4,}")
        lines += ["", "Best-effort distance mix (samples):", dist_mix.to_string(), ""]
        lines += [
            "PREPARATION (fitted on training data only)",
            "-" * 60,
            "Caps (training 99th percentile): " + ", ".join(f"{k} <= {v:.0f}" for k, v in params["caps"].items()),
            "ACWR / acwr_km missing -> 1.0 (neutral), flagged by acwr_valid",
            "Other missing -> training median + flag: " + ", ".join(params["missing_flags"]),
            f"Excluded from inputs (identifiers / target information): {', '.join(NON_FEATURES)}",
            "",
            "FILTER METHOD - CORRELATION",
            "-" * 60,
            f"Candidates: {len(candidates)} | screened out: {screened_out or 'none'}",
            f"Spearman |r| >= {CORR_THRESHOLD} treated as redundant; {len(decisions)} pair(s):",
        ]
        for r in decisions.itertuples(index=False):
            lines.append(f"  {r.feature_a} ~ {r.feature_b} (r = {r.spearman_r:.2f}): drop {r.dropped or 'none'} - {r.reason}")
        lines += [
            f"After correlation filter: {len(after_corr)} features",
            "Highest VIF after filter: " + ", ".join(f"{f} {v:.1f}" for f, v in top_vif.itertuples(index=False)),
            "",
            "TREE-BASED IMPORTANCE (XGBoost, fixed untuned settings, athlete-weighted, 5 seeds)",
            "-" * 60,
            "Rank  Feature                      Perm. importance (mean +/- SD)   Gain    Selected",
        ]
        for r in imp.itertuples(index=False):
            tag = "design" if r.kept_by_design and not r.clearly_useful else ("yes" if r.selected else "no")
            lines.append(f"{r.perm_rank:>4}  {r.feature:<28} {r.perm_mean:>9.5f} +/- {r.perm_std:.5f}       "
                         f"{r.gain_mean:.3f}   {tag}")
        lines += [
            "",
            "SELECTION CHECK (validation, athlete-weighted MAE in RPC units)",
            "-" * 60,
            f"All {len(after_corr)} post-filter features: {full_mae:.5f}",
            f"Selected {len(selected)} features:         {selected_mae:.5f}",
            f"Decision: {'use the selected set' if use_selected else 'keep all post-filter features (selected set > 1% worse)'}",
            "",
            f"FINAL TABULAR FEATURES (XGBoost) - {len(final_tabular)}:",
            "  " + ", ".join(final_tabular),
            "",
            f"FINAL SEQUENCE FEATURES (GRU / LSTM) - {len(final_sequence)}, input shape (N, {WINDOW_DAYS}, {len(final_sequence)}):",
            "  " + ", ".join(final_sequence),
            f"  (redundancy checked on {n_days:,} training-period run days; "
            f"{len(seq_decisions)} redundant pair(s), see stage4_sequence_redundancy_decisions.csv)",
            "",
            "LEAKAGE GUARDS",
            "-" * 60,
            "Split done before any selection; all fitting uses training data only.",
            "Validation set used only to score importance and the selection check; test set untouched.",
            "Per athlete, every validation/test sample is later in time than all training samples (checked).",
            "",
            "STATUS",
            "-" * 60,
            "Stage 4 splitting and feature selection completed.",
        ]
        (OUTPUT_DIR / "stage4_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"\nOutputs written to:\n{OUTPUT_DIR}")

    except Exception as exc:
        (LOG_DIR / "stage4_feature_selection_error.log").write_text(
            "STAGE 4 ERROR\n" + "=" * 60 + "\n\n" + str(exc) + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        print("\nERROR - see logs/stage4_feature_selection_error.log")
        raise


if __name__ == "__main__":
    main()
