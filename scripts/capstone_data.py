"""
capstone_data.py

Shared data helpers for the modelling stages (05_baseline_models onward).
Imported by the stage scripts; not meant to be run on its own.

Everything that is fitted (fill values, scaling) is fitted on the TRAINING
split from 06_stage4_feature_selection.py and saved, so validation, test and
deployment are transformed exactly the same way.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS AND CONSTANTS
# ============================================================

ROOT = Path(r"C:\Users\ACER\Desktop\CAPSTONE GAME TIME")

PREPARED_DIR = ROOT / "01_prepared"
FEATURE_DIR = ROOT / "03_feature_engineering"
SELECTION_DIR = ROOT / "04_feature_selection"

WINDOW_DAYS = 28
TARGET = "rpc"

# Must match 05_stage3_feature_engineering.py
RIEGEL_EXPONENT = 1.06
REFERENCE_KM = 10.0
REFERENCE_SECONDS = 1571.0
TARGET_COLUMNS = {
    5.0: "target_5_0km_sec",
    10.0: "target_10_0km_sec",
    15.0: "target_15_0km_sec",
    21.1: "target_21_1km_sec",
    42.2: "target_42_2km_sec",
}

# Past-performance inputs (removed in the "no past RPC" ablation)
PRIOR_TABULAR = ["last_rpc", "best_rpc_90", "best_rpc_90_missing", "days_since_last_target",
                 "prior_mean_rpc", "prior_mean_rpc_missing", "prior_rpc_count"]
PRIOR_SEQUENCE = ["known_rpc", "days_since_rpc", "mean_rpc_to_date"]


def configure(root):
    """Point all paths at another project folder (used for testing)."""
    global ROOT, PREPARED_DIR, FEATURE_DIR, SELECTION_DIR
    ROOT = Path(root)
    PREPARED_DIR = ROOT / "01_prepared"
    FEATURE_DIR = ROOT / "03_feature_engineering"
    SELECTION_DIR = ROOT / "04_feature_selection"


# ============================================================
# RPC <-> FINISH TIME
# ============================================================

def rpc_from_time(seconds, distance_km):
    return REFERENCE_SECONDS / (seconds * (REFERENCE_KM / distance_km) ** RIEGEL_EXPONENT)


def time_from_rpc(rpc, distance_km):
    return REFERENCE_SECONDS / (rpc * (REFERENCE_KM / distance_km) ** RIEGEL_EXPONENT)


# ============================================================
# LOADING
# ============================================================

def load_samples():
    """Stage 3 samples joined with the Stage 4 split."""
    samples = pd.read_csv(FEATURE_DIR / "stage3_samples.csv",
                          parse_dates=["target_date", "window_start", "window_end"])
    split = pd.read_csv(SELECTION_DIR / "stage4_split_assignment.csv")
    samples = samples.merge(split[["sample_id", "split"]], on="sample_id",
                            how="inner", validate="one_to_one")
    return samples.sort_values(["id", "target_date"]).reset_index(drop=True)


def load_params():
    return json.loads((SELECTION_DIR / "stage4_preprocessing_params.json").read_text(encoding="utf-8"))


def load_feature_list(name):
    text = (SELECTION_DIR / f"stage4_selected_features_{name}.txt").read_text(encoding="utf-8")
    return [line.strip() for line in text.splitlines() if line.strip()]


def athlete_weights(df):
    """Each athlete counts equally; scaled so the mean weight is 1."""
    w = 1.0 / df.groupby("id")["id"].transform("size").to_numpy()
    return w / w.mean()


# ============================================================
# TABULAR INPUTS (XGBoost, naive baselines)
# ============================================================

def prepare_tabular(samples, params):
    """Same transformation as 06_stage4_feature_selection.apply_preparation."""
    out = samples.copy()
    for col in params["missing_flags"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)
    for col, cap in params["caps"].items():
        out[col] = out[col].clip(upper=cap)
    for col, value in params["fill"].items():
        out[col] = out[col].fillna(value)
    return out


# ============================================================
# SEQUENCE INPUTS (GRU / LSTM)
# ============================================================

def daily_known_rpc():
    """
    Per athlete per day, from best efforts ON OR BEFORE that day:
      known_rpc          most recent RPC
      days_since_rpc     its age in days
      mean_rpc_to_date   mean of all RPCs so far (the athlete's long-run level)
      rpc_count_to_date  how many best-effort days that mean is based on
    Windows end the day before the target, so the target run is never included.
    """
    cols = ["id", "calendar_date"] + list(TARGET_COLUMNS.values())
    timeline = pd.read_csv(PREPARED_DIR / "stage1_daily_timeline.csv", usecols=cols,
                           parse_dates=["calendar_date"], low_memory=False)

    rpc = pd.concat(
        [rpc_from_time(pd.to_numeric(timeline[col], errors="coerce"), km)
         for km, col in TARGET_COLUMNS.items()],
        axis=1,
    ).max(axis=1)
    rpc = rpc.where(rpc <= 1.0)  # same rule as Stage 3 (faster than world record = GPS error)

    out = timeline[["id", "calendar_date"]].copy()
    out["rpc_day"] = rpc
    out = out.sort_values(["id", "calendar_date"])
    out["rpc_date"] = out["calendar_date"].where(out["rpc_day"].notna())
    g = out.groupby("id", sort=False)
    out["known_rpc"] = g["rpc_day"].ffill()
    out["rpc_date"] = g["rpc_date"].ffill()
    out["days_since_rpc"] = (out["calendar_date"] - out["rpc_date"]).dt.days
    out["rpc_count_to_date"] = out["rpc_day"].notna().astype(int).groupby(out["id"]).cumsum()
    rpc_sum = out["rpc_day"].fillna(0).groupby(out["id"]).cumsum()
    out["mean_rpc_to_date"] = (rpc_sum / out["rpc_count_to_date"]).where(out["rpc_count_to_date"] > 0)
    return out[["id", "calendar_date", "known_rpc", "days_since_rpc",
                "mean_rpc_to_date", "rpc_count_to_date"]]


def add_personal_level(samples):
    """
    Tabular personal-level features as of the day BEFORE the target
    (window_end): prior_mean_rpc (mean of all earlier RPCs) and
    prior_rpc_count. Uses only best efforts strictly before the target day.
    """
    daily = daily_known_rpc().rename(columns={
        "calendar_date": "window_end",
        "mean_rpc_to_date": "prior_mean_rpc",
        "rpc_count_to_date": "prior_rpc_count",
    })[["id", "window_end", "prior_mean_rpc", "prior_rpc_count"]]
    out = samples.merge(daily, on=["id", "window_end"], how="left", validate="many_to_one")
    assert len(out) == len(samples)
    out["prior_rpc_count"] = out["prior_rpc_count"].fillna(0).astype(int)
    return out


def load_daily(sequence_features):
    daily = pd.read_csv(FEATURE_DIR / "stage3_daily_features.csv", parse_dates=["calendar_date"])
    keep = ["id", "calendar_date"] + [c for c in sequence_features if c not in PRIOR_SEQUENCE]
    daily = daily[keep]
    if any(c in PRIOR_SEQUENCE for c in sequence_features):
        daily = daily.merge(daily_known_rpc(), on=["id", "calendar_date"], how="left",
                            validate="one_to_one")
    return daily.sort_values(["id", "calendar_date"]).reset_index(drop=True)


def build_windows(daily, samples, features):
    """
    Array (n_samples, 28, n_features): the 28 days before each target day.
    Each athlete's daily timeline is complete (one row per calendar day), so a
    window is 28 consecutive rows starting at window_start.
    """
    X = np.empty((len(samples), WINDOW_DAYS, len(features)), dtype=np.float32)
    by_athlete = {a: g for a, g in daily.groupby("id", sort=False)}

    for athlete, idx in samples.groupby("id", sort=False).groups.items():
        g = by_athlete[athlete]
        values = g[features].to_numpy(dtype=np.float32)
        first = g["calendar_date"].iloc[0]
        starts = (samples.loc[idx, "window_start"] - first).dt.days.to_numpy()
        for row, start in zip(idx, starts):
            X[row] = values[start:start + WINDOW_DAYS]

    return X


def fit_sequence_prep(X_train, features, params):
    """Fill values and scaling for the sequence inputs, from training windows only."""
    flat = X_train.reshape(-1, X_train.shape[-1]).astype(np.float64)
    caps = {
        "days_since_previous_activity": params["caps"]["days_since_last_run"],
        "days_since_rpc": params["caps"]["days_since_last_target"],
    }
    fill = {}
    for j, f in enumerate(features):
        if f in ("acwr", "acwr_km"):
            fill[f] = 1.0                                   # neutral; acwr_valid marks it
        elif f in ("known_rpc", "mean_rpc_to_date"):
            fill[f] = params["fill"]["last_rpc"]            # training median
        elif f in caps:
            fill[f] = caps[f]                               # no earlier run / RPC yet
        else:
            col = flat[:, j]
            fill[f] = float(np.nanmedian(col)) if np.isnan(col).any() else 0.0

    prepped = apply_sequence_prep(X_train, features, {"fill": fill, "caps": caps,
                                                      "mean": None, "std": None})
    flat = prepped.reshape(-1, prepped.shape[-1]).astype(np.float64)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    std[std == 0] = 1.0
    return {"features": features, "fill": fill, "caps": caps,
            "mean": mean.tolist(), "std": std.tolist()}


def apply_sequence_prep(X, features, prep):
    X = X.copy()
    for j, f in enumerate(features):
        col = X[:, :, j]
        col[np.isnan(col)] = prep["fill"][f]
        if f in prep["caps"]:
            np.minimum(col, prep["caps"][f], out=col)
    if prep["mean"] is not None:
        X = (X - np.array(prep["mean"], dtype=np.float32)) / np.array(prep["std"], dtype=np.float32)
    return X.astype(np.float32)


# ============================================================
# METRICS
# ============================================================

def regression_metrics(y, pred, w=None):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    err = pred - y
    mae = np.sum(w * np.abs(err)) / w.sum()
    rmse = np.sqrt(np.sum(w * err ** 2) / w.sum())
    ybar = np.sum(w * y) / w.sum()
    r2 = 1 - np.sum(w * err ** 2) / np.sum(w * (y - ybar) ** 2)
    mape = 100 * np.sum(w * np.abs(err) / np.abs(y)) / w.sum()
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
