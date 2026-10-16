r"""
05_stage3_feature_engineering.py

CAPSTONE
Stage 3 - Feature Engineering

Inputs (from 03_stage1_data_preparation.py):
    01_prepared/stage1_activity_index.csv
    01_prepared/stage1_daily_timeline.csv
    01_prepared/stage1_28day_window_index.csv
    00_raw/granular/<athlete>/<file>.csv   (heart rate per second)

Pipeline:
1. Per-run heart-rate load
     - HRmax = athlete's running max of per-run 99th-percentile HR,
       using only runs up to that day (causal, no future data)
     - Time in the 5 Strava %HRmax zones (moving-time clock)
     - Zone-weighted HR TRIMP = sum(minutes in zone x weight)
2. Per-run terrain normalisation
     - ITRA effort distance = distance + elevation gain / 100 (gain only)
     - Grade-adjusted speed (km/h) = effort distance / moving time
3. Daily feature sequence (one row per athlete per calendar day)
     - zero on rest days + is_activity_day flag
     - ATL (7-day mean TRIMP), CTL (28-day mean TRIMP), ACWR
     - distance-based ACWR for the Toresdahl-aligned alert framework
     - ACWR zone (Under-training / Sweet Spot / Monitoring / Danger)
4. Target: Running Performance Coefficient (RPC)
     - one sample per athlete per target day
     - best-effort time -> 10K-equivalent (Riegel, exponent 1.06)
     - RPC = 10K world record (1,571 s) / 10K-equivalent
5. Per-sample features from the 28 days BEFORE the target
     (volume, load at day before target, prior performance)
6. Checks + outputs to 03_feature_engineering/

The train/validation/test split, scaling and feature selection are
NOT done here: they must be fitted on training data only and belong
to the modelling stage.

Usage:
    python 05_stage3_feature_engineering.py
    python 05_stage3_feature_engineering.py --limit-athletes 5   (quick test)
"""

from pathlib import Path
import argparse
import traceback

import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(r"C:\Users\admin\Desktop\CAPSTONE GAME TIME")

PREPARED_DIR = ROOT / "01_prepared"
GRANULAR_ROOT = ROOT / "00_raw" / "granular"
OUTPUT_DIR = ROOT / "03_feature_engineering"
LOG_DIR = ROOT / "logs"

WINDOW_DAYS = 28
ACUTE_DAYS = 7
CHRONIC_DAYS = 28

# Must match 03_stage1_data_preparation.py (moving-time clock)
MAX_SAMPLE_GAP_SEC = 30

# Valid heart-rate range for a sample (bpm)
HR_MIN, HR_MAX = 30, 230
HR_BINS = np.arange(HR_MIN, HR_MAX + 2)  # 1-bpm histogram edges

# Strava %HRmax zones and manuscript weights (lower bound %, weight)
ZONES = [
    ("z1", 0.00, 0.2),   # Endurance / Active Recovery  0-65%
    ("z2", 0.65, 0.4),   # Aerobic / Moderate          65-80%
    ("z3", 0.80, 0.6),   # Tempo                       80-89%
    ("z4", 0.89, 1.0),   # Threshold                   89-97%
    ("z5", 0.97, 1.5),   # Anaerobic / Maximum          >97%
]

HRMAX_PERCENTILE = 99
HRMAX_PLAUSIBLE = 210   # per-run p99 above this = sensor artefact, ignored for HRmax

# Climb above this (m per km) is treated as unreliable for terrain normalisation
# (ITRA counts 100 m as 1 km, so noisy altitude would inflate effort distance)
MAX_GAIN_PER_KM_FOR_GAP = 100

# ACWR: valid only if the chronic window has at least this many run days
MIN_CHRONIC_RUN_DAYS = 4

ACWR_ZONES = [  # (upper bound exclusive, code, label)
    (0.8, 0, "Under-training"),
    (1.3, 1, "Sweet Spot"),
    (1.5, 2, "Monitoring"),
    (np.inf, 3, "Danger"),
]
ACWR_DANGER = 1.5

# RPC
RIEGEL_EXPONENT = 1.06
REFERENCE_KM = 10.0
REFERENCE_SECONDS = 1571.0   # men's 10K world record, 26:11

PRIOR_RPC_DAYS = 90

TARGET_COLUMNS = {
    5.0: "target_5_0km_sec",
    10.0: "target_10_0km_sec",
    15.0: "target_15_0km_sec",
    21.1: "target_21_1km_sec",
    42.2: "target_42_2km_sec",
}


# ============================================================
# HELPERS
# ============================================================

def rpc_from_time(seconds, distance_km):
    """Running Performance Coefficient from a best-effort time."""
    equivalent = seconds * (REFERENCE_KM / distance_km) ** RIEGEL_EXPONENT
    return REFERENCE_SECONDS / equivalent


def zone_weight(hr_fraction):
    """Weight of the zone that a %HRmax value falls in."""
    weight = ZONES[0][2]
    for _, lower, w in ZONES:
        if hr_fraction >= lower:
            weight = w
    return weight


def granular_hr_histogram(path):
    """
    Seconds of moving time spent at each bpm (1-bpm bins) for one run.

    Uses the same moving-time clock as 03_stage1_data_preparation.py:
    gaps longer than MAX_SAMPLE_GAP_SEC are credited only the time
    needed to cover that gap's distance at the run's typical pace.

    Returns (histogram, p99_hr) or (None, nan) if no usable HR.
    """

    g = pd.read_csv(path, usecols=["secs", "km", "hr"])
    g = g.apply(pd.to_numeric, errors="coerce").dropna(subset=["secs", "km"])
    g = g[(g["secs"] >= 0) & (g["km"] >= 0)]
    g = g.sort_values("secs").drop_duplicates("secs")

    if len(g) < 2:
        return None, np.nan

    km = g["km"].cummax().to_numpy()
    secs = g["secs"].to_numpy()
    dt = np.diff(secs, prepend=secs[0])
    dkm = np.diff(km, prepend=km[0])

    normal = dt <= MAX_SAMPLE_GAP_SEC
    if dkm[normal].sum() <= 0:
        return None, np.nan

    typical = dt[normal].sum() / dkm[normal].sum()
    credited = np.where(normal, dt, np.minimum(dt, dkm * typical))

    hr = g["hr"].to_numpy()
    valid = (hr >= HR_MIN) & (hr <= HR_MAX)

    if valid.sum() < 60:   # under ~1 minute of HR: unusable
        return None, np.nan

    hist, _ = np.histogram(hr[valid], bins=HR_BINS, weights=credited[valid])
    return hist, float(np.percentile(hr[valid], HRMAX_PERCENTILE))


# ============================================================
# 1-2. PER-RUN FEATURES
# ============================================================

def build_activity_features(activity):
    print("=" * 70)
    print("STEP 1 - PER-RUN HEART RATE LOAD")
    print("=" * 70)

    runs = activity[~activity["suspected_non_run"].astype(bool)].copy()
    runs["datetime_utc"] = pd.to_datetime(runs["datetime_utc"], utc=True)
    runs = runs.sort_values(["id", "datetime_utc"]).reset_index(drop=True)

    hists = []
    p99 = np.full(len(runs), np.nan)

    for i, row in enumerate(runs.itertuples(index=False)):
        if i == 0 or (i + 1) % 1000 == 0 or i + 1 == len(runs):
            print(f"Reading granular HR: {i + 1:,}/{len(runs):,}")

        hist = None
        if row.granular_status == "OK":
            path = GRANULAR_ROOT / str(row.id) / str(row.file)
            try:
                hist, p99[i] = granular_hr_histogram(path)
            except Exception:
                hist = None
        hists.append(hist)

    runs["hr_p99"] = p99
    plausible_p99 = runs["hr_p99"].where(runs["hr_p99"] <= HRMAX_PLAUSIBLE)

    # Causal HRmax: running max of per-run p99 up to and including this run
    runs["hrmax"] = plausible_p99.groupby(runs["id"]).cummax()
    runs["hrmax"] = runs.groupby("id")["hrmax"].ffill()
    # Default before an athlete's first usable HR run: median athlete HRmax
    population_hrmax = float(plausible_p99.groupby(runs["id"]).max().median())
    runs["hrmax_is_default"] = runs["hrmax"].isna()
    runs["hrmax"] = runs["hrmax"].fillna(population_hrmax)

    moving_min = runs["workout_time"] / 60
    bpm = HR_BINS[:-1]

    zone_minutes = np.zeros((len(runs), len(ZONES)))
    trimp = np.full(len(runs), np.nan)
    estimated = np.zeros(len(runs), dtype=bool)
    hr_pct = np.full(len(runs), np.nan)

    lowers = np.array([z[1] for z in ZONES])
    weights = np.array([z[2] for z in ZONES])

    for i, hist in enumerate(hists):
        hrmax = runs.at[i, "hrmax"]
        if hist is not None:
            zone_idx = np.searchsorted(lowers, bpm / hrmax, side="right") - 1
            minutes = np.bincount(zone_idx, weights=hist, minlength=len(ZONES)) / 60
            zone_minutes[i] = minutes
            trimp[i] = (minutes * weights).sum()
            hr_pct[i] = (hist * bpm).sum() / hist.sum() / hrmax
        else:
            # Fallback: whole run at the zone of its average HR
            avg_hr = runs.at[i, "average_hr"]
            if pd.notna(avg_hr) and pd.notna(moving_min[i]) and HR_MIN <= avg_hr <= HR_MAX:
                trimp[i] = moving_min[i] * zone_weight(avg_hr / hrmax)
                hr_pct[i] = avg_hr / hrmax
                estimated[i] = True

    for j, (name, _, _) in enumerate(ZONES):
        runs[f"{name}_min"] = zone_minutes[:, j]
    runs["trimp"] = trimp
    runs["trimp_estimated"] = estimated
    runs["hr_pct_max"] = hr_pct

    print(f"Runs: {len(runs):,} | TRIMP from granular HR: {int(np.isfinite(trimp).sum() - estimated.sum()):,} "
          f"| estimated from average HR: {int(estimated.sum()):,} | missing: {int(np.isnan(trimp).sum()):,}")
    print(f"Population median HRmax (default for runs before any HR): {population_hrmax:.1f} bpm")

    print()
    print("=" * 70)
    print("STEP 2 - TERRAIN NORMALISATION")
    print("=" * 70)

    gain_per_km = runs["elevation_gain"] / runs["total_distance"]
    gain_per_km = gain_per_km.where(gain_per_km <= MAX_GAIN_PER_KM_FOR_GAP)
    # Causal fill: median climb/km of the athlete's EARLIER runs
    prior_median = (
        gain_per_km.groupby(runs["id"])
        .transform(lambda s: s.expanding().median().shift(1))
    )
    runs["elevation_imputed"] = gain_per_km.isna()
    fill = prior_median.fillna(float(np.nanmedian(gain_per_km))) * runs["total_distance"]
    runs["elevation_gain_filled"] = (gain_per_km * runs["total_distance"]).fillna(fill)

    runs["effort_km"] = runs["total_distance"] + runs["elevation_gain_filled"] / 100
    runs["ga_speed_kmh"] = runs["effort_km"] / (runs["workout_time"] / 3600)

    print(f"Elevation imputed (missing, or > {MAX_GAIN_PER_KM_FOR_GAP} m/km): {int(runs['elevation_imputed'].sum()):,}")

    return runs, population_hrmax


# ============================================================
# 3. DAILY FEATURE SEQUENCE
# ============================================================

def acwr_zone_code(acwr):
    code = pd.Series(np.nan, index=acwr.index)
    lower = -np.inf
    for upper, c, _ in ACWR_ZONES:
        code[(acwr >= lower) & (acwr < upper)] = c
        lower = upper
    return code


def build_daily_features(runs, spine):
    print()
    print("=" * 70)
    print("STEP 3 - DAILY FEATURE SEQUENCE")
    print("=" * 70)

    runs = runs.copy()
    runs["calendar_date"] = pd.to_datetime(runs["calendar_date"])
    runs["hr_weighted"] = runs["hr_pct_max"] * runs["workout_time"]

    day = runs.groupby(["id", "calendar_date"]).agg(
        run_count=("file", "size"),
        distance_km=("total_distance", "sum"),
        max_run_km=("total_distance", "max"),
        moving_time_min=("workout_time", lambda s: s.sum(min_count=1) / 60),
        effort_km=("effort_km", "sum"),
        trimp=("trimp", lambda s: s.sum(min_count=1)),
        hr_weighted=("hr_weighted", lambda s: s.sum(min_count=1)),
        hr_time=("workout_time", "sum"),
    ).reset_index()

    daily = spine.merge(day, on=["id", "calendar_date"], how="left")

    zero_cols = ["run_count", "distance_km", "max_run_km", "moving_time_min",
                 "effort_km", "trimp", "hr_weighted", "hr_time"]
    rest = daily["is_activity_day"] == 0
    daily.loc[rest, zero_cols] = 0.0

    daily["ga_speed_kmh"] = np.where(
        rest, 0.0,
        daily["effort_km"] / (daily["moving_time_min"] / 60).where(daily["moving_time_min"] > 0)
    )
    daily["hr_pct_max"] = np.where(
        rest, 0.0, daily["hr_weighted"] / daily["hr_time"]
    )
    daily = daily.drop(columns=["hr_weighted", "hr_time"])

    daily = daily.sort_values(["id", "calendar_date"]).reset_index(drop=True)
    g = daily.groupby("id", sort=False)

    def roll(col, days, how):
        # trailing window INCLUDING the current day; missing load counts as 0
        return g[col].transform(
            lambda s: getattr(s.fillna(0).rolling(days, min_periods=days), how)()
        )

    daily["atl"] = roll("trimp", ACUTE_DAYS, "mean")
    daily["ctl"] = roll("trimp", CHRONIC_DAYS, "mean")
    daily["atl_km"] = roll("distance_km", ACUTE_DAYS, "mean")
    daily["ctl_km"] = roll("distance_km", CHRONIC_DAYS, "mean")
    daily["run_days_28"] = roll("is_activity_day", CHRONIC_DAYS, "sum")
    daily["run_days_7"] = roll("is_activity_day", ACUTE_DAYS, "sum")
    daily["km_7"] = roll("distance_km", ACUTE_DAYS, "sum")
    daily["km_28"] = roll("distance_km", CHRONIC_DAYS, "sum")
    daily["longest_run_km_28"] = roll("max_run_km", CHRONIC_DAYS, "max")
    effort_28 = roll("effort_km", CHRONIC_DAYS, "sum")
    time_28 = roll("moving_time_min", CHRONIC_DAYS, "sum")
    daily["mean_ga_speed_28"] = effort_28 / (time_28 / 60).where(time_28 > 0)

    valid = (daily["run_days_28"] >= MIN_CHRONIC_RUN_DAYS)
    daily["acwr"] = (daily["atl"] / daily["ctl"]).where(valid & (daily["ctl"] > 0))
    daily["acwr_km"] = (daily["atl_km"] / daily["ctl_km"]).where(valid & (daily["ctl_km"] > 0))
    daily["acwr_valid"] = daily["acwr"].notna().astype(int)
    daily["acwr_zone"] = acwr_zone_code(daily["acwr"])
    daily["acwr_km_zone"] = acwr_zone_code(daily["acwr_km"])

    daily["danger_days_28"] = (
        (daily["acwr"] >= ACWR_DANGER).astype(int)
        .groupby(daily["id"]).transform(lambda s: s.rolling(CHRONIC_DAYS, min_periods=1).sum())
    )

    print(f"Daily rows: {len(daily):,} | athletes: {daily['id'].nunique():,}")
    print(f"ACWR valid on {int(daily['acwr_valid'].sum()):,} days")

    return daily


# ============================================================
# 4-5. TARGET (RPC) AND PER-SAMPLE FEATURES
# ============================================================

def build_samples(daily, windows, runs):
    print()
    print("=" * 70)
    print("STEP 4 - TARGET: RUNNING PERFORMANCE COEFFICIENT")
    print("=" * 70)

    w = windows.copy()
    for c in ["window_start", "window_end", "target_date"]:
        w[c] = pd.to_datetime(w[c])
    w["rpc"] = rpc_from_time(w["target_seconds"], w["target_distance_km"])

    # One sample per athlete per target day: the best RPC across distances
    best = w.loc[w.groupby(["id", "target_date"])["rpc"].idxmax()]
    best = best[["id", "target_date", "window_start", "window_end",
                 "target_distance_km", "target_seconds", "rpc"]].reset_index(drop=True)
    print(f"Per-distance windows: {len(w):,} -> samples (athlete x target day): {len(best):,}")

    dropped = []

    above = best["rpc"] > 1.0
    dropped.append(best[above].assign(drop_reason="RPC_ABOVE_1"))
    best = best[~above]

    # Full chronic history: CTL on the day before the target needs
    # 28 days of timeline, i.e. history starting >= 27 days before window_start.
    first_day = daily.groupby("id")["calendar_date"].min()
    history_days = (best["window_start"] - best["id"].map(first_day)).dt.days
    short = history_days < CHRONIC_DAYS - 1
    dropped.append(best[short].assign(drop_reason="INSUFFICIENT_CHRONIC_HISTORY"))
    best = best[~short].reset_index(drop=True)

    dropped = pd.concat(dropped, ignore_index=True)
    print(f"Dropped: RPC > 1: {int(above.sum()):,} | insufficient chronic history: {int(short.sum()):,}")
    print(f"Samples kept: {len(best):,} | athletes: {best['id'].nunique():,}")

    print()
    print("=" * 70)
    print("STEP 5 - PER-SAMPLE FEATURES (28 DAYS BEFORE TARGET)")
    print("=" * 70)

    # Day-before-target lookups from the daily rolling features
    lookup_cols = ["atl", "ctl", "acwr", "acwr_km", "acwr_zone", "acwr_km_zone",
                   "acwr_valid", "danger_days_28", "run_days_7", "run_days_28",
                   "km_7", "km_28", "longest_run_km_28", "mean_ga_speed_28",
                   "days_since_previous_activity"]
    day_before = daily[["id", "calendar_date"] + lookup_cols].rename(
        columns={"calendar_date": "window_end"})
    samples = best.merge(day_before, on=["id", "window_end"], how="left",
                         validate="one_to_one")
    samples = samples.rename(columns={"days_since_previous_activity": "days_since_last_run"})

    # Mean HR intensity over run days in the window
    win_hr = []
    by_athlete = {k: v.set_index("calendar_date") for k, v in daily.groupby("id")}
    for r in samples.itertuples(index=False):
        d = by_athlete[r.id].loc[r.window_start:r.window_end]
        act = d[d["is_activity_day"] == 1]
        win_hr.append(act["hr_pct_max"].mean() if len(act) else np.nan)
    samples["mean_hr_pct_28"] = win_hr

    # Prior performance: every earlier best-effort target (not only kept samples)
    runs = runs.copy()
    runs["calendar_date"] = pd.to_datetime(runs["calendar_date"])
    perf = []
    for km, col in TARGET_COLUMNS.items():
        t = runs[["id", "calendar_date", col]].dropna()
        perf.append(pd.DataFrame({"id": t["id"], "date": t["calendar_date"],
                                  "rpc": rpc_from_time(t[col], km)}))
    perf = pd.concat(perf)
    perf = perf[perf["rpc"] <= 1.0].groupby(["id", "date"])["rpc"].max().reset_index()
    perf_by_athlete = {k: v.sort_values("date") for k, v in perf.groupby("id")}

    last_rpc, best_rpc_90, days_since_target = [], [], []
    for r in samples.itertuples(index=False):
        p = perf_by_athlete.get(r.id)
        prior = p[p["date"] < r.target_date] if p is not None else None
        if prior is None or prior.empty:
            last_rpc.append(np.nan); best_rpc_90.append(np.nan); days_since_target.append(np.nan)
            continue
        last_rpc.append(prior["rpc"].iloc[-1])
        days_since_target.append((r.target_date - prior["date"].iloc[-1]).days)
        recent = prior[prior["date"] >= r.target_date - pd.Timedelta(days=PRIOR_RPC_DAYS)]
        best_rpc_90.append(recent["rpc"].max() if len(recent) else np.nan)

    samples["last_rpc"] = last_rpc
    samples["best_rpc_90"] = best_rpc_90
    samples["days_since_last_target"] = days_since_target
    samples["has_prior_rpc"] = samples["last_rpc"].notna().astype(int)

    samples["sample_id"] = samples["id"] + "_" + samples["target_date"].dt.strftime("%Y-%m-%d")
    samples = samples.sort_values(["id", "target_date"]).reset_index(drop=True)

    print(f"Samples with a prior RPC: {int(samples['has_prior_rpc'].sum()):,}/{len(samples):,}")

    return samples, dropped


# ============================================================
# 6. CHECKS, DICTIONARY, OUTPUTS
# ============================================================

SEQUENCE_FEATURES = [
    "is_activity_day", "run_count", "distance_km", "moving_time_min", "effort_km",
    "ga_speed_kmh", "hr_pct_max", "trimp", "atl", "ctl", "acwr", "acwr_valid",
    "acwr_km", "days_since_previous_activity",
]

FEATURE_DICTIONARY = [
    # name, level, unit, definition
    ("is_activity_day", "daily", "0/1", "1 if at least one run that day, else 0 (rest day)"),
    ("run_count", "daily", "count", "Number of runs that day"),
    ("distance_km", "daily", "km", "Total running distance that day"),
    ("max_run_km", "daily", "km", "Longest single run that day"),
    ("moving_time_min", "daily", "min", "Total moving time (Stage 1 moving-time clock)"),
    ("effort_km", "daily", "km", "ITRA effort distance: distance + elevation gain / 100"),
    ("ga_speed_kmh", "daily", "km/h", "Grade-adjusted speed: effort_km / moving time; 0 on rest days"),
    ("hr_pct_max", "daily", "fraction", "Time-weighted mean HR as a fraction of the athlete's causal HRmax; 0 on rest days"),
    ("trimp", "daily", "a.u.", "Zone-weighted HR TRIMP: sum over zones of minutes x weight (0.2/0.4/0.6/1.0/1.5)"),
    ("atl", "daily", "a.u.", "Acute training load: 7-day rolling mean of daily TRIMP"),
    ("ctl", "daily", "a.u.", "Chronic training load: 28-day rolling mean of daily TRIMP"),
    ("acwr", "daily", "ratio", "ATL / CTL; missing if < 4 run days in the 28-day chronic window"),
    ("acwr_valid", "daily", "0/1", "1 if acwr is defined"),
    ("acwr_km", "daily", "ratio", "Distance-based ACWR (7-day / 28-day mean km), for the alert framework"),
    ("acwr_zone", "daily", "code", "0 Under-training (<0.8), 1 Sweet Spot (0.8-1.3), 2 Monitoring (1.3-1.5), 3 Danger (>=1.5)"),
    ("days_since_previous_activity", "daily", "days", "Days since the previous run"),
    ("rpc", "target", "0-1", "Running Performance Coefficient = 1571 s / Riegel 10K-equivalent of the best effort that day"),
    ("target_distance_km", "target", "km", "Standard distance of the best effort used for the RPC"),
    ("target_seconds", "target", "s", "Best-effort time at that distance (for back-conversion and evaluation)"),
    ("atl / ctl / acwr / acwr_km / acwr_zone", "sample", "-", "Value on the day before the target"),
    ("danger_days_28", "sample", "days", "Days with ACWR >= 1.5 in the 28 days up to the day before the target"),
    ("run_days_7 / run_days_28", "sample", "days", "Run days in the last 7 / 28 days before the target"),
    ("km_7 / km_28", "sample", "km", "Distance in the last 7 / 28 days before the target"),
    ("longest_run_km_28", "sample", "km", "Longest single run in the 28-day window"),
    ("mean_ga_speed_28", "sample", "km/h", "Grade-adjusted speed over all running in the 28-day window"),
    ("mean_hr_pct_28", "sample", "fraction", "Mean daily %HRmax over run days in the window"),
    ("days_since_last_run", "sample", "days", "Days between the last run and the day before the target"),
    ("last_rpc", "sample", "0-1", "Most recent RPC before the target day (any earlier best effort)"),
    ("best_rpc_90", "sample", "0-1", "Best RPC in the 90 days before the target day"),
    ("days_since_last_target", "sample", "days", "Days since that most recent earlier best effort"),
    ("has_prior_rpc", "sample", "0/1", "1 if the athlete has any earlier best effort"),
]


def run_checks(daily, samples):
    print()
    print("=" * 70)
    print("STEP 6 - CHECKS")
    print("=" * 70)

    assert (samples["window_end"] < samples["target_date"]).all(), "window includes target day"
    assert ((samples["window_end"] - samples["window_start"]).dt.days == WINDOW_DAYS - 1).all()
    assert samples["rpc"].between(0, 1, inclusive="right").all(), "RPC outside (0, 1]"
    for df, name in [(daily, "daily"), (samples, "samples")]:
        values = df.select_dtypes("number").to_numpy(dtype=float)
        assert not np.isinf(values).any(), f"infinite value in {name} features"
    assert samples["sample_id"].is_unique
    assert samples["ctl"].notna().all(), "sample without full chronic history"

    # Every sample has exactly 28 daily rows in its window
    counts = daily.merge(samples[["id", "window_start", "window_end"]], on="id")
    in_win = counts[(counts["calendar_date"] >= counts["window_start"]) &
                    (counts["calendar_date"] <= counts["window_end"])]
    per_sample = in_win.groupby(["id", "window_end"]).size()
    assert (per_sample == WINDOW_DAYS).all() and len(per_sample) == len(samples)

    # Prior performance never comes from the target day or later
    assert (samples["days_since_last_target"].dropna() > 0).all()

    print("All checks passed.")


def write_outputs(runs, daily, samples, dropped, population_hrmax, n_windows):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_cols = ["id", "file", "calendar_date", "total_distance", "workout_time",
                "elevation_gain_filled", "elevation_imputed", "effort_km", "ga_speed_kmh",
                "hr_p99", "hrmax", "hrmax_is_default", "hr_pct_max",
                "z1_min", "z2_min", "z3_min", "z4_min", "z5_min", "trimp", "trimp_estimated"]
    runs[run_cols].to_csv(OUTPUT_DIR / "stage3_run_features.csv", index=False)

    daily_cols = (["id", "calendar_date"] + SEQUENCE_FEATURES +
                  ["max_run_km", "acwr_zone", "acwr_km_zone", "km_7", "km_28",
                   "run_days_7", "run_days_28", "longest_run_km_28",
                   "mean_ga_speed_28", "danger_days_28", "ctl_km", "atl_km"])
    daily_cols = list(dict.fromkeys(daily_cols))
    daily[daily_cols].to_csv(OUTPUT_DIR / "stage3_daily_features.csv", index=False)

    samples.to_csv(OUTPUT_DIR / "stage3_samples.csv", index=False)
    dropped.to_csv(OUTPUT_DIR / "stage3_dropped_samples.csv", index=False)

    pd.DataFrame(FEATURE_DICTIONARY, columns=["feature", "level", "unit", "definition"]).to_csv(
        OUTPUT_DIR / "stage3_feature_dictionary.csv", index=False)

    zone_text = ", ".join(f"{n} >= {int(l * 100)}% x{w}" for n, l, w in ZONES)
    by_distance = samples["target_distance_km"].value_counts().sort_index()

    lines = [
        "CAPSTONE - STAGE 3 FEATURE ENGINEERING SUMMARY",
        "=" * 60,
        "",
        "RUN-LEVEL FEATURES",
        "-" * 60,
        f"Real runs: {len(runs):,}",
        f"TRIMP from granular HR: {int((runs['trimp'].notna() & ~runs['trimp_estimated']).sum()):,}",
        f"TRIMP estimated from average HR (no granular HR): {int(runs['trimp_estimated'].sum()):,}",
        f"TRIMP missing: {int(runs['trimp'].isna().sum()):,}",
        f"HRmax: causal running max of per-run {HRMAX_PERCENTILE}th-percentile HR; "
        f"population default {population_hrmax:.1f} bpm before an athlete's first HR run "
        f"({int(runs['hrmax_is_default'].sum()):,} runs)",
        f"Zones (%HRmax, weight): {zone_text}",
        f"Elevation imputed (missing or > {MAX_GAIN_PER_KM_FOR_GAP} m/km) from athlete's earlier median climb/km: {int(runs['elevation_imputed'].sum()):,}",
        f"Per-run p99 HR above {HRMAX_PLAUSIBLE} bpm ignored for HRmax (sensor artefact): {int((runs['hr_p99'] > HRMAX_PLAUSIBLE).sum()):,}",
        f"Samples with no runs in their 28-day window (kept): {int((samples['run_days_28'] == 0).sum()):,}",
        "",
        "DAILY SEQUENCE",
        "-" * 60,
        f"Daily rows: {len(daily):,} | athletes: {daily['id'].nunique():,}",
        f"ATL = {ACUTE_DAYS}-day mean TRIMP, CTL = {CHRONIC_DAYS}-day mean TRIMP, ACWR = ATL / CTL",
        f"ACWR valid only with >= {MIN_CHRONIC_RUN_DAYS} run days in the chronic window",
        f"RNN sequence features ({len(SEQUENCE_FEATURES)}): {', '.join(SEQUENCE_FEATURES)}",
        "",
        "TARGET (RPC)",
        "-" * 60,
        f"RPC = {REFERENCE_SECONDS:.0f} s / (best-effort time x ({REFERENCE_KM:.0f} / distance)^{RIEGEL_EXPONENT})",
        f"Per-distance windows from Stage 1: {n_windows:,}",
        f"Samples (one per athlete per target day): {len(samples) + len(dropped):,}",
        f"Dropped RPC > 1: {int((dropped['drop_reason'] == 'RPC_ABOVE_1').sum()):,}",
        f"Dropped insufficient chronic history: {int((dropped['drop_reason'] == 'INSUFFICIENT_CHRONIC_HISTORY').sum()):,}",
        f"Samples kept: {len(samples):,} | athletes: {samples['id'].nunique():,}",
        f"Athletes with >= 10 samples: {int((samples.groupby('id').size() >= 10).sum()):,}",
        f"RPC median {samples['rpc'].median():.3f}, 1-99%: "
        f"{samples['rpc'].quantile(0.01):.3f}-{samples['rpc'].quantile(0.99):.3f}",
        "Best-effort distance used: " + ", ".join(f"{k} km {v:,}" for k, v in by_distance.items()),
        f"Samples with a prior RPC: {int(samples['has_prior_rpc'].sum()):,}",
        "",
        "LEAKAGE GUARDS",
        "-" * 60,
        "Window = 28 days before the target day (target day excluded).",
        "HRmax and elevation fill use only the athlete's earlier runs.",
        "Prior-RPC features use only best efforts strictly before the target day.",
        "Split, scaling and feature selection are left to the modelling stage (training data only).",
        "",
        "STATUS",
        "-" * 60,
        "Stage 3 feature engineering completed.",
    ]
    (OUTPUT_DIR / "stage3_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nOutputs written to:\n{OUTPUT_DIR}")


# ============================================================
# MAIN
# ============================================================

def main(limit_athletes=None):
    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 3 FEATURE ENGINEERING")
    print("=" * 70)
    print()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    try:
        activity = pd.read_csv(PREPARED_DIR / "stage1_activity_index.csv", low_memory=False)
        spine = pd.read_csv(
            PREPARED_DIR / "stage1_daily_timeline.csv",
            usecols=["id", "calendar_date", "is_activity_day", "days_since_previous_activity"],
        )
        windows = pd.read_csv(PREPARED_DIR / "stage1_28day_window_index.csv")

        if limit_athletes:
            keep = windows["id"].drop_duplicates().head(limit_athletes)
            activity = activity[activity["id"].isin(keep)]
            spine = spine[spine["id"].isin(keep)]
            windows = windows[windows["id"].isin(keep)]
            print(f"TEST MODE: {limit_athletes} athletes\n")

        spine["calendar_date"] = pd.to_datetime(spine["calendar_date"])

        runs, population_hrmax = build_activity_features(activity)
        daily = build_daily_features(runs, spine)
        samples, dropped = build_samples(daily, windows, runs)
        run_checks(daily, samples)
        write_outputs(runs, daily, samples, dropped, population_hrmax, len(windows))

    except Exception as exc:
        (LOG_DIR / "stage3_feature_error.log").write_text(
            "STAGE 3 ERROR\n" + "=" * 60 + "\n\n" + str(exc) + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        print("\nERROR - see logs/stage3_feature_error.log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-athletes", type=int, default=None,
                        help="Run on the first N athletes only (quick test)")
    main(parser.parse_args().limit_athletes)
