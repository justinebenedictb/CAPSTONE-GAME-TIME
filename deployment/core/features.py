"""
deployment/core/features.py  -  Stage 9, Step 3: the runner pipeline

Stages 1-3 of the capstone pipeline for ONE runner, in memory. Every rule is a
line-by-line port of:
    scripts/03_stage1_data_preparation.py   (targets, moving time, quality checks, daily timeline, windows)
    scripts/05_stage3_feature_engineering.py (HRmax, TRIMP, grade-adjusted speed, ATL/CTL/ACWR, samples)
    scripts/capstone_data.py                 (prior_mean_rpc / prior_rpc_count)
Only two things change: data comes from deployment/core/ingest.py instead of files on
disk, and the two values Stage 3 computed over ALL training runners are fixed constants
(a single runner cannot recompute them).

Added for the dashboard: features_for_dates() builds the same model inputs for ANY
date (window = the 28 days before it), for "You now" and the weekly trend.

Check against the real pipeline outputs (needs the project data on your PC):
    py "deployment\\core\\features.py" --quick      (10 GoldenCheetah athletes + demo runner)
    py "deployment\\core\\features.py"              (all athletes + demo runner)
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .ingest import AthleteData, REQUIRED_SIGNATURE
except ImportError:  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ingest import AthleteData, REQUIRED_SIGNATURE

# ============================================================
# CONSTANTS (must match the pipeline scripts)
# ============================================================

REQUIRED_SPORT = "Run"
STANDARD_DISTANCES = {"5.0": 5.0, "10.0": 10.0, "15.0": 15.0, "21.1": 21.1, "42.2": 42.2}
WINDOW_DAYS = 28
MAX_SAMPLE_GAP_SEC = 30
MIN_PACE_SEC_PER_KM = 150
MAX_PACE_SEC_PER_KM = 900
MAX_DISTANCE_RESET_KM = 0.05
MAX_DISTANCE_MISMATCH = 0.05
TARGET_STATUS_OK = "OK"
MAX_ELEVATION_GAIN_PER_KM = 200
MAX_RUN_SPEED_KMH = 18
GRANULAR_REQUIRED_COLUMNS = ["secs", "km", "power", "hr", "cad", "alt"]

ACUTE_DAYS = 7
CHRONIC_DAYS = 28
HR_MIN, HR_MAX = 30, 230
HR_BINS = np.arange(HR_MIN, HR_MAX + 2)
ZONES = [("z1", 0.00, 0.2), ("z2", 0.65, 0.4), ("z3", 0.80, 0.6), ("z4", 0.89, 1.0), ("z5", 0.97, 1.5)]
HRMAX_PERCENTILE = 99
HRMAX_PLAUSIBLE = 210
MAX_GAIN_PER_KM_FOR_GAP = 100
MIN_CHRONIC_RUN_DAYS = 4
ACWR_ZONES = [(0.8, 0, "Under-training"), (1.3, 1, "Sweet Spot"), (1.5, 2, "Monitoring"), (np.inf, 3, "Danger")]
ACWR_DANGER = 1.5
RIEGEL_EXPONENT = 1.06
REFERENCE_KM = 10.0
REFERENCE_SECONDS = 1571.0
PRIOR_RPC_DAYS = 90
TARGET_COLUMNS = {5.0: "target_5_0km_sec", 10.0: "target_10_0km_sec", 15.0: "target_15_0km_sec",
                  21.1: "target_21_1km_sec", 42.2: "target_42_2km_sec"}

# Population values computed by Stage 3 over all 24,807 training-data runs
# (05_stage3_feature_engineering.py). Fixed here because one runner cannot recompute them.
POPULATION_HRMAX = 181.0                       # stage3_summary.txt: default HRmax before a runner's first HR run
POPULATION_GAIN_PER_KM = 12.285056157705839    # nanmedian climb (m/km, <= 100) used before a runner's first run with elevation

SAMPLE_COLUMNS = ["id", "target_date", "window_start", "window_end", "target_distance_km",
                  "target_seconds", "rpc", "atl", "ctl", "acwr", "acwr_km", "acwr_zone",
                  "acwr_km_zone", "acwr_valid", "danger_days_28", "run_days_7", "run_days_28",
                  "km_7", "km_28", "longest_run_km_28", "mean_ga_speed_28", "days_since_last_run",
                  "mean_hr_pct_28", "last_rpc", "best_rpc_90", "days_since_last_target",
                  "has_prior_rpc", "sample_id", "prior_mean_rpc", "prior_rpc_count"]


def rpc_from_time(seconds, distance_km):
    return REFERENCE_SECONDS / (seconds * (REFERENCE_KM / distance_km) ** RIEGEL_EXPONENT)


def time_from_rpc(rpc, distance_km):
    return REFERENCE_SECONDS / (rpc * (REFERENCE_KM / distance_km) ** RIEGEL_EXPONENT)


def _key(label: str) -> str:
    return f"target_{label.replace('.', '_')}km"


# ============================================================
# STAGE 1  (port of 03_stage1_data_preparation.py)
# ============================================================

def prepare_manifest(ad: AthleteData) -> pd.DataFrame:
    """load_and_filter_manifest for one runner.
    GoldenCheetah: the exact Stage 1 filter (sport Run + data signature).
    Strava: runs without a readable file are kept (treated like a Stage 1 activity
    whose granular file is missing: training load counts, no best-effort target)."""
    m = ad.runs.copy()
    m["sport"] = m["sport"].astype(str).str.strip()
    m["data"] = m["data"].astype(str).str.strip()
    keep = m["sport"].eq(REQUIRED_SPORT)
    if ad.source == "goldencheetah":
        keep &= m["data"].eq(REQUIRED_SIGNATURE)
    m = m.loc[keep].copy()
    m["datetime_utc"] = pd.to_datetime(m["date"], errors="coerce", utc=True)
    m = m.loc[m["datetime_utc"].notna()].copy()
    m["calendar_date"] = m["datetime_utc"].dt.floor("D").dt.date
    m["id"] = m["id"].astype(str).str.strip()
    m["file"] = m["file"].astype(str).str.strip()
    m = m.drop_duplicates(subset=["id", "file"], keep="first")
    m = m.sort_values(["id", "datetime_utc", "file"]).reset_index(drop=True)
    m["activity_sequence"] = m.groupby("id").cumcount() + 1
    return m


def clean_distance_series(granular):
    work = granular[["secs", "km"]].dropna()
    work = work[(work["secs"] >= 0) & (work["km"] >= 0)]
    work = work.sort_values("secs").drop_duplicates("secs").reset_index(drop=True)
    if len(work) < 2:
        return None
    km = work["km"].cummax().to_numpy()
    dt = np.diff(work["secs"].to_numpy(), prepend=work["secs"].iloc[0])
    dkm = np.diff(km, prepend=km[0])
    normal = dt <= MAX_SAMPLE_GAP_SEC
    normal_km = dkm[normal].sum()
    if normal_km <= 0:
        return None
    typical_sec_per_km = dt[normal].sum() / normal_km
    credited = np.where(normal, dt, np.minimum(dt, dkm * typical_sec_per_km))
    return km, np.cumsum(credited)


def interpolate_target_time(km, moving_secs, target_km):
    if km[-1] - km[0] < target_km:
        return np.nan
    grid_km, first_idx = np.unique(km, return_index=True)
    grid_t = moving_secs[first_idx]
    starts = km + target_km <= km[-1]
    end_t = np.interp(km[starts] + target_km, grid_km, grid_t)
    return float(np.min(end_t - moving_secs[starts]))


def parse_standard_distance_targets(activity_row, granular: pd.DataFrame | None) -> dict:
    result = {"id": str(activity_row["id"]), "file": str(activity_row["file"]),
              "granular_status": "OK", "granular_rows": np.nan,
              "granular_max_km": np.nan, "granular_moving_time": np.nan}
    for label in STANDARD_DISTANCES:
        result[f"{_key(label)}_sec"] = np.nan
        result[f"{_key(label)}_status"] = np.nan
    if granular is None:
        result["granular_status"] = "MISSING_FILE"
        return result
    granular = granular.copy()
    result["granular_rows"] = len(granular)
    missing = [c for c in GRANULAR_REQUIRED_COLUMNS if c not in granular.columns]
    if missing:
        result["granular_status"] = "MISSING_COLUMNS: " + ",".join(missing)
        return result
    for col in GRANULAR_REQUIRED_COLUMNS:
        granular[col] = pd.to_numeric(granular[col], errors="coerce")
    granular = granular.sort_values(["secs", "km"]).reset_index(drop=True)
    valid_distance = granular["km"].dropna()
    if valid_distance.empty:
        result["granular_status"] = "NO_VALID_DISTANCE"
        return result
    result["granular_max_km"] = valid_distance.max()
    raw_max_km = result["granular_max_km"]

    km_drops = -granular["km"].dropna().diff()
    file_problem = None
    if (km_drops > MAX_DISTANCE_RESET_KM).any():
        file_problem = "DISTANCE_RESET"
    manifest_km = pd.to_numeric(activity_row.get("total_distance"), errors="coerce")
    if (file_problem is None and pd.notna(manifest_km) and manifest_km > 0
            and abs(raw_max_km - manifest_km) / manifest_km > MAX_DISTANCE_MISMATCH):
        file_problem = "DISTANCE_MISMATCH"
    cleaned = None if file_problem else clean_distance_series(granular)
    if file_problem is None and cleaned is None:
        file_problem = "INSUFFICIENT_DATA"
    if cleaned is not None:
        result["granular_moving_time"] = float(cleaned[1][-1])

    for label, target_km in STANDARD_DISTANCES.items():
        key = _key(label)
        if raw_max_km < target_km:
            continue
        if file_problem:
            result[f"{key}_status"] = file_problem
            continue
        seconds = interpolate_target_time(*cleaned, target_km)
        if pd.isna(seconds):
            result[f"{key}_status"] = "NOT_REACHED_CLEAN"
            continue
        pace = seconds / target_km
        if not (MIN_PACE_SEC_PER_KM <= pace <= MAX_PACE_SEC_PER_KM):
            result[f"{key}_status"] = "PACE_OUT_OF_RANGE"
            continue
        result[f"{key}_status"] = TARGET_STATUS_OK
        result[f"{key}_sec"] = seconds
    return result


def build_activity_index(manifest: pd.DataFrame, streams: dict) -> pd.DataFrame:
    target_df = pd.DataFrame([parse_standard_distance_targets(row, streams.get(row["file"]))
                              for _, row in manifest.iterrows()])
    if target_df.empty:
        target_df = pd.DataFrame(columns=["id", "file"])
    activity = manifest.merge(target_df, on=["id", "file"], how="left", validate="one_to_one")
    for label in STANDARD_DISTANCES:   # all-empty status columns would otherwise be numeric
        activity[f"{_key(label)}_status"] = activity[f"{_key(label)}_status"].astype(object)

    distance = pd.to_numeric(activity["total_distance"], errors="coerce")
    raw_time = pd.to_numeric(activity["workout_time"], errors="coerce")
    activity["workout_time_raw"] = raw_time
    raw_pace = raw_time / distance
    activity["workout_time"] = activity["granular_moving_time"].where(
        activity["granular_moving_time"].notna(),
        raw_time.where(raw_pace.between(MIN_PACE_SEC_PER_KM, MAX_PACE_SEC_PER_KM)))
    raw_elev = pd.to_numeric(activity["elevation_gain"], errors="coerce")
    activity["elevation_gain_raw"] = raw_elev
    activity["elevation_gain"] = raw_elev.where(raw_elev / distance <= MAX_ELEVATION_GAIN_PER_KM)
    moving_speed_kmh = distance / (activity["workout_time"] / 3600)
    activity["suspected_non_run"] = moving_speed_kmh > MAX_RUN_SPEED_KMH
    activity["average_speed_raw"] = pd.to_numeric(activity["average_speed"], errors="coerce")
    activity["average_speed"] = moving_speed_kmh
    for label in STANDARD_DISTANCES:
        key = _key(label)
        hit = activity["suspected_non_run"] & activity[f"{key}_status"].eq(TARGET_STATUS_OK)
        activity.loc[hit, f"{key}_status"] = "NOT_A_RUN"
        activity.loc[hit, f"{key}_sec"] = np.nan
    activity = activity.sort_values(["id", "datetime_utc", "file"]).reset_index(drop=True)
    activity["activity_sequence"] = activity.groupby("id").cumcount() + 1
    return activity


def build_daily_timeline(activity: pd.DataFrame) -> pd.DataFrame:
    """Columns of stage1_daily_timeline.csv used downstream: is_activity_day,
    days_since_previous_activity and the fastest best effort per distance per day."""
    act = activity[~activity["suspected_non_run"].astype(bool)].copy()
    out = []
    for athlete_id, a in act.groupby("id", sort=False):
        dates = pd.date_range(pd.Timestamp(a["calendar_date"].min()),
                              pd.Timestamp(a["calendar_date"].max()), freq="D")
        d = pd.DataFrame({"id": athlete_id, "calendar_date": dates})
        a = a.assign(calendar_date=pd.to_datetime(a["calendar_date"]))
        per_day = a.groupby("calendar_date").agg(
            activity_count=("file", "size"),
            **{f"{_key(l)}_sec": (f"{_key(l)}_sec", "min") for l in STANDARD_DISTANCES})
        d = d.merge(per_day, on="calendar_date", how="left")
        d["activity_count"] = d["activity_count"].fillna(0).astype(int)
        d["is_activity_day"] = (d["activity_count"] > 0).astype(int)
        act_date = d["calendar_date"].where(d["is_activity_day"] == 1)
        prev = act_date.shift(1).ffill()
        d["days_since_previous_activity"] = (d["calendar_date"] - prev).dt.days.astype(float)
        out.append(d)
    daily = pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["id", "calendar_date", "is_activity_day", "days_since_previous_activity"])
    daily = daily.sort_values(["id", "calendar_date"]).reset_index(drop=True)
    daily["day_index"] = daily.groupby("id").cumcount()
    return daily


def build_target_windows(timeline: pd.DataFrame) -> pd.DataFrame:
    """build_28day_windows: one window per (target day, distance) with a complete
    28-day history ending the day before."""
    rows = []
    for athlete_id, d in timeline.groupby("id", sort=False):
        first = d["calendar_date"].min()
        for label, km in STANDARD_DISTANCES.items():
            col = f"{_key(label)}_sec"
            t = d.loc[d[col].notna(), ["calendar_date", col]]
            for target_date, secs in zip(t["calendar_date"], t[col]):
                window_start = target_date - pd.Timedelta(days=WINDOW_DAYS)
                if first > window_start:
                    continue
                rows.append({"id": athlete_id, "window_start": window_start,
                             "window_end": target_date - pd.Timedelta(days=1),
                             "target_date": target_date, "target_distance_km": float(km),
                             "target_seconds": float(secs)})
    cols = ["id", "window_start", "window_end", "target_date", "target_distance_km", "target_seconds"]
    w = pd.DataFrame(rows, columns=cols)
    return w.sort_values(["id", "target_date", "target_distance_km"]).reset_index(drop=True)


# ============================================================
# STAGE 3  (port of 05_stage3_feature_engineering.py)
# ============================================================

def zone_weight(hr_fraction):
    weight = ZONES[0][2]
    for _, lower, w in ZONES:
        if hr_fraction >= lower:
            weight = w
    return weight


def granular_hr_histogram(g: pd.DataFrame):
    g = g[["secs", "km", "hr"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["secs", "km"])
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
    if valid.sum() < 60:
        return None, np.nan
    hist, _ = np.histogram(hr[valid], bins=HR_BINS, weights=credited[valid])
    return hist, float(np.percentile(hr[valid], HRMAX_PERCENTILE))


def build_activity_features(activity: pd.DataFrame, streams: dict) -> pd.DataFrame:
    runs = activity[~activity["suspected_non_run"].astype(bool)].copy()
    runs["datetime_utc"] = pd.to_datetime(runs["datetime_utc"], utc=True)
    runs = runs.sort_values(["id", "datetime_utc"]).reset_index(drop=True)

    hists, p99 = [], np.full(len(runs), np.nan)
    for i, row in enumerate(runs.itertuples(index=False)):
        hist = None
        if row.granular_status == "OK" and row.file in streams:
            try:
                hist, p99[i] = granular_hr_histogram(streams[row.file])
            except Exception:
                hist = None
        hists.append(hist)

    runs["hr_p99"] = p99
    plausible_p99 = runs["hr_p99"].where(runs["hr_p99"] <= HRMAX_PLAUSIBLE)
    runs["hrmax"] = plausible_p99.groupby(runs["id"]).cummax()
    runs["hrmax"] = runs.groupby("id")["hrmax"].ffill()
    runs["hrmax_is_default"] = runs["hrmax"].isna()
    runs["hrmax"] = runs["hrmax"].fillna(POPULATION_HRMAX)

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

    gain_per_km = runs["elevation_gain"] / runs["total_distance"]
    gain_per_km = gain_per_km.where(gain_per_km <= MAX_GAIN_PER_KM_FOR_GAP)
    prior_median = gain_per_km.groupby(runs["id"]).transform(lambda s: s.expanding().median().shift(1))
    runs["elevation_imputed"] = gain_per_km.isna()
    fill = prior_median.fillna(POPULATION_GAIN_PER_KM) * runs["total_distance"]
    runs["elevation_gain_filled"] = (gain_per_km * runs["total_distance"]).fillna(fill)
    runs["effort_km"] = runs["total_distance"] + runs["elevation_gain_filled"] / 100
    runs["ga_speed_kmh"] = runs["effort_km"] / (runs["workout_time"] / 3600)
    return runs


def acwr_zone_code(acwr):
    code = pd.Series(np.nan, index=acwr.index)
    lower = -np.inf
    for upper, c, _ in ACWR_ZONES:
        code[(acwr >= lower) & (acwr < upper)] = c
        lower = upper
    return code


def build_daily_features(runs: pd.DataFrame, spine: pd.DataFrame) -> pd.DataFrame:
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
        rest, 0.0, daily["effort_km"] / (daily["moving_time_min"] / 60).where(daily["moving_time_min"] > 0))
    daily["hr_pct_max"] = np.where(rest, 0.0, daily["hr_weighted"] / daily["hr_time"])
    daily = daily.drop(columns=["hr_weighted", "hr_time"])
    daily = daily.sort_values(["id", "calendar_date"]).reset_index(drop=True)
    g = daily.groupby("id", sort=False)

    def roll(col, days, how):
        return g[col].transform(lambda s: getattr(s.fillna(0).rolling(days, min_periods=days), how)())

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
    valid = daily["run_days_28"] >= MIN_CHRONIC_RUN_DAYS
    daily["acwr"] = (daily["atl"] / daily["ctl"]).where(valid & (daily["ctl"] > 0))
    daily["acwr_km"] = (daily["atl_km"] / daily["ctl_km"]).where(valid & (daily["ctl_km"] > 0))
    daily["acwr_valid"] = daily["acwr"].notna().astype(int)
    daily["acwr_zone"] = acwr_zone_code(daily["acwr"])
    daily["acwr_km_zone"] = acwr_zone_code(daily["acwr_km"])
    daily["danger_days_28"] = (
        (daily["acwr"] >= ACWR_DANGER).astype(int)
        .groupby(daily["id"]).transform(lambda s: s.rolling(CHRONIC_DAYS, min_periods=1).sum()))
    return daily


def performance_history(runs: pd.DataFrame) -> pd.DataFrame:
    """Stage 3 'perf': every earlier best effort, best RPC per day (RPC <= 1)."""
    runs = runs.copy()
    runs["calendar_date"] = pd.to_datetime(runs["calendar_date"])
    perf = []
    for km, col in TARGET_COLUMNS.items():
        t = runs[["id", "calendar_date", col]].dropna()
        perf.append(pd.DataFrame({"id": t["id"], "date": t["calendar_date"],
                                  "rpc": rpc_from_time(t[col], km)}))
    perf = pd.concat(perf) if perf else pd.DataFrame(columns=["id", "date", "rpc"])
    return perf[perf["rpc"] <= 1.0].groupby(["id", "date"])["rpc"].max().reset_index()


def daily_known_rpc(timeline: pd.DataFrame) -> pd.DataFrame:
    """capstone_data.daily_known_rpc on this runner's Stage 1 timeline."""
    rpc = pd.concat([rpc_from_time(pd.to_numeric(timeline[col], errors="coerce"), km)
                     for km, col in TARGET_COLUMNS.items()], axis=1).max(axis=1)
    rpc = rpc.where(rpc <= 1.0)
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
    return out[["id", "calendar_date", "known_rpc", "days_since_rpc", "mean_rpc_to_date", "rpc_count_to_date"]]


LOOKUP_COLS = ["atl", "ctl", "acwr", "acwr_km", "acwr_zone", "acwr_km_zone", "acwr_valid",
               "danger_days_28", "run_days_7", "run_days_28", "km_7", "km_28",
               "longest_run_km_28", "mean_ga_speed_28", "days_since_previous_activity"]


def window_features(frame: pd.DataFrame, daily: pd.DataFrame, perf: pd.DataFrame,
                    known: pd.DataFrame) -> pd.DataFrame:
    """Stage 3 step 5 + capstone_data.add_personal_level for rows with
    id, target_date, window_start, window_end."""
    day_before = daily[["id", "calendar_date"] + LOOKUP_COLS].rename(columns={"calendar_date": "window_end"})
    s = frame.merge(day_before, on=["id", "window_end"], how="left", validate="many_to_one")
    s = s.rename(columns={"days_since_previous_activity": "days_since_last_run"})

    by_athlete = {k: v.set_index("calendar_date") for k, v in daily.groupby("id")}
    win_hr = []
    for r in s.itertuples(index=False):
        d = by_athlete[r.id].loc[r.window_start:r.window_end]
        act = d[d["is_activity_day"] == 1]
        win_hr.append(act["hr_pct_max"].mean() if len(act) else np.nan)
    s["mean_hr_pct_28"] = win_hr

    perf_by_athlete = {k: v.sort_values("date") for k, v in perf.groupby("id")}
    last_rpc, best_rpc_90, days_since_target = [], [], []
    for r in s.itertuples(index=False):
        p = perf_by_athlete.get(r.id)
        prior = p[p["date"] < r.target_date] if p is not None else None
        if prior is None or prior.empty:
            last_rpc.append(np.nan); best_rpc_90.append(np.nan); days_since_target.append(np.nan)
            continue
        last_rpc.append(prior["rpc"].iloc[-1])
        days_since_target.append((r.target_date - prior["date"].iloc[-1]).days)
        recent = prior[prior["date"] >= r.target_date - pd.Timedelta(days=PRIOR_RPC_DAYS)]
        best_rpc_90.append(recent["rpc"].max() if len(recent) else np.nan)
    s["last_rpc"] = last_rpc
    s["best_rpc_90"] = best_rpc_90
    s["days_since_last_target"] = days_since_target
    s["has_prior_rpc"] = s["last_rpc"].notna().astype(int)
    s["sample_id"] = s["id"] + "_" + s["target_date"].dt.strftime("%Y-%m-%d")

    k = known.rename(columns={"calendar_date": "window_end", "mean_rpc_to_date": "prior_mean_rpc",
                              "rpc_count_to_date": "prior_rpc_count"})[
        ["id", "window_end", "prior_mean_rpc", "prior_rpc_count"]]
    s = s.merge(k, on=["id", "window_end"], how="left", validate="many_to_one")
    s["prior_rpc_count"] = s["prior_rpc_count"].fillna(0).astype(int)
    return s


def build_samples(daily, windows, perf, known):
    """Stage 3 steps 4-5: best RPC per target day, drop RPC > 1 and short history."""
    w = windows.copy()
    w["rpc"] = rpc_from_time(w["target_seconds"], w["target_distance_km"])
    if w.empty:
        return pd.DataFrame(columns=SAMPLE_COLUMNS), pd.DataFrame()
    best = w.loc[w.groupby(["id", "target_date"])["rpc"].idxmax()]
    best = best[["id", "target_date", "window_start", "window_end",
                 "target_distance_km", "target_seconds", "rpc"]].reset_index(drop=True)
    above = best["rpc"] > 1.0
    dropped = [best[above].assign(drop_reason="RPC_ABOVE_1")]
    best = best[~above]
    first_day = daily.groupby("id")["calendar_date"].min()
    history_days = (best["window_start"] - best["id"].map(first_day)).dt.days
    short = history_days < CHRONIC_DAYS - 1
    dropped.append(best[short].assign(drop_reason="INSUFFICIENT_CHRONIC_HISTORY"))
    best = best[~short].reset_index(drop=True)
    samples = window_features(best, daily, perf, known)
    samples = samples.sort_values(["id", "target_date"]).reset_index(drop=True)
    return samples[SAMPLE_COLUMNS], pd.concat(dropped, ignore_index=True)


# ============================================================
# ONE RUNNER, END TO END
# ============================================================

@dataclass
class Runner:
    athlete_id: str
    source: str
    activity: pd.DataFrame          # Stage 1 activity index (targets, statuses, cleaned fields)
    timeline: pd.DataFrame          # Stage 1 daily timeline (one row per calendar day)
    runs: pd.DataFrame              # Stage 3 per-run features (HRmax, TRIMP, GAP)
    daily: pd.DataFrame             # Stage 3 daily features (ATL/CTL/ACWR, rolling 28-day)
    samples: pd.DataFrame           # best-effort days with complete history (+ actual RPC)
    dropped: pd.DataFrame
    perf: pd.DataFrame
    known: pd.DataFrame
    info: dict = field(default_factory=dict)

    @property
    def first_day(self) -> pd.Timestamp:
        return self.daily["calendar_date"].min()

    @property
    def last_day(self) -> pd.Timestamp:
        return self.daily["calendar_date"].max()

    @property
    def earliest_estimate_date(self) -> pd.Timestamp:
        """First date with the 28-day window AND 28 days of chronic history before it
        (Stage 3 drops samples with less): first run day + 55 days."""
        return self.first_day + pd.Timedelta(days=WINDOW_DAYS + CHRONIC_DAYS - 1)

    @property
    def you_now_date(self) -> pd.Timestamp:
        """'You now' = the day after the latest run (window ends on the latest run day)."""
        return self.last_day + pd.Timedelta(days=1)

    def features_for_dates(self, dates) -> pd.DataFrame:
        """Model inputs for any target dates (no best effort needed). Dates outside
        [earliest_estimate_date, you_now_date] are returned with ok=False."""
        dates = pd.to_datetime(pd.Series(list(dates))).dt.normalize().drop_duplicates()
        ok = (dates >= self.earliest_estimate_date) & (dates <= self.you_now_date)
        frame = pd.DataFrame({"id": self.athlete_id, "target_date": dates[ok].values})
        frame["window_start"] = frame["target_date"] - pd.Timedelta(days=WINDOW_DAYS)
        frame["window_end"] = frame["target_date"] - pd.Timedelta(days=1)
        if frame.empty:
            feats = frame
        else:
            feats = window_features(frame, self.daily, self.perf, self.known)
        out = pd.DataFrame({"target_date": dates.values}).merge(feats, on="target_date", how="left")
        out["ok"] = out["target_date"].between(self.earliest_estimate_date, self.you_now_date)
        return out


def build_runner(ad: AthleteData) -> Runner:
    t0 = time.time()
    manifest = prepare_manifest(ad)
    if manifest.empty:
        raise ValueError("No runs found.")
    activity = build_activity_index(manifest, ad.streams)
    timeline = build_daily_timeline(activity)
    runs = build_activity_features(activity, ad.streams)
    spine = timeline[["id", "calendar_date", "is_activity_day", "days_since_previous_activity"]]
    daily = build_daily_features(runs, spine)
    perf = performance_history(runs)
    known = daily_known_rpc(timeline)
    windows = build_target_windows(timeline)
    samples, dropped = build_samples(daily, windows, perf, known)
    info = {
        "runs": int(len(runs)),
        "excluded_as_non_run": int(activity["suspected_non_run"].astype(bool).sum()),
        "best_effort_days": int(windows["target_date"].nunique()) if len(windows) else 0,
        "samples": int(len(samples)),
        "trimp_from_hr_stream": int((runs["trimp"].notna() & ~runs["trimp_estimated"]).sum()),
        "trimp_estimated_from_avg_hr": int(runs["trimp_estimated"].sum()),
        "trimp_missing": int(runs["trimp"].isna().sum()),
        "days": int(len(daily)),
        "seconds_to_build": round(time.time() - t0, 1),
    }
    return Runner(ad.athlete_id, ad.source, activity, timeline, runs, daily, samples,
                  dropped, perf, known, info)


# ============================================================
# SELF-CHECK: match the real pipeline outputs
# ============================================================

def _compare(mine: pd.DataFrame, ref: pd.DataFrame, key: list, cols: list) -> dict:
    m = mine[key + cols].merge(ref[key + cols], on=key, how="outer", suffixes=("", "_ref"), indicator=True)
    res = {"rows_mine": int(len(mine)), "rows_ref": int(len(ref)),
           "only_mine": int((m["_merge"] == "left_only").sum()),
           "only_ref": int((m["_merge"] == "right_only").sum()), "max_abs_diff": {}, "mismatched_values": 0}
    both = m[m["_merge"] == "both"]
    for c in cols:
        a = pd.to_numeric(both[c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(both[c + "_ref"], errors="coerce").to_numpy(float)
        nan_mismatch = np.isnan(a) != np.isnan(b)
        diff = np.where(np.isnan(a) | np.isnan(b), 0.0, np.abs(a - b))
        bad = nan_mismatch | (diff > 1e-9 * np.maximum(1.0, np.abs(b)))
        res["mismatched_values"] += int(bad.sum())
        res["max_abs_diff"][c] = float(diff.max()) if len(diff) else 0.0
    return res


def _self_check(argv=None) -> int:
    import argparse
    try:
        from .ingest import load_goldencheetah, load_strava_export
    except ImportError:
        from ingest import load_goldencheetah, load_strava_export
    dep = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(dep.parent), help="project folder")
    ap.add_argument("--quick", action="store_true", help="10 athletes")
    ap.add_argument("--athletes", type=int, default=None)
    args = ap.parse_args(argv)
    root = Path(args.root)
    lines = ["Stage 9 Step 3 - runner pipeline check", "=" * 70]

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- A. GoldenCheetah athletes vs the real Stage 1/3 outputs ----
    ref_samples = pd.read_csv(root / "03_feature_engineering" / "stage3_samples.csv",
                              parse_dates=["target_date", "window_start", "window_end"])
    ref_runs = pd.read_csv(root / "03_feature_engineering" / "stage3_run_features.csv", low_memory=False)
    ref_act = pd.read_csv(root / "01_prepared" / "stage1_activity_index.csv", low_memory=False)
    ref_tl = pd.read_csv(root / "01_prepared" / "stage1_daily_timeline.csv", low_memory=False,
                         usecols=["id", "calendar_date"] + list(TARGET_COLUMNS.values()),
                         parse_dates=["calendar_date"])
    athletes = sorted(ref_samples["id"].unique())
    n = 10 if args.quick else (args.athletes or len(athletes))
    athletes = athletes[:n]
    say(f"A. GoldenCheetah match check: {len(athletes)} athletes")

    run_cols = ["workout_time", "hr_p99", "hrmax", "hr_pct_max", "trimp", "z1_min", "z2_min", "z3_min",
                "z4_min", "z5_min", "elevation_gain_filled", "effort_km", "ga_speed_kmh"]
    act_cols = [f"{_key(l)}_sec" for l in STANDARD_DISTANCES] + ["granular_moving_time"]
    samp_cols = [c for c in SAMPLE_COLUMNS if c not in ("id", "sample_id", "target_date",
                                                         "window_start", "window_end",
                                                         "prior_mean_rpc", "prior_rpc_count")]
    totals = {"activities": [], "runs": [], "samples": [], "prior": []}
    t0 = time.time()
    manifest_path = root / "manifest file" / "run_activities_manifest.csv"
    for i, a in enumerate(athletes, 1):
        print(f"  athlete {i}/{len(athletes)}", end="\r")
        ad = load_goldencheetah(a, manifest_path, root / "00_raw" / "granular")
        r = build_runner(ad)
        totals["activities"].append(_compare(r.activity, ref_act[ref_act["id"] == a], ["id", "file"], act_cols))
        totals["runs"].append(_compare(r.runs, ref_runs[ref_runs["id"] == a], ["id", "file"], run_cols))
        totals["samples"].append(_compare(r.samples, ref_samples[ref_samples["id"] == a], ["sample_id"], samp_cols))
        # prior_mean_rpc reference: capstone_data.daily_known_rpc on the REAL Stage 1 timeline
        ref_known = daily_known_rpc(ref_tl[ref_tl["id"] == a].copy())
        ref_prior = ref_samples[ref_samples["id"] == a][["sample_id", "id", "window_end"]].merge(
            ref_known.rename(columns={"calendar_date": "window_end", "mean_rpc_to_date": "prior_mean_rpc",
                                      "rpc_count_to_date": "prior_rpc_count"}), on=["id", "window_end"], how="left")
        totals["prior"].append(_compare(r.samples, ref_prior, ["sample_id"], ["prior_mean_rpc", "prior_rpc_count"]))
    print()
    all_ok = True
    for name, results in totals.items():
        mism = sum(x["mismatched_values"] for x in results)
        only_m = sum(x["only_mine"] for x in results)
        only_r = sum(x["only_ref"] for x in results)
        rows = sum(x["rows_ref"] for x in results)
        worst = max((max(x["max_abs_diff"].values()) for x in results if x["max_abs_diff"]), default=0.0)
        ok = mism == 0 and only_m == 0 and only_r == 0
        all_ok &= ok
        say(f"  {'OK ' if ok else 'BAD'} {name:10s} reference rows {rows:7,d} | missing {only_r} | extra {only_m} "
            f"| mismatched values {mism} | largest difference {worst:.2e}")
    say(f"  time: {time.time() - t0:.0f} s")

    # ---- B. Demo runner (justinhojt) ----
    say("")
    say("B. Demo runner (justinhojt Strava export)")
    demo = dep / "demo_data" / "justinhojt_strava_export.zip"
    ad = load_strava_export(demo)
    r = build_runner(ad)
    say(f"  {r.info}")
    say(f"  timeline {r.first_day.date()} -> {r.last_day.date()} | estimates possible from "
        f"{r.earliest_estimate_date.date()} | 'You now' date {r.you_now_date.date()}")
    if len(r.samples):
        s = r.samples
        say(f"  best-effort samples: {len(s)} | RPC median {s['rpc'].median():.3f} "
            f"(5K-equivalent {time_from_rpc(s['rpc'].median(), 5.0) / 60:.1f} min) | distances "
            f"{s['target_distance_km'].value_counts().sort_index().to_dict()}")
    now = r.features_for_dates([r.you_now_date])
    show = ["prior_mean_rpc", "last_rpc", "best_rpc_90", "mean_ga_speed_28", "mean_hr_pct_28",
            "ctl", "atl", "acwr", "longest_run_km_28", "days_since_last_run", "prior_rpc_count"]
    say("  'You now' inputs: " + ", ".join(f"{c}={now[c].iloc[0]:.3f}" for c in show))
    weekly = pd.date_range(r.earliest_estimate_date, r.you_now_date, freq="7D")
    wf = r.features_for_dates(weekly)
    say(f"  weekly trend dates: {len(wf)} (all with inputs: {bool(wf['ok'].all() and wf['ctl'].notna().all())})")
    # consistency: features_for_dates on the sample dates must equal the samples themselves
    if len(r.samples):
        again = r.features_for_dates(r.samples["target_date"])
        cmp = _compare(again.assign(id=r.athlete_id), r.samples, ["sample_id"],
                       [c for c in samp_cols if c not in ("target_distance_km", "target_seconds", "rpc")]
                       + ["prior_mean_rpc", "prior_rpc_count"])
        ok = cmp["mismatched_values"] == 0 and cmp["only_ref"] == 0
        all_ok &= ok
        say(f"  {'OK ' if ok else 'BAD'} any-date features == best-effort sample features "
            f"({cmp['rows_ref']} dates, mismatched values {cmp['mismatched_values']})")

    say("")
    say("RESULT: " + ("ALL GOOD - dashboard pipeline reproduces Stages 1-3 exactly" if all_ok
                      else "MISMATCH - send this report"))
    out = dep / "_checks"
    out.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    (out / f"step3_features_report{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    r.samples.to_csv(out / f"step3_demo_samples{tag}.csv", index=False)
    print(f"\nSaved: deployment/_checks/step3_features_report{tag}.txt")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
