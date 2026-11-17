"""
deployment/core/run_analytics.py  -  Stage 9, Step 5: per-run analytics ("This run" column)

Everything shown for one selected run. Uses the pipeline's own numbers where they exist
(Stage 1 moving-time clock and best efforts, Stage 3 TRIMP / zone minutes / grade-adjusted
speed), so the "This run" panel agrees with what the model saw.

    run_summary()     distance, moving time, pace, grade-adjusted pace, HR, elevation, TRIMP, zones
    km_splits()       time, pace, average HR and elevation change for each kilometre (moving time)
    best_efforts()    fastest 1K / 1 mile / 5K / 10K / half / marathon segment inside the run
                      (5K and longer = the pipeline's best-effort targets, with their status)
    pace_hr_series()  smoothed pace + HR + elevation along the run, for the chart
    private_route()   GPS for the map with PRIVACY ZONES: the first and last 500 m along the
                      route are cut AND every point within 500 m (straight line) of the start or
                      finish is removed (hides home even if a loop passes near it mid-run).
                      Coordinates are only returned for display, never stored.

Check:  py "deployment\\core\\run_analytics.py"   (all of the demo runner's runs)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .features import (Runner, clean_distance_series, interpolate_target_time,
                           STANDARD_DISTANCES, ZONES, rpc_from_time, MAX_SAMPLE_GAP_SEC)
    from .ingest import AthleteData, EARTH_RADIUS_M
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import (Runner, clean_distance_series, interpolate_target_time,
                          STANDARD_DISTANCES, ZONES, rpc_from_time, MAX_SAMPLE_GAP_SEC)
    from ingest import AthleteData, EARTH_RADIUS_M

PRIVACY_RADIUS_M = 500
PRIVACY_TRIM_KM = 0.5
MIN_ROUTE_KM_AFTER_TRIM = 0.5          # less than this left -> no map
MAP_MAX_POINTS = 1500                  # downsample for display
DISPLAY_EFFORTS = {"1K": 1.0, "1 mile": 1.609344, "5K": 5.0, "10K": 10.0,
                   "Half marathon": 21.0975, "Marathon": 42.195}
PIPELINE_KEY = {"5K": "5.0", "10K": "10.0", "Half marathon": "21.1", "Marathon": "42.2"}
ZONE_LABELS = {"z1": "Z1 Recovery (<65%)", "z2": "Z2 Aerobic (65-80%)", "z3": "Z3 Tempo (80-89%)",
               "z4": "Z4 Threshold (89-97%)", "z5": "Z5 Max (>97%)"}
PACE_SMOOTH_SEC = 30
MAX_DISPLAY_PACE = 15 * 60             # s/km; slower is shown as a gap (walking / stopped)


def _row(runner: Runner, file: str) -> pd.Series:
    r = runner.runs[runner.runs["file"] == file]
    if r.empty:
        raise KeyError(f"run not found: {file}")
    return r.iloc[0]


def _clock(stream: pd.DataFrame):
    """Stage 1 moving-time clock + matching HR/alt on the same (sorted, de-duplicated) samples."""
    g = stream[["secs", "km", "hr", "alt"]].apply(pd.to_numeric, errors="coerce").dropna(subset=["secs", "km"])
    g = g[(g["secs"] >= 0) & (g["km"] >= 0)].sort_values("secs").drop_duplicates("secs").reset_index(drop=True)
    c = clean_distance_series(g)
    if c is None:
        return None
    km, moving = c
    return pd.DataFrame({"secs": g["secs"], "km": km, "moving_s": moving, "hr": g["hr"], "alt": g["alt"]})


# ============================================================
# Summary
# ============================================================

def run_summary(runner: Runner, file: str) -> dict:
    r = _row(runner, file)
    dist = float(r["total_distance"])
    mt = float(r["workout_time"]) if pd.notna(r["workout_time"]) else np.nan
    zones = {ZONE_LABELS[z]: float(r[f"{z}_min"]) for z, _, _ in ZONES} if not r["trimp_estimated"] else {}
    if zones and sum(zones.values()) == 0:
        zones = {}
    return {
        "file": file, "name": r.get("activity_name", ""),
        "start_utc": r["datetime_utc"], "date": pd.Timestamp(r["calendar_date"]),
        "distance_km": dist, "moving_time_s": mt,
        "elapsed_time_s": float(r["elapsed_time"]) if "elapsed_time" in r and pd.notna(r.get("elapsed_time")) else np.nan,
        "pace_s_per_km": mt / dist if dist > 0 and np.isfinite(mt) else np.nan,
        "gap_s_per_km": 3600 / float(r["ga_speed_kmh"]) if pd.notna(r["ga_speed_kmh"]) and r["ga_speed_kmh"] > 0 else np.nan,
        "avg_hr": float(r["average_hr"]) if pd.notna(r["average_hr"]) else np.nan,
        "max_hr": float(r["max_hr"]) if "max_hr" in r and pd.notna(r.get("max_hr")) else np.nan,
        "hrmax_used": float(r["hrmax"]), "hrmax_is_default": bool(r["hrmax_is_default"]),
        "hr_pct_max": float(r["hr_pct_max"]) if pd.notna(r["hr_pct_max"]) else np.nan,
        "elevation_gain_m": float(r["elevation_gain"]) if pd.notna(r["elevation_gain"]) else np.nan,
        "elevation_imputed": bool(r["elevation_imputed"]),
        "trimp": float(r["trimp"]) if pd.notna(r["trimp"]) else np.nan,
        "trimp_estimated": bool(r["trimp_estimated"]),
        "zone_minutes": zones,
        "has_stream": bool(r.get("has_stream", False)), "has_hr": bool(r.get("has_hr", False)),
        "has_gps": bool(r.get("has_gps", False)),
    }


# ============================================================
# Splits, best efforts, chart
# ============================================================

def km_splits(ad: AthleteData, file: str) -> pd.DataFrame:
    cols = ["km", "split_km", "time_s", "pace_s_per_km", "avg_hr", "elev_change_m"]
    s = ad.streams.get(file)
    c = _clock(s) if s is not None else None
    if c is None or c["km"].iloc[-1] < 0.2:
        return pd.DataFrame(columns=cols)
    grid_km, idx = np.unique(c["km"].to_numpy(), return_index=True)
    grid_t = c["moving_s"].to_numpy()[idx]
    total = c["km"].iloc[-1]
    marks = list(np.arange(1, np.floor(total) + 1))
    if total - (marks[-1] if marks else 0) >= 0.1:
        marks.append(total)
    rows, prev_km, prev_t = [], 0.0, 0.0
    for m in marks:
        t = float(np.interp(m, grid_km, grid_t))
        seg = c[(c["km"] > prev_km) & (c["km"] <= m)]
        alt = seg["alt"].dropna()
        rows.append({"km": int(np.ceil(m)) if m == np.floor(m) else round(m, 2), "split_km": m - prev_km,
                     "time_s": t - prev_t, "pace_s_per_km": (t - prev_t) / (m - prev_km),
                     "avg_hr": float(seg["hr"].where(seg["hr"].between(30, 230)).mean()) if len(seg) else np.nan,
                     "elev_change_m": float(alt.iloc[-1] - alt.iloc[0]) if len(alt) > 1 else np.nan})
        prev_km, prev_t = m, t
    return pd.DataFrame(rows, columns=cols)


def best_efforts(runner: Runner, ad: AthleteData, file: str) -> pd.DataFrame:
    """Fastest segment per distance. 5K+ rows also show whether the pipeline counted it as a
    best-effort target (status OK) and its RPC; 1K / 1 mile are for display only."""
    s = ad.streams.get(file)
    act = runner.activity[runner.activity["file"] == file]
    a = act.iloc[0] if len(act) else None
    c = _clock(s) if s is not None else None
    rows = []
    for name, km in DISPLAY_EFFORTS.items():
        secs = np.nan
        if c is not None:
            secs = interpolate_target_time(c["km"].to_numpy(), c["moving_s"].to_numpy(), km)
        if not np.isfinite(secs):
            continue
        row = {"distance": name, "km": km, "seconds": float(secs), "pace_s_per_km": float(secs) / km,
               "model_target": False, "status": "display only", "rpc": np.nan}
        if name in PIPELINE_KEY and a is not None:
            key = f"target_{PIPELINE_KEY[name].replace('.', '_')}km"
            row["status"] = a.get(f"{key}_status")
            if row["status"] == "OK":
                row["model_target"] = True
                row["seconds"] = float(a[f"{key}_sec"])           # exactly the pipeline's value
                row["rpc"] = float(rpc_from_time(row["seconds"], float(PIPELINE_KEY[name])))
        rows.append(row)
    return pd.DataFrame(rows, columns=["distance", "km", "seconds", "pace_s_per_km", "model_target", "status", "rpc"])


def pace_hr_series(ad: AthleteData, file: str, every_s: int = 10) -> pd.DataFrame:
    cols = ["km", "moving_min", "pace_s_per_km", "hr", "alt"]
    s = ad.streams.get(file)
    c = _clock(s) if s is not None else None
    if c is None:
        return pd.DataFrame(columns=cols)
    secs = c["secs"].to_numpy(float)
    gap = np.diff(secs, prepend=secs[0]) > MAX_SAMPLE_GAP_SEC
    seg = np.cumsum(gap)                                   # no smoothing across pauses
    df = c.assign(seg=seg)
    t = pd.to_timedelta(df["secs"], unit="s")
    sp = []
    for _, g in df.groupby("seg"):
        gi = g.set_index(pd.to_timedelta(g["secs"], unit="s"))
        dk = gi["km"].rolling(f"{PACE_SMOOTH_SEC}s").apply(lambda x: x[-1] - x[0], raw=True)
        dt = gi["secs"].rolling(f"{PACE_SMOOTH_SEC}s").apply(lambda x: x[-1] - x[0], raw=True)
        sp.append(pd.Series((dt / dk).to_numpy(), index=g.index))
    pace = pd.concat(sp).reindex(df.index)
    pace = pace.where((pace > 0) & (pace <= MAX_DISPLAY_PACE))
    out = pd.DataFrame({"km": df["km"], "moving_min": df["moving_s"] / 60, "pace_s_per_km": pace,
                        "hr": df["hr"].where(df["hr"].between(30, 230)), "alt": df["alt"]})
    keep = (df["secs"] // every_s).drop_duplicates().index
    return out.loc[keep].reset_index(drop=True)


# ============================================================
# Route with privacy zones
# ============================================================

def _dist_m(lat0, lon0, lat, lon):
    p0, l0, p, l = map(np.radians, (lat0, lon0, lat, lon))
    a = np.sin((p - p0) / 2) ** 2 + np.cos(p0) * np.cos(p) * np.sin((l - l0) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def private_route(ad: AthleteData, file: str) -> dict:
    """Route for display only. Returns {'ok', 'segments' [list of (lat, lon) arrays], 'reason'}."""
    g = ad.gps.get(file)
    if g is None or len(g) < 2:
        return {"ok": False, "reason": "No GPS in this run.", "segments": []}
    lat, lon = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
    good = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 1e-6)
    lat, lon = lat[good], lon[good]
    if len(lat) < 2:
        return {"ok": False, "reason": "No GPS in this run.", "segments": []}
    step = _dist_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    along = np.concatenate([[0.0], np.cumsum(step)]) / 1000
    keep = (along >= PRIVACY_TRIM_KM) & (along <= along[-1] - PRIVACY_TRIM_KM)
    keep &= _dist_m(lat[0], lon[0], lat, lon) > PRIVACY_RADIUS_M
    keep &= _dist_m(lat[-1], lon[-1], lat, lon) > PRIVACY_RADIUS_M
    if keep.sum() < 2:
        return {"ok": False, "reason": "Run too short to show the route after hiding 500 m around start and finish.",
                "segments": []}
    # split where hidden points interrupt the line, so no line is drawn through a privacy zone
    idx = np.flatnonzero(keep)
    breaks = np.flatnonzero(np.diff(idx) > 1) + 1
    segments = []
    shown_km = 0.0
    for part in np.split(idx, breaks):
        if len(part) < 2:
            continue
        shown_km += along[part[-1]] - along[part[0]]
        segments.append(part)
    if shown_km < MIN_ROUTE_KM_AFTER_TRIM:
        return {"ok": False, "reason": "Too little of the route is left after the privacy zones.", "segments": []}
    total_pts = sum(len(p) for p in segments)
    stride = max(1, int(np.ceil(total_pts / MAP_MAX_POINTS)))
    segs = [np.column_stack([lat[p][::stride], lon[p][::stride]]) for p in segments]
    return {"ok": True, "segments": segs, "shown_km": round(shown_km, 2), "total_km": round(along[-1], 2),
            "reason": ""}


# ============================================================
# Self-check
# ============================================================

def _self_check(argv=None) -> int:
    import argparse
    import time
    try:
        from .ingest import load_strava_export
        from .features import build_runner
    except ImportError:
        from ingest import load_strava_export
        from features import build_runner
    dep = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="20 most recent runs")
    args = ap.parse_args(argv)
    lines = ["Stage 9 Step 5 - per-run analytics check", "=" * 70]

    def say(s=""):
        print(s)
        lines.append(s)

    ad = load_strava_export(dep / "demo_data" / "justinhojt_strava_export.zip",
                            limit_runs=20 if args.quick else None)
    runner = build_runner(ad)
    files = list(runner.runs["file"])
    say(f"Demo runner: {len(files)} runs")
    t0 = time.time()
    problems, n = [], {"splits": 0, "efforts": 0, "chart": 0, "map": 0, "no_map": {}}
    split_diffs, worst_eff = [], 0.0
    min_gap_start = np.inf
    for f in files:
        summ = run_summary(runner, f)
        sp = km_splits(ad, f)
        be = best_efforts(runner, ad, f)
        ch = pace_hr_series(ad, f)
        rt = private_route(ad, f)
        if len(sp):
            n["splits"] += 1
            a = runner.activity[runner.activity["file"] == f].iloc[0]
            if pd.notna(a["granular_moving_time"]):   # splits use the same clock as Stage 1
                split_diffs.append(abs(sp["time_s"].sum() - a["granular_moving_time"]))
        if len(be):
            n["efforts"] += 1
            # display-only recomputation of a 5K must equal the Stage 1 target where it counted
            ok5 = be[(be["distance"] == "5K") & be["model_target"]]
            if len(ok5):
                c = _clock(ad.streams[f])
                recomputed = interpolate_target_time(c["km"].to_numpy(), c["moving_s"].to_numpy(), 5.0)
                worst_eff = max(worst_eff, abs(recomputed - ok5["seconds"].iloc[0]))
        if len(ch):
            n["chart"] += 1
        if rt["ok"]:
            n["map"] += 1
            g = ad.gps[f]
            pts = np.vstack(rt["segments"])
            lat0, lon0 = g["lat"].dropna().iloc[0], g["lon"].dropna().iloc[0]
            lat1, lon1 = g["lat"].dropna().iloc[-1], g["lon"].dropna().iloc[-1]
            dmin = min(_dist_m(lat0, lon0, pts[:, 0], pts[:, 1]).min(), _dist_m(lat1, lon1, pts[:, 0], pts[:, 1]).min())
            min_gap_start = min(min_gap_start, dmin)
            if dmin <= PRIVACY_RADIUS_M:
                problems.append(f"{f}: map point within {dmin:.0f} m of start/finish")
        else:
            n["no_map"][rt["reason"]] = n["no_map"].get(rt["reason"], 0) + 1
    say(f"  splits {n['splits']} | best efforts {n['efforts']} | pace/HR chart {n['chart']} | map {n['map']} | "
        f"no map {n['no_map']}")
    sd = np.array(split_diffs) if split_diffs else np.array([0.0])
    say(f"  splits add up to Stage 1 moving time: median difference {np.median(sd):.2f} s, largest {sd.max():.1f} s "
        f"(standing still after the last metre is not part of a split)")
    say(f"  best 5K recomputed here vs Stage 1 target: largest difference {worst_eff:.2e} s")
    say(f"  privacy: closest shown map point to any start/finish = {min_gap_start:.0f} m (must be > {PRIVACY_RADIUS_M})")
    say(f"  time for all runs: {time.time() - t0:.1f} s")
    # example: most recent run with a 5K best effort
    ex = runner.activity[runner.activity["target_5_0km_status"].eq("OK")]["file"].iloc[-1]
    s = run_summary(runner, ex)
    say("")
    say(f"Example run {s['date'].date()} '{s['name']}': {s['distance_km']:.2f} km, moving {s['moving_time_s'] / 60:.1f} min, "
        f"pace {s['pace_s_per_km'] / 60:.2f} min/km, GAP {s['gap_s_per_km'] / 60:.2f}, HR {s['avg_hr']:.0f}, "
        f"TRIMP {s['trimp']:.1f}")
    say(f"  zones (min): {{{', '.join(f'{k.split()[0]} {v:.1f}' for k, v in s['zone_minutes'].items())}}}")
    sp = km_splits(ad, ex)
    say("  splits: " + ", ".join(f"{int(r.km) if float(r.km).is_integer() else r.km}: {r.pace_s_per_km / 60:.2f}"
                                for r in sp.itertuples()))
    be = best_efforts(runner, ad, ex)
    say("  best efforts: " + ", ".join(f"{r.distance} {r.seconds / 60:.2f} min{' (model target, RPC %.3f)' % r.rpc if r.model_target else ''}"
                                      for r in be.itertuples()))
    rt = private_route(ad, ex)
    say(f"  map: {'shown ' + str(rt['shown_km']) + ' of ' + str(rt['total_km']) + ' km in ' + str(len(rt['segments'])) + ' segment(s)' if rt['ok'] else rt['reason']}")
    ok = not problems and worst_eff < 1e-6 and np.median(sd) < 1.0
    for p in problems[:10]:
        say("  " + p)
    say("")
    say("RESULT: " + ("ALL GOOD" if ok else "CHECK THE PROBLEMS ABOVE"))
    out = dep / "_checks"
    out.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    (out / f"step5_run_analytics_report{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved: deployment/_checks/step5_run_analytics_report{tag}.txt  (no GPS written)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
