"""
deployment/core/ingest.py  -  Stage 9, Step 2: reader / converter

Turns a runner's data into the SAME format the pipeline was built on
(GoldenCheetah OpenData), so every later step can reuse the pipeline's rules
unchanged:

    runs     one row per run, manifest columns:
             id, file, date ("YYYY/MM/DD HH:MM:SS UTC"), sport, data,
             workout_time (s), total_distance (km), elevation_gain (m),
             average_speed (km/h), average_hr (+ display extras)
    streams  per-second data per run:  secs, km, power, hr, cad, alt
             (resampled to 1 sample per second like GoldenCheetah; pauses longer
             than 30 s are left as gaps so the pipeline's moving-time rule applies)
    gps      lat/lon per run, FOR THE ROUTE MAP ONLY (never used by the model,
             never written to disk)

Three ways in:
    load_strava_export(zip)      Strava bulk export ZIP (activities.csv + FIT/GPX/TCX files)
    load_summary_table(file)     activities.csv / .xlsx only, no per-second files (summary-only)
    load_goldencheetah(id, ...)  one GoldenCheetah athlete (for the Stage 3/7 match check)

Run this file directly to check it on the demo export:
    py "deployment\\core\\ingest.py"            (all runs)
    py "deployment\\core\\ingest.py" --quick    (first 20 runs)
"""
from __future__ import annotations

import gzip
import io
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

# Must match the pipeline
REQUIRED_SIGNATURE = "TDS-HC-AGL----R"   # GoldenCheetah data signature used by Stage 1
MAX_SAMPLE_GAP_SEC = 30                  # 03_stage1_data_preparation.py / 05_stage3_feature_engineering.py
GRANULAR_COLUMNS = ["secs", "km", "power", "hr", "cad", "alt"]

# Strava activity types treated as runs. Virtual Run (treadmill) is left out:
# no GPS distance or altitude, unlike the outdoor runs the model was trained on.
RUN_TYPES = {"Run", "Trail Run"}

SEMICIRCLE_TO_DEG = 180.0 / 2 ** 31
EARTH_RADIUS_M = 6_371_008.8
MAX_UNCOMPRESSED_BYTES = 4 * 1024 ** 3   # refuse absurd archives (zip bombs)


@dataclass
class AthleteData:
    athlete_id: str
    source: str                                   # strava_export | summary_only | goldencheetah
    runs: pd.DataFrame
    streams: dict[str, pd.DataFrame] = field(default_factory=dict)
    gps: dict[str, pd.DataFrame] = field(default_factory=dict)
    report: dict = field(default_factory=dict)


# ============================================================
# Per-second helpers
# ============================================================

def _interp_column(t: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Linear interpolation of one column onto grid t, only between valid samples
    that are at most MAX_SAMPLE_GAP_SEC apart (no values invented across pauses)."""
    ok = np.isfinite(v)
    if ok.sum() == 0:
        return np.full(len(t), np.nan)
    sv, vv = s[ok], v[ok]
    if len(sv) == 1:
        out = np.full(len(t), np.nan)
        out[t == sv[0]] = vv[0]
        return out
    out = np.interp(t, sv, vv)
    i = np.clip(np.searchsorted(sv, t, side="right") - 1, 0, len(sv) - 2)
    gap = sv[i + 1] - sv[i]
    exact = (t == sv[i]) | (t == sv[i + 1])
    out[(gap > MAX_SAMPLE_GAP_SEC) & ~exact] = np.nan
    out[(t < sv[0]) | (t > sv[-1])] = np.nan
    return out


def to_one_hz(df: pd.DataFrame) -> pd.DataFrame:
    """Resample a recording (secs + data columns) to 1 sample per second.
    Grid seconds inside a gap longer than 30 s are not created, so a pause stays a
    gap and the pipeline's moving-time clock handles it exactly as for GoldenCheetah."""
    df = df.dropna(subset=["secs"]).sort_values("secs").drop_duplicates("secs")
    if len(df) < 2:
        return pd.DataFrame(columns=GRANULAR_COLUMNS)
    s = df["secs"].to_numpy(float)
    grid = np.arange(np.floor(s[0]), np.floor(s[-1]) + 1)
    i = np.clip(np.searchsorted(s, grid, side="right") - 1, 0, len(s) - 2)
    keep = ((s[i + 1] - s[i]) <= MAX_SAMPLE_GAP_SEC) | np.isin(grid, s)
    t = np.union1d(grid[keep], s)               # always keep the original sample times
    out = pd.DataFrame({"secs": t - t[0]})
    for col in GRANULAR_COLUMNS[1:]:
        v = df[col].to_numpy(float) if col in df else np.full(len(s), np.nan)
        out[col] = _interp_column(t, s, v)
    return out[GRANULAR_COLUMNS]


def _haversine_cum_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat, lon = np.radians(lat), np.radians(lon)
    dlat, dlon = np.diff(lat), np.diff(lon)
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(dlon / 2) ** 2
    d = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    d = np.where(np.isfinite(d), d, 0.0)
    return np.concatenate([[0.0], np.cumsum(d)]) / 1000.0


def _finish(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """raw: secs, dist_m (may be NaN), hr, cad, power, alt, lat, lon -> (stream, gps)."""
    raw = raw.dropna(subset=["secs"]).sort_values("secs").drop_duplicates("secs").reset_index(drop=True)
    has_gps = raw[["lat", "lon"]].notna().all(axis=1)
    if raw["dist_m"].notna().sum() >= 2:
        raw["km"] = raw["dist_m"] / 1000.0
    elif has_gps.sum() >= 2:                     # no distance field: derive from GPS
        g = raw[has_gps]
        raw["km"] = np.nan
        raw.loc[has_gps, "km"] = _haversine_cum_km(g["lat"].to_numpy(), g["lon"].to_numpy())
    else:
        raw["km"] = np.nan
    from_gps = raw["dist_m"].notna().sum() < 2 and raw["km"].notna().sum() >= 2
    stream = to_one_hz(raw)
    stream.attrs["distance_from_gps"] = bool(from_gps)
    gps = raw.loc[has_gps, ["secs", "lat", "lon"]].copy()
    gps["secs"] = gps["secs"] - raw["secs"].iloc[0]
    return stream, gps.reset_index(drop=True)


# ============================================================
# File parsers (FIT / GPX / TCX, optionally .gz)
# ============================================================

def parse_fit(data: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    import fitparse
    fit = fitparse.FitFile(data)
    rows = []
    for msg in fit.get_messages("record"):
        v = msg.get_values()
        ts = v.get("timestamp")
        if ts is None:
            continue
        lat, lon = v.get("position_lat"), v.get("position_long")
        alt = v.get("enhanced_altitude", v.get("altitude"))
        cad = v.get("cadence")
        if cad is not None and v.get("fractional_cadence") is not None:
            cad = cad + v["fractional_cadence"]
        rows.append({
            "ts": ts,
            "dist_m": v.get("distance"),
            "hr": v.get("heart_rate"),
            "cad": cad,
            "power": v.get("power"),
            "alt": alt,
            "lat": lat * SEMICIRCLE_TO_DEG if lat is not None else None,
            "lon": lon * SEMICIRCLE_TO_DEG if lon is not None else None,
        })
    if not rows:
        raise ValueError("no record messages")
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df.pop("ts"), utc=True)
    df.insert(0, "secs", (ts - ts.iloc[0]).dt.total_seconds())
    df = df.apply(pd.to_numeric, errors="coerce")
    return _finish(df)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_gpx(data: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    import gpxpy
    gpx = gpxpy.parse(data.decode("utf-8", errors="replace").lstrip())
    rows = []
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                if p.time is None:
                    continue
                hr = cad = power = None
                for ext in p.extensions:
                    for el in ext.iter():
                        name = _local(el.tag).lower()
                        if name == "hr":
                            hr = el.text
                        elif name in ("cad", "cadence"):
                            cad = el.text
                        elif name == "power":
                            power = el.text
                rows.append({"ts": p.time, "lat": p.latitude, "lon": p.longitude,
                             "alt": p.elevation, "hr": hr, "cad": cad, "power": power})
    if not rows:
        raise ValueError("no timed track points")
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df.pop("ts"), utc=True)
    df.insert(0, "secs", (ts - ts.iloc[0]).dt.total_seconds())
    df["dist_m"] = np.nan
    df = df.apply(pd.to_numeric, errors="coerce")
    return _finish(df)


def parse_tcx(data: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = ET.fromstring(data.decode("utf-8", errors="replace").strip())
    rows = []
    for tp in root.iter():
        if _local(tp.tag) != "Trackpoint":
            continue
        r = {"ts": None, "dist_m": None, "alt": None, "hr": None, "cad": None,
             "power": None, "lat": None, "lon": None}
        for el in tp.iter():
            n = _local(el.tag)
            if n == "Time":
                r["ts"] = el.text
            elif n == "DistanceMeters":
                r["dist_m"] = el.text
            elif n == "AltitudeMeters":
                r["alt"] = el.text
            elif n == "HeartRateBpm":
                r["hr"] = next((c.text for c in el if _local(c.tag) == "Value"), None)
            elif n in ("Cadence", "RunCadence"):
                r["cad"] = el.text
            elif n == "Watts":
                r["power"] = el.text
            elif n == "LatitudeDegrees":
                r["lat"] = el.text
            elif n == "LongitudeDegrees":
                r["lon"] = el.text
        if r["ts"]:
            rows.append(r)
    if not rows:
        raise ValueError("no trackpoints")
    df = pd.DataFrame(rows)
    ts = pd.to_datetime(df.pop("ts"), utc=True, format="ISO8601")
    df.insert(0, "secs", (ts - ts.iloc[0]).dt.total_seconds())
    df = df.apply(pd.to_numeric, errors="coerce")
    return _finish(df)


def parse_activity_file(name: str, data: bytes) -> tuple[pd.DataFrame, pd.DataFrame]:
    low = name.lower()
    if low.endswith(".gz"):
        data = gzip.decompress(data)
        low = low[:-3]
    if low.endswith(".fit"):
        return parse_fit(data)
    if low.endswith(".gpx"):
        return parse_gpx(data)
    if low.endswith(".tcx"):
        return parse_tcx(data)
    raise ValueError(f"unsupported file type: {name}")


# ============================================================
# activities.csv  ->  manifest rows
# ============================================================

def _num(series: pd.Series) -> pd.Series:
    if series.dtype == object or str(series.dtype).startswith("str"):
        series = series.astype(str).str.replace(",", "", regex=False)
    return pd.to_numeric(series, errors="coerce")


def _first(df: pd.DataFrame, *names: str) -> pd.Series:
    for n in names:
        if n in df.columns:
            return df[n]
    return pd.Series(np.nan, index=df.index)


def activities_to_manifest(acts: pd.DataFrame, athlete_id: str) -> tuple[pd.DataFrame, dict]:
    """Strava activities.csv (export format) -> manifest-format run rows."""
    if "Activity Type" not in acts.columns or "Activity Date" not in acts.columns:
        raise ValueError("This does not look like a Strava activities file "
                         "(missing 'Activity Type' / 'Activity Date').")
    types = acts["Activity Type"].astype(str).str.strip()
    is_run = types.isin(RUN_TYPES)
    info = {"activities_total": int(len(acts)),
            "activity_types": types.value_counts().to_dict(),
            "runs": int(is_run.sum()),
            "virtual_runs_excluded": int((types == "Virtual Run").sum())}
    a = acts.loc[is_run].copy()

    dt = pd.to_datetime(a["Activity Date"], format="%b %d, %Y, %I:%M:%S %p", errors="coerce", utc=True)
    if dt.isna().any():   # other export locales / formats
        dt = dt.fillna(pd.to_datetime(a["Activity Date"], format="mixed", errors="coerce", utc=True))
    info["bad_dates_dropped"] = int(dt.isna().sum())

    # Strava repeats column names: the second "Distance" is metres, the first is km
    dist_km = _num(_first(a, "Distance.1")) / 1000.0
    dist_km = dist_km.fillna(_num(_first(a, "Distance")))
    moving = _num(_first(a, "Moving Time")).fillna(_num(_first(a, "Elapsed Time")))
    elapsed = _num(_first(a, "Elapsed Time.1", "Elapsed Time"))
    avg_speed_ms = _num(_first(a, "Average Speed"))
    avg_speed_kmh = (avg_speed_ms * 3.6).where(avg_speed_ms.notna(), dist_km / (moving / 3600))

    runs = pd.DataFrame({
        "id": athlete_id,
        "file": _first(a, "Filename").fillna("").astype(str).str.strip(),
        "date": dt.dt.strftime("%Y/%m/%d %H:%M:%S UTC"),
        "sport": "Run",
        "data": "summary",                       # set to REQUIRED_SIGNATURE when a stream is read
        "workout_time": moving,
        "total_distance": dist_km,
        "elevation_gain": _num(_first(a, "Elevation Gain")),
        "average_speed": avg_speed_kmh,
        "average_hr": _num(_first(a, "Average Heart Rate")),
        "average_power": _num(_first(a, "Average Watts")),
        # display extras (not used by the pipeline rules)
        "activity_id": _first(a, "Activity ID").astype(str),
        "activity_name": _first(a, "Activity Name").fillna("").astype(str),
        "strava_type": types[is_run],
        "elapsed_time": elapsed,
        "max_hr": _num(_first(a, "Max Heart Rate.1", "Max Heart Rate")),
        "datetime_utc": dt,
    })
    runs = runs[runs["datetime_utc"].notna()]
    # A run without a file still needs a unique key
    nofile = runs["file"].eq("") | runs["file"].eq("nan")
    runs.loc[nofile, "file"] = "nofile_" + runs.loc[nofile, "activity_id"]
    runs = runs.sort_values("datetime_utc").drop_duplicates("file").reset_index(drop=True)
    for col in ["has_stream", "has_hr", "has_gps", "distance_scaled_to_strava"]:
        runs[col] = False
    runs["stream_format"] = ""
    runs["stream_error"] = ""
    return runs, info


# ============================================================
# Public loaders
# ============================================================

def _read_table(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data), low_memory=False)


def load_summary_table(source, athlete_id: str = "user") -> AthleteData:
    """activities.csv or .xlsx only (no per-second files): summary-only mode."""
    if isinstance(source, (str, Path)):
        name, data = str(source), Path(source).read_bytes()
    else:
        name, data = getattr(source, "name", "activities.csv"), source.read()
    runs, info = activities_to_manifest(_read_table(name, data), athlete_id)
    info.update(_coverage(runs))
    return AthleteData(athlete_id, "summary_only", runs, {}, {}, info)


def load_strava_export(source, athlete_id: str = "user", limit_runs: int | None = None,
                       progress=None) -> AthleteData:
    """Strava bulk export ZIP (path, bytes or file-like). progress(i, n) is optional."""
    t0 = time.time()
    if isinstance(source, (str, Path)):
        zf = zipfile.ZipFile(source)
    elif isinstance(source, (bytes, bytearray)):
        zf = zipfile.ZipFile(io.BytesIO(source))
    else:
        zf = zipfile.ZipFile(source)
    with zf:
        if sum(i.file_size for i in zf.infolist()) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("Archive is too large once unpacked.")
        names = [n for n in zf.namelist() if not n.endswith("/")]
        csvs = [n for n in names if PurePosixPath(n).name == "activities.csv"]
        if not csvs:
            raise ValueError("No activities.csv found. Upload the ZIP from Strava's "
                             "'Download or Delete Your Account' > 'Request your archive'.")
        csv_name = min(csvs, key=len)               # top-most activities.csv
        base = str(PurePosixPath(csv_name).parent)
        base = "" if base == "." else base + "/"
        runs, info = activities_to_manifest(pd.read_csv(io.BytesIO(zf.read(csv_name)),
                                                        low_memory=False), athlete_id)
        if limit_runs:
            runs = runs.tail(limit_runs).reset_index(drop=True)
        lookup = {n.lower(): n for n in names}
        streams, gps = {}, {}
        failures = {}
        for i, row in runs.iterrows():
            if progress:
                progress(i, len(runs))
            if row["file"].startswith("nofile_"):
                runs.at[i, "stream_error"] = "no file listed"
                continue
            member = lookup.get((base + row["file"]).lower()) or lookup.get(row["file"].lower())
            if member is None:
                runs.at[i, "stream_error"] = "file missing from ZIP"
                failures["file missing from ZIP"] = failures.get("file missing from ZIP", 0) + 1
                continue
            fmt = re.sub(r"\.gz$", "", member.lower()).rsplit(".", 1)[-1]
            runs.at[i, "stream_format"] = fmt
            try:
                stream, route = parse_activity_file(member, zf.read(member))
            except Exception as e:  # corrupt / unsupported file: keep the run as summary-only
                reason = f"{fmt}: {type(e).__name__}"
                runs.at[i, "stream_error"] = reason
                failures[reason] = failures.get(reason, 0) + 1
                continue
            if stream["km"].notna().sum() < 2:
                runs.at[i, "stream_error"] = "no distance in file"
                failures["no distance in file"] = failures.get("no distance in file", 0) + 1
                continue
            # Distance measured from GPS points (e.g. GPX) runs slightly long because of
            # GPS jitter; scale it so the run's total equals Strava's distance.
            total = row["total_distance"]
            if stream.attrs.get("distance_from_gps") and np.isfinite(total) and total > 0:
                stream["km"] = stream["km"] * (total / stream["km"].max())
                runs.at[i, "distance_scaled_to_strava"] = True
            streams[row["file"]] = stream
            if len(route) >= 2:
                gps[row["file"]] = route
            runs.at[i, "has_stream"] = True
            runs.at[i, "data"] = REQUIRED_SIGNATURE
            runs.at[i, "has_hr"] = bool(stream["hr"].between(30, 230).sum() >= 60)
            runs.at[i, "has_gps"] = len(route) >= 2
    info["stream_failures"] = failures
    info["seconds_to_read"] = round(time.time() - t0, 1)
    info.update(_coverage(runs))
    return AthleteData(athlete_id, "strava_export", runs, streams, gps, info)


def load_goldencheetah(athlete_id: str, manifest_path, granular_root) -> AthleteData:
    """One GoldenCheetah athlete, read exactly as Stage 1 reads it (for the match check).
    'data' keeps its original signature so the Stage 1 filter can be applied unchanged."""
    m = pd.read_csv(manifest_path, low_memory=False)
    m = m[m["id"].astype(str).str.strip() == athlete_id].copy()
    m["sport"] = m["sport"].astype(str).str.strip()
    m["data"] = m["data"].astype(str).str.strip()
    m["file"] = m["file"].astype(str).str.strip()
    m["datetime_utc"] = pd.to_datetime(m["date"], errors="coerce", utc=True)
    streams = {}
    for f in m["file"]:
        if not isinstance(f, str) or not f:
            continue
        p = Path(granular_root) / athlete_id / f
        if p.exists():
            streams[f] = pd.read_csv(p)
    m["has_stream"] = m["file"].isin(streams)
    m["has_gps"] = False
    m["has_hr"] = [bool(f in streams and pd.to_numeric(streams[f]["hr"], errors="coerce")
                        .between(30, 230).sum() >= 60) for f in m["file"]]
    info = {"activities_total": int(len(m)), "runs": int(m["sport"].eq("Run").sum())}
    info.update(_coverage(m))
    return AthleteData(athlete_id, "goldencheetah", m.reset_index(drop=True), streams, {}, info)


def _coverage(runs: pd.DataFrame) -> dict:
    if runs.empty:
        return {"runs_kept": 0}
    d = runs["datetime_utc"].sort_values()
    gaps = d.diff().dt.days
    return {
        "runs_kept": int(len(runs)),
        "runs_with_stream": int(runs["has_stream"].sum()),
        "runs_with_hr": int(runs["has_hr"].sum()),
        "runs_with_gps": int(runs["has_gps"].sum()),
        "runs_distance_scaled_from_gps": int(runs.get("distance_scaled_to_strava", pd.Series(dtype=bool)).sum()),
        "runs_5k_plus": int((runs["total_distance"] >= 5).sum()),
        "first_run": str(d.iloc[0].date()),
        "last_run": str(d.iloc[-1].date()),
        "longest_gap_days": int(gaps.max()) if gaps.notna().any() else 0,
    }


# ============================================================
# Self-check (run this file directly)
# ============================================================

def _self_check(argv=None) -> int:
    import argparse
    import json
    here = Path(__file__).resolve().parent.parent          # deployment/
    ap = argparse.ArgumentParser()
    ap.add_argument("zip", nargs="?", default=str(here / "demo_data" / "justinhojt_strava_export.zip"))
    ap.add_argument("--quick", action="store_true", help="only the 20 most recent runs")
    args = ap.parse_args(argv)

    print(f"Reading {args.zip}")
    ad = load_strava_export(args.zip, limit_runs=20 if args.quick else None,
                            progress=lambda i, n: print(f"  run {i + 1}/{n}", end="\r"))
    print()
    r = ad.runs
    lines = ["Stage 9 Step 2 - ingest check", "=" * 60,
             json.dumps(ad.report, indent=2, default=str), ""]
    problems = []
    for f, s in ad.streams.items():
        if list(s.columns) != GRANULAR_COLUMNS:
            problems.append(f"{f}: wrong columns")
        d = s["secs"].diff().dropna()
        if (d <= 0).any():
            problems.append(f"{f}: time not increasing")
        if not ((d == 1) | (d > MAX_SAMPLE_GAP_SEC)).all():
            problems.append(f"{f}: not 1 Hz outside pauses")
    # stream distance vs activities.csv distance (Stage 1 rejects > 5%)
    km_file = r["file"].map(lambda f: ad.streams[f]["km"].max() if f in ad.streams else np.nan)
    rel = (km_file - r["total_distance"]).abs() / r["total_distance"]
    lines += [f"Streams checked: {len(ad.streams)}  problems: {len(problems)}",
              *[f"  {p}" for p in problems[:20]],
              f"Distance file vs activities.csv: median diff {np.nanmedian(rel) * 100:.2f}%, "
              f"runs > 5% (Stage 1 would reject): {int((rel > 0.05).sum())}",
              f"Samples per run: median {int(np.median([len(s) for s in ad.streams.values()])) if ad.streams else 0}",
              "", "Most recent runs:",
              r[["date", "total_distance", "workout_time", "average_hr", "stream_format",
                 "has_stream", "has_hr", "has_gps", "stream_error"]].tail(8).to_string(index=False)]
    ok = not problems and len(ad.streams) > 0
    lines += ["", "RESULT: " + ("ALL GOOD" if ok else "CHECK THE PROBLEMS ABOVE")]
    text = "\n".join(lines)
    print(text)
    out = here / "_checks"
    out.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    (out / f"step2_ingest_report{tag}.txt").write_text(text + "\n", encoding="utf-8")
    r.drop(columns=["datetime_utc"]).to_csv(out / f"step2_runs{tag}.csv", index=False)
    print(f"\nSaved: deployment/_checks/step2_ingest_report{tag}.txt and step2_runs{tag}.csv")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
