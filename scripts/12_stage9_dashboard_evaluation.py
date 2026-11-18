"""
12_stage9_dashboard_evaluation.py  -  Stage 9, Step 7: technical evaluation of the RunState dashboard

Evidence for Research Objective 3 (design, develop and technically evaluate the dashboard prototype).
Runs the dashboard's own code (deployment/core) without the web page and checks five things:

    A. Accuracy     every Stage 7 test runner through the dashboard path (raw files -> reader ->
                    Stages 1-3 for one runner -> deployed XGBoost) reproduces the Stage 7 test
                    predictions; MAE and likely-range coverage on the test set
    B. Edge cases   no heart rate, short history, broken / missing activity files, ZIP without
                    activities.csv, wrong table, treadmill runs, activity list only (Kaggle),
                    stale data
    C. ACWR alert   zone boundaries 0.8 / 1.3 / 1.5, the "at least 4 run days in 28" rule,
                    distance ACWR = 7-day / 28-day mean distance, both demos
    D. Privacy      with "Hide" on, no route point or marker position sent to the browser lies within
                    500 m of a run's start or finish; coordinate columns dropped from activity lists
    E. Speed        time to load each demo, build the estimate, open a run, move the slider panel

Nothing is trained or tuned here; the models are the frozen Stage 6/7 models in deployment/models.

Run from the project folder:
    py scripts\\12_stage9_dashboard_evaluation.py            (all 216 test runners, ~10-15 min)
    py scripts\\12_stage9_dashboard_evaluation.py --quick    (10 test runners, ~2 min)

Outputs (09_dashboard_evaluation/):
    stage9_evaluation_report.txt      summary for the paper (Chapter 4 / 5)
    stage9_A_stage7_match.csv         one row per test sample: Stage 7 vs dashboard prediction
    stage9_B_edge_cases.csv           one row per edge case: expected vs observed, pass/fail
    stage9_C_acwr_checks.csv          zone boundary and rule checks
    stage9_D_privacy_by_run.csv       closest shown point to start/finish, per run
    stage9_E_speed.csv                timings (seconds)
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / "deployment"
sys.path.insert(0, str(DEP))

from core import ingest as IN            # noqa: E402
from core import features as FE          # noqa: E402
from core import predict as PR           # noqa: E402
from core import run_analytics as RA     # noqa: E402
from core import run_view as RV          # noqa: E402

DEMO_ZIP = DEP / "demo_data" / "justinhojt_strava_export.zip"
KAGGLE_DIR = DEP / "demo_data" / "kaggle"
OUT = ROOT / "09_dashboard_evaluation"
STALE_AFTER_DAYS = 28          # same setting as deployment/app.py
TODAY = "2026-11-17"           # fixed so the report does not change from day to day
PRIVACY_M = RA.PRIVACY_RADIUS_M

LINES: list[str] = []


def say(s: str = "") -> None:
    print(s)
    LINES.append(s)


def mark(ok: bool) -> str:
    return "OK " if ok else "BAD"


def timed(fn, *a, **k):
    t0 = time.perf_counter()
    v = fn(*a, **k)
    return v, time.perf_counter() - t0


# ============================================================
# A. Accuracy: Stage 7 reproduced through the dashboard path
# ============================================================

def part_a(model: PR.DeployedModel, quick: bool) -> bool:
    ref = pd.read_csv(ROOT / "07_test_evaluation" / "stage7_test_predictions.csv")
    split = pd.read_csv(ROOT / "04_feature_selection" / "stage4_split_assignment.csv")
    manifest = ROOT / "manifest file" / "run_activities_manifest.csv"
    ids = sorted(ref["id"].astype(str).unique())
    if quick:
        ids = ids[:10]
    say(f"A. ACCURACY - {len(ids)} test runners through the dashboard path "
        f"(raw files -> reader -> Stages 1-3 for one runner -> deployed model)")
    t0 = time.time()
    got = []
    for i, a in enumerate(ids, 1):
        print(f"  runner {i}/{len(ids)}", end="\r")
        r = FE.build_runner(IN.load_goldencheetah(a, manifest, ROOT / "00_raw" / "granular"))
        s = r.samples.merge(split[["sample_id", "split"]], on="sample_id", how="inner")
        s = s[s["split"] == "test"]
        got.append(pd.DataFrame({"sample_id": s["sample_id"], "dashboard": model.predict(s)}))
    print()
    got = pd.concat(got, ignore_index=True)
    ref = ref[ref["id"].astype(str).isin(ids)]
    m = ref[["sample_id", "id", "rpc", "tuned_xgb_full"]].merge(got, on="sample_id", how="outer")
    missing, extra = int(m["dashboard"].isna().sum()), int(m["tuned_xgb_full"].isna().sum())
    diff = (m["dashboard"] - m["tuned_xgb_full"]).abs()
    ok = missing == 0 and extra == 0 and float(diff.max()) < 1e-6
    w = 1 / m.groupby("id")["sample_id"].transform("size")
    mae = float(np.average((m["dashboard"] - m["rpc"]).abs(), weights=w))
    inside = (m["rpc"] >= m["dashboard"] * model.range["lo"]) & (m["rpc"] <= m["dashboard"] * model.range["hi"])
    cov_r = float(np.average(inside, weights=w))
    sec5 = float(np.average(np.abs(FE.time_from_rpc(m["dashboard"], 5.0) - FE.time_from_rpc(m["rpc"], 5.0)),
                            weights=w))
    m.assign(abs_difference=diff).to_csv(OUT / "stage9_A_stage7_match.csv", index=False)
    say(f"  {mark(ok)} test samples {len(ref):,} | missing {missing} | extra {extra} | "
        f"largest difference {diff.max():.2e} RPC")
    say(f"  dashboard MAE {mae:.4f} RPC (each runner equal; Stage 7 reported 0.0380 on all 216 runners) "
        f"(average 5K time error about {sec5:.0f} s)")
    say(f"  likely range (80%) covers {cov_r:.1%} of test best efforts (each runner equal) | {time.time() - t0:.0f} s")
    return ok


# ============================================================
# B. Edge cases
# ============================================================

def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, b in files.items():
            z.writestr(n, b)
    return buf.getvalue()


def _demo_members() -> dict[str, bytes]:
    with zipfile.ZipFile(DEMO_ZIP) as z:
        return {n: z.read(n) for n in z.namelist() if not n.endswith("/")}


def _rows(csv_bytes: bytes):
    rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
    return rows[0], rows[1:]


def _csv_bytes(header, rows) -> bytes:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")


class _Upload(io.BytesIO):
    """Stands in for a Streamlit upload (bytes with a file name)."""
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def part_b(model: PR.DeployedModel, demo_ad: IN.AthleteData, demo_r: FE.Runner) -> bool:
    say("B. EDGE CASES")
    rows = []

    def case(name, expected, fn):
        try:
            observed, ok = fn()
        except Exception as e:                                   # noqa: BLE001
            observed, ok = f"unhandled error {type(e).__name__}: {e}", False
        rows.append({"case": name, "expected": expected, "observed": observed, "pass": ok})
        say(f"  {mark(ok)} {name}: {observed}")

    members = _demo_members()
    header, body = _rows(members["activities.csv"])
    col = {h: i for i, h in enumerate(header)}

    # 1. no heart rate anywhere
    def no_hr():
        ad = copy.deepcopy(demo_ad)
        for f, s in ad.streams.items():
            s["hr"] = np.nan
        ad.runs["average_hr"] = np.nan
        ad.runs["max_hr"] = np.nan
        ad.runs["has_hr"] = False
        r = FE.build_runner(ad)
        now = PR.you_now(model, r, today=TODAY)
        ld = now["load"]
        ok = (r.runs["trimp"].isna().all() and now["ok"] and pd.isna(ld["acwr_hr"])
              and pd.notna(ld["acwr_km"]) and not bool(ld["has_hr_load"]))
        return (f"estimate shown {now['ok']} (5K {PR.fmt_time(FE.time_from_rpc(now['rpc'], 5.0))}); "
                f"HR training load missing on {int(r.runs['trimp'].isna().sum())}/{len(r.runs)} runs; "
                f"distance ACWR {ld['acwr_km']:.2f} still shown; HR ACWR not shown"), ok
    case("No heart-rate data", "estimate still shown; HR load and HR ACWR marked unavailable; distance ACWR shown",
         no_hr)

    # 2. short history (first 40 days of the demo runner)
    def short():
        ad = copy.deepcopy(demo_ad)
        dt = pd.to_datetime(ad.runs["datetime_utc"])
        ad.runs = ad.runs[dt <= dt.min() + pd.Timedelta(days=40)].reset_index(drop=True)
        r = FE.build_runner(ad)
        now = PR.you_now(model, r, today=TODAY)
        tr = PR.weekly_trend(model, r)
        ok = (not now["ok"]) and "Needs at least" in now.get("reason", "") and len(tr) == 0
        return f"{len(ad.runs)} runs over 40 days -> no estimate, message: \"{now.get('reason', '')}\"", ok
    case("Short history (< 55 days)", "no estimate; message says how much history is needed", short)

    # 3. broken and missing activity files inside the ZIP
    def broken():
        files = [r[col["Filename"]] for r in body if r[col["Activity Type"]] == "Run" and r[col["Filename"]]]
        bad, gone = files[-1], files[-2]
        m = dict(members)
        m[bad] = b"this is not a FIT file"
        m.pop(gone)
        ad = IN.load_strava_export(_Upload(_zip_bytes(m), "export.zip"))
        r = FE.build_runner(ad)
        now = PR.you_now(model, r, today=TODAY)
        err = ad.runs.set_index("file")["stream_error"]
        ok = (len(ad.runs) == len(demo_ad.runs) and err.get(bad, "") != "" and err.get(gone, "") ==
              "file missing from ZIP" and now["ok"])
        return (f"all {len(ad.runs)} runs kept; corrupt file -> \"{err.get(bad)}\", missing file -> "
                f"\"{err.get(gone)}\" (both kept as summary-only); estimate still shown"), ok
    case("Corrupt file and file missing from ZIP", "runs kept as summary-only with a reason; app keeps working",
         broken)

    # 4. ZIP without activities.csv
    def no_csv():
        m = {k: v for k, v in members.items() if k != "activities.csv"}
        try:
            IN.load_strava_export(_Upload(_zip_bytes(m), "export.zip"))
        except ValueError as e:
            return f"friendly error: \"{e}\"", "activities.csv" in str(e)
        return "loaded without an error", False
    case("ZIP without activities.csv", "clear error message, no crash", no_csv)

    # 5. a table that is not a Strava activity list
    def wrong_table():
        data = b"name,score\nalice,3\nbob,5\n"
        try:
            IN.load_summary_table(_Upload(data, "scores.csv"))
        except ValueError as e:
            return f"friendly error: \"{str(e)[:110]}\"", True
        return "loaded without an error", False
    case("Wrong table uploaded", "clear error message, no crash", wrong_table)

    # 6. treadmill (Virtual Run) excluded
    def treadmill():
        src = next(r for r in body if r[col["Activity Type"]] == "Run")
        extra = list(src)
        extra[col["Activity ID"]] = "999999999999"
        extra[col["Activity Type"]] = "Virtual Run"
        extra[col["Filename"]] = ""
        m = dict(members)
        m["activities.csv"] = _csv_bytes(header, body + [extra])
        ad = IN.load_strava_export(_Upload(_zip_bytes(m), "export.zip"))
        ok = ad.report["virtual_runs_excluded"] == 1 and len(ad.runs) == len(demo_ad.runs)
        return (f"{ad.report['virtual_runs_excluded']} Virtual Run excluded; runs {len(ad.runs)} "
                f"(= demo {len(demo_ad.runs)})"), ok
    case("Treadmill run (Virtual Run)", "not counted as a run", treadmill)

    # 7. activity list only (Kaggle demo)
    kag = sorted(KAGGLE_DIR.glob("*.xls*")) + sorted(KAGGLE_DIR.glob("*.csv"))

    def summary_only():
        ka = IN.load_summary_table(kag[0])
        kr = FE.build_runner(ka)
        now = PR.you_now(model, kr, today=TODAY)
        ld = now["load"]
        ok = (not now["ok"]) and now.get("mode") == "summary" and len(PR.weekly_trend(model, kr)) == 0 \
            and pd.notna(ld["acwr_km"])
        return (f"{len(ka.runs)} runs, no per-second files -> no estimate (message shown); distance ACWR "
                f"{ld['acwr_km']:.2f} ({ld['zone']}); weekly load trend {len(PR.load_trend(kr))} weeks"), ok
    if kag:
        case("Activity list only (Kaggle demo)", "no estimate; training load and ACWR alert shown", summary_only)

    # 8. stale data
    def stale():
        later = (demo_r.last_day + pd.Timedelta(days=40)).date().isoformat()
        now = PR.you_now(model, demo_r, today=later)
        ok = now["days_since_last_run"] > STALE_AFTER_DAYS and now["date"] == demo_r.you_now_date
        return (f"opened {now['days_since_last_run']} days after the last run -> stale warning "
                f"(> {STALE_AFTER_DAYS} days); estimate stays dated {now['date'].date()}"), ok
    case("Stale data (last run 40 days ago)", "warning shown; estimate keeps its own date", stale)

    pd.DataFrame(rows).to_csv(OUT / "stage9_B_edge_cases.csv", index=False)
    n_ok = sum(r["pass"] for r in rows)
    say(f"  {n_ok}/{len(rows)} edge cases behave as designed")
    return n_ok == len(rows)


# ============================================================
# C. ACWR alert logic
# ============================================================

def part_c(runners: dict[str, FE.Runner]) -> bool:
    say("C. ACWR ALERT LOGIC")
    rows = []
    vals = pd.Series([0.0, 0.79, 0.8, 1.0, 1.29, 1.3, 1.49, 1.5, 2.1, np.nan])
    want = [0, 0, 1, 1, 1, 2, 2, 3, 3, np.nan]
    got = FE.acwr_zone_code(vals)
    for v, w_, g in zip(vals, want, got):
        ok = (pd.isna(w_) and pd.isna(g)) or w_ == g
        rows.append({"check": f"zone of ACWR {v}", "expected": PR.ZONE_NAMES.get(w_, "none"),
                     "observed": PR.ZONE_NAMES.get(g, "none") if pd.notna(g) else "none", "pass": bool(ok)})
    zone_ok = all(r["pass"] for r in rows)
    say(f"  {mark(zone_ok)} zone boundaries: < 0.8 Under-training | 0.8-1.3 Sweet spot | 1.3-1.5 Monitoring | "
        f">= 1.5 Danger ({len(rows)} values incl. the exact boundaries)")
    all_ok = zone_ok
    for name, r in runners.items():
        d = r.daily
        few = d["run_days_28"] < FE.MIN_CHRONIC_RUN_DAYS
        rule = bool(d.loc[few, "acwr_km"].isna().all())
        ok_d = d["acwr_km"].notna() & d["ctl_km"].gt(0)
        recomputed = (d["atl_km"] / d["ctl_km"])[ok_d]
        formula = bool(np.allclose(recomputed, d.loc[ok_d, "acwr_km"]))
        zones = d["acwr_km_zone"].map(PR.ZONE_NAMES).value_counts().to_dict()
        n_valid = int(d["acwr_km"].notna().sum())
        rows += [{"check": f"{name}: no ACWR when < 4 run days in 28", "expected": "ACWR missing",
                  "observed": f"{int(few.sum())} such days, ACWR missing on all: {rule}", "pass": rule},
                 {"check": f"{name}: distance ACWR = 7-day / 28-day mean km", "expected": "equal",
                  "observed": f"{n_valid} days recomputed, equal: {formula}", "pass": formula}]
        all_ok &= rule and formula
        say(f"  {mark(rule and formula)} {name}: {n_valid} days with distance ACWR, {int(few.sum())} days with "
            f"< 4 run days -> no ACWR; zones {zones}")
    pd.DataFrame(rows).to_csv(OUT / "stage9_C_acwr_checks.csv", index=False)
    return all_ok


# ============================================================
# D. Privacy
# ============================================================

def part_d(ad: IN.AthleteData) -> bool:
    say("D. PRIVACY (route map)")
    rows = []
    for f in ad.runs["file"]:
        g = RA._clean_gps(ad.gps.get(f))
        if g is None:
            continue
        lat0, lon0, lat1, lon1 = g["lat"].iloc[0], g["lon"].iloc[0], g["lat"].iloc[-1], g["lon"].iloc[-1]
        row = {"file": f, "gps_points": len(g)}
        for label, priv in (("hide", True), ("full", False)):
            steps = RA.scrub_steps(ad, f, priv)
            if not len(steps):
                row[f"{label}_map"] = False
                continue
            p = RV.build_payload(steps, RA.route_track(ad, f, priv), RA.hidden_ranges(steps))
            pts = [pt for r_ in p["route"] for seg in r_["s"] for pt in seg]
            pts += [(s["la"], s["lo"]) for s in p["steps"] if s["la"] is not None and not s["x"]]
            row[f"{label}_map"] = bool(p["has_map"])
            row[f"{label}_points_sent"] = len(pts)
            if pts:
                P = np.array(pts, float)
                row[f"{label}_closest_m"] = float(min(RA._dist_m(lat0, lon0, P[:, 0], P[:, 1]).min(),
                                                      RA._dist_m(lat1, lon1, P[:, 0], P[:, 1]).min()))
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "stage9_D_privacy_by_run.csv", index=False)
    hide_close = df["hide_closest_m"].min()
    ok_hide = bool(hide_close > PRIVACY_M)
    say(f"  {mark(ok_hide)} Hide on (upload default): {int(df['hide_map'].sum())} of {len(df)} GPS runs have a map; "
        f"closest point sent to the browser {hide_close:.1f} m from a start/finish (rule: > {PRIVACY_M:.0f} m)")
    say(f"      Show full route (demo data): {int(df['full_map'].sum())} of {len(df)} GPS runs have a map")
    ok_cols = True
    kag = sorted(KAGGLE_DIR.glob("*.xls*")) + sorted(KAGGLE_DIR.glob("*.csv"))
    if kag:
        raw = pd.read_excel(kag[0]) if kag[0].suffix.startswith(".xls") else pd.read_csv(kag[0])
        coord = [c for c in raw.columns if any(k in c.lower() for k in ("latlng", "polyline", "map"))]
        kept = [c for c in IN.load_summary_table(kag[0]).runs.columns
                if any(k in c.lower() for k in ("lat", "lng", "lon", "polyline", "map"))]
        ok_cols = not kept
        say(f"  {mark(ok_cols)} activity list (Kaggle): coordinate columns in the file {coord or 'none'} -> "
            f"kept after loading: {kept or 'none'}")
    return ok_hide and ok_cols


# ============================================================
# E. Speed
# ============================================================

def part_e(model: PR.DeployedModel, quick: bool):
    say("E. SPEED (this computer)")
    rows = []

    def rec(step, secs, note=""):
        rows.append({"step": step, "seconds": round(secs, 3), "note": note})
        say(f"  {step:<52} {secs:7.2f} s  {note}")

    ad, t = timed(IN.load_strava_export, DEMO_ZIP)
    rec("Demo 1 (full): read export ZIP", t, f"{len(ad.runs)} runs, {len(ad.streams)} files")
    r, t = timed(FE.build_runner, ad)
    rec("Demo 1: Stages 1-3 for one runner", t)

    def estimate_bundle():
        now = PR.you_now(model, r, today=TODAY)
        return now, PR.weekly_trend(model, r), PR.run_day_estimates(model, r), PR.load_trend(r)
    (now, tr, rd, lt), t = timed(estimate_bundle)
    rec("Demo 1: You now + weekly trend + run-day estimates", t, f"{len(tr)} trend points, {len(rd)} runs")
    f_now = r.features_for_dates([r.you_now_date])
    _, t = timed(model.volume_curve, f_now, list(range(-30, 31, 5)))
    rec("Demo 1: sensitivity curve (13 settings)", t)

    recent = list(ad.runs["file"].tail(10 if quick else 30))
    per_run = []
    for f in recent:
        def panel():
            RA.run_summary(r, f)
            RA.km_splits(ad, f)
            RA.best_efforts(r, ad, f)
            steps = RA.scrub_steps(ad, f, True)
            if len(steps):
                RV.build_html(steps, RA.route_track(ad, f, True), RA.hidden_ranges(steps))
        per_run.append(timed(panel)[1])
    rec("Demo 1: open one run (stats, splits, best efforts, map)", float(np.median(per_run)),
        f"median of {len(per_run)} runs, slowest {max(per_run):.2f} s")
    total = sum(x["seconds"] for x in rows[:3])
    rec("Demo 1: first load in total (read + build + estimate)", total)

    kag = sorted(KAGGLE_DIR.glob("*.xls*")) + sorted(KAGGLE_DIR.glob("*.csv"))
    if kag:
        ka, t1 = timed(IN.load_summary_table, kag[0])
        kr, t2 = timed(FE.build_runner, ka)
        _, t3 = timed(lambda: (PR.you_now(model, kr, today=TODAY), PR.load_trend(kr)))
        rec("Demo 2 (activity list): read + build + load trend", t1 + t2 + t3, f"{len(ka.runs)} runs")
    pd.DataFrame(rows).to_csv(OUT / "stage9_E_speed.csv", index=False)
    say("  Note: the browser also loads the map library and map tiles; slider movement runs in the browser "
        "and needs no server work.")
    return ad, r


# ============================================================

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="10 test runners in part A")
    args = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    say("Stage 9 - technical evaluation of the RunState dashboard" + ("  (QUICK: 10 test runners)" if args.quick else ""))
    say("=" * 78)
    model = PR.DeployedModel()
    say(f"Deployed model: tuned XGBoost 'full', mean of seeds {PR.SEEDS}; {len(model.features)} inputs; "
        f"likely range x{model.range['lo']:.4f} - x{model.range['hi']:.4f}")
    say("")
    ok_a = part_a(model, args.quick)
    say("")
    print("Reading demo 1 for parts B-D ...")
    demo_ad = IN.load_strava_export(DEMO_ZIP)
    demo_r = FE.build_runner(demo_ad)
    ok_b = part_b(model, demo_ad, demo_r)
    say("")
    runners = {"Demo 1 (justinhojt)": demo_r}
    kag = sorted(KAGGLE_DIR.glob("*.xls*")) + sorted(KAGGLE_DIR.glob("*.csv"))
    if kag:
        runners["Demo 2 (Kaggle)"] = FE.build_runner(IN.load_summary_table(kag[0]))
    ok_c = part_c(runners)
    say("")
    ok_d = part_d(demo_ad)
    say("")
    part_e(model, args.quick)
    say("")
    all_ok = ok_a and ok_b and ok_c and ok_d
    say(f"RESULT: {'ALL GOOD' if all_ok else 'PROBLEMS - send this report'}  "
        f"(A accuracy {mark(ok_a).strip()}, B edge cases {mark(ok_b).strip()}, C ACWR {mark(ok_c).strip()}, "
        f"D privacy {mark(ok_d).strip()}; E speed is descriptive)  | {time.time() - t0:.0f} s")
    tag = "_quick" if args.quick else ""
    (OUT / f"stage9_evaluation_report{tag}.txt").write_text("\n".join(LINES) + "\n", encoding="utf-8")
    print(f"\nSaved: 09_dashboard_evaluation/stage9_evaluation_report{tag}.txt (+ CSVs A-E)")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
