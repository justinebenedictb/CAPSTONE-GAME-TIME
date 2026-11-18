"""
deployment/core/predict.py  -  Stage 9, Step 4: predictions

The deployed model exactly as evaluated in Stage 7:
    tuned XGBoost "full" (06_tuning/models/xgb_full_seed11/22/33.json, copied to deployment/models/),
    Stage 4 preparation (missing flags -> caps -> training fill values, as capstone_data.prepare_tabular),
    the 10 selected features in the Stage 4 order, prediction = mean of the 3 seeds
    (XGBRegressor.load_model + predict, as 09_stage7_test_evaluation.py).

Added for the dashboard:
    likely range  80% range from the Stage 7 TEST errors of this model (relative: actual / predicted,
                  10th-90th percentile, each runner counts equally). The models are frozen; the test
                  errors are only used to describe measured accuracy.
    finish times  RPC -> 5K / 10K / half / marathon (inverse Riegel, as the pipeline)
    what-if       same scaling as Stage 8 (training load: ATL, CTL, longest run; training speed:
                  grade-adjusted speed), applied to the prepared inputs

Check:
    py "deployment\\core\\predict.py" --quick   (10 test runners through the full dashboard path)
    py "deployment\\core\\predict.py"           (all 216 test runners)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .features import Runner, time_from_rpc, rpc_from_time
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from features import Runner, time_from_rpc, rpc_from_time

DEP = Path(__file__).resolve().parent.parent
MODEL_DIR = DEP / "models"
SEEDS = (11, 22, 33)
RANGE_FILE = "prediction_range.json"
RANGE_QUANTILES = (0.10, 0.90)          # 80% range

# Distances shown on the estimate card: all have real best-effort targets in training, validation and test.
# The marathon is not shown (8 test samples; Riegel tends to be optimistic for recreational marathons).
DISTANCES = {"5K": 5.0, "10K": 10.0, "15K": 15.0, "Half marathon": 21.0975}

# Stage 8 what-if scaling (10_stage8_interpretation.what_if)
LOAD_FEATURES = ["atl", "ctl", "longest_run_km_28"]      # ACWR = ATL/CTL unchanged
SPEED_FEATURES = ["mean_ga_speed_28"]


# ============================================================
# Likely range (from Stage 7 test errors)
# ============================================================

def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights, float)[order]
    cum = np.cumsum(w)
    cum /= cum[-1]
    return float(np.interp(q, cum, v))


def compute_range(stage7_predictions_csv, column="tuned_xgb_full") -> dict:
    """80% relative range of actual/predicted on the Stage 7 test set, each runner weighted equally."""
    d = pd.read_csv(stage7_predictions_csv)
    ratio = d["rpc"] / d[column]
    w = 1.0 / d.groupby("id")["id"].transform("size")
    lo = weighted_quantile(ratio, w, RANGE_QUANTILES[0])
    hi = weighted_quantile(ratio, w, RANGE_QUANTILES[1])
    inside = ratio.between(lo, hi)
    level = pd.qcut(d[column], 3, labels=["slower third", "middle third", "faster third"])
    cov_level = {str(k): round(float(np.average(inside[level == k], weights=w[level == k])), 3)
                 for k in level.cat.categories}
    return {
        "description": "80% likely range: actual best-effort RPC / predicted RPC, 10th-90th percentile "
                       "on the Stage 7 test set (each runner counts equally). Range = prediction x [lo, hi].",
        "model": column, "source": "07_test_evaluation/stage7_test_predictions.csv",
        "quantiles": list(RANGE_QUANTILES), "lo": lo, "hi": hi,
        "n_samples": int(len(d)), "n_runners": int(d["id"].nunique()),
        "coverage_runner_weighted": round(float(np.average(inside, weights=w)), 3),
        "coverage_per_sample": round(float(inside.mean()), 3),
        "coverage_by_level_runner_weighted": cov_level,
        "time_range_pct": {"faster_by": round(100 * (1 - 1 / hi), 1), "slower_by": round(100 * (1 / lo - 1), 1)},
    }


# ============================================================
# The deployed model
# ============================================================

class DeployedModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        from xgboost import XGBRegressor
        model_dir = Path(model_dir)
        self.features = [l.strip() for l in (model_dir / "stage4_selected_features_tabular.txt")
                         .read_text(encoding="utf-8").splitlines() if l.strip()]
        self.params = json.loads((model_dir / "stage4_preprocessing_params.json").read_text(encoding="utf-8"))
        self.models = []
        for s in SEEDS:
            m = XGBRegressor()
            m.load_model(str(model_dir / f"xgb_full_seed{s}.json"))
            self.models.append(m)
        rf = model_dir / RANGE_FILE
        self.range = json.loads(rf.read_text(encoding="utf-8")) if rf.exists() else None

    # capstone_data.prepare_tabular (Stage 4 parameters, fitted on training data)
    def prepare(self, feats: pd.DataFrame) -> pd.DataFrame:
        out = feats.copy()
        for col in self.params["missing_flags"]:
            out[f"{col}_missing"] = out[col].isna().astype(int)
        for col, cap in self.params["caps"].items():
            if col in out:
                out[col] = out[col].clip(upper=cap)
        for col, value in self.params["fill"].items():
            if col in out:
                out[col] = out[col].fillna(value)
        return out

    def predict_prepared(self, X: pd.DataFrame) -> np.ndarray:
        return np.mean([np.asarray(m.predict(X[self.features]), dtype=np.float64) for m in self.models], axis=0)

    def predict(self, feats: pd.DataFrame) -> np.ndarray:
        return self.predict_prepared(self.prepare(feats))

    def with_range(self, rpc) -> tuple[np.ndarray, np.ndarray]:
        rpc = np.asarray(rpc, float)
        if not self.range:
            return np.full_like(rpc, np.nan), np.full_like(rpc, np.nan)
        return rpc * self.range["lo"], rpc * self.range["hi"]

    def what_if(self, feats: pd.DataFrame, load_pct: float = 0.0, speed_pct: float = 0.0) -> np.ndarray:
        """Predicted RPC if training load / training speed had been load_pct / speed_pct % different."""
        X = self.prepare(feats)
        for c in LOAD_FEATURES:
            X[c] = X[c] * (1 + load_pct / 100)
        for c in SPEED_FEATURES:
            X[c] = X[c] * (1 + speed_pct / 100)
        return self.predict_prepared(X)

    def volume_curve(self, feats: pd.DataFrame, pcts) -> np.ndarray:
        """Sensitivity panel: predicted RPC with training volume scaled by each % in pcts (one batch).
        Scales acute and chronic load and the longest run together (ACWR unchanged); everything else fixed."""
        X = self.prepare(feats.iloc[[0]])
        X = pd.concat([X] * len(pcts), ignore_index=True)
        f = 1 + np.asarray(pcts, float) / 100
        for c in LOAD_FEATURES:
            X[c] = X[c].to_numpy(float) * f
        return self.predict_prepared(X)


def finish_times(rpc: float, lo: float | None = None, hi: float | None = None) -> pd.DataFrame:
    """Estimated best-effort time per distance, with the likely range (fast = rpc*hi, slow = rpc*lo)."""
    rows = []
    for name, km in DISTANCES.items():
        rows.append({"distance": name, "km": km, "seconds": time_from_rpc(rpc, km),
                     "fast_seconds": time_from_rpc(rpc * hi, km) if hi else np.nan,
                     "slow_seconds": time_from_rpc(rpc * lo, km) if lo else np.nan})
    return pd.DataFrame(rows)


def fmt_time(seconds: float) -> str:
    if not np.isfinite(seconds):
        return "-"
    s = int(round(seconds))
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ============================================================
# Runner-level outputs for the dashboard
# ============================================================

ZONE_NAMES = {0: "Under-training", 1: "Sweet spot", 2: "Monitoring", 3: "Danger"}
NO_ESTIMATE_REASON = ("A fitness estimate needs at least one best effort (a run of 5 km or more with "
                      "per-second data: FIT, GPX or TCX files). This data has no best efforts, so only "
                      "training load and the injury-risk alert are shown.")


def estimate_available(runner: Runner) -> bool:
    """An estimate is shown only when the runner has at least one valid best effort.
    Without one, every past-performance input (82% of importance in Stage 8) would be a
    population fill value, so the number would describe an average runner, not this one."""
    return len(runner.perf) > 0


def _with_prior(f: pd.DataFrame) -> pd.Series:
    return f["ok"] & (f["prior_rpc_count"].fillna(0) >= 1)


def training_load_at(runner: Runner, dates) -> pd.DataFrame:
    """Training load on each date (= the day before, as the model sees it).
    Alert measure = distance ACWR (acwr_km), which Stage 3 defined for the alert framework;
    heart-rate ACWR is reported alongside when the runner has heart-rate data."""
    dates = pd.to_datetime(pd.Series(list(dates))).dt.normalize()
    d = runner.daily.set_index("calendar_date")
    rows = []
    for t in dates:
        day = t - pd.Timedelta(days=1)
        if day not in d.index:
            rows.append({"date": t})
            continue
        r = d.loc[day]
        z = r["acwr_km_zone"]
        rows.append({"date": t, "km_7": r["km_7"], "km_28": r["km_28"], "run_days_7": r["run_days_7"],
                     "run_days_28": r["run_days_28"], "longest_run_km_28": r["longest_run_km_28"],
                     "acwr_km": r["acwr_km"], "zone": ZONE_NAMES.get(int(z)) if pd.notna(z) else None,
                     "acwr_hr": r["acwr"], "atl": r["atl"], "ctl": r["ctl"]})
    out = pd.DataFrame(rows)
    out["has_hr_load"] = bool(runner.runs["trimp"].notna().any())
    return out


def load_trend(runner: Runner) -> pd.DataFrame:
    """Weekly training load ending on the 'You now' date (both modes)."""
    start = runner.first_day + pd.Timedelta(days=27)          # first day with a full 28-day window
    n = max(0, (runner.you_now_date - start).days // 7 + 1)
    dates = [runner.you_now_date - pd.Timedelta(days=7 * k) for k in range(n)][::-1]
    return training_load_at(runner, dates)


def you_now(model: DeployedModel, runner: Runner, today=None) -> dict:
    f = runner.features_for_dates([runner.you_now_date])
    today = pd.Timestamp(today or pd.Timestamp.now().normalize())
    out = {"date": runner.you_now_date, "last_run": runner.last_day,
           "days_since_last_run": int((today - runner.last_day).days), "ok": bool(f["ok"].iloc[0])}
    out["load"] = training_load_at(runner, [runner.you_now_date]).iloc[0].to_dict()
    if not estimate_available(runner):
        out.update(ok=False, reason=NO_ESTIMATE_REASON, mode="summary")
        return out
    out["mode"] = "full"
    if not out["ok"]:
        out["reason"] = (f"Needs at least {(runner.earliest_estimate_date - runner.first_day).days} days of "
                         f"history (first run {runner.first_day.date()}).")
        return out
    if not _with_prior(f).iloc[0]:
        out.update(ok=False, reason="No best effort before this date yet.")
        return out
    rpc = float(model.predict(f)[0])
    lo, hi = model.with_range([rpc])
    out.update({"rpc": rpc, "rpc_lo": float(lo[0]), "rpc_hi": float(hi[0]),
                "times": finish_times(rpc, model.range and model.range["lo"], model.range and model.range["hi"]),
                "inputs": model.prepare(f).iloc[0][model.features],
                "raw_inputs": f.iloc[0]})
    return out


def weekly_trend(model: DeployedModel, runner: Runner) -> pd.DataFrame:
    """Estimate every 7 days, ending on the 'You now' date."""
    if runner.you_now_date < runner.earliest_estimate_date:
        return pd.DataFrame(columns=["date", "rpc", "rpc_lo", "rpc_hi", "acwr", "acwr_zone"])
    n = (runner.you_now_date - runner.earliest_estimate_date).days // 7 + 1
    dates = [runner.you_now_date - pd.Timedelta(days=7 * k) for k in range(n)][::-1]
    f = runner.features_for_dates(dates)
    f = f[_with_prior(f)].reset_index(drop=True)
    if f.empty:
        return pd.DataFrame(columns=["date", "rpc", "rpc_lo", "rpc_hi", "acwr", "acwr_zone"])
    rpc = model.predict(f)
    lo, hi = model.with_range(rpc)
    return pd.DataFrame({"date": f["target_date"], "rpc": rpc, "rpc_lo": lo, "rpc_hi": hi,
                         "acwr": f["acwr"], "acwr_zone": f["acwr_zone"]})


def run_day_estimates(model: DeployedModel, runner: Runner) -> pd.DataFrame:
    """For every run: the fitness estimate for that day (inputs = the 28 days before it) and,
    if the run contained a valid best effort, its actual RPC."""
    runs = runner.runs[["file", "calendar_date"]].copy()
    runs["calendar_date"] = pd.to_datetime(runs["calendar_date"])
    f = runner.features_for_dates(runs["calendar_date"].unique())
    ok = f[_with_prior(f)].reset_index(drop=True)
    est = pd.DataFrame({"calendar_date": ok["target_date"], "rpc_est": np.nan, "rpc_est_lo": np.nan,
                        "rpc_est_hi": np.nan})
    if len(ok):
        est["rpc_est"] = model.predict(ok)
        est["rpc_est_lo"], est["rpc_est_hi"] = model.with_range(est["rpc_est"])
    actual = runner.perf.rename(columns={"date": "calendar_date", "rpc": "rpc_actual_day"})[
        ["calendar_date", "rpc_actual_day"]]
    out = runs.merge(est, on="calendar_date", how="left").merge(actual, on="calendar_date", how="left")
    return out


# ============================================================
# Self-check
# ============================================================

def _self_check(argv=None) -> int:
    import argparse
    try:
        from .ingest import load_goldencheetah, load_strava_export
        from .features import build_runner
    except ImportError:
        from ingest import load_goldencheetah, load_strava_export
        from features import build_runner
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEP.parent))
    ap.add_argument("--quick", action="store_true", help="10 test runners")
    args = ap.parse_args(argv)
    root = Path(args.root)
    lines = ["Stage 9 Step 4 - prediction check", "=" * 70]

    def say(s=""):
        print(s)
        lines.append(s)

    # ---- likely range ----
    rng = compute_range(root / "07_test_evaluation" / "stage7_test_predictions.csv")
    (MODEL_DIR / RANGE_FILE).write_text(json.dumps(rng, indent=2), encoding="utf-8")
    say("A. Likely range (80%) from the Stage 7 test errors -> deployment/models/prediction_range.json")
    say(f"  actual / predicted between {rng['lo']:.4f} and {rng['hi']:.4f}  "
        f"(times: up to {rng['time_range_pct']['faster_by']}% faster / {rng['time_range_pct']['slower_by']}% slower)")
    say(f"  coverage: {rng['coverage_runner_weighted']:.1%} (each runner equal), {rng['coverage_per_sample']:.1%} "
        f"(per sample); by level {rng['coverage_by_level_runner_weighted']}")

    model = DeployedModel()

    # ---- Stage 7 match through the full dashboard path ----
    ref = pd.read_csv(root / "07_test_evaluation" / "stage7_test_predictions.csv")
    split = pd.read_csv(root / "04_feature_selection" / "stage4_split_assignment.csv")
    test_ids = sorted(ref["id"].unique())
    if args.quick:
        test_ids = test_ids[:10]
    say("")
    say(f"B. Stage 7 match: {len(test_ids)} test runners, Strava-style path "
        f"(raw files -> ingest -> features -> predict)")
    t0 = time.time()
    got = []
    manifest = root / "manifest file" / "run_activities_manifest.csv"
    for i, a in enumerate(test_ids, 1):
        print(f"  runner {i}/{len(test_ids)}", end="\r")
        r = build_runner(load_goldencheetah(a, manifest, root / "00_raw" / "granular"))
        s = r.samples.merge(split[["sample_id", "split"]], on="sample_id", how="inner")
        s = s[s["split"] == "test"]
        got.append(pd.DataFrame({"sample_id": s["sample_id"], "dash": model.predict(s)}))
    print()
    got = pd.concat(got)
    m = ref[ref["id"].isin(test_ids)][["sample_id", "tuned_xgb_full", "rpc"]].merge(got, on="sample_id", how="outer")
    missing = int(m["dash"].isna().sum())
    extra = int(m["tuned_xgb_full"].isna().sum())
    diff = (m["dash"] - m["tuned_xgb_full"]).abs()
    ok_b = missing == 0 and extra == 0 and float(diff.max()) < 1e-6
    w = 1 / m.groupby(m["sample_id"].str.rsplit("_", n=1).str[0])["sample_id"].transform("size")
    mae = float(np.average((m["dash"] - m["rpc"]).abs(), weights=w))
    say(f"  {'OK ' if ok_b else 'BAD'} test samples {len(ref[ref['id'].isin(test_ids)]):,} | missing {missing} | "
        f"extra {extra} | largest difference {diff.max():.2e} RPC | dashboard MAE {mae:.4f} "
        f"(Stage 7: 0.0380 on all 216 runners) | {time.time() - t0:.0f} s")

    # ---- demo runner ----
    say("")
    say("C. Demo runner (justinhojt)")
    r = build_runner(load_strava_export(DEP / "demo_data" / "justinhojt_strava_export.zip"))
    now = you_now(model, r, today="2026-11-17")
    say(f"  'You now' ({now['date'].date()}, last run {now['last_run'].date()}): RPC {now['rpc']:.3f} "
        f"(likely {now['rpc_lo']:.3f}-{now['rpc_hi']:.3f})")
    for _, t in now["times"].iterrows():
        say(f"    {t['distance']:<14} {fmt_time(t['seconds']):>8}   likely {fmt_time(t['fast_seconds'])} - "
            f"{fmt_time(t['slow_seconds'])}")
    raw = now["raw_inputs"]
    say(f"  inputs: personal average {raw['prior_mean_rpc']:.3f} | last {raw['last_rpc']:.3f} | "
        f"best 90 d {raw['best_rpc_90']:.3f} | CTL {raw['ctl']:.1f} ATL {raw['atl']:.1f} ACWR {raw['acwr']:.2f}")
    recent = r.samples[r.samples["target_date"] >= now["date"] - pd.Timedelta(days=90)]
    say(f"  his actual best efforts in the last 90 days: {len(recent)} | median RPC {recent['rpc'].median():.3f} "
        f"| best {recent['rpc'].max():.3f}")
    tr = weekly_trend(model, r)
    say(f"  weekly trend: {len(tr)} points, ends {tr['date'].iloc[-1].date()}; last 6: "
        + ", ".join(f"{d:%b %d} {v:.3f}" for d, v in zip(tr['date'].tail(6), tr['rpc'].tail(6))))
    rd = run_day_estimates(model, r)
    s = r.samples.assign(pred=model.predict(r.samples))
    say(f"  run-day estimates: {int(rd['rpc_est'].notna().sum())}/{len(rd)} runs | on his {len(s)} best-effort days: "
        f"MAE {np.mean(np.abs(s['pred'] - s['rpc'])):.4f}, inside likely range "
        f"{np.mean((s['rpc'] >= s['pred'] * model.range['lo']) & (s['rpc'] <= s['pred'] * model.range['hi'])):.0%}")
    f_now = r.features_for_dates([r.you_now_date])
    base5 = time_from_rpc(now["rpc"], 5.0)
    say("  what-if on a 5K (seconds vs now; negative = faster):")
    for name, lp, sp in [("train 20% more", 20, 0), ("train 20% less", -20, 0), ("detraining (load -50%)", -50, 0),
                         ("training runs 5% faster", 0, 5)]:
        v = float(model.what_if(f_now, lp, sp)[0])
        say(f"    {name:<26} {time_from_rpc(v, 5.0) - base5:+6.1f} s")
    ld = now["load"]
    say(f"  training load now: {ld['km_7']:.1f} km last 7 d, {ld['km_28']:.1f} km last 28 d, distance ACWR "
        f"{ld['acwr_km']:.2f} ({ld['zone']}), HR ACWR {ld['acwr_hr']:.2f}")

    # ---- summary-only mode (Kaggle) ----
    ok_d = True
    kag = sorted((DEP / "demo_data" / "kaggle").glob("*.xls*")) + sorted((DEP / "demo_data" / "kaggle").glob("*.csv"))
    say("")
    if kag:
        try:
            from .ingest import load_summary_table
        except ImportError:
            from ingest import load_summary_table
        say(f"D. Summary-only mode: {kag[0].name}")
        ka = load_summary_table(kag[0])
        kr = build_runner(ka)
        know = you_now(model, kr, today="2026-11-17")
        lt = load_trend(kr)
        say(f"  {ka.report['activities_total']} activities -> {ka.report['runs']} runs ({ka.report['table_layout']} layout), "
            f"{kr.info['runs']} kept, {kr.info['excluded_as_non_run']} excluded as non-run; heart rate: "
            f"{'yes' if lt['has_hr_load'].iloc[0] else 'none'}; {kr.first_day.date()} -> {kr.last_day.date()}")
        say(f"  estimate shown: {know['ok']} | message: {know.get('reason', '')[:80]}...")
        kl = know["load"]
        say(f"  training load now ({know['date'].date()}): {kl['km_7']:.1f} km last 7 d, {kl['km_28']:.1f} km last 28 d, "
            f"distance ACWR {kl['acwr_km']:.2f} ({kl['zone']})")
        say(f"  weekly load trend: {len(lt)} weeks; zones {lt['zone'].value_counts(dropna=False).to_dict()}")
        ok_d = (not know["ok"]) and know.get("mode") == "summary" and len(weekly_trend(model, kr)) == 0 \
            and not any("lat" in c.lower() for c in ka.runs.columns)
        say(f"  {'OK ' if ok_d else 'BAD'} no estimate in summary mode, no coordinates kept")
    else:
        say("D. Summary-only mode: no Kaggle file in deployment/demo_data/kaggle (skipped)")
    say("")
    all_ok = ok_b and ok_d
    say("RESULT: " + ("ALL GOOD - dashboard predictions reproduce Stage 7" if all_ok else "MISMATCH - send this report"))
    out = DEP / "_checks"
    out.mkdir(exist_ok=True)
    tag = "_quick" if args.quick else ""
    (out / f"step4_predict_report{tag}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    m.to_csv(out / f"step4_stage7_match{tag}.csv", index=False)
    print(f"\nSaved: deployment/_checks/step4_predict_report{tag}.txt and deployment/models/{RANGE_FILE}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
