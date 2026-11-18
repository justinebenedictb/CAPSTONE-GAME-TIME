"""
RunState - Stage 9 dashboard (layout C2: split cockpit with a distinct control bar)

    streamlit run deployment/app.py

Everything shown comes from the verified Stage 9 modules in deployment/core/
(ingest -> features -> predict -> run_analytics), which reproduce Stages 1-7 of the
capstone pipeline exactly. Pass 1: everything except the route map (pass 2).
"""
from __future__ import annotations

import sys
from html import escape as esc
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

DEP = Path(__file__).resolve().parent
sys.path.insert(0, str(DEP))

from core.ingest import load_strava_export, load_summary_table               # noqa: E402
from core.features import build_runner, time_from_rpc                         # noqa: E402
from core import predict as P                                                 # noqa: E402
from core import run_analytics as RA                                          # noqa: E402
from core import charts as C                                                  # noqa: E402

# ============================================================
# SETTINGS (one line each, safe to change)
# ============================================================
APP_TITLE = "RunState"
APP_SUBTITLE = "Where your running fitness stands today, and how your training is moving it"
TREND_UNITS = "5K time"          # "5K time" or "RPC"
STALE_AFTER_DAYS = 28            # warn when the latest run is older than this
RANGE_TYPE = "relative"          # likely range from Stage 7 test errors (actual / predicted)
FOOTER_CREDIT = ""               # e.g. project / university line, shown in the footer if set
UPLOAD_HIDE_BY_DEFAULT = True    # uploads: "Hide 500 m around start and finish" preselected (demos show full routes)
PRIVACY_OPTIONS = ["Hide 500 m around start and finish", "Show full route"]

DEMOS = {
    "Demo · Strava export (justinhojt, MIT)": ("zip", DEP / "demo_data" / "justinhojt_strava_export.zip"),
    "Demo · Kaggle activity list (CC0)": ("table", DEP / "demo_data" / "kaggle" / "Strava Running Data.xlsx"),
}

# ============================================================
# STYLE (C2 mockup)
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
.stApp { font-family: Inter, system-ui, sans-serif; }
.block-container { max-width: 1400px; padding-top: 3.4rem; }
.rs-title { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 28px; color: #15181D; }
.rs-sub { font-size: 15px; color: #5A6270; margin-left: 10px; }
/* control bar */
.st-key-controlbar { background: #15181D; border-radius: 16px; padding: 18px 20px 20px 20px; }
.st-key-controlbar p, .st-key-controlbar label, .st-key-controlbar span, .st-key-controlbar small { color: #F2F4F7; }
.st-key-controlbar [data-testid="stWidgetLabel"] p { color: #B4BCC8 !important; font-size: 13px; }
.st-key-controlbar [data-testid="stFileUploaderDropzone"] { background: #1E2633; border: 2px dashed #5B8FE8; border-radius: 12px; }
.st-key-controlbar [data-testid="stFileUploaderDropzone"] * { color: #F2F4F7; }
.st-key-controlbar [data-testid="stCaptionContainer"] p { color: #B4BCC8 !important; }
.st-key-controlbar [data-testid="stButton"] button { background: #222730; color: #F2F4F7; border: 1px solid #3A414D;
  height: 40px; min-height: 40px; width: 100%; padding: 0; font-size: 18px; }
.st-key-controlbar [data-testid="stButton"] button:disabled { opacity: 0.35; background: #222730; border: 1px solid #3A414D; color: #F2F4F7; }
.st-key-controlbar [data-testid="stFileUploaderDropzone"] { padding: 10px 14px; min-height: 64px; }
.st-key-controlbar [data-testid="stFileUploaderDropzone"] button { background: #222730; color: #F2F4F7; border: 1px solid #5B8FE8; }
.st-key-controlbar [data-testid="stFileUploader"] section + div { color: #F2F4F7; }
.st-key-controlbar [data-testid="stSelectbox"] > div { margin-top: 0; }
.rs-step { display: flex; align-items: center; gap: 10px; font-weight: 600; font-size: 16px; margin-bottom: 4px; color: #F2F4F7; }
.rs-badge { width: 26px; height: 26px; border-radius: 13px; display: inline-flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; }
.rs-chips { display: flex; flex-wrap: wrap; gap: 8px; row-gap: 8px; font-size: 13px; margin: 6px 0 2px 0; }
.rs-chip { background: #2A303B; color: #D4D9E1; border-radius: 999px; padding: 4px 10px; }
.rs-chip.ok { background: #1F3B2C; color: #A7E3C1; }
.rs-chip.full { background: #1D3A66; color: #CFE0FA; }
.rs-chip.summary { background: #4A3216; color: #FBD9B8; }
/* sections and panels */
.rs-sec { font-size: 15px; letter-spacing: 0.08em; font-weight: 600; padding-top: 10px; margin: 6px 0 2px 0; }
.rs-sec.blue { border-top: 4px solid #3B82F6; color: #1D5FBF; }
.rs-sec.orange { border-top: 4px solid #F08A4B; color: #B4530F; }
[class*="st-key-blue_"] { background: #EEF4FD; border: 1px solid #C9DBF5; border-radius: 14px; padding: 16px 18px; }
[class*="st-key-orange_"] { background: #FFF5EE; border: 1px solid #F3D3BD; border-radius: 14px; padding: 16px 18px 18px 18px; }
.rs-label { display: block; font-size: 17px; font-weight: 600; color: #15181D; margin: 4px 0 4px 0; }
.rs-h3 { display: inline-block; font-size: 19px; font-weight: 600; color: #15181D; margin: 10px 0 6px 0; }
.rs-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 16px; margin: 6px 0; }
.rs-grid4 { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
.rs-k { font-size: 14px; color: #5A6270; }
.rs-big { font-family: 'Space Grotesk', sans-serif; font-size: 32px; font-weight: 700; color: #15181D; line-height: 1.15; }
.rs-mid { font-size: 20px; font-weight: 700; color: #15181D; }
.rs-range { font-size: 13px; color: #3B4250; }
.rs-note { font-size: 13px; color: #5A6270; line-height: 1.5; }
.rs-warn { background: #FFF4D6; border: 1px solid #F2C46B; border-radius: 10px; padding: 8px 12px; font-size: 14px; color: #5A3E00; }
.rs-msg { background: #FFFFFF; border: 1px solid #C9DBF5; border-radius: 10px; padding: 12px 14px; font-size: 15px; color: #15181D; }
.rs-box { background: #FFFFFF; border: 1px solid #F3D3BD; border-radius: 10px; padding: 10px 12px; font-size: 14px; color: #5A2A0A; margin: 8px 0 4px 0; }
.rs-zonebar { display: grid; height: 10px; border-radius: 5px; overflow: hidden; }
.rs-hrbar { display: flex; height: 18px; border-radius: 5px; overflow: hidden; }
.rs-mapnote { background: #EEF1EC; border-radius: 10px; padding: 10px 12px; color: #5A6270; font-size: 14px; }
.rs-legend { display: inline-block; width: 120px; height: 10px; border-radius: 5px; vertical-align: middle;
  background: linear-gradient(90deg, #85B7EB, #D85A30); }
.rs-zonebar2 { display: flex; height: 12px; border-radius: 6px; overflow: hidden; }
.rs-marker { position: absolute; top: -4px; width: 4px; height: 20px; background: #15181D; border-radius: 2px; }
.rs-ticks { position: relative; height: 16px; font-size: 12px; color: #3B4250; }
.rs-tick { position: absolute; transform: translateX(-50%); }
.rs-zlegend { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0 4px 0; }
.rs-zitem { display: inline-flex; align-items: center; gap: 5px; font-size: 13px; color: #3B4250; padding: 2px 8px;
  border-radius: 999px; background: #FFFFFF; border: 1px solid #DCE3EE; }
.rs-zitem.on { font-weight: 700; color: #15181D; border: 1.5px solid #15181D; }
.rs-zr { color: #5A6270; font-weight: 400; }
.rs-dot { width: 10px; height: 10px; border-radius: 5px; display: inline-block; }
.rs-two { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 4px 0; }
.rs-tip { cursor: help; color: #1D5FBF; font-size: 14px; margin-left: 2px; }
.rs-sub { font-size: 13px; color: #3B4250; margin-top: 1px; }
.rs-paceleg { display: flex; align-items: center; gap: 8px; font-size: 14px; color: #15181D; margin: 2px 0 4px 0; }
.rs-paceleg-t { font-weight: 600; margin-right: 4px; }
.rs-footer { font-size: 13px; color: #5A6270; border-top: 1px solid #D9DEE6; padding-top: 10px; margin-top: 14px; line-height: 1.6; }
</style>
"""

ZONE_TEXT_COLOR = {"Under-training": "#4A5A73", "Sweet spot": "#1E6B45", "Monitoring": "#8A5A00", "Danger": "#B4530F"}
HR_ZONE_COLORS = ["#D5DEEA", "#A9C2E4", "#6E9BD6", "#E59464", "#C2410C"]


# ============================================================
# HELPERS
# ============================================================

def html(s: str):
    st.markdown(s, unsafe_allow_html=True)


CHART_BUTTONS = ["zoomIn2d", "zoomOut2d", "resetScale2d"]
RUN_FIGURE_BUTTONS = ["zoomInMap", "zoomOutMap", "resetViewMap", "resetScale2d"]


def show_chart(fig, buttons=None):
    """Only the zoom / reset buttons we want, always visible; the mouse wheel keeps scrolling the page."""
    config = {"displayModeBar": True, "displaylogo": False, "scrollZoom": False,
              "modeBarButtons": [buttons or CHART_BUTTONS]}
    try:
        st.plotly_chart(fig, width="stretch", config=config)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, config=config)


def show_table(df: pd.DataFrame):
    try:
        st.dataframe(df, hide_index=True, width="stretch")
    except TypeError:
        st.dataframe(df, hide_index=True, use_container_width=True)


def fmt_time(s) -> str:
    return P.fmt_time(float(s)) if s is not None and pd.notna(s) else "–"


def fmt_pace(s_per_km) -> str:
    return f"{P.fmt_time(float(s_per_km))} /km" if s_per_km is not None and pd.notna(s_per_km) else "–"


def fmt_num(v, fmt="{:.0f}", suffix="") -> str:
    return (fmt.format(v) + suffix) if v is not None and pd.notna(v) else "–"


def fmt_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d:%b} {d.day}, {d.year}"


def run_label(row) -> str:
    name = str(row.get("activity_name") or "Run").strip() or "Run"
    return f"{fmt_date(row['datetime_utc'])} · {name} · {row['total_distance']:.2f} km"


# ============================================================
# DATA (processed once per dataset)
# ============================================================

@st.cache_resource(show_spinner=False)
def get_model():
    return P.DeployedModel()


def process(ad) -> dict:
    """Everything the dashboard needs for one runner."""
    model = get_model()
    runner = build_runner(ad)
    now = P.you_now(model, runner)
    mode = now.get("mode", "full" if P.estimate_available(runner) else "summary")
    runs = runner.runs.sort_values("datetime_utc", ascending=False).reset_index(drop=True)
    b = {"ad": ad, "runner": runner, "now": now, "mode": mode, "runs": runs,
         "load": P.load_trend(runner)}
    if mode == "full":
        b["trend"] = P.weekly_trend(model, runner)
        b["runday"] = P.run_day_estimates(model, runner)
        b["f_now"] = runner.features_for_dates([runner.you_now_date])
    return b


@st.cache_data(show_spinner=False, max_entries=4)
def process_demo(name: str) -> dict:
    kind, path = DEMOS[name]
    ad = load_strava_export(path) if kind == "zip" else load_summary_table(path)
    return process(ad)


def process_upload(file) -> dict:
    """Uploads are processed per session (st.session_state), never in the shared cache."""
    key = f"{file.name}:{file.size}"
    if st.session_state.get("upload_key") == key:
        return st.session_state["upload_bundle"]
    data = file.getvalue()
    bar = st.progress(0.0, text="Reading your activity files…")
    if file.name.lower().endswith(".zip"):
        ad = load_strava_export(data, progress=lambda i, n: bar.progress((i + 1) / max(n, 1),
                                                                         text=f"Reading run {i + 1} of {n}…"))
    else:
        class _F:  # file-like with a name, for load_summary_table
            def __init__(self, n, d):
                self.name, self._d = n, d

            def read(self):
                return self._d
        ad = load_summary_table(_F(file.name, data))
    bar.progress(1.0, text="Building features and estimates…")
    b = process(ad)
    bar.empty()
    st.session_state["upload_key"], st.session_state["upload_bundle"] = key, b
    return b


# ============================================================
# CONTROL BAR
# ============================================================

def step(n: int, text: str, color: str, fg: str):
    html(f'<div class="rs-step"><span class="rs-badge" style="background:{color};color:{fg}">{n}</span>{text}</div>')


def control_bar():
    with st.container(key="controlbar"):
        left, right = st.columns([1.25, 1], gap="large")
        with left:
            step(1, "Load your data", "#3B82F6", "#FFFFFF")
            c1, c2 = st.columns([1.2, 1], vertical_alignment="center")
            with c1:
                up = st.file_uploader("Drop Strava export (.zip) or an activity list (.csv / .xlsx)",
                                      type=["zip", "csv", "xlsx"], key="upload")
            with c2:
                demo = st.selectbox("or try a demo dataset", list(DEMOS), key="demo",
                                    disabled=up is not None)
                if up is not None:
                    st.caption("Using your upload. Remove it to go back to the demos.")
            privacy = False                                   # demo data is public: full routes
            if up is not None:
                choice = st.radio("Route privacy", PRIVACY_OPTIONS, index=0 if UPLOAD_HIDE_BY_DEFAULT else 1,
                                  key="route_privacy", horizontal=True)
                st.caption("Only you see your data. Hiding helps if you'll share your screen.")
                privacy = choice == PRIVACY_OPTIONS[0]
            bundle, source = None, None
            try:
                if up is not None:
                    bundle, source = process_upload(up), f"upload:{up.name}:{up.size}"
                else:
                    with st.spinner("Processing the demo data (about 30 s the first time)…"):
                        bundle, source = process_demo(demo), f"demo:{demo}"
            except ValueError as e:
                st.error(f"Could not read this file: {e}")
            except Exception as e:  # unexpected file problems: never crash the page
                st.error(f"Could not process this file ({type(e).__name__}). Please upload a Strava export ZIP "
                         "or an activity list (.csv / .xlsx).")
            if bundle is not None:
                chips(bundle, privacy)
        with right:
            step(2, "Pick a run to explore", "#F08A4B", "#15181D")
            run = run_picker(bundle, source)
    return bundle, run, privacy


def chips(b: dict, privacy: bool = False):
    r, runs = b["runner"], b["runs"]
    n = len(runs)
    hr_share = runs["has_hr"].mean() if "has_hr" in runs and n else 0
    span = f"{r.first_day:%b %Y} – {r.last_day:%b %Y}"
    mode = ('<span class="rs-chip full" title="Activity files with per-second data: fitness estimate, '
            'trend, what-if and run analytics are on">Full data</span>' if b["mode"] == "full" else
            '<span class="rs-chip summary" title="Activity list only: training load and the ACWR alert; '
            'a fitness estimate needs activity files (FIT, GPX or TCX)">Summary only</span>')
    hr = f"Heart rate on {hr_share:.0%}" if hr_share > 0 else "No heart rate"
    html(f'<div class="rs-chips"><span class="rs-chip ok">✓ {n} runs loaded</span>{mode}'
         f'<span class="rs-chip">{span}</span><span class="rs-chip">{hr}</span>'
         + (('<span class="rs-chip">Route: start/finish hidden</span>' if privacy else
             '<span class="rs-chip">Route: full</span>') if runs["has_gps"].any() else "")
         + f'<span class="rs-chip" title="Files are processed in memory and never written to disk">'
         f'In memory only · not stored</span></div>')


def run_picker(b: dict | None, source: str | None):
    if b is None or b["runs"].empty:
        st.selectbox("Activity", ["No runs loaded"], disabled=True, label_visibility="collapsed")
        return None
    runs = b["runs"]
    n = len(runs)
    if st.session_state.get("run_source") != source:          # new dataset -> latest run
        st.session_state["run_source"] = source
        st.session_state["run_sel"] = 0
    st.session_state["run_sel"] = min(st.session_state.get("run_sel", 0), n - 1)

    def older():
        st.session_state["run_sel"] = min(st.session_state["run_sel"] + 1, n - 1)

    def newer():
        st.session_state["run_sel"] = max(st.session_state["run_sel"] - 1, 0)

    c1, c2, c3 = st.columns([1, 8, 1], vertical_alignment="center")
    with c1:
        st.button("‹", key="older", on_click=older, help="Older run", disabled=st.session_state["run_sel"] >= n - 1)
    with c2:
        idx = st.selectbox("Activity", list(range(n)), key="run_sel", label_visibility="collapsed",
                           format_func=lambda i: run_label(runs.iloc[i]))
    with c3:
        st.button("›", key="newer", on_click=newer, help="Newer run", disabled=st.session_state["run_sel"] <= 0)
    st.caption(f'Run {n - idx} of {n} · newest first · "You now" always uses your latest 28 days')
    return runs.iloc[idx]


# ============================================================
# YOU NOW (blue)
# ============================================================

def you_now_column(b: dict, run):
    now, r = b["now"], b["runner"]
    html(f'<div class="rs-sec blue">YOU NOW · FROM YOUR LATEST 28 DAYS (TO {fmt_date(r.last_day).upper()})</div>')
    model = get_model()
    with st.container(key="blue_estimate"):
        if b["mode"] == "full" and now.get("ok"):
            html(f'<span class="rs-label">Current fitness · best-effort estimate · as of {fmt_date(now["date"])}</span>')
            cells = "".join(
                f'<div><div class="rs-k">{t["distance"]}</div><div class="rs-big">{fmt_time(t["seconds"])}</div>'
                f'<div class="rs-range">likely {fmt_time(t["fast_seconds"])} – {fmt_time(t["slow_seconds"])}</div></div>'
                for _, t in now["times"].iterrows())
            html(f'<div class="rs-grid">{cells}</div>')
            recent = r.perf[r.perf["date"] >= now["date"] - pd.Timedelta(days=90)]
            if len(recent):
                med = time_from_rpc(recent["rpc"].median(), 5.0)
                best = time_from_rpc(recent["rpc"].max(), 5.0)
                html(f'<span class="rs-note">Your actual best efforts in the last 90 days ({len(recent)}): '
                     f'median {fmt_time(med)}, best {fmt_time(best)} (5K-equivalent). The estimate leans on your '
                     f'long-run level, so after a quick improvement it can sit below your recent bests.</span>')
            html('<span class="rs-note">Likely range: 8 in 10 test runners\' best efforts landed inside it. '
                 'Estimates describe best efforts within any run, not race results.</span>')
        elif b["mode"] == "full":
            html(f'<div class="rs-msg">{now.get("reason", "No estimate available yet.")}</div>')
        else:
            html('<span class="rs-label">Current fitness · best-effort estimate</span>')
            html(f'<div class="rs-msg"><b>No fitness estimate for this data.</b><br>{P.NO_ESTIMATE_REASON}</div>')
        if now.get("days_since_last_run", 0) > STALE_AFTER_DAYS:
            html(f'<div class="rs-warn">Your latest run was {now["days_since_last_run"]} days ago '
                 f'({fmt_date(r.last_day)}). Everything here describes you as of then, not today.</div>')

    with st.container(key="blue_trend"):
        if b["mode"] == "full":
            html('<div style="display:flex;justify-content:space-between;flex-wrap:wrap"><span class="rs-h3">Fitness trend</span>'
                 '<span class="rs-note">band = likely range · dots = your best efforts · shaded = ACWR danger weeks · '
                 'orange = selected run · drag to zoom, ↺ resets</span></div>')
            danger = [d for d, z in zip(b["load"]["date"], b["load"]["zone"]) if z == "Danger"]
            sel = None
            if run is not None:
                rd = b["runday"]
                hit = rd[rd["file"] == run["file"]]
                sel = {"date": pd.Timestamp(run["calendar_date"]),
                       "rpc": float(hit["rpc_est"].iloc[0]) if len(hit) and pd.notna(hit["rpc_est"].iloc[0]) else None}
            efforts = r.perf.rename(columns={"date": "date"})[["date", "rpc"]]
            line = trend_90(b["trend"])
            if line:
                html(line)
            if len(b["trend"]):
                show_chart(C.fitness_trend(b["trend"], efforts, danger, sel, units=TREND_UNITS))
            else:
                html('<div class="rs-msg">Not enough history yet for a trend.</div>')
        else:
            html('<div style="display:flex;justify-content:space-between;flex-wrap:wrap"><span class="rs-h3">Training load trend</span>'
                 '<span class="rs-note">bars = km in the last 7 days (coloured by ACWR zone) · line = distance ACWR · '
                 'drag to zoom, ↺ resets</span></div>')
            sel_date = pd.Timestamp(run["calendar_date"]) if run is not None else None
            if len(b["load"]):
                show_chart(C.load_chart(b["load"], sel_date))
            missing = int(b["load"]["acwr_km"].isna().sum())
            if missing:
                html(f'<span class="rs-note">ACWR not shown for {missing} of {len(b["load"])} weeks: it needs at '
                     f'least 4 run days in the previous 28.</span>')

    a, w = st.columns(2)
    with a:
        with st.container(key="blue_acwr"):
            acwr_card(now.get("load", {}))
    with w:
        with st.container(key="blue_whatif"):
            what_if_card(b, model)


def trend_90(trend: pd.DataFrame) -> str:
    """'Last 90 days' line: latest weekly estimate vs the one at or before 90 days earlier (as a 5K time)."""
    if trend is None or len(trend) < 2:
        return ""
    t = trend.sort_values("date")
    last = t.iloc[-1]
    prior = t[t["date"] <= last["date"] - pd.Timedelta(days=90)]
    if prior.empty:
        return ""
    then = prior.iloc[-1]
    a, z = time_from_rpc(then["rpc"], 5.0), time_from_rpc(last["rpc"], 5.0)
    d = z - a
    if abs(d) < NOISE_S:
        msg, col = f"no real change (within ±{NOISE_S} s)", "#5A6270"
    elif d < 0:
        msg, col = f"{fmt_time(-d)} faster", "#1E8E5A"
    else:
        msg, col = f"{fmt_time(d)} slower", "#C2410C"
    return (f'<div class="rs-note" style="font-size:14px;margin:2px 0 4px 0">Last 90 days: 5K estimate '
            f'<b style="color:{col}">{msg}</b> ({fmt_time(a)} on {fmt_date(then["date"])} → {fmt_time(z)})</div>')


ZONES_UI = [("Under-training", "under 0.8", "#B7C6DB", 0.0, 0.8), ("Sweet spot", "0.8–1.3", "#7FC3A0", 0.8, 1.3),
            ("Monitoring", "1.3–1.5", "#F2C46B", 1.3, 1.5), ("Danger", "1.5+", "#E59464", 1.5, 2.0)]


def acwr_card(ld: dict):
    html('<span class="rs-label">Workload (ACWR) <span class="rs-tip" title="Acute:chronic workload ratio. '
         'Distance in the last 7 days divided by your average week over the last 28 days. Above 1.5 means a '
         'sudden jump in training, linked in published research to higher injury risk.">ⓘ</span></span>')
    v, zone = ld.get("acwr_km"), ld.get("zone")
    scale = 2.0
    if v is not None and pd.notna(v):
        html(f'<div style="display:flex;align-items:baseline;gap:8px"><span class="rs-big" style="font-size:36px">{v:.2f}</span>'
             f'<span style="font-weight:600;color:{ZONE_TEXT_COLOR.get(zone, "#15181D")}">{zone}</span></div>')
        pos = min(max(v, 0), scale) / scale * 100
        segs = "".join(f'<div style="flex:{(b - a)};background:{c}"></div>' for _, _, c, a, b in ZONES_UI)
        ticks = "".join(f'<span class="rs-tick" style="left:{t / scale * 100:.1f}%">{t:g}</span>' for t in (0.8, 1.3, 1.5))
        html('<div style="position:relative;margin:4px 0 2px 0"><div class="rs-zonebar2">' + segs + '</div>'
             f'<div class="rs-marker" style="left:calc({pos:.1f}% - 2px)"></div></div>'
             f'<div class="rs-ticks">{ticks}</div>')
    else:
        html('<div class="rs-msg" style="font-size:14px">Not enough runs to calculate: ACWR needs at least '
             '4 run days in the last 28.</div>')
    items = "".join(
        f'<span class="rs-zitem{" on" if name == zone else ""}"><span class="rs-dot" style="background:{c}"></span>'
        f'{name} <span class="rs-zr">{rng}</span></span>' for name, rng, c, _, _ in ZONES_UI)
    html(f'<div class="rs-zlegend">{items}</div>')
    km7, km28 = ld.get("km_7"), ld.get("km_28")
    if km7 is not None and pd.notna(km7):
        html('<div class="rs-two"><div><div class="rs-k">Last 7 days</div>'
             f'<div class="rs-mid">{km7:.1f} km</div></div><div><div class="rs-k">28-day weekly average</div>'
             f'<div class="rs-mid">{km28 / 4:.1f} km</div></div></div>')
        hr_acwr = ld.get("acwr_hr", np.nan)
        if hr_acwr is not None and pd.notna(hr_acwr):
            html(f'<span class="rs-note">Heart-rate ACWR (training load instead of distance): {hr_acwr:.2f}</span>')
    html('<span class="rs-note">ACWR = last 7 days ÷ average week of the last 28 days.</span>')


SENS_PCTS = list(range(-30, 31, 5))
NOISE_S = 10          # differences under this many seconds are treated as no meaningful change


def what_if_card(b: dict, model):
    """Sensitivity panel: how the model's estimate responds to training volume (not advice, not causal)."""
    html('<span class="rs-label">How the estimate responds</span>')
    html('<span class="rs-note" style="margin-top:0;display:block">How the model\'s estimate reacts to your training inputs, '
         'not training advice.</span>')
    if b["mode"] != "full" or not b["now"].get("ok"):
        html('<div class="rs-msg" style="font-size:14px">Needs a fitness estimate (full data with activity files).</div>')
        return
    load = st.slider("Training volume", -30, 30, 0, step=5, format="%d%%", key="wi_load")
    rpc = model.volume_curve(b["f_now"], SENS_PCTS)
    secs = [time_from_rpc(x, 5.0) for x in rpc]
    t0, t1 = secs[SENS_PCTS.index(0)], secs[SENS_PCTS.index(int(load))]
    d = t1 - t0
    if load == 0:
        res = "<b>your current estimate</b>"
    elif abs(d) < NOISE_S:
        res = (f"<b>≈ no meaningful change</b> (differences under {NOISE_S} s are within the model's noise)")
    else:
        res = f"<b>≈ {abs(d):.0f} s {'faster' if d < 0 else 'slower'}</b>"
    html(f'<span style="font-size:15px">5K: {res} · {fmt_time(t0)} → {fmt_time(t1)}</span>')
    show_chart(C.volume_response(SENS_PCTS, secs, load), ["resetScale2d"])
    html('<span class="rs-note">Line = the model\'s 5K estimate across −30% to +30% volume (faster is up); '
         f'shaded = within ±{NOISE_S} s of now; dot = slider. Volume scales acute and chronic load and the longest '
         'run together (ACWR unchanged); past performance is held fixed. Model sensitivity, not cause and effect.</span>')


# ============================================================
# THIS RUN (orange)
# ============================================================

TIP_TRIMP = ("Training load (TRIMP): heart-rate-weighted minutes. Each minute counts by its heart-rate zone: "
             "Z1 x0.2, Z2 x0.4, Z3 x0.6, Z4 x1.0, Z5 x1.5, so longer and harder runs score more. "
             "Daily training load feeds your acute (7-day) and chronic (28-day) load.")
TIP_ELEV = "Elevation gain: total climbing during the run (all uphill metres added up; downhills not subtracted)."
TIP_GAP = ("GAP, grade-adjusted pace: your pace with climbing counted as extra distance (100 m up = 1 km), "
           "so hilly runs compare fairly with flat ones.")


def trimp_context(r, value) -> tuple[str, int]:
    """Where this run's training load sits among the runner's own runs with a training load."""
    all_t = r.runs["trimp"].dropna().to_numpy(float)
    if value is None or pd.isna(value) or len(all_t) < 5:
        return ("needs heart-rate data" if value is None or pd.isna(value) else ""), len(all_t)
    p = float((all_t < value).mean() * 100)                 # share of the runner's runs easier than this one
    if p < 40:
        return f"easier than {int(round((100 - p) / 10) * 10)}% of your runs", len(all_t)
    if p <= 60:
        return "about your typical run", len(all_t)
    if p < 90:
        return f"harder than {int(round(p / 10) * 10)}% of your runs", len(all_t)
    return "one of your hardest 10%", len(all_t)


def run_view(steps, track, hidden) -> dict:
    """Linked map + chart + slider. Smooth browser component (Leaflet); Plotly figure as the fallback."""
    try:
        import streamlit.components.v1 as components
        from core import run_view as RV
        page, height, leg = RV.build_html(steps, track, hidden)
        components.html(page, height=height)
        return leg
    except Exception:                                            # noqa: BLE001  fallback keeps the app working
        fig, leg = C.run_scrubber(steps, track, hidden)
        show_chart(fig, RUN_FIGURE_BUTTONS if leg else ["resetScale2d"])
        return leg


def this_run_column(b: dict, run, privacy: bool = False):
    if run is None:
        html('<div class="rs-sec orange">THIS RUN</div>')
        return
    name = str(run.get("activity_name") or "Run").strip() or "Run"
    html(f'<div class="rs-sec orange">THIS RUN · {fmt_date(run["datetime_utc"]).upper()} · {name.upper()}</div>')
    r, ad, f = b["runner"], b["ad"], run["file"]
    s = RA.run_summary(r, f)
    with st.container(key="orange_run"):
        tl_ctx, tl_n = trimp_context(r, s["trimp"])
        stats = [("Distance", f"{s['distance_km']:.2f} km", "", ""),
                 ("Moving time", fmt_time(s["moving_time_s"]), "", ""),
                 ("Avg pace", fmt_pace(s["pace_s_per_km"]), "", ""),
                 ("GAP", fmt_pace(s["gap_s_per_km"]), "", TIP_GAP),
                 ("Avg HR", fmt_num(s["avg_hr"], suffix=" bpm"), "", ""),
                 ("Max HR", fmt_num(s["max_hr"], suffix=" bpm"), "", ""),
                 ("Elevation gain", fmt_num(s["elevation_gain_m"], suffix=" m") + (" *" if s["elevation_imputed"] else ""),
                  "", TIP_ELEV),
                 ('Training load <span class="rs-k" style="font-size:12px">(TRIMP)</span>',
                  fmt_num(s["trimp"]) + (" *" if s["trimp_estimated"] else ""), tl_ctx,
                  TIP_TRIMP + (f" Compared with your {tl_n} runs that have a training load." if tl_n else
                               " Needs heart-rate data."))]
        cells = []
        for k, v, sub, tip in stats:
            tip_html = f' <span class="rs-tip" title="{esc(tip)}">ⓘ</span>' if tip else ""
            sub_html = f'<div class="rs-sub">{sub}</div>' if sub else ""
            cells.append(f'<div><div class="rs-k">{k}{tip_html}</div><div class="rs-mid">{v}</div>{sub_html}</div>')
        html('<div class="rs-grid4">' + "".join(cells) + "</div>")
        notes = []
        if s["elevation_imputed"]:
            notes.append("elevation filled from your earlier runs (missing or unreliable)")
        if s["trimp_estimated"]:
            notes.append("TRIMP estimated from average heart rate")
        if s["hrmax_is_default"] and pd.notna(s["avg_hr"]):
            notes.append("HR zones use a default max HR of 181 bpm until a run with heart-rate data")
        if notes:
            html('<span class="rs-note">* ' + "; ".join(notes) + ".</span>")

        if not s["has_stream"]:
            html('<div class="rs-msg" style="font-size:14px">Route, pace chart, zones, splits and best efforts need '
                 'this run\'s activity file (FIT, GPX or TCX). This data only has the activity list.</div>')
        else:
            steps = RA.scrub_steps(ad, f, privacy)
            track = RA.route_track(ad, f, privacy)
            if not s["has_gps"] or track.empty:
                html('<div class="rs-mapnote">No GPS in this run, so there is no route map. The chart and slider '
                     'below still work.</div>')
            elif not track["visible"].any():
                html('<div class="rs-mapnote">Route hidden for privacy: this run never goes more than 500 m from its '
                     'start or finish (for example laps near home or on a track). The chart and slider below still work.</div>')
            if len(steps):
                leg = run_view(steps, track, RA.hidden_ranges(steps))
                if leg:
                    html('<div class="rs-paceleg"><span class="rs-paceleg-t">Route colour = pace</span>'
                         f'<span>slower {fmt_pace(leg.get("slow"))}</span><span class="rs-legend"></span>'
                         f'<span>{fmt_pace(leg.get("fast"))} faster</span></div>')
                bits = ["Drag the slider, click the chart or press ▶ to move through the run",
                        "zoom the map with + / −, ↺ resets"]
                if leg:
                    bits.append(("500 m around start and finish hidden · " if privacy else "")
                                + "map © OpenStreetMap contributors")
                html('<span class="rs-note">' + " · ".join(bits) + "</span>")
            zones = s["zone_minutes"]
            if zones:
                tot = sum(zones.values())
                bar = "".join(f'<div style="flex:{m};background:{c}"></div>'
                              for m, c in zip(zones.values(), HR_ZONE_COLORS) if m > 0)
                lab = " · ".join(f"{k.split()[0]} {m / tot:.0%}" for k, m in zones.items())
                html(f'<div class="rs-hrbar">{bar}</div>')
                html(f'<span class="rs-range">HR zones (% of moving time): {lab}</span>')
            else:
                html('<span class="rs-note">No heart-rate zones for this run (no heart-rate recording).</span>')
            sp = RA.km_splits(ad, f)
            if len(sp):
                html('<span class="rs-h3">Splits</span>')
                show_table(pd.DataFrame({
                    "km": [str(k) for k in sp["km"]],
                    "Time": [fmt_time(t) for t in sp["time_s"]],
                    "Pace": [fmt_pace(p) for p in sp["pace_s_per_km"]],
                    "Avg HR": [fmt_num(h) for h in sp["avg_hr"]],
                    "Elev. change": [fmt_num(e, "{:+.0f}", " m") for e in sp["elev_change_m"]]}))
            be = RA.best_efforts(r, ad, f)
            if len(be):
                html('<span class="rs-h3">Best efforts in this run</span>')
                show_table(pd.DataFrame({
                    "Distance": be["distance"], "Time": [fmt_time(t) for t in be["seconds"]],
                    "Pace": [fmt_pace(p) for p in be["pace_s_per_km"]],
                    "Used by the model": [f"✓ RPC {x:.3f}" if m else "" for m, x in zip(be["model_target"], be["rpc"])]}))

        # fitness estimate on this day
        if b["mode"] == "full":
            rd = b["runday"]
            hit = rd[rd["file"] == f]
            if len(hit) and pd.notna(hit["rpc_est"].iloc[0]):
                e = hit.iloc[0]
                txt = (f'Fitness estimate on this day: <b>5K {fmt_time(time_from_rpc(e["rpc_est"], 5.0))}</b> '
                       f'({fmt_time(time_from_rpc(e["rpc_est_hi"], 5.0))} – {fmt_time(time_from_rpc(e["rpc_est_lo"], 5.0))})')
                if pd.notna(e.get("rpc_actual_day")):
                    txt += (f'<br>Best effort that day: 5K-equivalent {fmt_time(time_from_rpc(e["rpc_actual_day"], 5.0))} '
                            f'(RPC {e["rpc_actual_day"]:.3f})')
                html(f'<div class="rs-box">{txt}</div>')
            else:
                html('<div class="rs-box">No fitness estimate for this day yet (needs 55 days of history and an '
                     'earlier best effort).</div>')


# ============================================================
# PAGE
# ============================================================

def footer(b: dict | None):
    rng = get_model().range or {}
    lo, hi = rng.get("time_range_pct", {}).get("faster_by"), rng.get("time_range_pct", {}).get("slower_by")
    lines = [
        "<b>Model.</b> XGBoost (mean of 3 seeds) trained on GoldenCheetah OpenData (329 runners); test error "
        "MAE 0.038 RPC (about 8% of finish time). Inputs: the 28 days before the date (training load, heart-rate "
        "intensity, grade-adjusted speed) and earlier best efforts. "
        + (f"Likely range: 80% of test runners' best efforts fell within about {lo:.0f}% faster to {hi:.0f}% slower "
           "than the estimate." if lo else ""),
        "<b>What it is.</b> An estimate of the best effort you would produce within a run now, converted to "
        "5K–half-marathon times with Riegel's formula. Not a race prediction; not medical advice. ACWR alert "
        "thresholds follow published workload guidance.",
        "<b>Privacy.</b> Files are processed in memory and not stored; GPS is used only to draw the route map. "
        "Demo routes are shown in full (public data). For your own upload you choose whether to hide 500 m around "
        "the start and finish. Map tiles are loaded from OpenStreetMap (© OpenStreetMap contributors, shown in grey) and the map library (Leaflet) from unpkg.",
        "<b>Demo data.</b> Strava export of Justin Ho Junting (github.com/justinhojt/strava-viz, MIT License) · "
        "\"Strava Running Activity Data\" by ajitjadhav1 on Kaggle (CC0 1.0).",
    ]
    if FOOTER_CREDIT:
        lines.append(FOOTER_CREDIT)
    html('<div class="rs-footer">' + "<br>".join(lines) + "</div>")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    html(CSS)
    html(f'<div><span class="rs-title">{APP_TITLE}</span><span class="rs-sub">{APP_SUBTITLE}</span></div>')
    bundle, run, privacy = control_bar()
    if bundle is not None:
        left, right = st.columns(2, gap="large")
        with left:
            you_now_column(bundle, run)
        with right:
            this_run_column(bundle, run, privacy)
    footer(bundle)


if __name__ == "__main__":
    main()
