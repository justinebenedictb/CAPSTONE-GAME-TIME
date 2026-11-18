"""
deployment/core/charts.py  -  Stage 9, Step 6: chart builders for the RunState dashboard

Plotly figures only (no Streamlit here), styled with the C2 mockup colours.
    fitness_trend()   weekly fitness estimate (5K-equivalent time or RPC) with the likely-range band,
                      actual best-effort dots, ACWR danger weeks shaded, selected run marked
    load_chart()      summary mode: weekly distance bars + distance-ACWR line with zone lines
    pace_hr_chart()   pace and heart rate along the selected run
    volume_response() sensitivity panel: 5K estimate across training-volume changes, current setting marked
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from .features import time_from_rpc
except ImportError:
    from features import time_from_rpc

# C2 mockup palette
BLUE = "#1D5FBF"
BLUE_BAND = "#DCE7F7"
ORANGE = "#F08A4B"
RUST = "#C2410C"
DANGER_FILL = "#F6C9A8"
INK = "#15181D"
MUTED = "#5A6270"
GRID = "#D9DEE6"
ZONE_COLORS = {"Under-training": "#B7C6DB", "Sweet spot": "#7FC3A0", "Monitoring": "#F2C46B", "Danger": "#E59464"}
FONT = "Inter, system-ui, sans-serif"


def _base(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=13, color=INK)),
        modebar=dict(orientation="h", bgcolor="rgba(255,255,255,0.85)", color=MUTED, activecolor=ORANGE),
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, tickfont=dict(color=MUTED))
    return fig


def _mmss_ticks(lo_min: float, hi_min: float, n: int = 5):
    """Tick values (minutes) and m:ss / h:mm:ss labels for a time axis."""
    if not np.isfinite(lo_min) or not np.isfinite(hi_min) or hi_min <= lo_min:
        return None, None
    span = hi_min - lo_min
    steps = [0.25, 0.5, 1, 2, 5, 10, 15, 20, 30, 60, 120]
    step = next((s for s in steps if span / s <= n), steps[-1])
    start = np.floor(lo_min / step) * step
    vals = np.arange(start, hi_min + step, step)

    def lab(m):
        s = int(round(m * 60))
        h, mm, ss = s // 3600, (s % 3600) // 60, s % 60
        return f"{h}:{mm:02d}:{ss:02d}" if h else f"{mm}:{ss:02d}"
    return list(vals), [lab(v) for v in vals]


def fitness_trend(trend: pd.DataFrame, efforts: pd.DataFrame | None = None,
                  danger_weeks: list | None = None, selected: dict | None = None,
                  units: str = "5K time", height: int = 280) -> go.Figure:
    """trend: date, rpc, rpc_lo, rpc_hi  ·  efforts: date, rpc (actual best-effort days)
    danger_weeks: list of week-end dates whose distance ACWR was in the danger zone
    selected: {"date": ..., "rpc": ... or None}"""
    fig = go.Figure()
    as_time = units == "5K time"

    def conv(r):
        r = np.asarray(r, float)
        return time_from_rpc(r, 5.0) / 60 if as_time else r

    for wk in danger_weeks or []:
        wk = pd.Timestamp(wk)
        fig.add_vrect(x0=wk - pd.Timedelta(days=7), x1=wk, fillcolor=DANGER_FILL, opacity=0.55,
                      line_width=0, layer="below")
    if len(trend):
        x = list(trend["date"]) + list(trend["date"])[::-1]
        # faster end of the range = rpc_hi -> shorter time
        band = list(conv(trend["rpc_hi"])) + list(conv(trend["rpc_lo"]))[::-1]
        fig.add_trace(go.Scatter(x=x, y=band, fill="toself", fillcolor=BLUE_BAND, line=dict(width=0),
                                 hoverinfo="skip", name="Likely range (80%)"))
        hover = ("%{x|%b %d, %Y}<br>Estimate: %{customdata[0]}<br>RPC %{customdata[1]:.3f}<extra></extra>")
        fig.add_trace(go.Scatter(
            x=trend["date"], y=conv(trend["rpc"]), mode="lines", line=dict(color=BLUE, width=3),
            name="Fitness estimate", hovertemplate=hover,
            customdata=np.column_stack([[_fmt(t) for t in time_from_rpc(trend["rpc"].to_numpy(float), 5.0)],
                                        trend["rpc"].to_numpy(float)])))
    if efforts is not None and len(efforts):
        fig.add_trace(go.Scatter(
            x=efforts["date"], y=conv(efforts["rpc"]), mode="markers",
            marker=dict(size=6, color="rgba(21,24,29,0.45)", line=dict(width=0)),
            name="Your actual best efforts",
            customdata=np.column_stack([[_fmt(t) for t in time_from_rpc(efforts["rpc"].to_numpy(float), 5.0)],
                                        efforts["rpc"].to_numpy(float)]),
            hovertemplate="%{x|%b %d, %Y}<br>Best effort: %{customdata[0]} (5K-equivalent)<br>RPC %{customdata[1]:.3f}<extra></extra>"))
    if selected and selected.get("date") is not None:
        d = pd.Timestamp(selected["date"])
        if selected.get("rpc") is not None and np.isfinite(selected["rpc"]):
            fig.add_trace(go.Scatter(x=[d], y=conv([selected["rpc"]]), mode="markers", name="Selected run",
                                     marker=dict(size=13, color=ORANGE, line=dict(color="white", width=2)),
                                     hovertemplate="Selected run<extra></extra>"))
        else:
            fig.add_vline(x=d, line=dict(color=ORANGE, width=2, dash="dot"))
    _base(fig, height)
    if as_time:
        ys = [v for tr in fig.data for v in (tr.y if tr.y is not None else []) if v is not None and np.isfinite(v)]
        if ys:
            lo, hi = float(np.min(ys)), float(np.max(ys))
            pad = (hi - lo) * 0.06 or 0.5
            vals, labs = _mmss_ticks(lo - pad, hi + pad)
            fig.update_yaxes(autorange="reversed", range=[hi + pad, lo - pad], tickvals=vals, ticktext=labs,
                             title=dict(text="5K-equivalent (faster ↑)", font=dict(size=12, color=MUTED)))
    else:
        fig.update_yaxes(title=dict(text="RPC (higher = fitter)", font=dict(size=12, color=MUTED)))
    return fig


def load_chart(load: pd.DataFrame, selected_date=None, height: int = 280) -> go.Figure:
    """Summary mode: km in the last 7 days (bars, coloured by ACWR zone) + distance ACWR (line)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    colors = [ZONE_COLORS.get(z, "#CBD3DF") if isinstance(z, str) else "#CBD3DF" for z in load["zone"]]
    fig.add_trace(go.Bar(x=load["date"], y=load["km_7"], marker_color=colors, name="km in last 7 days",
                         hovertemplate="%{x|%b %d, %Y}<br>%{y:.1f} km in the last 7 days<extra></extra>"),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=load["date"], y=load["acwr_km"], mode="lines+markers", name="Distance ACWR",
                             line=dict(color=BLUE, width=2), marker=dict(size=4), connectgaps=False,
                             hovertemplate="ACWR %{y:.2f}<extra></extra>"), secondary_y=True)
    for v, c in [(0.8, "#8A97AB"), (1.3, "#4F9E77"), (1.5, RUST)]:
        fig.add_hline(y=v, line=dict(color=c, width=1, dash="dot"), secondary_y=True)
    if selected_date is not None:
        fig.add_vline(x=pd.Timestamp(selected_date), line=dict(color=ORANGE, width=2, dash="dot"))
    _base(fig, height)
    fig.update_yaxes(title=dict(text="km / 7 days", font=dict(size=12, color=MUTED)), secondary_y=False)
    fig.update_yaxes(title=dict(text="ACWR", font=dict(size=12, color=MUTED)), showgrid=False,
                     rangemode="tozero", secondary_y=True)
    fig.update_layout(bargap=0.15)
    return fig


def pace_hr_chart(series: pd.DataFrame, height: int = 230) -> go.Figure:
    """series: km, pace_s_per_km, hr, alt (from run_analytics.pace_hr_series)."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    pace_min = series["pace_s_per_km"] / 60
    fig.add_trace(go.Scatter(x=series["km"], y=pace_min, mode="lines", name="Pace", connectgaps=False,
                             line=dict(color=BLUE, width=2),
                             customdata=[_fmt(p) for p in series["pace_s_per_km"]],
                             hovertemplate="%{x:.2f} km · pace %{customdata} /km<extra></extra>"), secondary_y=False)
    has_hr = series["hr"].notna().sum() > 10
    if has_hr:
        fig.add_trace(go.Scatter(x=series["km"], y=series["hr"], mode="lines", name="Heart rate",
                                 line=dict(color=RUST, width=2), connectgaps=False,
                                 hovertemplate="HR %{y:.0f} bpm<extra></extra>"), secondary_y=True)
    _base(fig, height)
    p = pace_min.dropna()
    if len(p):
        lo, hi = float(p.quantile(0.02)), float(p.quantile(0.98))
        pad = max((hi - lo) * 0.1, 0.1)
        vals, labs = _mmss_ticks(lo - pad, hi + pad)
        fig.update_yaxes(range=[hi + pad, lo - pad], tickvals=vals, ticktext=labs,
                         title=dict(text="pace /km (faster ↑)", font=dict(size=12, color=MUTED)), secondary_y=False)
    if has_hr:
        fig.update_yaxes(title=dict(text="bpm", font=dict(size=12, color=MUTED)), showgrid=False, secondary_y=True)
    fig.update_xaxes(title=dict(text="km", font=dict(size=12, color=MUTED)), ticksuffix="")
    return fig


def _fmt(seconds) -> str:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(s):
        return "-"
    s = int(round(s))
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# ============================================================
# Pass 2: linked run figure (map + chart + slider), all in the browser
# ============================================================
PACE_SLOW = (133, 183, 235)      # light blue
PACE_FAST = (216, 90, 48)        # orange
N_PACE_BINS = 6
READOUT_BORDER = "#F3D3BD"
MAP_STYLE = "open-street-map"     # fallback only (no key needed; © OpenStreetMap contributors)
MAP_BOTTOM = 0.44                 # map fills the figure width from here to the top
CHART_TOP = 0.36                  # pace / HR / elevation chart below it
HIDDEN_FILL = "#C9CED6"


def _mix(t: float) -> str:
    return "rgb(" + ",".join(str(round(a + (b - a) * t)) for a, b in zip(PACE_SLOW, PACE_FAST)) + ")"


def _zoom(lat: np.ndarray, lon: np.ndarray) -> float:
    span = max(float(np.ptp(lat)), float(np.ptp(lon)) * np.cos(np.radians(float(np.mean(lat)))), 1e-4)
    return float(np.clip(np.log2(360 / span) - 1.6, 10, 16.5))


def _readout(r, show_hidden: bool = True) -> str:
    def t(s):
        return _fmt(s) if s is not None and np.isfinite(s) else "-"
    parts = [f"<b>At {r['km']:.2f} km</b> · {t(r['moving_s'])}"]
    line2 = [f"pace {t(r['pace_s_per_km'])} /km" if np.isfinite(r["pace_s_per_km"]) else "pace -",
             f"HR {r['hr']:.0f}" if np.isfinite(r["hr"]) else None,
             f"{r['alt']:.0f} m" if np.isfinite(r["alt"]) else None]
    parts.append(" · ".join(p for p in line2 if p))
    if show_hidden and r["hidden"]:
        parts.append("<i>position hidden (privacy zone)</i>")
    return "<br>".join(parts)


def run_scrubber(steps: pd.DataFrame, track: pd.DataFrame, hidden: list, height: int = 640) -> tuple[go.Figure, dict]:
    """One linked figure: route map coloured by pace (if any route can be shown), pace / HR / elevation
    chart, a distance slider with play button, and an 'At X km' readout. Returns (figure, legend info)."""
    vis = track[track["visible"]] if len(track) else track
    has_map = len(vis) >= 2
    if has_map:
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "map"}], [{"secondary_y": True}]],
                            row_heights=[0.6, 0.4], vertical_spacing=0.05)
        row_xy = dict(row=2, col=1)
    else:
        fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])
        row_xy = dict(row=1, col=1)
    legend = {}
    if not has_map:
        hidden = []                       # no route shown at all: the app explains why instead

    # ---- map: route coloured by pace (relative to this run), only visible stretches ----
    if has_map:
        p = vis["pace_s_per_km"].to_numpy(float)
        ok = np.isfinite(p)
        lo, hi = (np.nanpercentile(p[ok], 5), np.nanpercentile(p[ok], 95)) if ok.sum() > 2 else (np.nan, np.nan)
        legend = {"fast": lo, "slow": hi}
        tr = track.reset_index(drop=True)
        stride = max(1, int(np.ceil(len(tr) / 1500)))
        tr = tr.iloc[::stride].reset_index(drop=True)
        bins = {b: ([], []) for b in range(N_PACE_BINS)}
        v = tr["visible"].to_numpy(bool)
        pc = tr["pace_s_per_km"].to_numpy(float)
        for i in range(len(tr) - 1):
            if not (v[i] and v[i + 1]):
                continue                                       # never draw across a hidden stretch
            frac = 0.5 if not (np.isfinite(pc[i]) and np.isfinite(lo) and hi > lo) else (hi - pc[i]) / (hi - lo)
            b = int(np.clip(frac, 0, 0.999) * N_PACE_BINS)
            bins[b][0].extend([tr.at[i, "lat"], tr.at[i + 1, "lat"], None])
            bins[b][1].extend([tr.at[i, "lon"], tr.at[i + 1, "lon"], None])
        for b, (la, lo_) in bins.items():
            if la:
                fig.add_trace(go.Scattermap(lat=la, lon=lo_, mode="lines", line=dict(width=6, color=_mix((b + 0.5) / N_PACE_BINS)),
                                            hoverinfo="skip", showlegend=False), row=1, col=1)
        fig.update_layout(map=dict(style=MAP_STYLE, domain=dict(x=[0, 1], y=[MAP_BOTTOM, 1]), center=dict(lat=float(vis["lat"].mean()), lon=float(vis["lon"].mean())),
                                   zoom=_zoom(vis["lat"].to_numpy(float), vis["lon"].to_numpy(float))))

    # ---- chart: pace (y), HR (y2), elevation profile (y3, bottom third), cursor (y4) ----
    km = steps["km"]
    pace_min = steps["pace_s_per_km"] / 60
    fig.add_trace(go.Scatter(x=km, y=pace_min, mode="lines", line=dict(color=BLUE, width=2), name="Pace",
                             connectgaps=False, customdata=[_fmt(v) for v in steps["pace_s_per_km"]],
                             hovertemplate="%{x:.2f} km · pace %{customdata} /km<extra></extra>"), **row_xy, secondary_y=False)
    has_hr = steps["hr"].notna().sum() > 3
    if has_hr:
        fig.add_trace(go.Scatter(x=km, y=steps["hr"], mode="lines", line=dict(color=RUST, width=2), name="Heart rate",
                                 connectgaps=False, hovertemplate="HR %{y:.0f} bpm<extra></extra>"), **row_xy, secondary_y=True)
    alt = steps["alt"]
    if alt.notna().sum() > 3:
        a0, a1 = float(alt.min()), float(alt.max())
        span = max(a1 - a0, 5.0)
        fig.add_trace(go.Scatter(x=km, y=alt, mode="lines", fill="tozeroy", line=dict(color="#9AA3AF", width=1),
                                 fillcolor="rgba(154,163,175,0.25)", name="Elevation", yaxis="y3", xaxis="x",
                                 hovertemplate="%{y:.0f} m<extra></extra>"))
        fig.update_layout(yaxis3=dict(overlaying="y", anchor="x", visible=False, range=[a0 - 0.02 * span, a0 + span / 0.33]))
    for a, b in hidden:
        fig.add_shape(type="rect", xref="x", yref="y domain", x0=a, x1=max(b, a + 0.02), y0=0, y1=1,
                      fillcolor=HIDDEN_FILL, opacity=0.35, line_width=0, layer="below",
                      label=dict(text="hidden on map", textposition="top center", font=dict(size=10, color=MUTED)))

    # dynamic traces: cursor (+ marker on the map)
    k0 = 0
    fig.add_trace(go.Scatter(x=[km.iloc[k0]] * 2, y=[0, 1], mode="lines", yaxis="y4", xaxis="x",
                             line=dict(color=ORANGE, width=2, dash="dot"), hoverinfo="skip", showlegend=False))
    fig.update_layout(yaxis4=dict(overlaying="y", anchor="x", visible=False, range=[0, 1], fixedrange=True))
    dyn = [len(fig.data) - 1]
    if has_map:
        def mk(r, size, color):
            return go.Scattermap(lat=[r["lat"]] if r["lat"] is not None else [None],
                                 lon=[r["lon"]] if r["lon"] is not None else [None], mode="markers",
                                 marker=dict(size=size, color=color), hoverinfo="skip", showlegend=False)
        r0 = steps.iloc[k0]
        fig.add_trace(mk(r0, 17, "#FFFFFF"), row=1, col=1)
        fig.add_trace(mk(r0, 11, INK), row=1, col=1)
        dyn += [len(fig.data) - 2, len(fig.data) - 1]

    readout_pos = dict(xref="paper", yref="paper", x=0.01, y=0.995, xanchor="left", yanchor="top", showarrow=False,
                       align="left", bgcolor="rgba(255,255,255,0.95)", bordercolor=READOUT_BORDER, borderwidth=1,
                       borderpad=6, font=dict(size=12, color=INK))

    frames, slider_steps = [], []
    for i, r in steps.iterrows():
        data = [go.Scatter(x=[r["km"]] * 2, y=[0, 1])]
        if has_map:
            lat = [r["lat"]] if r["lat"] is not None else [None]
            lon = [r["lon"]] if r["lon"] is not None else [None]
            data += [go.Scattermap(lat=lat, lon=lon), go.Scattermap(lat=lat, lon=lon)]
        name = str(i)
        frames.append(go.Frame(name=name, data=data, traces=dyn,
                               layout=dict(annotations=[dict(text=_readout(r, has_map), **readout_pos)])))
        whole = abs(r["km"] - round(r["km"])) < (steps["km"].iloc[1] - steps["km"].iloc[0]) / 2 if len(steps) > 1 else True
        slider_steps.append(dict(method="animate", label=f"{round(r['km']):d}" if whole else "",
                                 args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                    transition=dict(duration=0))]))
    fig.frames = frames
    fig.update_layout(
        annotations=[dict(text=_readout(steps.iloc[k0], has_map), **readout_pos)],
        sliders=[dict(active=k0, steps=slider_steps, x=0.09, len=0.91, y=0, yanchor="top", pad=dict(t=34, b=4),
                      currentvalue=dict(visible=False), ticklen=3, tickcolor=MUTED, font=dict(size=10, color=MUTED),
                      bgcolor="#F3D3BD", activebgcolor=ORANGE, bordercolor="#E7B99A")],
        updatemenus=[dict(type="buttons", direction="left", showactive=False, x=0, y=0, xanchor="left", yanchor="top",
                          pad=dict(t=30, r=6), bgcolor="#FFFFFF", bordercolor="#E7B99A", font=dict(size=12, color=INK),
                          buttons=[dict(label="▶", method="animate",
                                        args=[None, dict(frame=dict(duration=90, redraw=True), transition=dict(duration=0),
                                                         fromcurrent=True, mode="immediate")]),
                                   dict(label="❚❚", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                                           transition=dict(duration=0))])])],
    )
    _base(fig, height)
    fig.update_layout(margin=dict(l=4, r=4, t=4, b=70), hovermode="x", uirevision="run")
    if has_map:   # full-width map; chart tick labels inside so no side margins are reserved
        fig.update_layout(xaxis=dict(domain=[0, 1]), yaxis=dict(domain=[0, CHART_TOP]),
                          legend=dict(orientation="h", x=0, xanchor="left", y=CHART_TOP + 0.005, yanchor="bottom"))
    else:
        fig.update_layout(xaxis=dict(domain=[0, 1]))
    p = pace_min.dropna()
    if len(p):
        lo, hi = float(p.quantile(0.02)), float(p.quantile(0.98))
        pad = max((hi - lo) * 0.1, 0.1)
        vals, labs = _mmss_ticks(lo - pad, hi + pad)
        fig.update_yaxes(range=[hi + pad, lo - pad], tickvals=vals, ticktext=labs, ticklabelposition="inside",
                         ticksuffix=" /km", **row_xy, secondary_y=False)
    if has_hr:
        fig.update_yaxes(showgrid=False, tickformat=".0f", ticksuffix=" bpm", ticklabelposition="inside",
                         **row_xy, secondary_y=True)
    fig.update_xaxes(title=dict(text="km into the run", font=dict(size=12, color=MUTED)), **row_xy)
    return fig, legend


def volume_response(pcts, seconds, current: float, height: int = 170) -> go.Figure:
    """5K estimate (m:ss) across training-volume changes; the dot marks the slider position."""
    pcts = np.asarray(pcts, float)
    mins = np.asarray(seconds, float) / 60
    base = float(mins[np.argmin(np.abs(pcts))])
    fig = go.Figure()
    fig.add_hrect(y0=base - 10 / 60, y1=base + 10 / 60, fillcolor=BLUE_BAND, opacity=0.6, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=pcts, y=mins, mode="lines+markers", line=dict(color=BLUE, width=2.5), marker=dict(size=4, color=BLUE),
                             customdata=[_fmt(m * 60) for m in mins],
                             hovertemplate="volume %{x:+.0f}%: 5K %{customdata}<extra></extra>", showlegend=False))
    cur = float(mins[np.argmin(np.abs(pcts - current))])
    fig.add_trace(go.Scatter(x=[current], y=[cur], mode="markers", marker=dict(size=11, color=ORANGE,
                             line=dict(color="white", width=2)), hoverinfo="skip", showlegend=False))
    fig = _base(fig, height)
    fig.update_layout(hovermode="closest", margin=dict(l=8, r=8, t=4, b=4))
    lo, hi = min(mins.min(), base - 10 / 60), max(mins.max(), base + 10 / 60)
    pad = max((hi - lo) * 0.15, 0.05)
    tv, tt = _mmss_ticks(lo - pad, hi + pad, 4)
    fig.update_yaxes(range=[hi + pad, lo - pad], tickvals=tv, ticktext=tt, title=None)   # faster = up
    fig.update_xaxes(ticksuffix="%", tickvals=[-30, -15, 0, 15, 30], range=[-32, 32], title=None)
    return fig
