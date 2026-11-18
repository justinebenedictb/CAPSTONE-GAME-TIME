"""
deployment/core/run_view.py  -  Stage 9, smooth linked run view (option B)

Builds one self-contained HTML component for the "This run" panel:
    - route map (Leaflet, OpenStreetMap basemap shown in greyscale; plain grey if tiles fail), route coloured by pace (relative to the run)
    - pace / heart-rate / elevation chart
    - distance slider with play/pause; a marker GLIDES along the route (no redraw), the chart
      cursor and the "At X km" readout follow it; click or drag on the chart to scrub
    - + / - / reset zoom on the map; mouse wheel left for page scrolling

Privacy: only what the Python side already allows is embedded. Route points come from
run_analytics.route_track (visible points only) and marker positions from run_analytics.scrub_steps
(None inside a privacy zone). Nothing else about location is sent to the browser.
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

PACE_SLOW = (133, 183, 235)      # light blue
PACE_FAST = (216, 90, 48)        # orange
N_PACE_BINS = 6
MAX_ROUTE_POINTS = 1500


def _mix(t: float) -> str:
    return "rgb(" + ",".join(str(round(a + (b - a) * t)) for a, b in zip(PACE_SLOW, PACE_FAST)) + ")"


def _num(v, nd=2):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(f) else round(f, nd)


def build_payload(steps: pd.DataFrame, track: pd.DataFrame, hidden: list) -> dict:
    vis = track[track["visible"]] if len(track) else track
    has_map = len(vis) >= 2
    legend, route = {}, []
    if has_map:
        p = vis["pace_s_per_km"].to_numpy(float)
        ok = np.isfinite(p)
        lo, hi = (float(np.nanpercentile(p[ok], 5)), float(np.nanpercentile(p[ok], 95))) if ok.sum() > 2 else (np.nan, np.nan)
        legend = {"fast": lo, "slow": hi}
        tr = track.reset_index(drop=True)
        stride = max(1, int(np.ceil(len(tr) / MAX_ROUTE_POINTS)))
        tr = tr.iloc[::stride].reset_index(drop=True)
        v = tr["visible"].to_numpy(bool)
        pc = tr["pace_s_per_km"].to_numpy(float)
        la, lo_ = tr["lat"].to_numpy(float), tr["lon"].to_numpy(float)
        bins = {b: [] for b in range(N_PACE_BINS)}
        for i in range(len(tr) - 1):
            if not (v[i] and v[i + 1]):
                continue                                   # never draw across a hidden stretch
            frac = 0.5 if not (np.isfinite(pc[i]) and np.isfinite(lo) and hi > lo) else (hi - pc[i]) / (hi - lo)
            b = int(np.clip(frac, 0, 0.999) * N_PACE_BINS)
            bins[b].append([[round(la[i], 6), round(lo_[i], 6)], [round(la[i + 1], 6), round(lo_[i + 1], 6)]])
        route = [{"c": _mix((b + 0.5) / N_PACE_BINS), "s": segs} for b, segs in bins.items() if segs]
    st = [{"k": _num(r["km"], 3), "t": _num(r["moving_s"], 1), "p": _num(r["pace_s_per_km"], 1),
           "h": _num(r["hr"], 1), "a": _num(r["alt"], 1),
           "la": _num(r["lat"], 6) if r["lat"] is not None else None,
           "lo": _num(r["lon"], 6) if r["lon"] is not None else None,
           "x": bool(r["hidden"]) and has_map} for _, r in steps.iterrows()]
    return {"has_map": has_map, "route": route, "steps": st,
            "hidden": [[round(a, 3), round(b, 3)] for a, b in (hidden if has_map else [])], "legend": legend}


def build_html(steps: pd.DataFrame, track: pd.DataFrame, hidden: list) -> tuple[str, int, dict]:
    """Returns (html, iframe height in px, pace legend {'fast','slow'} or {})."""
    payload = build_payload(steps, track, hidden)
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    height = 630 if payload["has_map"] else 300
    return _TEMPLATE.replace("__DATA__", data), height, payload["legend"]


_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html,body{margin:0;padding:0;background:transparent;font-family:Inter,system-ui,-apple-system,"Segoe UI",sans-serif;color:#15181D}
  #wrap{position:relative}
  #map{height:330px;border-radius:10px;background:#EEF0F2}
  .leaflet-tile-pane{filter:grayscale(1) contrast(.9) brightness(1.04)}
  #readout{position:absolute;left:10px;top:10px;z-index:1000;background:rgba(255,255,255,.95);border:1px solid #F3D3BD;
    border-radius:8px;padding:7px 11px;font-size:14px;line-height:1.5;pointer-events:none}
  .leg{display:flex;gap:14px;font-size:13px;margin:8px 2px 2px}
  #chartbox{position:relative;background:#fff;border-radius:8px;margin-top:2px}
  #chart{display:block;width:100%;height:170px;cursor:pointer;touch-action:none}
  .ctl{display:flex;align-items:center;gap:10px;margin-top:10px}
  #play{width:38px;height:30px;border:1px solid #E7B99A;border-radius:8px;background:#fff;font-size:13px;cursor:pointer;color:#15181D}
  #sl{flex:1;accent-color:#F08A4B}
  #kml{font-size:14px;min-width:62px;text-align:right}
  .reset{background:#fff;width:30px;height:30px;line-height:30px;text-align:center;font-size:15px;cursor:pointer;color:#15181D}
  .nomap #readout{position:static;display:inline-block;margin:0 0 6px 0}
</style></head>
<body><div id="wrap">
<div id="mapbox" style="position:relative"><div id="map"></div><div id="readout"></div></div>
<div class="leg"><span style="color:#1D5FBF">━ Pace</span><span style="color:#C2410C" id="hrleg">━ Heart rate</span>
<span style="color:#9AA3AF">▬ Elevation</span><span style="color:#5A6270" id="hidleg"></span></div>
<div id="chartbox"><svg id="chart"></svg></div>
<div class="ctl"><button id="play" aria-label="Play">▶</button>
<input type="range" id="sl" min="0" max="1" step="any" value="0"><span id="kml"></span></div>
</div>
<script type="application/json" id="rs-data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById('rs-data').textContent);
const S = D.steps, N = S.length, TOT = S[N-1].k;
const fmt = s => { if (s == null || !isFinite(s)) return '-'; s = Math.round(s);
  const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), x = s%60;
  return h ? `${h}:${String(m).padStart(2,'0')}:${String(x).padStart(2,'0')}` : `${m}:${String(x).padStart(2,'0')}`; };
const lerp = (a, b, f) => (a == null || b == null) ? (f < .5 ? a : b) : a + (b - a) * f;
const sl = document.getElementById('sl'); sl.max = N - 1;
let map = null, marker = null;
if (D.has_map && window.L) {
  map = L.map('map', {zoomControl:false, scrollWheelZoom:false, attributionControl:true});
  // OpenStreetMap standard tiles (no key), shown in greyscale by CSS. If tiles fail (offline, blocked),
  // the layer is removed and the route stays on the plain grey map background.
  const tiles = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {maxZoom:19,
    referrerPolicy:'strict-origin-when-cross-origin',
    attribution:'&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors'}).addTo(map);
  let tileErrors = 0;
  tiles.on('tileerror', () => { if (++tileErrors >= 4 && map.hasLayer(tiles)) map.removeLayer(tiles); });
  L.control.zoom({position:'topright'}).addTo(map);
  const Reset = L.Control.extend({options:{position:'topright'}, onAdd(){ const d = L.DomUtil.create('div','leaflet-bar');
    const a = L.DomUtil.create('a','reset',d); a.href='#'; a.title='Reset map view'; a.innerHTML='&#8634;';
    L.DomEvent.on(a,'click',e=>{L.DomEvent.preventDefault(e); fit();}); return d; }});
  new Reset().addTo(map);
  const grp = L.featureGroup(D.route.map(r => L.polyline(r.s, {color:r.c, weight:6, opacity:.95, lineCap:'round'}))).addTo(map);
  const BOUNDS = grp.getBounds();                 // from the drawn (visible) route only
  const fit = () => { if (BOUNDS.isValid()) map.fitBounds(BOUNDS, {padding:[24,24]}); else map.setView([0, 0], 1); }; fit();
  marker = L.circleMarker([0,0], {radius:7, color:'#FFFFFF', weight:3, fillColor:'#15181D', fillOpacity:1});
} else {
  document.getElementById('wrap').classList.add('nomap');
  document.getElementById('map').style.display = 'none';
  document.getElementById('chartbox').before(document.getElementById('readout'));
}
if (D.hidden.length) document.getElementById('hidleg').innerHTML = '<span style="color:#C9CED6">▬</span> hidden on map';
if (!S.some(s => s.h != null)) document.getElementById('hrleg').style.display = 'none';

// ---- chart (drawn in pixels; redrawn on resize) ----
const svg = document.getElementById('chart'), NS = 'http://www.w3.org/2000/svg';
const q = (arr, p) => { const a = arr.filter(v => v != null).sort((x,y)=>x-y); return a.length ? a[Math.min(a.length-1, Math.max(0, Math.round(p*(a.length-1))))] : null; };
const paces = S.map(s => s.p), hrs = S.map(s => s.h), alts = S.map(s => s.a);
let pLo = q(paces,.02), pHi = q(paces,.98); if (pLo != null) { const pad = Math.max((pHi-pLo)*.1, 6); pLo -= pad; pHi += pad; }
const hLo = q(hrs,0), hHi = q(hrs,1), aLo = q(alts,0), aHi = q(alts,1);
let W = 600, H = 170, cursor = null;
function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent||svg).appendChild(e); return e; }
function draw() {
  W = svg.clientWidth || 600; svg.setAttribute('viewBox', `0 0 ${W} ${H}`); svg.innerHTML = '';
  const X = k => 4 + (W - 8) * k / TOT;
  D.hidden.forEach(([a,b]) => { el('rect', {x:X(a), y:0, width:Math.max(2, X(b)-X(a)), height:H, fill:'#C9CED6', opacity:.45});
    if (X(b) - X(a) > 70) { const t = el('text', {x:(X(a)+X(b))/2, y:H-16, 'font-size':12, fill:'#5A6270', 'text-anchor':'middle'}); t.textContent = 'hidden on map'; } });
  if (aLo != null) { const span = Math.max(aHi - aLo, 5), Y = a => H - 4 - (a - aLo) / span * (H * .3);
    let d = `M${X(0)} ${H}`; S.forEach(s => { if (s.a != null) d += ` L${X(s.k)} ${Y(s.a)}`; }); d += ` L${X(TOT)} ${H} Z`;
    el('path', {d, fill:'rgba(154,163,175,.28)', stroke:'#9AA3AF', 'stroke-width':1}); }
  const line = (key, Y, color) => { let d = '', pen = false; S.forEach(s => { const v = s[key];
      if (v == null) { pen = false; return; } d += (pen ? ' L' : ' M') + X(s.k) + ' ' + Y(v); pen = true; });
    el('path', {d, fill:'none', stroke:color, 'stroke-width':2}); };
  if (pLo != null) { const Y = p => 10 + (Math.min(Math.max(p, pLo), pHi) - pLo) / (pHi - pLo) * (H * .62); line('p', Y, '#1D5FBF');
    [pLo + (pHi-pLo)*.15, pHi - (pHi-pLo)*.15].forEach(v => { const t = el('text', {x:6, y:Y(v)+4, 'font-size':12, fill:'#5A6270'}); t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 3); t.textContent = fmt(v) + ' /km'; }); }
  if (hLo != null) { const span = Math.max(hHi - hLo, 5), Y = h => 10 + (1 - (h - hLo) / span) * (H * .62); line('h', Y, '#C2410C');
    [hLo, hHi].forEach(v => { const t = el('text', {x:W-6, y:Y(v)+4, 'font-size':12, fill:'#5A6270', 'text-anchor':'end'}); t.setAttribute('paint-order', 'stroke'); t.setAttribute('stroke', '#fff'); t.setAttribute('stroke-width', 3); t.textContent = Math.round(v) + ' bpm'; }); }
  const step = TOT > 20 ? 5 : TOT > 8 ? 2 : 1;
  for (let k = step; k < TOT; k += step) { const t = el('text', {x:X(k), y:H-3, 'font-size':12, fill:'#8A93A0', 'text-anchor':'middle'}); t.textContent = k + ' km'; }
  cursor = el('line', {x1:0, x2:0, y1:0, y2:H, stroke:'#F08A4B', 'stroke-width':2, 'stroke-dasharray':'4 3'});
  update();
}
// ---- linked update ----
function update() {
  const p = Math.min(Math.max(+sl.value, 0), N - 1), i = Math.min(Math.floor(p), N - 2 < 0 ? 0 : N - 2), f = N > 1 ? p - i : 0;
  const a = S[i], b = S[Math.min(i + 1, N - 1)];
  const k = lerp(a.k, b.k, f), t = lerp(a.t, b.t, f), pc = lerp(a.p, b.p, f), h = lerp(a.h, b.h, f), al = lerp(a.a, b.a, f);
  const hidden = (f < .5 ? a.x : b.x);
  if (cursor) { const x = 4 + (W - 8) * k / TOT; cursor.setAttribute('x1', x); cursor.setAttribute('x2', x); }
  if (marker) {
    let ll = null;
    if (a.la != null && b.la != null) ll = [a.la + (b.la - a.la) * f, a.lo + (b.lo - a.lo) * f];
    else { const n = f < .5 ? a : b; if (n.la != null) ll = [n.la, n.lo]; }
    if (ll && !hidden) { marker.setLatLng(ll); if (!map.hasLayer(marker)) marker.addTo(map); }
    else if (map.hasLayer(marker)) map.removeLayer(marker);
  }
  const parts = [`pace ${fmt(pc)} /km`]; if (h != null) parts.push(`HR ${Math.round(h)}`); if (al != null) parts.push(`${Math.round(al)} m`);
  document.getElementById('readout').innerHTML = `<b>At ${k.toFixed(2)} km</b> · ${fmt(t)}<br>${parts.join(' · ')}` + (hidden ? '<br><i>position hidden (privacy zone)</i>' : '');
  document.getElementById('kml').textContent = k.toFixed(2) + ' km';
}
sl.addEventListener('input', update);
// click / drag on the chart to scrub
const scrub = e => { const r = svg.getBoundingClientRect(), k = Math.min(Math.max((e.clientX - r.left - 4) / (r.width - 8) * TOT, 0), TOT);
  let j = 0; while (j < N - 1 && S[j + 1].k < k) j++; const f = j < N - 1 ? (k - S[j].k) / Math.max(S[j + 1].k - S[j].k, 1e-9) : 0;
  sl.value = Math.min(j + f, N - 1); update(); };
let dragging = false;
svg.addEventListener('pointerdown', e => { dragging = true; svg.setPointerCapture(e.pointerId); scrub(e); });
svg.addEventListener('pointermove', e => { if (dragging) scrub(e); });
svg.addEventListener('pointerup', () => { dragging = false; });
// play / pause: glide through the whole run in 10-30 s
let raf = null, last = 0; const btn = document.getElementById('play'), dur = Math.min(30, Math.max(10, TOT * 2.0));
const setBtn = on => { btn.textContent = on ? '❚❚' : '▶'; btn.setAttribute('aria-label', on ? 'Pause' : 'Play'); };
function tick(ts) { if (!last) last = ts; const dt = (ts - last) / 1000; last = ts;
  let v = +sl.value + dt * (N - 1) / dur; if (v >= N - 1) { sl.value = N - 1; update(); raf = null; setBtn(false); return; }
  sl.value = v; update(); raf = requestAnimationFrame(tick); }
btn.addEventListener('click', () => { if (raf) { cancelAnimationFrame(raf); raf = null; setBtn(false); return; }
  if (+sl.value >= N - 1) sl.value = 0; last = 0; setBtn(true); raf = requestAnimationFrame(tick); });
window.addEventListener('resize', draw);
draw();
</script></body></html>
"""
