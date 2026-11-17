"""
11_stage9_setup.py  -  Stage 9, Step 1: dashboard setup

What it does (safe to re-run; it never deletes anything):
  1. Checks that the dashboard libraries are installed and prints the install command if any are missing.
  2. Creates the deployment/ folder tree.
  3. Copies the deployed model (tuned XGBoost "full", seeds 11/22/33) and the
     Stage 4 preprocessing settings + feature list into deployment/models/.
     deployment/ must be self-contained so it can run on Streamlit Cloud.
  4. Checks the demo data:
       deployment/demo_data/justinhojt_strava_export.zip   (MIT)
       deployment/demo_data/kaggle/<xlsx>                  (CC0, downloaded by you)
  5. Writes pinned requirement files (reproducibility package):
       requirements.txt               full pipeline (Stages 1-9)
       deployment/requirements.txt    dashboard only (what Streamlit Cloud installs)
  6. Writes a setup report to logs/stage9_setup_report.txt

Run:  py "scripts\\11_stage9_setup.py"
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as md
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEP = ROOT / "deployment"

# pip name -> import name
PIPELINE_LIBS = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "scikit-learn": "sklearn",
    "xgboost": "xgboost", "tensorflow": "tensorflow", "matplotlib": "matplotlib",
    "seaborn": "seaborn", "statsmodels": "statsmodels", "openpyxl": "openpyxl",
    "requests": "requests", "tqdm": "tqdm", "pyarrow": "pyarrow",
}
DASHBOARD_LIBS = {
    "streamlit": "streamlit", "plotly": "plotly", "folium": "folium",
    "streamlit-folium": "streamlit_folium", "fitparse": "fitparse", "gpxpy": "gpxpy",
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "xgboost": "xgboost",
    "openpyxl": "openpyxl",
}
REQUIRED_FOR_DASHBOARD = ["streamlit", "plotly", "folium", "streamlit-folium",
                          "fitparse", "gpxpy", "numpy", "pandas", "scipy", "xgboost", "openpyxl"]

MODEL_FILES = [ROOT / "06_tuning" / "models" / f"xgb_full_seed{s}.json" for s in (11, 22, 33)]
STAGE4_FILES = [ROOT / "04_feature_selection" / "stage4_preprocessing_params.json",
                ROOT / "04_feature_selection" / "stage4_selected_features_tabular.txt"]

report: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    report.append(msg)


def version(pip_name: str, import_name: str) -> str | None:
    try:
        return md.version(pip_name)
    except md.PackageNotFoundError:
        try:
            mod = importlib.import_module(import_name)
            return getattr(mod, "__version__", None)
        except Exception:
            return None


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main() -> int:
    ok = True
    say(f"Stage 9 setup  |  {datetime.now():%Y-%m-%d %H:%M}")
    say(f"Python {platform.python_version()}  |  {platform.system()} {platform.release()}  |  project: {ROOT}")
    say("=" * 70)

    # 1. libraries
    say("\n[1] Libraries")
    missing = []
    for pip_name in REQUIRED_FOR_DASHBOARD:
        v = version(pip_name, DASHBOARD_LIBS[pip_name])
        say(f"  {'OK ' if v else 'MISSING'}  {pip_name:18s} {v or ''}")
        if not v:
            missing.append(pip_name)
    if missing:
        ok = False
        say("\n  Install the missing ones with:")
        say(f"    py -m pip install {' '.join(missing)}")

    # 2. folders
    say("\n[2] Folders")
    for sub in ["core", "models", "demo_data", "demo_data/kaggle", ".streamlit"]:
        (DEP / sub).mkdir(parents=True, exist_ok=True)
        say(f"  ok  deployment/{sub}/")
    init = DEP / "core" / "__init__.py"
    if not init.exists():
        init.write_text('"""Dashboard processing modules (Stage 9)."""\n', encoding="utf-8")
    cfg = DEP / ".streamlit" / "config.toml"
    if not cfg.exists():
        cfg.write_text('[server]\nmaxUploadSize = 1024\n\n[theme]\nbase = "light"\n', encoding="utf-8")
        say("  ok  deployment/.streamlit/config.toml  (upload limit 1 GB for big Strava exports)")

    # 3. models + Stage 4 settings
    say("\n[3] Model files -> deployment/models/")
    manifest = {"created": datetime.now().isoformat(timespec="seconds"), "files": {}}
    for src in MODEL_FILES + STAGE4_FILES:
        if not src.exists():
            ok = False
            say(f"  MISSING  {src.relative_to(ROOT)}")
            continue
        dst = DEP / "models" / src.name
        shutil.copy2(src, dst)
        digest = sha256(dst)
        if digest != sha256(src):
            ok = False
            say(f"  COPY MISMATCH  {src.name}")
        manifest["files"][src.name] = {"from": str(src.relative_to(ROOT)).replace("\\", "/"),
                                       "sha256_16": digest, "bytes": dst.stat().st_size}
        say(f"  ok  {src.name:42s} {dst.stat().st_size/1e6:6.2f} MB  sha {digest}")
    (DEP / "models" / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    try:
        params = json.loads(STAGE4_FILES[0].read_text(encoding="utf-8"))
        say(f"  Stage 4 params keys: {list(params)[:8]}")
        feats = STAGE4_FILES[1].read_text(encoding="utf-8").split()
        say(f"  tabular features ({len(feats)}): {feats}")
    except Exception as e:
        say(f"  (could not preview Stage 4 settings: {e})")

    # 4. demo data
    say("\n[4] Demo data")
    zp = DEP / "demo_data" / "justinhojt_strava_export.zip"
    if zp.exists():
        with zipfile.ZipFile(zp) as z:
            names = z.namelist()
            has_csv = "activities.csv" in names
            n_files = sum(n.startswith("activities/") and not n.endswith("/") for n in names)
            has_lic = "LICENSE" in names
        good = has_csv and has_lic and n_files > 0
        ok &= good
        say(f"  {'ok ' if good else 'BAD'}  justinhojt_strava_export.zip  activities.csv={has_csv}  "
            f"activity files={n_files}  LICENSE={has_lic}")
    else:
        ok = False
        say("  MISSING  deployment/demo_data/justinhojt_strava_export.zip")
    kag = list((DEP / "demo_data" / "kaggle").glob("*.xls*")) + list((DEP / "demo_data" / "kaggle").glob("*.csv"))
    if kag:
        say(f"  ok   Kaggle file(s): {[k.name for k in kag]}")
        try:
            import pandas as pd
            df = pd.read_excel(kag[0]) if kag[0].suffix.startswith(".xls") else pd.read_csv(kag[0])
            say(f"       shape {df.shape}; columns: {list(df.columns)[:12]}{' ...' if df.shape[1] > 12 else ''}")
        except Exception as e:
            say(f"       (could not open: {e})")
    else:
        say("  TODO  put the Kaggle xlsx in deployment/demo_data/kaggle/  (not blocking for Steps 2-4)")
    note = DEP / "demo_data" / "kaggle" / "CC0_NOTE.txt"
    if not note.exists():
        note.write_text(
            "Demo dataset 2: 'Strava Running Activity Data' by ajitjadhav1 on Kaggle.\n"
            "https://www.kaggle.com/datasets/ajitjadhav1/strava-running-activity-data\n"
            "License: CC0 1.0 Universal (Public Domain Dedication).\n"
            "Summary-level activity data (no per-second streams). Used as public secondary data;\n"
            "no participants were recruited.\n", encoding="utf-8")

    # 5. pinned requirements
    say("\n[5] Pinned requirements")
    def pins(libs: dict[str, str]) -> list[str]:
        out = []
        for pip_name, imp in libs.items():
            v = version(pip_name, imp)
            if v:
                out.append(f"{pip_name}=={v}")
        return out
    full = pins(PIPELINE_LIBS | DASHBOARD_LIBS)
    dash = pins(DASHBOARD_LIBS)
    hdr = f"# Pinned {datetime.now():%Y-%m-%d} on Python {platform.python_version()} ({platform.system()}), CPU only\n"
    (ROOT / "requirements.txt").write_text(hdr + "\n".join(sorted(set(full))) + "\n", encoding="utf-8")
    (DEP / "requirements.txt").write_text(hdr + "\n".join(sorted(set(dash))) + "\n", encoding="utf-8")
    say(f"  ok  requirements.txt             ({len(set(full))} packages)")
    say(f"  ok  deployment/requirements.txt  ({len(set(dash))} packages)")

    say("\n" + "=" * 70)
    say("RESULT: " + ("ALL GOOD - ready for Step 2 (ingest)" if ok else "FIX THE ITEMS MARKED MISSING/BAD, then re-run"))
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "stage9_setup_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nReport saved: logs/stage9_setup_report.txt")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
