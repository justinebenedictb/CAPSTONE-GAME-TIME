"""
07_stage5_baseline_models.py

CAPSTONE
Stage 5 - Baseline Model Training

Inputs:
    03_feature_engineering/  (samples, daily features)   - 05_stage3_feature_engineering.py
    04_feature_selection/    (split, prep params, lists)  - 06_stage4_feature_selection.py
    01_prepared/stage1_daily_timeline.csv                 (best efforts for known_rpc)

Pipeline:
1. Build inputs (all fitting on the TRAINING split only)
     - XGBoost: Stage 4 tabular features (10)
     - GRU / LSTM: 28 days before the target x (Stage 4 sequence features
       + known_rpc + days_since_rpc), filled and scaled on training windows
2. Naive baselines: last RPC, athlete's training-mean RPC, overall training mean
3. Base models (manuscript 3.4.4 initial configurations), athlete-weighted,
   MAE loss, 3 seeds each:
     - XGBoost: depth 4, lr 0.05, subsample 0.8, up to 1000 trees, early stop
     - GRU / LSTM: 1 recurrent layer (32 units, dropout 0.2) -> Dense 16 -> 1,
       Adam 1e-3, batch 64, up to 100 epochs, early stop (patience 10)
4. Ablation: every model also trained WITHOUT past-RPC inputs
5. Evaluation on the VALIDATION split only (test is reserved for Stage 7):
   MAE (primary), RMSE, R2, MAPE on RPC, athlete-weighted and unweighted;
   per-athlete MAE; per best-effort distance; finish-time error
6. Outputs to 05_baseline_models/

Usage:
    python 07_stage5_baseline_models.py                 (everything)
    python 07_stage5_baseline_models.py --models gru    (one model; others kept)
    python 07_stage5_baseline_models.py --quick         (smoke test, ~1-2 min,
                                                          writes to 05_baseline_models/_quick)
Requires: pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib
"""

from pathlib import Path
import argparse
import json
import time
import traceback

import numpy as np
import pandas as pd

import capstone_data as cd


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = cd.ROOT / "05_baseline_models"
LOG_DIR = cd.ROOT / "logs"

SEEDS = [11, 22, 33]
VARIANTS = ["full", "no_past_rpc"]
MODELS = ["xgb", "gru", "lstm"]

XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    objective="reg:absoluteerror",
    eval_metric="mae",
    early_stopping_rounds=50,
    n_jobs=-1,
)

RNN_UNITS = 32
RNN_DROPOUT = 0.2
DENSE_UNITS = 16
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 100
PATIENCE = 10

MODEL_LABELS = {"xgb": "XGBoost", "gru": "GRU", "lstm": "LSTM"}
COLORS = {"gru": "#2a78d6", "lstm": "#eb6834", "xgb": "#1baf7a"}
TEXT, MUTED, GRID = "#0b0b0b", "#6b6b68", "#e6e5e1"


# ============================================================
# 1. INPUTS
# ============================================================

def build_inputs(quick):
    print("=" * 70)
    print("STEP 1 - BUILDING INPUTS (FITTED ON TRAINING ONLY)")
    print("=" * 70)

    samples = cd.add_personal_level(cd.load_samples())
    samples = samples[samples["split"].isin(["train", "val"])]  # test reserved for Stage 7
    if quick:
        keep = samples["id"].drop_duplicates().head(25)
        samples = samples[samples["id"].isin(keep)]
    params = cd.load_params()

    tab = cd.prepare_tabular(samples, params)
    train = tab[tab["split"] == "train"].reset_index(drop=True)
    val = tab[tab["split"] == "val"].reset_index(drop=True)

    tab_full = cd.load_feature_list("tabular")
    seq_full = cd.load_feature_list("sequence") + cd.PRIOR_SEQUENCE
    features = {
        "xgb": {"full": tab_full,
                "no_past_rpc": [f for f in tab_full if f not in cd.PRIOR_TABULAR]},
        "seq": {"full": seq_full,
                "no_past_rpc": [f for f in seq_full if f not in cd.PRIOR_SEQUENCE]},
    }

    daily = cd.load_daily(seq_full)
    X_train_raw = cd.build_windows(daily, train, seq_full)
    X_val_raw = cd.build_windows(daily, val, seq_full)

    seq = {}
    for variant, feats in features["seq"].items():
        cols = [seq_full.index(f) for f in feats]
        prep = cd.fit_sequence_prep(X_train_raw[:, :, cols], feats, params)
        seq[variant] = {
            "prep": prep,
            "train": cd.apply_sequence_prep(X_train_raw[:, :, cols], feats, prep),
            "val": cd.apply_sequence_prep(X_val_raw[:, :, cols], feats, prep),
        }
        assert np.isfinite(seq[variant]["train"]).all() and np.isfinite(seq[variant]["val"]).all()
        (OUTPUT_DIR / f"stage5_sequence_prep_{variant}.json").write_text(
            json.dumps(prep, indent=2), encoding="utf-8")

    print(f"Train samples: {len(train):,} ({train['id'].nunique()} athletes) | "
          f"validation samples: {len(val):,} ({val['id'].nunique()} athletes)")
    print(f"XGBoost features: full {len(features['xgb']['full'])}, "
          f"no_past_rpc {len(features['xgb']['no_past_rpc'])}")
    print(f"Sequence input: full {seq['full']['train'].shape[1:]}, "
          f"no_past_rpc {seq['no_past_rpc']['train'].shape[1:]}")

    return train, val, features, seq


# ============================================================
# 2. NAIVE BASELINES
# ============================================================

def naive_predictions(train, val):
    athlete_mean = train.groupby("id")[cd.TARGET].mean()
    return {
        "naive_last_rpc": val["last_rpc"].to_numpy(),
        "naive_athlete_mean": val["id"].map(athlete_mean).fillna(train[cd.TARGET].mean()).to_numpy(),
        "naive_global_mean": np.full(len(val), train[cd.TARGET].mean()),
    }


# ============================================================
# 3. MODELS
# ============================================================

def train_xgb(train, val, feats, seed, path):
    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=seed, **XGB_PARAMS)
    model.fit(train[feats], train[cd.TARGET], sample_weight=cd.athlete_weights(train),
              eval_set=[(val[feats], val[cd.TARGET])],
              sample_weight_eval_set=[cd.athlete_weights(val)], verbose=False)
    model.save_model(str(path.with_suffix(".json")))
    history = model.evals_result()["validation_0"]["mae"]
    return model.predict(val[feats]), {"best_iteration": int(model.best_iteration),
                                       "val_curve": history}


def train_rnn(kind, X_train, y_train, w_train, X_val, y_val, w_val, seed, path, quick):
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    recurrent = keras.layers.GRU if kind == "gru" else keras.layers.LSTM
    inputs = keras.Input(shape=X_train.shape[1:])
    x = recurrent(RNN_UNITS, dropout=RNN_DROPOUT)(inputs)          # many-to-one
    x = keras.layers.Dense(DENSE_UNITS, activation="relu")(x)
    # Start the output at the training mean RPC so training refines it instead of
    # first climbing from 0 to ~0.5 (much faster convergence with MAE loss).
    start = float(np.average(y_train, weights=w_train))
    outputs = keras.layers.Dense(1, bias_initializer=keras.initializers.Constant(start))(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mae")

    history = model.fit(
        X_train, y_train, sample_weight=w_train,
        validation_data=(X_val, y_val, w_val),
        epochs=3 if quick else MAX_EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                 restore_best_weights=True)],
    )
    model.save(str(path.with_suffix(".keras")))
    h = history.history
    return model.predict(X_val, verbose=0).ravel(), {
        "best_epoch": int(np.argmin(h["val_loss"]) + 1),
        "epochs_run": len(h["val_loss"]),
        "train_curve": h["loss"],
        "val_curve": h["val_loss"],
    }


# ============================================================
# 5. EVALUATION
# ============================================================

def evaluate(name, model, variant, seed, val, pred):
    y = val[cd.TARGET].to_numpy()
    w = cd.athlete_weights(val)
    weighted = cd.regression_metrics(y, pred, w)
    plain = cd.regression_metrics(y, pred)

    per_athlete = (pd.DataFrame({"id": val["id"], "ae": np.abs(pred - y)})
                   .groupby("id")["ae"].mean())
    seconds = cd.time_from_rpc(np.clip(pred, 1e-3, None), val["target_distance_km"].to_numpy())
    time_err = np.abs(seconds - val["target_seconds"].to_numpy())

    return {
        "run": name, "model": model, "variant": variant, "seed": seed,
        "mae": weighted["mae"], "rmse": weighted["rmse"], "r2": weighted["r2"],
        "mape": weighted["mape"],
        "mae_unweighted": plain["mae"], "rmse_unweighted": plain["rmse"],
        "r2_unweighted": plain["r2"], "mape_unweighted": plain["mape"],
        "athlete_mae_median": per_athlete.median(),
        "athlete_mae_q1": per_athlete.quantile(0.25),
        "athlete_mae_q3": per_athlete.quantile(0.75),
        "finish_time_mae_s": float(np.sum(w * time_err) / w.sum()),
        "finish_time_mape": float(100 * np.sum(w * time_err / val["target_seconds"]) / w.sum()),
    }


def by_distance(val, preds):
    rows = []
    for name, pred in preds.items():
        df = pd.DataFrame({
            "km": val["target_distance_km"], "y": val[cd.TARGET], "pred": pred,
            "sec": val["target_seconds"],
        })
        df["pred_sec"] = cd.time_from_rpc(np.clip(df["pred"], 1e-3, None), df["km"])
        for km, g in df.groupby("km"):
            rows.append({
                "prediction": name, "distance_km": km, "n": len(g),
                "rpc_mae": np.mean(np.abs(g["pred"] - g["y"])),
                "finish_time_mae_s": np.mean(np.abs(g["pred_sec"] - g["sec"])),
                "finish_time_mape": 100 * np.mean(np.abs(g["pred_sec"] - g["sec"]) / g["sec"]),
            })
    return pd.DataFrame(rows)


# ============================================================
# PLOTS
# ============================================================

def plot_learning_curves(curves, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rnn = {k: v for k, v in curves.items() if v["model"] in ("gru", "lstm")}
    if not rnn:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        for info in rnn.values():
            if info["variant"] != variant:
                continue
            ax.plot(range(1, len(info["val_curve"]) + 1), info["val_curve"], color=COLORS[info["model"]], lw=2, alpha=0.85,
                    label=MODEL_LABELS[info["model"]])
        handles, labels = ax.get_legend_handles_labels()
        uniq = dict(zip(labels, handles))
        ax.legend(uniq.values(), uniq.keys(), frameon=False, fontsize=8)
        ax.set_title(f"{variant} (one line per seed)", fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("Epoch", color=MUTED, fontsize=8)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(color=GRID, lw=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Validation MAE (RPC, athlete-weighted)", color=MUTED, fontsize=8)
    fig.suptitle("GRU / LSTM validation loss by epoch", fontsize=10, color=TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_pred_vs_actual(val, preds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("xgb_full", "gru_full", "lstm_full") if n in preds]
    if not names:
        return
    y = val[cd.TARGET].to_numpy()
    lo, hi = np.percentile(y, 0.5) - 0.05, np.percentile(y, 99.5) + 0.05
    fig, axes = plt.subplots(1, len(names), figsize=(3.6 * len(names), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        model = name.split("_")[0]
        ax.scatter(y, preds[name], s=8, alpha=0.35, color=COLORS[model], linewidths=0)
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{MODEL_LABELS[model]} (full, seed mean)", fontsize=9, color=TEXT, loc="left")
        ax.set_xlabel("Actual RPC", color=MUTED, fontsize=8)
        ax.grid(color=GRID, lw=0.6)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("Predicted RPC", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main(models, quick):
    global OUTPUT_DIR
    if quick:
        OUTPUT_DIR = OUTPUT_DIR / "_quick"
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS[:1] if quick else SEEDS

    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 5 BASELINE MODEL TRAINING" + (" (QUICK TEST)" if quick else ""))
    print("=" * 70)
    print()

    try:
        train, val, features, seq = build_inputs(quick)
        y_train = train[cd.TARGET].to_numpy(dtype=np.float32)
        y_val = val[cd.TARGET].to_numpy(dtype=np.float32)
        w_train = cd.athlete_weights(train).astype(np.float32)
        w_val = cd.athlete_weights(val).astype(np.float32)

        print()
        print("=" * 70)
        print("STEP 2 - NAIVE BASELINES")
        print("=" * 70)
        runs, preds, curves = [], {}, {}
        for name, pred in naive_predictions(train, val).items():
            preds[name] = pred
            runs.append(evaluate(name, name, "naive", 0, val, pred))
            print(f"  {name:<20} MAE {runs[-1]['mae']:.4f}")

        print()
        print("=" * 70)
        print("STEP 3-4 - BASE MODELS (WITH AND WITHOUT PAST RPC)")
        print("=" * 70)
        seed_preds = {}
        for model in models:
            for variant in VARIANTS:
                for seed in seeds:
                    name = f"{model}_{variant}_seed{seed}"
                    start = time.time()
                    path = OUTPUT_DIR / "models" / name
                    if model == "xgb":
                        pred, info = train_xgb(train, val, features["xgb"][variant], seed, path)
                        detail = f"trees {info['best_iteration'] + 1}"
                    else:
                        pred, info = train_rnn(model, seq[variant]["train"], y_train, w_train,
                                               seq[variant]["val"], y_val, w_val, seed, path, quick)
                        detail = f"best epoch {info['best_epoch']}/{info['epochs_run']}"
                    curves[name] = {"model": model, "variant": variant, **info}
                    runs.append(evaluate(name, model, variant, seed, val, pred))
                    seed_preds.setdefault(f"{model}_{variant}", []).append(pred)
                    print(f"  {name:<26} MAE {runs[-1]['mae']:.4f}  ({detail}, {time.time() - start:.0f}s)")
        for key, plist in seed_preds.items():
            preds[key] = np.mean(plist, axis=0)

        # ---- merge with earlier runs of models not re-run this time ----
        runs = pd.DataFrame(runs)
        runs_path = OUTPUT_DIR / "stage5_metrics_runs.csv"
        pred_path = OUTPUT_DIR / "stage5_val_predictions.csv"
        pred_df = val[["sample_id", "id", "target_date", "target_distance_km",
                       "target_seconds", cd.TARGET]].copy()
        for name, pred in preds.items():
            pred_df[name] = pred
        if runs_path.exists() and set(models) != set(MODELS):
            old = pd.read_csv(runs_path)
            old = old[~old["model"].isin(models + list(naive_predictions(train, val)))]
            runs = pd.concat([old, runs], ignore_index=True)
            if pred_path.exists():
                old_pred = pd.read_csv(pred_path)
                keep = [c for c in old_pred.columns if c not in pred_df.columns]
                pred_df = pred_df.merge(old_pred[["sample_id"] + keep], on="sample_id", how="left")
        runs.to_csv(runs_path, index=False)
        pred_df.to_csv(pred_path, index=False)

        metric_cols = ["mae", "rmse", "r2", "mape", "mae_unweighted", "athlete_mae_median",
                       "finish_time_mae_s", "finish_time_mape"]
        summary = (runs.groupby(["model", "variant"])[metric_cols]
                   .agg(["mean", "std"]).sort_values(("mae", "mean")))
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index()
        summary.to_csv(OUTPUT_DIR / "stage5_metrics_summary.csv", index=False)

        pred_cols = [c for c in pred_df.columns if c.startswith(("naive_", "xgb_", "gru_", "lstm_"))]
        dist = by_distance(val, {c: pred_df[c].to_numpy() for c in pred_cols})
        dist.to_csv(OUTPUT_DIR / "stage5_metrics_by_distance.csv", index=False)

        (OUTPUT_DIR / "stage5_training_info.json").write_text(json.dumps(
            {k: {kk: vv for kk, vv in v.items() if kk not in ("train_curve",)}
             for k, v in curves.items()}, indent=1), encoding="utf-8")
        plot_learning_curves(curves, OUTPUT_DIR / "stage5_learning_curves.png")
        plot_pred_vs_actual(val, {c: pred_df[c].to_numpy() for c in pred_cols},
                            OUTPUT_DIR / "stage5_pred_vs_actual.png")

        # ---- checks ----
        print()
        print("=" * 70)
        print("STEP 6 - CHECKS")
        print("=" * 70)
        assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}
        assert not set(train["sample_id"]) & set(val["sample_id"])
        for name, pred in preds.items():
            assert np.isfinite(pred).all(), f"non-finite predictions from {name}"
        print("All checks passed.")

        # ---- summary ----
        def fmt(row, col, digits=4):
            sd = row.get(f"{col}_std")
            sd = "" if pd.isna(sd) else f" ± {sd:.{digits}f}"
            return f"{row[f'{col}_mean']:.{digits}f}{sd}"

        best_naive = summary[summary["variant"] == "naive"].iloc[0]
        lines = [
            "CAPSTONE - STAGE 5 BASELINE MODEL TRAINING SUMMARY" + (" (QUICK TEST)" if quick else ""),
            "=" * 60,
            "",
            "DATA (validation only - test set reserved for Stage 7)",
            "-" * 60,
            f"Train: {len(train):,} samples, {train['id'].nunique()} athletes | "
            f"Validation: {len(val):,} samples, {val['id'].nunique()} athletes",
            f"XGBoost features (full, {len(features['xgb']['full'])}): {', '.join(features['xgb']['full'])}",
            f"XGBoost features (no_past_rpc, {len(features['xgb']['no_past_rpc'])}): "
            f"{', '.join(features['xgb']['no_past_rpc'])}",
            f"RNN input (full): (N, {cd.WINDOW_DAYS}, {len(features['seq']['full'])}) - {', '.join(features['seq']['full'])}",
            f"RNN input (no_past_rpc): (N, {cd.WINDOW_DAYS}, {len(features['seq']['no_past_rpc'])})",
            "",
            "SETTINGS",
            "-" * 60,
            f"Seeds: {seeds} | loss: MAE | sample weight: 1 / athlete's sample count",
            f"XGBoost: depth {XGB_PARAMS['max_depth']}, lr {XGB_PARAMS['learning_rate']}, "
            f"subsample {XGB_PARAMS['subsample']}, colsample {XGB_PARAMS['colsample_bytree']}, "
            f"up to {XGB_PARAMS['n_estimators']} trees, early stop {XGB_PARAMS['early_stopping_rounds']}",
            f"GRU/LSTM: 1 layer x {RNN_UNITS} units, dropout {RNN_DROPOUT}, Dense {DENSE_UNITS} (ReLU) -> 1, "
            f"Adam {LEARNING_RATE}, batch {BATCH_SIZE}, up to {MAX_EPOCHS} epochs, early stop {PATIENCE}",
            "",
            "RESULTS (validation; athlete-weighted unless stated; mean ± SD over seeds)",
            "-" * 60,
            f"{'Model':<20}{'Variant':<14}{'MAE':<20}{'RMSE':<20}{'R2':<20}{'MAPE %':<18}{'Finish-time MAPE*'}",
        ]
        for _, r in summary.iterrows():
            lines.append(f"{r['model']:<20}{r['variant']:<14}{fmt(r, 'mae'):<20}{fmt(r, 'rmse'):<20}"
                         f"{fmt(r, 'r2', 3):<20}{fmt(r, 'mape', 2):<18}{fmt(r, 'finish_time_mape', 1)} %")
        lines += [
            "* finish-time error at each sample's own best-effort distance, as % of the actual time",
            "",
            f"Best naive baseline: {best_naive['model']} (MAE {best_naive['mae_mean']:.4f})",
        ]
        for model in models:
            full = summary[(summary["model"] == model) & (summary["variant"] == "full")]
            abl = summary[(summary["model"] == model) & (summary["variant"] == "no_past_rpc")]
            if len(full) and len(abl):
                f, a = full.iloc[0]["mae_mean"], abl.iloc[0]["mae_mean"]
                vs = 100 * (best_naive["mae_mean"] - f) / best_naive["mae_mean"]
                verdict = f"{abs(vs):.1f}% {'better' if vs > 0 else 'worse'} than best naive"
                lines.append(f"{MODEL_LABELS[model]}: full {f:.4f} ({verdict}) | "
                             f"without past RPC {a:.4f} (past RPC changes MAE by {100 * (a - f) / f:+.1f}%)")
        lines += [
            "",
            "MAE BY BEST-EFFORT DISTANCE (seed-mean predictions; see stage5_metrics_by_distance.csv)",
            "-" * 60,
        ]
        piv = dist.pivot_table(index="prediction", columns="distance_km", values="rpc_mae")
        n_by = dist.groupby("distance_km")["n"].first()
        lines.append("n per distance: " + ", ".join(f"{k:g} km {int(v)}" for k, v in n_by.items()))
        lines.append(piv.round(4).to_string())
        lines += [
            "",
            "STATUS",
            "-" * 60,
            "Stage 5 baseline training completed. Next: Stage 6 hyperparameter tuning (Random Search).",
        ]
        (OUTPUT_DIR / "stage5_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        print(f"\nOutputs written to:\n{OUTPUT_DIR}")

    except Exception as exc:
        (LOG_DIR / "stage5_baseline_error.log").write_text(
            "STAGE 5 ERROR\n" + "=" * 60 + "\n\n" + str(exc) + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        print("\nERROR - see logs/stage5_baseline_error.log")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS),
                        help="comma-separated subset of: xgb,gru,lstm")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 25 athletes, 1 seed, 3 epochs")
    args = parser.parse_args()
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    assert set(chosen) <= set(MODELS), f"unknown model in {chosen}"
    main(chosen, args.quick)
