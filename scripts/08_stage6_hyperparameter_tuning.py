"""
08_stage6_hyperparameter_tuning.py

CAPSTONE
Stage 6 - Hyperparameter Tuning (Random Search)

Inputs: the same as 07_stage5_baseline_models.py (Stage 3 samples and daily
features, Stage 4 split / preparation / feature lists). The test split is not
loaded.

Pipeline, for every model (XGBoost, GRU, LSTM) x version (full, no_past_rpc):
1. Random Search: TRIALS settings drawn from the search space below.
   Trial 0 is always the Stage 5 baseline setting, so tuning can never end up
   worse than the baseline. Each trial: one seed (11), trained on the training
   split, early stopping and scoring on validation MAE (athlete-weighted).
2. Seed check: the TOP_K best trials are re-trained with seeds 22 and 33.
   Winner = lowest mean validation MAE over the 3 seeds (protects against a
   setting that was only lucky with one seed).
3. The winner's three seed models are the tuned models (same 3 seeds as
   Stage 5, so tuned vs baseline is like-for-like).
4. Report: tuned vs baseline (Stage 5) vs naive baselines; average of the three
   tuned full models as a secondary result; by-distance errors; plots.

Every finished trial is saved immediately. Re-running the same command skips
finished trials and continues, so the run can be stopped (Ctrl+C, sleep,
shutdown) and resumed. Once everything is finished, re-running only rebuilds
the report.

Usage:
    py 08_stage6_hyperparameter_tuning.py                  (everything)
    py 08_stage6_hyperparameter_tuning.py --models xgb     (subset; resumable)
    py 08_stage6_hyperparameter_tuning.py --trials 60      (more trials)
    py 08_stage6_hyperparameter_tuning.py --quick          (smoke test, ~2-4 min,
                                                            writes to 06_tuning/_quick)
Requires: pandas, numpy, scikit-learn, xgboost, tensorflow, matplotlib
"""

from pathlib import Path
import argparse
import importlib
import json
import math
import shutil
import sys
import time
import traceback

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capstone_data as cd

s5 = importlib.import_module("07_stage5_baseline_models")  # reuse Stage 5 inputs / metrics


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = cd.ROOT / "06_tuning"
BASELINE_DIR = cd.ROOT / "05_baseline_models"
LOG_DIR = cd.ROOT / "logs"

MODELS = ["xgb", "gru", "lstm"]
VARIANTS = ["full", "no_past_rpc"]
SEARCH_SEED = 11            # seed used for every search trial
CHECK_SEEDS = [22, 33]      # extra seeds for the top-K trials
DEFAULT_TRIALS = 40         # per model x version (trial 0 = Stage 5 baseline setting)
TOP_K = 3

XGB_FIXED = dict(n_estimators=3000, early_stopping_rounds=100,
                 objective="reg:absoluteerror", eval_metric="mae", n_jobs=-1)
RNN_MAX_EPOCHS = 150
RNN_PATIENCE = 10

# Stage 5 settings (trial 0)
XGB_BASELINE = dict(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    min_child_weight=5.0, reg_lambda=1.0, reg_alpha=0.0, gamma=0.0)
RNN_BASELINE = dict(units=32, layers=1, dropout=0.2, dense=16, learning_rate=1e-3,
                    batch_size=64, l2=0.0)


def log_uniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_xgb(rng):
    return dict(
        max_depth=int(rng.integers(2, 11)),                       # 2-10
        learning_rate=log_uniform(rng, 0.005, 0.3),
        subsample=float(rng.uniform(0.5, 1.0)),
        colsample_bytree=float(rng.uniform(0.5, 1.0)),
        min_child_weight=log_uniform(rng, 1, 50),
        reg_lambda=log_uniform(rng, 0.01, 100),
        reg_alpha=float(rng.choice([0.0, 1e-4, 1e-3, 1e-2, 1e-1])),
        gamma=float(rng.choice([0.0, 1e-4, 1e-3, 3e-3])),          # RPC errors are ~0.04
    )


def sample_rnn(rng):
    return dict(
        units=int(rng.choice([16, 32, 64, 128])),
        layers=int(rng.choice([1, 2])),
        dropout=float(rng.uniform(0.0, 0.4)),
        dense=int(rng.choice([0, 8, 16, 32])),                     # 0 = no hidden dense layer
        learning_rate=log_uniform(rng, 3e-4, 3e-3),
        batch_size=int(rng.choice([32, 64, 128])),
        l2=float(rng.choice([0.0, 1e-6, 1e-5, 1e-4])),
    )


def trial_configs(model, variant, n_trials):
    """Same list every run (fixed seed), so a resumed run continues the same search."""
    rng = np.random.default_rng(1000 + 10 * MODELS.index(model) + VARIANTS.index(variant))
    sampler, base = (sample_xgb, XGB_BASELINE) if model == "xgb" else (sample_rnn, RNN_BASELINE)
    draws = [sampler(rng) for _ in range(max(n_trials, DEFAULT_TRIALS) - 1)]
    return ([dict(base)] + draws)[:n_trials]


# ============================================================
# TRAINING
# ============================================================

def fit_xgb(cfg, train, val, feats, seed, path):
    from xgboost import XGBRegressor
    model = XGBRegressor(random_state=seed, **XGB_FIXED, **cfg)
    model.fit(train[feats], train[cd.TARGET], sample_weight=cd.athlete_weights(train),
              eval_set=[(val[feats], val[cd.TARGET])],
              sample_weight_eval_set=[cd.athlete_weights(val)], verbose=False)
    model.save_model(str(path.with_suffix(".json")))
    return model.predict(val[feats]), {"trees": int(model.best_iteration) + 1}


def fit_rnn(kind, cfg, data, seed, path, max_epochs):
    import tensorflow as tf
    from tensorflow import keras

    X_train, y_train, w_train, X_val, y_val, w_val = data
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    reg = keras.regularizers.l2(cfg["l2"]) if cfg["l2"] > 0 else None
    recurrent = keras.layers.GRU if kind == "gru" else keras.layers.LSTM

    inputs = keras.Input(shape=X_train.shape[1:])
    x = inputs
    for i in range(cfg["layers"]):
        x = recurrent(cfg["units"], dropout=cfg["dropout"], kernel_regularizer=reg,
                      return_sequences=i < cfg["layers"] - 1)(x)
    if cfg["dense"]:
        x = keras.layers.Dense(cfg["dense"], activation="relu", kernel_regularizer=reg)(x)
    start = float(np.average(y_train, weights=w_train))      # same output start as Stage 5
    outputs = keras.layers.Dense(1, bias_initializer=keras.initializers.Constant(start))(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(cfg["learning_rate"]), loss="mae")

    history = model.fit(
        X_train, y_train, sample_weight=w_train, validation_data=(X_val, y_val, w_val),
        epochs=max_epochs, batch_size=cfg["batch_size"], verbose=0,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=RNN_PATIENCE,
                                                 restore_best_weights=True)],
    )
    model.save(str(path.with_suffix(".keras")))
    h = history.history["val_loss"]
    return model.predict(X_val, verbose=0).ravel(), {"best_epoch": int(np.argmin(h)) + 1,
                                                     "epochs_run": len(h)}


# ============================================================
# SEARCH (resumable)
# ============================================================

class TrialStore:
    """One row per (model, variant, trial, seed) in stage6_trials.csv; predictions in .npy."""

    def __init__(self, out_dir):
        self.csv = out_dir / "stage6_trials.csv"
        self.pred_dir = out_dir / "trials"
        self.pred_dir.mkdir(parents=True, exist_ok=True)
        self.rows = pd.read_csv(self.csv).to_dict("records") if self.csv.exists() else []

    def key(self, model, variant, trial, seed):
        return f"{model}_{variant}_t{trial:03d}_s{seed}"

    def done(self, model, variant, trial, seed):
        return any(r["model"] == model and r["variant"] == variant and r["trial"] == trial
                   and r["seed"] == seed for r in self.rows) and \
            (self.pred_dir / f"{self.key(model, variant, trial, seed)}.npy").exists()

    def add(self, row, pred):
        np.save(self.pred_dir / f"{self.key(row['model'], row['variant'], row['trial'], row['seed'])}.npy",
                np.asarray(pred, dtype=np.float64))
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)   # saved after every trial

    def pred(self, model, variant, trial, seed):
        return np.load(self.pred_dir / f"{self.key(model, variant, trial, seed)}.npy")

    def frame(self, model, variant):
        df = pd.DataFrame(self.rows)
        if df.empty:
            return df
        return df[(df["model"] == model) & (df["variant"] == variant)]


def run_one(store, model, variant, trial, seed, cfg, ctx):
    if store.done(model, variant, trial, seed):
        return False
    start = time.time()
    path = ctx["out"] / "trials" / store.key(model, variant, trial, seed)
    if model == "xgb":
        pred, info = fit_xgb(cfg, ctx["train"], ctx["val"], ctx["features"]["xgb"][variant], seed, path)
    else:
        pred, info = fit_rnn(model, cfg, ctx["rnn_data"][variant], seed, path, ctx["max_epochs"])
    m = s5.evaluate("", model, variant, seed, ctx["val"], pred)
    row = {"model": model, "variant": variant, "trial": trial, "seed": seed,
           "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"], "seconds": round(time.time() - start, 1),
           **info, "params": json.dumps(cfg)}
    store.add(row, pred)
    best = store.frame(model, variant)
    best = best[best["seed"] == SEARCH_SEED]["mae"].min()
    detail = f"trees {info['trees']}" if model == "xgb" else f"epoch {info['best_epoch']}/{info['epochs_run']}"
    print(f"  {model:<4} {variant:<11} trial {trial:>3} seed {seed}  MAE {m['mae']:.4f}  "
          f"(best {best:.4f}; {detail}; {row['seconds']:.0f}s)")
    return True


def search(store, model, variant, n_trials, top_k, ctx):
    configs = trial_configs(model, variant, n_trials)
    todo = [t for t in range(n_trials) if not store.done(model, variant, t, SEARCH_SEED)]
    print(f"\n{s5.MODEL_LABELS[model]} / {variant}: {n_trials} trials, "
          f"{n_trials - len(todo)} already done")
    t0 = time.time()
    for i, t in enumerate(todo, 1):
        run_one(store, model, variant, t, SEARCH_SEED, configs[t], ctx)
        if i < len(todo):
            eta = (time.time() - t0) / i * (len(todo) - i)
            print(f"      ~{eta / 60:.0f} min left for this model/version")

    # seed check on the top-K trials
    df = store.frame(model, variant)
    first = df[(df["seed"] == SEARCH_SEED) & (df["trial"] < n_trials)].sort_values("mae")
    top = first["trial"].head(top_k).tolist()
    for t in top:
        for seed in ctx["check_seeds"]:
            run_one(store, model, variant, t, seed, configs[t], ctx)

    df = store.frame(model, variant)
    seeds = [SEARCH_SEED] + ctx["check_seeds"]
    cand = (df[df["trial"].isin(top) & df["seed"].isin(seeds)]
            .groupby("trial")["mae"].agg(["mean", "std", "count"]))
    cand = cand[cand["count"] == len(seeds)].sort_values("mean")
    winner = int(cand.index[0])
    print(f"  -> winner: trial {winner} (mean MAE {cand.loc[winner, 'mean']:.4f} over {len(seeds)} seeds)")
    return winner, configs[winner], cand


# ============================================================
# REPORT
# ============================================================

def plot_progress(store, n_trials, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, variant in zip(axes, VARIANTS):
        for model in MODELS:
            df = store.frame(model, variant)
            if df.empty:
                continue
            df = df[(df["seed"] == SEARCH_SEED) & (df["trial"] < n_trials)].sort_values("trial")
            ax.scatter(df["trial"], df["mae"], s=10, alpha=0.35, color=s5.COLORS[model], linewidths=0)
            ax.plot(df["trial"], df["mae"].cummin(), color=s5.COLORS[model], lw=2,
                    label=s5.MODEL_LABELS[model])
        ax.set_title(f"{variant}: best so far (line), each trial (dots)", fontsize=9,
                     color=s5.TEXT, loc="left")
        ax.set_xlabel("Trial (0 = Stage 5 baseline setting)", color=s5.MUTED, fontsize=8)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.grid(color=s5.GRID, lw=0.6)
        ax.legend(frameon=False, fontsize=8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    lo = min(store.frame(m, v)["mae"].min() for m in MODELS for v in VARIANTS
             if not store.frame(m, v).empty)
    axes[0].set_ylim(lo - 0.002, lo + 0.03)    # zoom on the useful range; very bad trials cut off
    axes[0].set_ylabel("Validation MAE (RPC, athlete-weighted, seed 11)", color=s5.MUTED, fontsize=8)
    fig.suptitle("Random Search progress", fontsize=10, color=s5.TEXT, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def describe(model, cfg):
    if model == "xgb":
        return (f"depth {cfg['max_depth']}, lr {cfg['learning_rate']:.4f}, subsample {cfg['subsample']:.2f}, "
                f"colsample {cfg['colsample_bytree']:.2f}, min_child_weight {cfg['min_child_weight']:.1f}, "
                f"lambda {cfg['reg_lambda']:.3g}, alpha {cfg['reg_alpha']:g}, gamma {cfg['gamma']:g}")
    return (f"{cfg['layers']} layer(s) x {cfg['units']} units, dropout {cfg['dropout']:.2f}, "
            f"dense {cfg['dense'] or 'none'}, lr {cfg['learning_rate']:.5f}, batch {cfg['batch_size']}, "
            f"L2 {cfg['l2']:g}")


def main(models, n_trials, top_k, quick):
    global OUTPUT_DIR
    if quick:
        OUTPUT_DIR = OUTPUT_DIR / "_quick"
        n_trials, top_k = min(n_trials, 3), min(top_k, 2)
    (OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 70)
    print("CAPSTONE - STAGE 6 HYPERPARAMETER TUNING (RANDOM SEARCH)" + (" (QUICK TEST)" if quick else ""))
    print("=" * 70)
    print(f"Trials per model x version: {n_trials} | seed check on top {top_k} | models: {', '.join(models)}")
    print()

    try:
        s5.OUTPUT_DIR = OUTPUT_DIR                       # Stage 5 helpers write sequence prep here
        train, val, features, seq = s5.build_inputs(quick)
        for v in VARIANTS:
            src = OUTPUT_DIR / f"stage5_sequence_prep_{v}.json"
            src.replace(OUTPUT_DIR / f"stage6_sequence_prep_{v}.json")
        y_train = train[cd.TARGET].to_numpy(dtype=np.float32)
        y_val = val[cd.TARGET].to_numpy(dtype=np.float32)
        w_train = cd.athlete_weights(train).astype(np.float32)
        w_val = cd.athlete_weights(val).astype(np.float32)
        ctx = {
            "out": OUTPUT_DIR, "train": train, "val": val, "features": features,
            "rnn_data": {v: (seq[v]["train"], y_train, w_train, seq[v]["val"], y_val, w_val)
                         for v in VARIANTS},
            "max_epochs": 3 if quick else RNN_MAX_EPOCHS,
            "check_seeds": CHECK_SEEDS[:1] if quick else CHECK_SEEDS,
        }
        seeds = [SEARCH_SEED] + ctx["check_seeds"]

        print()
        print("=" * 70)
        print("STEP 1-2 - RANDOM SEARCH + SEED CHECK")
        print("=" * 70)
        store = TrialStore(OUTPUT_DIR)
        best = {}
        best_path = OUTPUT_DIR / "stage6_best_params.json"
        if best_path.exists():
            best = json.loads(best_path.read_text(encoding="utf-8"))
        for model in models:
            for variant in VARIANTS:
                winner, cfg, cand = search(store, model, variant, n_trials, top_k, ctx)
                best[f"{model}_{variant}"] = {
                    "model": model, "variant": variant, "trial": winner, "params": cfg,
                    "is_stage5_setting": winner == 0,
                    "val_mae_mean": float(cand.loc[winner, "mean"]),
                    "val_mae_std": float(cand.loc[winner, "std"]),
                    "top_k": {int(t): {"mean": float(r["mean"]), "std": float(r["std"])}
                              for t, r in cand.iterrows()},
                }
                # the winner's seed models are the tuned models
                for seed in seeds:
                    key = store.key(model, variant, winner, seed)
                    ext = ".json" if model == "xgb" else ".keras"
                    shutil.copyfile(OUTPUT_DIR / "trials" / f"{key}{ext}",
                                    OUTPUT_DIR / "models" / f"{model}_{variant}_seed{seed}{ext}")
        best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")

        # ---------------- report ----------------
        print()
        print("=" * 70)
        print("STEP 3-4 - TUNED vs BASELINE")
        print("=" * 70)
        runs, preds = [], {}
        for name, pred in s5.naive_predictions(train, val).items():
            preds[name] = pred
            runs.append({**s5.evaluate(name, name, "naive", 0, val, pred), "stage": "naive"})
        for key, info in best.items():
            m, v = info["model"], info["variant"]
            plist = []
            for seed in seeds:
                p = store.pred(m, v, info["trial"], seed)
                plist.append(p)
                runs.append({**s5.evaluate(f"{key}_seed{seed}", m, v, seed, val, p), "stage": "tuned"})
            preds[f"{key}"] = np.mean(plist, axis=0)
        full_keys = [f"{m}_full" for m in MODELS if f"{m}_full" in preds]
        if len(full_keys) == 3:                               # secondary result
            preds["average_full"] = np.mean([preds[k] for k in full_keys], axis=0)
            runs.append({**s5.evaluate("average_full", "average_3_models", "full", 0, val,
                                       preds["average_full"]), "stage": "tuned"})
        runs = pd.DataFrame(runs)

        baseline_path = BASELINE_DIR / "stage5_metrics_runs.csv"
        if baseline_path.exists() and not quick:
            base = pd.read_csv(baseline_path)
            base = base[base["variant"] != "naive"].assign(stage="baseline")
            runs = pd.concat([runs, base], ignore_index=True)
        runs.to_csv(OUTPUT_DIR / "stage6_metrics_runs.csv", index=False)

        metric_cols = ["mae", "rmse", "r2", "mape", "mae_unweighted", "athlete_mae_median",
                       "finish_time_mae_s", "finish_time_mape"]
        summary = (runs.groupby(["stage", "model", "variant"])[metric_cols]
                   .agg(["mean", "std"]))
        summary.columns = [f"{a}_{b}" for a, b in summary.columns]
        summary = summary.reset_index().sort_values("mae_mean")
        summary.to_csv(OUTPUT_DIR / "stage6_metrics_summary.csv", index=False)

        pred_df = val[["sample_id", "id", "target_date", "target_distance_km",
                       "target_seconds", cd.TARGET]].copy()
        for name, pred in preds.items():
            pred_df[name] = pred
        pred_df.to_csv(OUTPUT_DIR / "stage6_val_predictions.csv", index=False)
        dist = s5.by_distance(val, preds)
        dist.to_csv(OUTPUT_DIR / "stage6_metrics_by_distance.csv", index=False)
        plot_progress(store, n_trials, OUTPUT_DIR / "stage6_search_progress.png")
        s5.plot_pred_vs_actual(val, preds, OUTPUT_DIR / "stage6_pred_vs_actual.png")

        # ---------------- checks ----------------
        assert set(train["split"]) == {"train"} and set(val["split"]) == {"val"}
        for name, pred in preds.items():
            assert np.isfinite(pred).all(), f"non-finite predictions from {name}"
        for key, info in best.items():
            n = len(store.frame(info["model"], info["variant"]).query("seed == @SEARCH_SEED"))
            assert n >= 1
        print("All checks passed.")

        # ---------------- summary text ----------------
        def get(stage, model, variant):
            r = summary[(summary["stage"] == stage) & (summary["model"] == model) &
                        (summary["variant"] == variant)]
            return r.iloc[0] if len(r) else None

        naive = summary[summary["stage"] == "naive"].sort_values("mae_mean")
        best_naive = naive.iloc[0]
        tr = pd.DataFrame(store.rows)
        lines = [
            "CAPSTONE - STAGE 6 HYPERPARAMETER TUNING SUMMARY" + (" (QUICK TEST)" if quick else ""),
            "=" * 60,
            "",
            "METHOD",
            "-" * 60,
            f"Random Search, {n_trials} trials per model x version (trial 0 = Stage 5 setting), seed {SEARCH_SEED}.",
            f"Top {top_k} trials re-trained with seeds {ctx['check_seeds']}; winner = lowest mean validation MAE "
            f"over seeds {seeds}.",
            "Score: validation MAE (RPC, athlete-weighted). Training split for fitting, validation split for "
            "early stopping and selection. Test split not used.",
            f"XGBoost: up to {XGB_FIXED['n_estimators']} trees, early stop {XGB_FIXED['early_stopping_rounds']}, "
            "MAE objective.",
            f"GRU/LSTM: up to {3 if quick else RNN_MAX_EPOCHS} epochs, early stop {RNN_PATIENCE}, MAE loss, "
            "output starts at training mean RPC.",
            f"Trials run: {len(tr)} model fits, {tr['seconds'].sum() / 3600:.1f} h of training in total.",
            "",
            "SEARCH SPACE",
            "-" * 60,
            "XGBoost: max_depth 2-10; learning_rate 0.005-0.3 (log); subsample 0.5-1.0; colsample_bytree 0.5-1.0;",
            "         min_child_weight 1-50 (log); reg_lambda 0.01-100 (log); reg_alpha {0,1e-4,1e-3,1e-2,0.1};",
            "         gamma {0,1e-4,1e-3,3e-3}",
            "GRU/LSTM: units {16,32,64,128}; layers {1,2}; dropout 0-0.4; dense {none,8,16,32};",
            "          learning_rate 3e-4-3e-3 (log); batch {32,64,128}; L2 {0,1e-6,1e-5,1e-4}",
            "",
            "BEST SETTINGS",
            "-" * 60,
        ]
        for key, info in best.items():
            tag = " (= Stage 5 setting)" if info["is_stage5_setting"] else ""
            lines.append(f"{s5.MODEL_LABELS[info['model']]} {info['variant']}: trial {info['trial']}{tag}")
            lines.append(f"    {describe(info['model'], info['params'])}")
        lines += [
            "",
            "RESULTS (validation; athlete-weighted; mean ± SD over seeds)",
            "-" * 60,
            f"{'Model':<10}{'Version':<13}{'Baseline MAE':<22}{'Tuned MAE':<22}{'Change':<10}{'Tuned R2':<10}"
            f"{'Finish-time MAPE'}",
        ]
        for model in MODELS:
            for variant in VARIANTS:
                t, b = get("tuned", model, variant), get("baseline", model, variant)
                if t is None:
                    continue
                bt = f"{b['mae_mean']:.4f} ± {b['mae_std']:.4f}" if b is not None else "-"
                ch = f"{100 * (b['mae_mean'] - t['mae_mean']) / b['mae_mean']:+.1f}%" if b is not None else "-"
                tt = f"{t['mae_mean']:.4f} ± {t['mae_std']:.4f}"
                lines.append(f"{s5.MODEL_LABELS[model]:<10}{variant:<13}{bt:<22}{tt:<22}{ch:<10}"
                             f"{t['r2_mean']:<10.3f}{t['finish_time_mape_mean']:.1f} %")
        avg = get("tuned", "average_3_models", "full")
        if avg is not None:
            lines.append(f"{'Average of the 3 tuned full models (secondary)':<45}MAE {avg['mae_mean']:.4f}  "
                         f"R2 {avg['r2_mean']:.3f}")
        lines += ["", "Naive baselines:"]
        for _, r in naive.iterrows():
            lines.append(f"  {r['model']:<22} MAE {r['mae_mean']:.4f}")
        lines += ["", "Error reduction vs best naive baseline "
                  f"({best_naive['model']}, MAE {best_naive['mae_mean']:.4f}), tuned full models:"]
        for model in MODELS + ["average_3_models"]:
            t = get("tuned", model, "full")
            if t is not None:
                red = 100 * (best_naive["mae_mean"] - t["mae_mean"]) / best_naive["mae_mean"]
                lines.append(f"  {model:<18} {red:+.1f}%")
        lines += [
            "(error reduction = (baseline MAE - model MAE) / baseline MAE; positive = better)",
            "",
            "MAE BY BEST-EFFORT DISTANCE (seed-mean predictions)",
            "-" * 60,
        ]
        piv = dist.pivot_table(index="prediction", columns="distance_km", values="rpc_mae")
        n_by = dist.groupby("distance_km")["n"].first()
        lines.append("n per distance: " + ", ".join(f"{k:g} km {int(v)}" for k, v in n_by.items()))
        lines.append(piv.round(4).to_string())
        lines += [
            "",
            "NOTE: selection used the validation set, so these validation scores are slightly optimistic;",
            "the unbiased comparison is the test set (Stage 7).",
            "",
            "STATUS",
            "-" * 60,
            "Stage 6 tuning completed for: " + ", ".join(sorted(best)) + ". Next: Stage 7 test-set evaluation.",
        ]
        (OUTPUT_DIR / "stage6_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines[lines.index("BEST SETTINGS"):]))
        print(f"\nOutputs written to:\n{OUTPUT_DIR}")

    except KeyboardInterrupt:
        print("\nStopped. Finished trials are saved - run the same command again to continue.")
    except Exception as exc:
        (LOG_DIR / "stage6_tuning_error.log").write_text(
            "STAGE 6 ERROR\n" + "=" * 60 + "\n\n" + str(exc) + "\n\n" + traceback.format_exc(),
            encoding="utf-8",
        )
        print("\nERROR - see logs/stage6_tuning_error.log (finished trials are saved)")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS),
                        help="comma-separated subset of: xgb,gru,lstm")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help=f"trials per model x version (default {DEFAULT_TRIALS})")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"trials re-checked with extra seeds (default {TOP_K})")
    parser.add_argument("--quick", action="store_true",
                        help="smoke test: 25 athletes, 3 trials, 3 epochs, 2 seeds")
    args = parser.parse_args()
    chosen = [m.strip() for m in args.models.split(",") if m.strip()]
    assert set(chosen) <= set(MODELS), f"unknown model in {chosen}"
    assert args.trials >= 1 and args.top_k >= 1
    main(chosen, args.trials, args.top_k, args.quick)
