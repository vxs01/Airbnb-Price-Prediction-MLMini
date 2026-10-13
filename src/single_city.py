"""Single-city experiments: five models per city + ablations (paper Tables 1-4).

    python -m src.single_city                 # all cities, 3 seeds, all models + ablations
    python -m src.single_city --quick         # 1 seed, no ablations
"""
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from . import config, plots
from .data import load_city, log_price, split
from .features import FeatureBuilder
from .models import LABELS, MODELS, get_model

ALL_MODELS = MODELS + ["xgb_tuned"]


def fit_predict(name, seed, Xtr, ytr, Xdv, ydv, *Xs):
    m = get_model(name, seed)
    if name == "xgb_tuned":
        m.fit(Xtr, ytr, eval_set=[(Xdv, ydv)], verbose=False)
    else:
        m.fit(Xtr, ytr)
    return m, [m.predict(X) for X in (Xtr, Xdv, *Xs)]


def run_models(city, df, seed, models, groups="C+O+T+D"):
    train, dev, test = split(df, seed)
    fb = FeatureBuilder(groups, seed=seed)
    Xtr, Xdv, Xte = fb.fit_transform(train), fb.transform(dev), fb.transform(test)
    ytr, ydv, yte = log_price(train), log_price(dev), log_price(test)
    rows, extras = [], {}
    for name in models:
        t0 = time.time()
        m, (ptr, pdv, pte) = fit_predict(name, seed, Xtr, ytr, Xdv, ydv, Xte)
        rows.append(dict(city=city, model=name, seed=seed, n_features=Xtr.shape[1],
                         train_r2=r2_score(ytr, ptr), dev_r2=r2_score(ydv, pdv),
                         test_mse=mean_squared_error(yte, pte), test_r2=r2_score(yte, pte),
                         test_mae_price=np.mean(np.abs(np.exp(pte) - test["price"])),
                         seconds=round(time.time() - t0, 1)))
        print(f"  [{city} seed {seed}] {name:9s} test R2={rows[-1]['test_r2']:.3f} "
              f"MSE={rows[-1]['test_mse']:.4f}  ({rows[-1]['seconds']}s)", flush=True)
        if name == "xgb_tuned":
            extras["pred"] = (test["price"].to_numpy(), np.exp(pte))
            gain = m.get_booster().get_score(importance_type="total_gain")
            extras["importance"] = pd.Series({fb.names[int(k[1:])]: v for k, v in gain.items()})
    return rows, extras


def ablations(city, df, seed=config.SEED):
    """XGBoost (notebook settings) with different labels and feature groups."""
    train, dev, test = split(df, seed)
    lab_rows, feat_rows = [], []
    fb = FeatureBuilder("C", seed=seed)
    Xtr, Xte = fb.fit_transform(train), fb.transform(test)
    fwd = {"none": lambda p: p, "sqrt": np.sqrt, "log": np.log}
    inv = {"none": lambda y: y, "sqrt": lambda y: np.clip(y, 0, None) ** 2, "log": np.exp}
    for lab in ("none", "sqrt", "log"):
        m = get_model("xgb", seed).fit(Xtr, fwd[lab](train["price"].to_numpy(float)))
        pte = m.predict(Xte)
        lab_rows.append(dict(city=city, label=lab,
                             test_r2_own_scale=r2_score(fwd[lab](test["price"]), pte),
                             test_r2_price=r2_score(test["price"], inv[lab](pte)),
                             test_mae_price=np.mean(np.abs(inv[lab](pte) - test["price"]))))
    for groups in ("C", "O", "T", "D", "C+O", "C+O+T", "C+O+T+D"):
        fb = FeatureBuilder(groups, seed=seed)
        Xtr, Xte = fb.fit_transform(train), fb.transform(test)
        pte = get_model("xgb", seed).fit(Xtr, log_price(train)).predict(Xte)
        feat_rows.append(dict(city=city, features=groups, n_features=Xtr.shape[1],
                              test_mse=mean_squared_error(log_price(test), pte),
                              test_r2=r2_score(log_price(test), pte)))
    print(f"  [{city}] ablations done", flush=True)
    return lab_rows, feat_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cities", nargs="+", default=list(config.CITIES))
    p.add_argument("--models", nargs="+", default=ALL_MODELS)
    p.add_argument("--quick", action="store_true", help="1 seed, no ablations")
    p.add_argument("--data-dir", default=None)
    a = p.parse_args(argv)
    seeds = config.SEEDS[:1] if a.quick else config.SEEDS
    config.RESULTS_DIR.mkdir(exist_ok=True)

    frames = {c: load_city(c, a.data_dir) for c in a.cities}
    rows, preds, imps, lab, feat = [], {}, {}, [], []
    for city, df in frames.items():
        print(f"{city}: {len(df):,} listings after cleaning", flush=True)
        for seed in seeds:
            r, extras = run_models(city, df, seed, a.models)
            rows += r
            if seed == config.SEED and extras:
                preds[city], imps[city] = extras["pred"], extras["importance"]
        if not a.quick:
            l, f = ablations(city, df)
            lab += l
            feat += f

    res = pd.DataFrame(rows)
    res.to_csv(config.RESULTS_DIR / "single_city_runs.csv", index=False)
    summary = (res.groupby(["city", "model"], sort=False)
                  .agg(n_features=("n_features", "first"), train_r2=("train_r2", "mean"),
                       test_mse=("test_mse", "mean"), r2_mean=("test_r2", "mean"),
                       r2_std=("test_r2", "std"), mae_price=("test_mae_price", "mean"))
                  .reset_index().fillna({"r2_std": 0}))
    summary.round(4).to_csv(config.RESULTS_DIR / "single_city_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))

    if set(frames) == set(config.CITIES):
        plots.price_distribution(frames)
        plots.single_city_r2(summary)
        if len(preds) == 3:
            plots.pred_vs_actual(preds, LABELS["xgb_tuned"])
            plots.feature_importance(imps)
        if lab:
            lab_df, feat_df = pd.DataFrame(lab), pd.DataFrame(feat)
            lab_df.round(4).to_csv(config.RESULTS_DIR / "ablation_label.csv", index=False)
            feat_df.round(4).to_csv(config.RESULTS_DIR / "ablation_features.csv", index=False)
            plots.ablations(lab_df, feat_df)
    return summary


if __name__ == "__main__":
    main()
