"""Cross-city learning (paper Tables 6 and 7): train on NYC / Paris / NYC+Paris, test on Berlin.

Three setups, same data, same C+O features (neighbourhoods left out - they are city-specific):
  original    - the old notebook's method: every column, including 0/1 one-hots, z-scored
                with the training cities' statistics
  fixed       - only real-valued columns are scaled; 0/1 columns stay 0/1
  calibrated  - fixed + the model learns log price relative to each training city's median
                and is mapped onto Berlin by a linear fit on N_CALIB labelled Berlin listings

Berlin metrics use all Berlin listings except the N_CALIB calibration ones; Berlin is never
used for training. Repeated for 3 seeds (split, model init and calibration sample change).

    python -m src.cross_city
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from . import config, plots
from .data import load_city, log_price, split
from .features import FeatureBuilder
from .models import get_model

N_CALIB = 100
TRAIN_SETS = {"NYC": ["nyc"], "Paris": ["paris"], "NYC + Paris": ["nyc", "paris"]}
SETUPS = ["original", "fixed", "calibrated"]


def run(setup, model_name, trained_on, splits, berlin_eval, berlin_calib, seed):
    cities = TRAIN_SETS[trained_on]
    train = pd.concat([splits[c][0] for c in cities], ignore_index=True)
    y = log_price(train)
    fb = FeatureBuilder("C+O", cross_city=True, seed=seed).fit(train)
    if setup == "original":
        scaler = StandardScaler().fit(fb.transform(train))
        feats = lambda d: scaler.transform(fb.transform(d))
    else:
        feats = fb.transform

    level = {c: 0.0 for c in config.CITIES}           # city price level (log), 0 = not used
    if setup == "calibrated":
        level.update({c: float(np.median(log_price(splits[c][0]))) for c in cities})
    offset = lambda d: d["city"].map(level).to_numpy(float)
    model = get_model(model_name, seed).fit(feats(train), y - offset(train))
    a, b = 0.0, 1.0
    if setup == "calibrated":                          # y = a + b * prediction on Berlin
        b, a = np.polyfit(model.predict(feats(berlin_calib)), log_price(berlin_calib), 1)

    def pred(d):
        p = model.predict(feats(d))
        return a + b * p if d["city"].iloc[0] == "berlin" else p + offset(d)

    yb, pb = log_price(berlin_eval), pred(berlin_eval)
    row = dict(setup=setup, model=model_name, trained_on=trained_on, seed=seed,
               n_features=len(fb.names), train_r2=r2_score(y, pred(train)))
    for c in cities:                                   # Table 6: own held-out test split
        row[f"{c}_test_r2"] = r2_score(log_price(splits[c][2]), pred(splits[c][2]))
    row.update(berlin_mse=mean_squared_error(yb, pb), berlin_r2=r2_score(yb, pb),
               berlin_bias=float(np.mean(pb - yb)),
               berlin_mae_price=float(np.mean(np.abs(np.exp(np.clip(pb, -20, 20)) - berlin_eval["price"]))))
    print(f"  seed {seed} {setup:10s} {model_name:8s} {trained_on:12s} Berlin R2={row['berlin_r2']:>10.3f}",
          flush=True)
    return row


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["nn_small", "xgb"])
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.SEEDS))
    p.add_argument("--data-dir", default=None)
    p.add_argument("--gdrive-folder", default=None, help="Google Drive folder URL")
    a = p.parse_args(argv)

    data = {c: load_city(c, a.data_dir, a.gdrive_folder) for c in config.CITIES}
    rows = []
    for seed in a.seeds:
        splits = {c: split(data[c], seed) for c in ("nyc", "paris")}
        berlin = data["berlin"].sample(frac=1, random_state=seed).reset_index(drop=True)
        calib, evaluation = berlin.iloc[:N_CALIB], berlin.iloc[N_CALIB:].reset_index(drop=True)
        rows += [run(s, m, t, splits, evaluation, calib, seed)
                 for m in a.models for s in SETUPS for t in TRAIN_SETS]

    runs = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    runs.to_csv(config.RESULTS_DIR / "cross_city_runs.csv", index=False)
    metrics = [c for c in runs.columns if c.endswith(("_r2", "_mse", "_bias", "_price"))]
    keys = ["model", "setup", "trained_on"]
    summary = runs.groupby(keys, sort=False)[metrics].mean().reset_index()
    summary["berlin_r2_std"] = runs.groupby(keys, sort=False)["berlin_r2"].std().fillna(0).values
    summary.round(4).to_csv(config.RESULTS_DIR / "cross_city_summary.csv", index=False)
    print(summary.round(3).to_string(index=False))
    for m in a.models:            # 'original' is kept in the CSV; the figure shows the two fixed setups
        plots.cross_city(summary[summary.setup != "original"], m)
    return summary


if __name__ == "__main__":
    main()
