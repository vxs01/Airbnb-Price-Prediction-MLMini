"""All figures used in the README (matplotlib, saved to results/figures/)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # validated categorical slots 1-3
CITY_COLOR = dict(zip(config.CITIES, SERIES))
CITY_NAME = {"berlin": "Berlin", "paris": "Paris", "nyc": "NYC"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 9,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
})


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / name, dpi=150)
    plt.close(fig)


def _grouped_bars(ax, groups, series, values, errors=None, colors=SERIES, fmt="{:.2f}", floor=None):
    """values[s][g] -> bars; one colour per series, a 2px surface gap between bars."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        pos = x - 0.4 + width * (i + 0.5)
        v = np.asarray(values[i], float)
        err = None if errors is None else errors[i]
        ax.bar(pos, v, width * 0.92, color=colors[i], label=s, yerr=err,
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=2), edgecolor=SURFACE, linewidth=1)
        e = np.zeros_like(v) if err is None else np.nan_to_num(np.asarray(err, float))
        for p, val, ev in zip(pos, v, e):
            if not np.isfinite(val) or (floor is not None and val <= floor):
                continue
            if val >= 0:
                ax.text(p, val + ev + 0.012, fmt.format(val), ha="center", va="bottom", fontsize=7, color=INK2)
            else:
                ax.text(p, val - ev - 0.012, fmt.format(val), ha="center", va="top", fontsize=7, color=INK2)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def price_distribution(frames):
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.linspace(np.log(5), np.log(500), 60)
    for city, df in frames.items():
        lp = np.log(df["price"])
        ax.hist(lp, bins=bins, histtype="step", lw=2, color=CITY_COLOR[city], density=True,
                label=f"{CITY_NAME[city]}  (n={len(df):,}, median {df['price'].median():.0f})")
        ax.axvline(np.median(lp), color=CITY_COLOR[city], lw=1, ls="--")
    ticks = [10, 25, 50, 100, 200, 500]
    ax.set_xticks(np.log(ticks), ticks)
    ax.set_xlabel("Nightly price (log scale, local currency)")
    ax.set_ylabel("Density")
    ax.set_title("Price distribution per city (after the 500 cap)")
    ax.legend(loc="upper left")
    _save(fig, "price_distribution.png")


def single_city_r2(summary):
    """summary: DataFrame with city, model, r2_mean, r2_std."""
    from .models import LABELS
    models = list(dict.fromkeys(summary["model"]))
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    vals, errs = [], []
    for city in config.CITIES:
        s = summary[summary.city == city].set_index("model").reindex(models)
        vals.append(s["r2_mean"].to_numpy())
        errs.append(s["r2_std"].to_numpy())
    _grouped_bars(ax, [LABELS[m].replace(" (", "\n(") for m in models],
                  [CITY_NAME[c] for c in config.CITIES], vals, errs)
    ax.set_ylabel("Test R²  (log price)")
    ax.set_ylim(min(0, np.nanmin(vals) - 0.12), 1)
    ax.set_title("Single-city test R² (mean ± std over 3 random splits)")
    ax.legend(ncol=3, loc="upper left")
    _save(fig, "single_city_r2.png")


def pred_vs_actual(preds, model_label):
    """preds: {city: (y_true_price, y_pred_price)}"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharex=True, sharey=True)
    for ax, city in zip(axes, config.CITIES):
        yt, yp = preds[city]
        ax.scatter(yt, yp, s=6, alpha=0.35, color=CITY_COLOR[city], edgecolors="none")
        ax.plot([5, 500], [5, 500], color=INK2, lw=1, ls="--")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(8, 520); ax.set_ylim(8, 520)
        mae = np.mean(np.abs(yt - yp))
        ax.set_title(f"{CITY_NAME[city]}   (MAE {mae:.0f} per night)")
        ax.set_xlabel("Actual price")
    axes[0].set_ylabel("Predicted price")
    fig.suptitle(f"{model_label}: predicted vs actual on the test set", fontsize=11, fontweight="bold")
    _save(fig, "pred_vs_actual.png")


def feature_importance(imps, top=12):
    """imps: {city: pd.Series(name -> gain)}"""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, city in zip(axes, config.CITIES):
        s = imps[city].sort_values().tail(top)
        s = s / imps[city].sum() * 100
        ax.barh([n[:32] for n in s.index], s.values, color=CITY_COLOR[city], height=0.7)
        ax.set_title(CITY_NAME[city])
        ax.set_xlabel("Share of total gain (%)")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7.5)
    fig.suptitle(f"Top {top} XGBoost features", fontsize=11, fontweight="bold")
    _save(fig, "feature_importance.png")


def ablations(label_df, feat_df):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1, 2.2]))
    labels = list(dict.fromkeys(label_df["label"]))
    _grouped_bars(a1, labels, [CITY_NAME[c] for c in config.CITIES],
                  [label_df[label_df.city == c].set_index("label").reindex(labels)["test_r2_price"]
                   for c in config.CITIES])
    a1.set_title("Label transform (C features)")
    a1.set_ylabel("Test R²  (price in currency units)")
    a1.set_ylim(0, 1)
    feats = list(dict.fromkeys(feat_df["features"]))
    _grouped_bars(a2, feats, [CITY_NAME[c] for c in config.CITIES],
                  [feat_df[feat_df.city == c].set_index("features").reindex(feats)["test_r2"]
                   for c in config.CITIES])
    a2.set_title("Feature groups (log label)")
    a2.set_ylabel("Test R²  (log price)")
    a2.set_ylim(min(0, feat_df["test_r2"].min() - 0.12), 1)
    a2.legend(ncol=3, loc="upper left")
    fig.suptitle("XGBoost ablations", fontsize=11, fontweight="bold")
    _save(fig, "ablations.png")


def cross_city(df, model):
    """Berlin R² for each training set and setup; values below the axis floor are labelled."""
    from .models import LABELS
    names = {"original": "Original notebook (one-hots z-scored)", "fixed": "Scaling fixed (zero-shot)",
             "calibrated": "Fixed + calibrated on 100 Berlin listings"}
    d = df[df.model == model].assign(setup=lambda x: x["setup"].map(names))
    trains = list(dict.fromkeys(d["trained_on"]))
    setups = list(dict.fromkeys(d["setup"]))
    floor = -1.0
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = [d[d.setup == s].set_index("trained_on").reindex(trains)["berlin_r2"].to_numpy()
            for s in setups]
    errs = [np.where(v > floor, d[d.setup == s].set_index("trained_on").reindex(trains)["berlin_r2_std"], 0)
            for s, v in zip(setups, vals)]
    _grouped_bars(ax, trains, setups, [np.maximum(v, floor) for v in vals], errs, floor=floor)
    width = 0.8 / len(setups)
    for i, v in enumerate(vals):                 # print the true value of clipped bars
        for j, val in enumerate(v):
            if val < floor:
                ax.text(j - 0.4 + width * (i + 0.5), floor + 0.03, f"{val:,.0f}", ha="center",
                        va="bottom", fontsize=7, color=INK, rotation=90)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylim(floor, 1.05)
    ax.set_ylabel("R² on Berlin (log price)")
    ax.set_title(f"Transfer to unseen Berlin — {LABELS[model]} (mean ± std, 3 seeds)")
    ax.legend(loc="upper left", ncol=1)
    _save(fig, f"cross_city_{model}.png")
