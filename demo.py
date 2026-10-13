"""Live demo: train the tuned XGBoost on one city and price a few unseen test listings.

    python demo.py --city paris --n 8
"""
import argparse

import numpy as np
import pandas as pd

from src import config
from src.data import load_city, log_price, split
from src.features import FeatureBuilder
from src.models import get_model


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--city", default="paris", choices=config.CITIES)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--data-dir", default=None)
    a = p.parse_args(argv)

    train, dev, test = split(load_city(a.city, a.data_dir))
    fb = FeatureBuilder("C+O+T+D").fit(train)
    model = get_model("xgb_tuned").fit(fb.transform(train), log_price(train),
                                       eval_set=[(fb.transform(dev), log_price(dev))], verbose=False)
    sample = test.sample(a.n, random_state=1)
    pred = np.exp(model.predict(fb.transform(sample)))

    out = pd.DataFrame({
        "neighbourhood": sample["neighbourhood_cleansed"].str[:22],
        "room_type": sample["room_type"],
        "guests": sample["accommodates"],
        "actual": sample["price"].round(0).astype(int),
        "predicted": pred.round(0).astype(int),
    })
    out["error_%"] = ((out["predicted"] - out["actual"]).abs() / out["actual"] * 100).round(0).astype(int)
    print(f"\n{a.city.upper()} - {len(train):,} training listings, {model.best_iteration + 1} trees\n")
    print(out.to_string(index=False))
    all_pred = np.exp(model.predict(fb.transform(test)))
    print(f"\nWhole test set ({len(test):,} listings): median absolute error "
          f"{np.median(np.abs(all_pred - test['price'])):.0f} per night")


if __name__ == "__main__":
    main()
