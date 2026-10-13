"""Load, clean and split the Inside Airbnb listing files."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def to_number(s: pd.Series) -> pd.Series:
    """'$1,234.00' / '95%' -> 1234.0 / 95.0; anything unparsable -> NaN."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,%\s]", "", regex=True), errors="coerce")


def find_file(city: str, data_dir=None) -> Path:
    d = Path(data_dir or config.DATA_DIR)
    for name in config.CITY_FILES[city]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"No {city} file in {d}. Expected one of {config.CITY_FILES[city]} "
                            "(see data/README.md).")


def load_city(city: str, data_dir=None, cap=config.PRICE_CAP) -> pd.DataFrame:
    """Read one city, parse the price and keep 0 < price <= cap."""
    df = pd.read_csv(find_file(city, data_dir), low_memory=False)
    df["price"] = to_number(df["price"])
    df = df[df["price"].gt(0)]
    if cap:
        df = df[df["price"].le(cap)]
    df = df.reset_index(drop=True)
    df["city"] = city
    return df


def split(df: pd.DataFrame, seed=config.SEED):
    """70 / 20 / 10 train / dev / test split."""
    tr, dv, te = config.SPLIT
    train, rest = train_test_split(df, train_size=tr, random_state=seed)
    dev, test = train_test_split(rest, train_size=dv / (dv + te), random_state=seed)
    return [x.reset_index(drop=True) for x in (train, dev, test)]


def log_price(df: pd.DataFrame) -> np.ndarray:
    return np.log(df["price"].to_numpy(float))
