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


def _download_gdrive_folder(folder_url: str) -> Path:
    """Download a shared Drive folder once and return its local cache path."""
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError("Google Drive input requires gdown; install requirements.txt") from exc

    cache_dir = config.GDRIVE_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    gdown.download_folder(folder_url, output=str(cache_dir), quiet=False, resume=True)
    return cache_dir


def find_file(city: str, data_dir=None, gdrive_folder=None) -> Path:
    if data_dir is not None:
        d = Path(data_dir)
    elif gdrive_folder or config.GDRIVE_FOLDER_URL:
        d = _download_gdrive_folder(gdrive_folder or config.GDRIVE_FOLDER_URL)
    else:
        d = config.DATA_DIR
    for name in config.CITY_FILES[city]:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(
        f"No {city} file in {d}. Expected one of {config.CITY_FILES[city]} "
        "(see data/README.md or the Google Drive folder)."
    )


def load_city(city: str, data_dir=None, gdrive_folder=None, cap=config.PRICE_CAP) -> pd.DataFrame:
    """Read one city, parse the price and keep 0 < price <= cap."""
    df = pd.read_csv(find_file(city, data_dir, gdrive_folder), low_memory=False)
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
