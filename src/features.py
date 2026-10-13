"""Feature groups from the paper: C (continuous), O (categorical/amenities), T (text), D (dates).

`FeatureBuilder` is fitted on the training rows only and then applied unchanged to
dev / test / other cities, so nothing leaks from evaluation data. Real-valued
columns are clipped to the training 1st-99th percentile and standardised; 0/1 indicator
columns are left as 0/1 (z-scoring a rare indicator turns it into a value of ~100 on
another city and breaks the network).
"""
import ast
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .data import to_number


def parse_list(v) -> list:
    """'["Wifi", "TV"]' or "['email', 'phone']" -> ['wifi', 'tv']."""
    if not isinstance(v, str):
        return []
    try:
        x = ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return []
    return [str(i).strip().lower() for i in x] if isinstance(x, (list, tuple, set)) else []


def distance_to_centre_km(df: pd.DataFrame) -> np.ndarray:
    """Haversine distance from each listing to its own city's centre."""
    centre = np.array([config.CITY_CENTRE[c] for c in df["city"]])
    lat1, lon1 = np.radians(df["latitude"].to_numpy(float)), np.radians(df["longitude"].to_numpy(float))
    lat2, lon2 = np.radians(centre[:, 0]), np.radians(centre[:, 1])
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _numeric(df: pd.DataFrame, cols) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        out[c] = to_number(df[c]) if c in df.columns else np.nan
    if "bathrooms" in cols and "bathrooms_text" in df.columns:   # '1.5 baths' -> 1.5
        from_text = pd.to_numeric(df["bathrooms_text"].astype(str).str.extract(r"(\d+\.?\d*)")[0],
                                  errors="coerce")
        out["bathrooms"] = out["bathrooms"].fillna(from_text)
    return out


class FeatureBuilder:
    def __init__(self, groups="C+O+T+D", cross_city=False, latlon=True, seed=config.SEED):
        self.groups = [g for g in groups.upper().split("+") if g]
        assert set(self.groups) <= set("COTD"), f"unknown feature group in {groups}"
        self.cross_city = cross_city      # drop neighbourhoods (city-specific categories)
        self.latlon = latlon              # raw coordinates (also tell the cities apart)
        self.seed = seed

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame):
        self.names = []
        if "C" in self.groups:
            cols = config.CONTINUOUS + (config.LAT_LON if self.latlon else [])
            raw = _numeric(df, cols)
            self.c_cols = [c for c in cols if raw[c].notna().any()]
            self.c_median = raw[self.c_cols].median()
            x = self._continuous(df, clip=False)
            # winsorise to the training 1st-99th percentile: the raw files contain sentinels such
            # as maximum_nights = 2,147,483,647 that would otherwise dominate the scaling
            self.c_lo, self.c_hi = np.percentile(x, 1, axis=0), np.percentile(x, 99, axis=0)
            self.c_scaler = StandardScaler().fit(np.clip(x, self.c_lo, self.c_hi))
            self.names += self.c_cols + ["dist_centre_km"]
        if "O" in self.groups:
            cats = [c for c in config.CATEGORICAL
                    if c in df.columns and df[c].notna().any()
                    and not (self.cross_city and c in config.CITY_SPECIFIC)]
            self.o_cols = cats
            self.ohe = OneHotEncoder(handle_unknown="infrequent_if_exist", sparse_output=False,
                                     min_frequency=config.MIN_CATEGORY_COUNT)
            self.ohe.fit(self._cats(df))
            self.names += list(self.ohe.get_feature_names_out(cats))
            self.vocab = {}
            for col in ("amenities", "host_verifications"):
                if col in df.columns:
                    counts = Counter(t for v in df[col] for t in set(parse_list(v)))
                    keep = [t for t, n in counts.items() if n >= config.AMENITY_MIN_SHARE * len(df)]
                    self.vocab[col] = sorted(keep)
                    self.names += [f"{col}={t}" for t in self.vocab[col]]
        if "T" in self.groups:
            self.text = {}
            for col in config.TEXT:
                if col not in df.columns or df[col].notna().sum() < 100:
                    continue
                tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=config.TFIDF_MAX_FEATURES,
                                        stop_words="english", min_df=2, sublinear_tf=True)
                m = tfidf.fit_transform(df[col].fillna("").astype(str))
                svd = TruncatedSVD(config.SVD_DIM, random_state=self.seed).fit(m)
                self.text[col] = (tfidf, svd)
                self.names += [f"{col}_svd{i}" for i in range(config.SVD_DIM)]
            if self.text:
                self.t_scaler = StandardScaler().fit(self._text(df))
        if "D" in self.groups:
            days = self._days(df)
            self.d_cols = [c for c in days if days[c].notna().any()]
            self.d_median = days[self.d_cols].median()
            if self.d_cols:
                self.d_scaler = StandardScaler().fit(days[self.d_cols].fillna(self.d_median))
                self.names += [f"{c}_days" for c in self.d_cols]
        return self

    # -------------------------------------------------------------- helpers
    def _continuous(self, df, clip=True):
        x = _numeric(df, self.c_cols).fillna(self.c_median).fillna(0)
        x["dist_centre_km"] = distance_to_centre_km(df)
        x = x.to_numpy(float)
        return np.clip(x, self.c_lo, self.c_hi) if clip else x

    def _cats(self, df):
        return pd.DataFrame({c: df[c].fillna("missing").astype(str) if c in df.columns else "missing"
                             for c in self.o_cols}, index=df.index)

    def _text(self, df):
        return np.hstack([svd.transform(tfidf.transform(df[c].fillna("").astype(str)
                                                         if c in df.columns else [""] * len(df)))
                          for c, (tfidf, svd) in self.text.items()])

    def _days(self, df):
        origin = pd.Timestamp(config.DATE_ORIGIN)
        return pd.DataFrame({c: (pd.to_datetime(df[c], errors="coerce", format="%Y-%m-%d") - origin).dt.days
                             if c in df.columns else np.nan for c in config.DATES}, index=df.index)

    # ------------------------------------------------------------ transform
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        parts = []
        if "C" in self.groups:
            parts.append(self.c_scaler.transform(self._continuous(df)))
        if "O" in self.groups:
            parts.append(self.ohe.transform(self._cats(df)))
            for col, vocab in self.vocab.items():
                idx = {t: i for i, t in enumerate(vocab)}
                m = np.zeros((len(df), len(vocab)))
                if col in df.columns:
                    for r, v in enumerate(df[col]):
                        for t in parse_list(v):
                            if t in idx:
                                m[r, idx[t]] = 1
                parts.append(m)
        if "T" in self.groups and self.text:
            parts.append(self.t_scaler.transform(self._text(df)))
        if "D" in self.groups and self.d_cols:
            d = self._days(df)[self.d_cols].fillna(self.d_median)
            parts.append(self.d_scaler.transform(d))
        return np.hstack(parts).astype(np.float32)

    def fit_transform(self, df):
        return self.fit(df).transform(df)
