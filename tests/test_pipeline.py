"""Smoke tests on synthetic data (no real files needed):  pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_city, split, to_number
from src.features import FeatureBuilder, distance_to_centre_km
from src.models import get_model
from tests.synthetic import write_all


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    write_all(d)
    return d


def test_to_number():
    s = to_number(pd.Series(["$1,234.50", "95%", None, "abc"]))
    assert s.iloc[0] == 1234.5 and s.iloc[1] == 95 and s.iloc[2:].isna().all()


def test_cleaning_and_split(data_dir):
    df = load_city("paris", data_dir)
    assert df["price"].between(0, config.PRICE_CAP).all()
    tr, dv, te = split(df)
    assert len(tr) + len(dv) + len(te) == len(df)


def test_distance_to_centre():
    d = pd.DataFrame({"city": ["paris"], "latitude": [48.8566], "longitude": [2.3522]})
    assert distance_to_centre_km(d)[0] < 1e-6


@pytest.mark.parametrize("groups", ["C", "O", "T", "D", "C+O+T+D"])
def test_features_no_nan_and_fixed_width(data_dir, groups):
    tr, _, te = split(load_city("berlin", data_dir))
    fb = FeatureBuilder(groups).fit(tr)
    a, b = fb.transform(tr), fb.transform(te)
    assert a.shape[1] == b.shape[1] == len(fb.names) and np.isfinite(a).all() and np.isfinite(b).all()


def test_indicators_not_scaled(data_dir):
    tr, _, _ = split(load_city("paris", data_dir))
    fb = FeatureBuilder("O").fit(tr)
    assert set(np.unique(fb.transform(tr))) <= {0.0, 1.0}


def test_cross_city_transform(data_dir):
    tr = pd.concat([load_city(c, data_dir) for c in ("nyc", "paris")], ignore_index=True)
    fb = FeatureBuilder("C+O", cross_city=True).fit(tr)
    X = fb.transform(load_city("berlin", data_dir))
    assert X.shape[1] == len(fb.names) and np.isfinite(X).all()
    assert not any(n.startswith("neighbourhood") for n in fb.names)


@pytest.mark.parametrize("name", ["linear", "rf", "xgb", "nn", "nn_small"])
def test_models_learn(data_dir, name):
    tr, _, te = split(load_city("nyc", data_dir))
    fb = FeatureBuilder("C+O").fit(tr)
    m = get_model(name).fit(fb.transform(tr), np.log(tr["price"]))
    r2 = 1 - np.mean((m.predict(fb.transform(te)) - np.log(te["price"])) ** 2) / np.var(np.log(te["price"]))
    assert r2 > 0.3, f"{name}: R2={r2:.2f}"


def test_scripts_end_to_end(data_dir, tmp_path, monkeypatch):
    from src import cross_city, single_city
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "FIG_DIR", tmp_path / "figures")
    s = single_city.main(["--quick", "--data-dir", str(data_dir)])
    assert len(s) == 3 * len(single_city.ALL_MODELS)
    c = cross_city.main(["--models", "xgb", "--seeds", "42", "--data-dir", str(data_dir)])
    assert len(c) == 9 and (tmp_path / "figures" / "cross_city_xgb.png").exists()
