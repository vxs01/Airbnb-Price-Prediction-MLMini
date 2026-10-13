"""The five models from the paper plus a tuned XGBoost."""
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from . import config

MODELS = ["linear", "knn", "rf", "xgb", "nn"]
LABELS = {"linear": "Linear (Ridge)", "knn": "KNN", "rf": "Random Forest", "xgb": "XGBoost",
          "xgb_tuned": "XGBoost (tuned)", "nn": "Neural Network", "nn_small": "Neural Network (64-32)"}


def _std_target(model):
    # the network's output starts near 0; centring the label (log price ~ 4.8) makes training stable
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def get_model(name: str, seed=config.SEED):
    if name == "linear":      # linear regression with L2 penalty
        return Ridge(alpha=1.0)
    if name == "knn":         # unweighted, Euclidean, k = 5
        return KNeighborsRegressor(n_neighbors=5, n_jobs=-1)
    if name == "rf":
        return RandomForestRegressor(n_estimators=100, max_features=0.33, n_jobs=-1, random_state=seed)
    if name == "xgb":         # same settings as the original notebook
        return XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=seed)
    if name == "xgb_tuned":   # more, smaller steps + row/column subsampling; stops on the dev set
        return XGBRegressor(n_estimators=2000, max_depth=7, learning_rate=0.03, subsample=0.8,
                            colsample_bytree=0.5, min_child_weight=3, reg_lambda=1.0,
                            early_stopping_rounds=100, n_jobs=-1, random_state=seed)
    if name == "nn":          # paper's first network: 128-512-64, Adam lr 1e-4, weight decay 1e-5, batch 64
        return _std_target(MLPRegressor(
            hidden_layer_sizes=(128, 512, 64), solver="adam", learning_rate_init=1e-4, alpha=1e-5,
            batch_size=64, max_iter=200, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10, tol=1e-4, random_state=seed))
    if name == "nn_small":    # paper's transfer network: 64-32, SGD + Nesterov (0.9, lr 0.005), L2 0.005
        return _std_target(MLPRegressor(
            hidden_layer_sizes=(64, 32), solver="sgd", learning_rate_init=0.005, momentum=0.9,
            nesterovs_momentum=True, alpha=0.005, batch_size=64, max_iter=200,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4,
            random_state=seed))
    raise ValueError(f"unknown model {name}")
