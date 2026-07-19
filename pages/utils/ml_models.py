"""
Gradient boosting benchmark models: XGBoost and LightGBM.
Uses lag features, rolling stats, and technical indicators (from
features.py) to predict next-day Close, then rolls that prediction
forward recursively to build a multi-day forecast.

Optional hyperparameter tuning (tune=True) uses RandomizedSearchCV
with TimeSeriesSplit — cross-validation folds that respect
chronological order (never trains on the future to predict the
past), unlike a naive random K-fold split.
"""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV

from pages.utils.features import build_features, feature_columns, make_supervised
from pages.utils.metrics import compute_metrics

MODEL_BUILDERS = {
    'xgboost': lambda: XGBRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1
    ),
    'lightgbm': lambda: LGBMRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
        n_jobs=-1, verbose=-1, min_child_samples=5
    ),
}

# Search space for tuning; kept modest so RandomizedSearchCV finishes
# quickly inside a Streamlit app.
PARAM_DISTRIBUTIONS = {
    'xgboost': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    },
    'lightgbm': {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, -1],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'min_child_samples': [5, 10, 20],
    },
}

TUNE_N_ITER = 15
TUNE_CV_SPLITS = 3


def _tune(model_type: str, X: pd.DataFrame, y: pd.Series):
    """Randomized search over PARAM_DISTRIBUTIONS, scored on RMSE
    across TimeSeriesSplit folds (chronological, no look-ahead)."""
    base_model = {
        'xgboost': XGBRegressor(random_state=42, n_jobs=-1),
        'lightgbm': LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    }[model_type]

    n_splits = min(TUNE_CV_SPLITS, max(2, len(X) // 30))
    tscv = TimeSeriesSplit(n_splits=n_splits)

    search = RandomizedSearchCV(
        base_model, PARAM_DISTRIBUTIONS[model_type],
        n_iter=TUNE_N_ITER, cv=tscv,
        scoring='neg_root_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    return search.best_estimator_


def _fit(model_type: str, X: pd.DataFrame, y: pd.Series, tune: bool = False):
    if tune:
        return _tune(model_type, X, y)
    model = MODEL_BUILDERS[model_type]()
    model.fit(X, y)
    return model


def _recursive_forecast(model, close: pd.Series, steps: int) -> np.ndarray:
    """Roll the fitted model forward `steps` days.

    At each step, features are rebuilt from the extended series
    (real history + predictions so far), and the model predicts the
    next day's close.
    """
    history = close.copy()
    preds = []
    for _ in range(steps):
        feat = build_features(history)
        cols = feature_columns(feat)
        x_last = feat[cols].iloc[[-1]]
        if x_last.isna().any(axis=None):
            # Not enough history to build all indicators; fall back
            # to the last known value.
            next_val = history.iloc[-1]
        else:
            next_val = float(model.predict(x_last)[0])
        preds.append(next_val)
        next_index = history.index[-1] + pd.offsets.BDay()
        history.loc[next_index] = next_val
    return np.array(preds)


def evaluate_window(train_close: pd.Series, test_close: pd.Series, model_type: str, tune: bool = False) -> dict:
    """Fit on `train_close`, forecast len(test_close) days ahead,
    score against `test_close`. Used both for the single last-30-day
    holdout and for walk-forward validation across multiple windows."""
    X_train, y_train, _ = make_supervised(train_close)
    model = _fit(model_type, X_train, y_train, tune=tune)
    preds = _recursive_forecast(model, train_close, len(test_close))
    return compute_metrics(test_close.values, preds)


def evaluate_gbm(close: pd.Series, model_type: str, horizon: int = 30, tune: bool = False) -> dict:
    """Holdout evaluation: train on all but the last `horizon` rows,
    forecast forward, compare against the held-out actuals (same
    convention as the ARIMA baseline's evaluate_model). Returns the
    full metrics dict (RMSE, MAE, MAPE, R2, directional accuracy)."""
    train_close, test_close = close.iloc[:-horizon], close.iloc[-horizon:]
    return evaluate_window(train_close, test_close, model_type, tune=tune)


def forecast_gbm(close: pd.Series, model_type: str, horizon: int = 30, tune: bool = False) -> pd.DataFrame:
    """Fit on the full available series and forecast `horizon` days ahead."""
    X, y, _ = make_supervised(close)
    model = _fit(model_type, X, y, tune=tune)
    preds = _recursive_forecast(model, close, horizon)

    last_date = close.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)
    forecast_index = pd.date_range(start=last_date + pd.offsets.BDay(), periods=horizon, freq='B')
    return pd.DataFrame({'Close': preds}, index=forecast_index)
