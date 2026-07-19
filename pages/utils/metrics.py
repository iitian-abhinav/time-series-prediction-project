"""
Shared evaluation metrics for comparing forecasting models on the
same holdout window. Every model in the app is scored with the same
function so the comparison table is apples-to-apples.
"""
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def compute_metrics(actual, predicted) -> dict:
    """Compute RMSE, MAE, MAPE, R^2 and directional accuracy.

    - RMSE / MAE: error magnitude, in price units.
    - MAPE: error as a percentage of actual price (scale-free).
    - R^2: how much of the variance in the actuals the forecast explains.
    - Directional Accuracy: % of days the forecast got the up/down move right —
      often more useful than price error for a trading-relevant model.
    """
    actual = np.asarray(actual, dtype=float).flatten()
    predicted = np.asarray(predicted, dtype=float).flatten()

    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)

    nonzero = actual != 0
    mape = (
        np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100
        if nonzero.any() else np.nan
    )

    r2 = r2_score(actual, predicted) if len(actual) > 1 else np.nan

    if len(actual) > 1:
        actual_dir = np.sign(np.diff(actual))
        pred_dir = np.sign(np.diff(predicted))
        directional_acc = np.mean(actual_dir == pred_dir) * 100
    else:
        directional_acc = np.nan

    def _r(v, nd=3):
        return round(float(v), nd) if v is not None and not np.isnan(v) else None

    return {
        'RMSE': _r(rmse),
        'MAE': _r(mae),
        'MAPE (%)': _r(mape, 2),
        'R2': _r(r2),
        'Directional Accuracy (%)': _r(directional_acc, 2),
    }
