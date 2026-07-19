"""
A single, intentionally small deep-learning model: a 1-layer GRU.
This is kept light on purpose (small hidden size, few epochs, no
attention/transformer) so it trains quickly in a Streamlit app on CPU,
while still giving a genuine sequence-model comparison point against
ARIMA and the gradient boosting models.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from pages.utils.metrics import compute_metrics

WINDOW = 20
HIDDEN_UNITS = 32
EPOCHS = 25
BATCH_SIZE = 16


def _build_model():
    # Imported lazily so the rest of the app works even if
    # tensorflow isn't installed / is still loading.
    import tensorflow as tf
    from tensorflow.keras import Sequential, Input
    from tensorflow.keras.layers import GRU, Dense

    tf.random.set_seed(42)
    model = Sequential([
        Input(shape=(WINDOW, 1)),
        GRU(HIDDEN_UNITS),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model


def _make_sequences(scaled: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i - window:i, 0])
        y.append(scaled[i, 0])
    X = np.array(X).reshape(-1, window, 1)
    y = np.array(y)
    return X, y


def _fit_gru(train_close: pd.Series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(train_close.values.reshape(-1, 1))
    X, y = _make_sequences(scaled, WINDOW)
    model = _build_model()
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    return model, scaler, scaled


def _recursive_forecast_gru(model, scaler, scaled_history: np.ndarray, steps: int) -> np.ndarray:
    window = scaled_history[-WINDOW:].reshape(1, WINDOW, 1)
    preds_scaled = []
    for _ in range(steps):
        next_scaled = model.predict(window, verbose=0)[0, 0]
        preds_scaled.append(next_scaled)
        window = np.append(window[:, 1:, :], [[[next_scaled]]], axis=1)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    return scaler.inverse_transform(preds_scaled).flatten()


def evaluate_window(train_close: pd.Series, test_close: pd.Series) -> dict:
    """Fit on `train_close`, forecast len(test_close) days ahead,
    score against `test_close`. Used both for the single last-30-day
    holdout and for walk-forward validation across multiple windows."""
    if len(train_close) <= WINDOW + 5:
        raise ValueError('Not enough history to train the GRU model.')
    model, scaler, scaled = _fit_gru(train_close)
    preds = _recursive_forecast_gru(model, scaler, scaled, len(test_close))
    return compute_metrics(test_close.values, preds)


def evaluate_gru(close: pd.Series, horizon: int = 30) -> dict:
    """Train on all but the last `horizon` rows, forecast forward,
    compare against held-out actuals. Returns the full metrics dict."""
    train_close, test_close = close.iloc[:-horizon], close.iloc[-horizon:]
    return evaluate_window(train_close, test_close)


def forecast_gru(close: pd.Series, horizon: int = 30) -> pd.DataFrame:
    if len(close) <= WINDOW + 5:
        raise ValueError('Not enough history to train the GRU model.')
    model, scaler, scaled = _fit_gru(close)
    preds = _recursive_forecast_gru(model, scaler, scaled, horizon)

    last_date = close.index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)
    forecast_index = pd.date_range(start=last_date + pd.offsets.BDay(), periods=horizon, freq='B')
    return pd.DataFrame({'Close': preds}, index=forecast_index)
