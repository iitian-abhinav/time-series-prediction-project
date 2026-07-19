import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def get_data(ticker, start_date):
    stock_data=yf.download(ticker, start=start_date)
    if isinstance(stock_data.columns, pd.MultiIndex):
        close_col = ('Close', ticker)
        if close_col in stock_data.columns:
            close_price = stock_data.loc[:, close_col].to_frame(name='Close')
        else:
            close_price = stock_data.loc[:, stock_data.columns.get_level_values(0) == 'Close']
            close_price.columns = ['Close']
    else:
        close_price = stock_data[['Close']]
    return close_price

def stationary_check(close_price):
    if close_price is None or len(close_price) == 0:
        raise ValueError('No price data available for stationarity check.')
    adf_test=adfuller(close_price)
    p_value=round(adf_test[1],3)
    return p_value

def get_rolling_mean(close_price):
    rolling_price=close_price.rolling(window=7).mean().dropna()
    return rolling_price

def get_differencing_order(close_price):
    p_value=stationary_check(close_price)
    d=0
    while True:
        if p_value>0.05:
            d+=1
            close_price=close_price.diff().dropna()
            p_value=stationary_check(close_price)
        else:
            break
    return d

def fit_model(data, differencing_order):

    model = ARIMA(data, order=(30, differencing_order, 30))

    model_fit = model.fit()

    forecast_steps = 30

    forecast = model_fit.get_forecast(steps=forecast_steps)

    predictions = forecast.predicted_mean

    return predictions

def evaluate_model(original_price, differencing_order):

    train_data, test_data = original_price[:-30], original_price[-30:]

    predictions = fit_model(train_data, differencing_order)

    rmse = np.sqrt(mean_squared_error(test_data, predictions))

    return round(rmse, 2)


def evaluate_model_metrics(original_price, differencing_order):
    """Same holdout split as evaluate_model, but returns the full
    metrics dict (RMSE, MAE, MAPE, R2, directional accuracy) from
    pages.utils.metrics, for the model comparison page."""
    train_data, test_data = original_price[:-30], original_price[-30:]
    return evaluate_window(train_data, test_data)


def evaluate_window(train_data, test_data):
    """Fit on `train_data`, forecast len(test_data) days ahead, score
    against `test_data`. Used both for the single last-30-day holdout
    and for walk-forward validation across multiple windows."""
    from pages.utils.metrics import compute_metrics

    d = get_differencing_order(train_data)
    predictions = fit_model(train_data, d)
    return compute_metrics(np.asarray(test_data).flatten(), np.asarray(predictions).flatten())

def scaling(close_price):

    scaler = StandardScaler()

    scaled_data = scaler.fit_transform(

        np.array(close_price).reshape(-1, 1)

    )

    return scaled_data, scaler

def get_forecast(scaled_data, differencing_order, original_index):

    predictions = fit_model(scaled_data, differencing_order)

    last_date = original_index[-1]
    if not isinstance(last_date, pd.Timestamp):
        last_date = pd.to_datetime(last_date)

    forecast_index = pd.date_range(
        start=last_date + pd.offsets.BDay(),
        periods=len(predictions),
        freq='B'
    )

    forecast_df = pd.DataFrame(
        {'Close': np.asarray(predictions)},
        index=forecast_index
    )

    return forecast_df

def inverse_scaling(scaler, scaled_data):

    close_price = scaler.inverse_transform(

        np.array(scaled_data).reshape(-1, 1)

    )

    return close_price