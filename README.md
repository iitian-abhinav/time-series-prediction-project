# Time Series Prediction Project

Streamlit app for stock prediction and volatility forecasting.

Pages:
- **Stock Information** — general stock data browser.
- **Stock Prediction** — original ARIMA forecast (kept as the classical baseline).
- **Advanced Forecasting** — model comparison page:
  - **ARIMA** (`statsmodels`) — classical baseline, kept so the upgrade is measurable against it.
  - **XGBoost / LightGBM** — gradient boosting on lag features, rolling stats, and technical
    indicators (RSI, MACD, Bollinger Bands).
  - **GRU** (`tensorflow`/Keras) — a single, deliberately small deep learning model (1 GRU
    layer, 32 units) for sequence forecasting.
  - **GARCH / EGARCH** (`arch`) — volatility forecasting alongside the price forecast, since
    price-only forecasts ignore risk.

All price models are evaluated with RMSE on the same last-30-trading-day holdout window so
results are directly comparable.

Run locally:
pip install -r requirements.txt
streamlit run main.py
