import streamlit as st
from datetime import date, timedelta
import pandas as pd
import plotly.graph_objects as go

from pages.utils.model_train import get_data, get_rolling_mean, get_differencing_order, evaluate_model_metrics, get_forecast
from pages.utils.model_train import evaluate_window as arima_evaluate_window
from pages.utils.plotlyfigure import plotly_table
from pages.utils.ml_models import evaluate_gbm, forecast_gbm, evaluate_window as gbm_evaluate_window
from pages.utils.dl_models import evaluate_gru, forecast_gru, evaluate_window as gru_evaluate_window
from pages.utils.volatility_models import forecast_volatility, realized_volatility
from pages.utils.backtest import run_walk_forward

st.set_page_config(
    page_title='Advanced Forecasting',
    page_icon='chart_with_upward_trend',
    layout='wide'
)

st.title("Advanced Forecasting: Model Comparison")
st.write(
    "Compares the classical ARIMA baseline against gradient boosting "
    "(XGBoost / LightGBM), a lightweight GRU deep learning model, and "
    "adds a GARCH/EGARCH volatility forecast alongside the price forecast."
)

col1, col2, col3 = st.columns(3)
with col1:
    ticker = st.text_input('Stock Ticker', 'TSLA')
    start_date = st.date_input('Start Date', value=date.today() - timedelta(days=1095))
with col2:
    models_selected = st.multiselect(
        'Price models to compare',
        ['ARIMA (baseline)', 'XGBoost', 'LightGBM', 'GRU (deep learning)'],
        default=['ARIMA (baseline)', 'XGBoost', 'LightGBM', 'GRU (deep learning)']
    )
with col3:
    vol_model = st.selectbox('Volatility model', ['GARCH', 'EGARCH'])

wf_col1, wf_col2, wf_col3 = st.columns([1, 1.5, 1.5])
with wf_col1:
    use_walk_forward = st.checkbox(
        'Walk-forward validation',
        value=False,
        help='Instead of scoring on just the last 30 days, re-test on several rolling '
             '30-day windows further back and average the results — a more reliable '
             'measure than a single lucky/unlucky holdout, at the cost of more training time.'
    )
with wf_col2:
    n_windows = st.slider('Number of windows', min_value=2, max_value=8, value=5, disabled=not use_walk_forward)
with wf_col3:
    tune_gbm = st.checkbox(
        'Tune XGBoost/LightGBM',
        value=False,
        help='Runs a randomized search (15 candidates x time-series cross-validation) over '
             'tree depth, learning rate, n_estimators, and sampling ratios instead of using '
             'fixed defaults. Slower, but gives these models a fairer shot vs ARIMA.'
    )

HORIZON = 30

close_price_df = get_data(ticker, start_date.strftime('%Y-%m-%d'))
if close_price_df is None or close_price_df.empty:
    st.error('No price data was found for this ticker and date. Please choose an earlier start date or verify the ticker symbol.')
    st.stop()

close = close_price_df['Close']

if len(close) < 100:
    st.warning('Less than 100 trading days of history. Results (especially the GRU) may be unreliable — consider an earlier start date.')

# ---------------------------------------------------------------------
# Price forecasting: run each selected model, collect RMSE + forecasts
# ---------------------------------------------------------------------
st.subheader(f'Price Forecast Comparison — Next {HORIZON} Days for {ticker}')

rolling_price = get_rolling_mean(close_price_df)
results = {}
metrics_by_model = {}
per_window_by_model = {}

wf_label = f'{n_windows}-window walk-forward validation' if use_walk_forward else 'last 30 held-out trading days'
tune_note = ' (XGBoost/LightGBM tuned)' if tune_gbm else ''

with st.spinner(f'Training models ({wf_label}{tune_note})...'):
    if 'ARIMA (baseline)' in models_selected and rolling_price is not None and not rolling_price.empty:
        try:
            if use_walk_forward:
                avg, per_window = run_walk_forward(arima_evaluate_window, rolling_price, HORIZON, n_windows)
                if avg is None:
                    raise ValueError('Not enough history for any walk-forward window.')
                metrics_by_model['ARIMA (baseline)'] = avg
                per_window_by_model['ARIMA (baseline)'] = per_window
            else:
                d_order = get_differencing_order(rolling_price)
                metrics_by_model['ARIMA (baseline)'] = evaluate_model_metrics(rolling_price, d_order)
            d_order = get_differencing_order(rolling_price)
            fc = get_forecast(rolling_price, d_order, rolling_price.index)
            results['ARIMA (baseline)'] = fc['Close']
        except Exception as e:
            st.warning(f'ARIMA failed: {e}')

    if 'XGBoost' in models_selected:
        try:
            if use_walk_forward:
                avg, per_window = run_walk_forward(lambda tr, te: gbm_evaluate_window(tr, te, 'xgboost', tune=tune_gbm), close, HORIZON, n_windows)
                if avg is None:
                    raise ValueError('Not enough history for any walk-forward window.')
                metrics_by_model['XGBoost'] = avg
                per_window_by_model['XGBoost'] = per_window
            else:
                metrics_by_model['XGBoost'] = evaluate_gbm(close, 'xgboost', HORIZON, tune=tune_gbm)
            fc = forecast_gbm(close, 'xgboost', HORIZON, tune=tune_gbm)
            results['XGBoost'] = fc['Close']
        except Exception as e:
            st.warning(f'XGBoost failed: {e}')

    if 'LightGBM' in models_selected:
        try:
            if use_walk_forward:
                avg, per_window = run_walk_forward(lambda tr, te: gbm_evaluate_window(tr, te, 'lightgbm', tune=tune_gbm), close, HORIZON, n_windows)
                if avg is None:
                    raise ValueError('Not enough history for any walk-forward window.')
                metrics_by_model['LightGBM'] = avg
                per_window_by_model['LightGBM'] = per_window
            else:
                metrics_by_model['LightGBM'] = evaluate_gbm(close, 'lightgbm', HORIZON, tune=tune_gbm)
            fc = forecast_gbm(close, 'lightgbm', HORIZON, tune=tune_gbm)
            results['LightGBM'] = fc['Close']
        except Exception as e:
            st.warning(f'LightGBM failed: {e}')

    if 'GRU (deep learning)' in models_selected:
        try:
            if use_walk_forward:
                avg, per_window = run_walk_forward(gru_evaluate_window, close, HORIZON, n_windows)
                if avg is None:
                    raise ValueError('Not enough history for any walk-forward window.')
                metrics_by_model['GRU (deep learning)'] = avg
                per_window_by_model['GRU (deep learning)'] = per_window
            else:
                metrics_by_model['GRU (deep learning)'] = evaluate_gru(close, HORIZON)
            fc = forecast_gru(close, HORIZON)
            results['GRU (deep learning)'] = fc['Close']
        except Exception as e:
            st.warning(f'GRU failed: {e}')

if metrics_by_model:
    st.write(f'##### Evaluation Metrics — {wf_label}')
    st.caption(
        'RMSE / MAE / MAPE: lower is better (error magnitude). '
        'R²: closer to 1 is better (variance explained). '
        'Directional Accuracy: higher is better (% of days the up/down move was called correctly).'
        + (' Values below are averaged across all windows.' if use_walk_forward else '')
    )
    metrics_df = pd.DataFrame(metrics_by_model).T
    metrics_df = metrics_df[['RMSE', 'MAE', 'MAPE (%)', 'R2', 'Directional Accuracy (%)']]
    metrics_df = metrics_df.sort_values('RMSE')
    best_model = metrics_df.index[0]
    fig_metrics = plotly_table(metrics_df)
    fig_metrics.update_layout(height=45 * (len(metrics_df) + 1))
    st.plotly_chart(fig_metrics, use_container_width=True)
    st.success(f'Lowest {"average " if use_walk_forward else ""}RMSE: **{best_model}**')

    if use_walk_forward and per_window_by_model:
        st.write('##### Stability Across Windows')
        st.caption(
            'A model with consistently low RMSE across windows is more trustworthy than '
            'one that only did well in a single window.'
        )
        fig_stability = go.Figure()
        colors = {'ARIMA (baseline)': 'red', 'XGBoost': 'orange', 'LightGBM': 'green', 'GRU (deep learning)': 'blue'}
        for name, pw_df in per_window_by_model.items():
            fig_stability.add_trace(go.Scatter(
                x=pw_df.index, y=pw_df['RMSE'], mode='lines+markers', name=name,
                line=dict(color=colors.get(name, 'gray'), width=2)
            ))
        fig_stability.update_layout(
            title='RMSE by Window (Window 1 = most recent)',
            xaxis_title='Window', yaxis_title='RMSE', height=380,
            plot_bgcolor='white', paper_bgcolor='white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        st.plotly_chart(fig_stability, use_container_width=True)

if results:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index, y=close.values, mode='lines',
        name='Historical Close', line=dict(color='black', width=2)
    ))
    colors = {
        'ARIMA (baseline)': 'red',
        'XGBoost': 'orange',
        'LightGBM': 'green',
        'GRU (deep learning)': 'blue',
    }
    for name, series in results.items():
        fig.add_trace(go.Scatter(
            x=series.index, y=series.values, mode='lines', name=name,
            line=dict(color=colors.get(name, 'gray'), width=2, dash='dash')
        ))
    fig.update_layout(
        title=f'{ticker}: Actual vs Forecast Close Price ({HORIZON}-day horizon)',
        xaxis_title='Date', yaxis_title='Price', height=520,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('Select at least one price model above to see a forecast.')

# ---------------------------------------------------------------------
# Volatility forecasting
# ---------------------------------------------------------------------
st.subheader(f'Volatility Forecast — {vol_model}')
st.write(
    "A price forecast alone says nothing about risk. The chart below "
    f"forecasts *volatility* (not price) for the next {HORIZON} trading "
    f"days using a {vol_model} model fit to daily returns, alongside the "
    "30-day rolling realized volatility for context."
)

try:
    vol_forecast = forecast_volatility(close, vol_model, HORIZON)
    hist_vol = realized_volatility(close, window=30).dropna()

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=hist_vol.index, y=hist_vol.values, mode='lines',
        name='Realized Volatility (30d rolling, %)', line=dict(color='black', width=2)
    ))
    fig_vol.add_trace(go.Scatter(
        x=vol_forecast.index, y=vol_forecast['Daily Volatility (%)'], mode='lines',
        name=f'{vol_model} Forecast Daily Volatility (%)', line=dict(color='purple', width=2, dash='dash')
    ))
    fig_vol.update_layout(
        title=f'{ticker}: Daily Volatility — Realized vs {vol_model} Forecast',
        xaxis_title='Date', yaxis_title='Daily Volatility (%)', height=450,
        plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.write('##### Forecast Volatility Table (Next 30 Days)')
    fig_vol_table = plotly_table(vol_forecast.round(3))
    fig_vol_table.update_layout(height=220)
    st.plotly_chart(fig_vol_table, use_container_width=True)
except Exception as e:
    st.warning(f'{vol_model} volatility forecast failed: {e}')

st.caption(
    "Notes: RMSE is computed on the same last-30-trading-day holdout for every "
    "model so the comparison is apples-to-apples. The GBM and GRU forecasts are "
    "produced recursively (each predicted day feeds into the features/window for "
    "the next), which is the standard way to extend single-step models to a "
    "multi-day horizon and will drift more than a true multi-output model on long horizons."
)
