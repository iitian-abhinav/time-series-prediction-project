import streamlit as st
from datetime import date, timedelta
from pages.utils.model_train import get_data,get_rolling_mean,get_differencing_order, evaluate_model,stationary_check,fit_model, get_forecast
import pandas as pd
import plotly.graph_objects as go
from pages.utils.plotlyfigure import plotly_table

st.set_page_config(
    page_title='Stock Prediction',
    page_icon='chart_with_upward_trend',
    layout='wide'
)

st.title("Stock Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    ticker = st.text_input('Stock Ticker', 'TSLA')
    start_date = st.date_input('Start Date', value=date.today() - timedelta(days=365))

st.subheader('Predicting Next 30 days Close Price for: ' + ticker)

close_price = get_data(ticker, start_date.strftime('%Y-%m-%d'))
if close_price is None or close_price.empty:
    st.error('No price data was found for this ticker and date. Please choose an earlier start date or verify the ticker symbol.')
    st.stop()

rolling_price = get_rolling_mean(close_price)
if rolling_price is None or rolling_price.empty:
    st.error('Not enough historical data available after applying the 7-day rolling mean. Please choose an earlier start date.')
    st.stop()

differencing_order = get_differencing_order(rolling_price)

rmse = evaluate_model(rolling_price, differencing_order)

st.write("**Model RMSE Score:**", rmse)

forecast = get_forecast(rolling_price, differencing_order, rolling_price.index)

st.write('##### Forecast Data (Next 30 days)')

fig_tail = plotly_table(

    forecast.sort_index(ascending=True).round(3)

)

fig_tail.update_layout(height=220)

st.plotly_chart(fig_tail, use_container_width=True)

historical = close_price[['Close']].copy()
forecast_df = forecast[['Close']].copy()

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=historical.index,
        y=historical['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='black', width=2)
    )
)
fig.add_trace(
    go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Close'],
        mode='lines',
        name='Future Close Price',
        line=dict(color='red', width=2, dash='dash')
    )
)
fig.update_layout(
    title='Actual and Forecast Close Price',
    xaxis_title='Date',
    yaxis_title='Price',
    height=500,
    plot_bgcolor='white',
    paper_bgcolor='white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
)

st.plotly_chart(fig, use_container_width=True)