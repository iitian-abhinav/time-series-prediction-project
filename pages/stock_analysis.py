import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import datetime
import tabnanny
from pages.utils.plotlyfigure import (
    plotly_table,
    close_chart,
    candlestick,
    RSI,
    Moving_average,
    MACD,
)

st.set_page_config(
    page_title="Stock Analysis",
    page_icon="page_with_curl",
    layout='wide'
)

st.title("Stock Analysis")

col1,col2,col3=st.columns(3)

today=datetime.date.today()

with col1:
    ticker=st.text_input("Stock Ticker","TSLA")
with col2:
    start_date=st.date_input("Choose start date", datetime.date(today.year-1,today.month,today.day))
with col3:
    end_date=st.date_input("Choose end date", datetime.date(today.year,today.month,today.day))

st.subheader(ticker)

stock=yf.Ticker(ticker)
stock_info = stock.info or {}

st.write(stock_info.get('longBusinessSummary', 'No business summary available.'))
st.write("**Stock**", stock_info.get('sector', 'N/A'))
st.write("**Total No. of Employees**", stock_info.get('fullTimeEmployees', 'N/A'))
st.write("**Website Link**", stock_info.get('website', 'N/A'))

col1,col2=st.columns(2)

with col1:
    df=pd.DataFrame(index=['Market Cap','Beta','EPS','PE Ratio'])
    df['']=[
        stock_info.get('marketCap', 'N/A'),
        stock_info.get('beta', 'N/A'),
        stock_info.get('trailingEps', 'N/A'),
        stock_info.get('trailingPE', 'N/A')
    ]
    fig_df=plotly_table(df)
    st.plotly_chart(fig_df,use_container_width=True)

with col2:

    df = pd.DataFrame(

        index=[

            'Quick Ratio',

            'Revenue per share',

            'Profit Margins',

            'Debt to Equity',

            'Return on Equity'

        ]

    )

    df[''] = [

        stock_info.get("quickRatio", 'N/A'),

        stock_info.get("revenuePerShare", 'N/A'),

        stock_info.get("profitMargins", 'N/A'),

        stock_info.get("debtToEquity", 'N/A'),

        stock_info.get("returnOnEquity", 'N/A')

    ]

    fig_df = plotly_table(df)

    st.plotly_chart(fig_df, use_container_width=True)

data=yf.download(ticker,start=start_date,end=end_date)

if data.empty:
    st.error(f"No data available for ticker '{ticker}' in the selected date range. Please check the ticker and date range.")
    st.stop()

col1,col2,col3=st.columns(3)
last_close = data['Close'].iloc[-1]
prev_close = data['Close'].iloc[-2]

if isinstance(last_close, pd.Series):
    last_close = last_close.iloc[0]
    prev_close = prev_close.iloc[0]

daily_change = last_close - prev_close
col1.metric("Daily Change", round(last_close, 2), round(daily_change, 2))

last_10_df=data.tail(10).sort_index(ascending=False).round(3)
fig_df=plotly_table(last_10_df)

st.write("#### Historical Data (Last 10 Days)")
st.plotly_chart(fig_df,use_container_width=True)

col1,col2,col3,col4,col5,col6=st.columns(6)
num_period=''
with col1:
    if st.button('5D'):
        num_period='5d'
with col2:
    if st.button('1M'):
        num_period='1mo'
with col3:
    if st.button('6m'):
        num_period='6mo'
with col4:
    if st.button('YTD'):
        num_period='ytd'
with col5:
    if st.button('1Y'):
        num_period='1y'
with col6:
    if st.button('MAX'):
        num_period='max'

col1,col2,col3=st.columns(3)

with col1:
    chart_type=st.selectbox("",('Candle','Line'))
with col2:
    if chart_type=='Candle':
        indicators=st.selectbox("",("RSI","MACD"))
    else:
        indicators=st.selectbox('',('RSI','Moving Average','MACD'))

ticker_=yf.Ticker(ticker)
new_df1=ticker_.history(period='max')
data1=ticker_.history(period='max')

if num_period=='':
    if chart_type=='Candle' and indicators=='RSI':
        st.plotly_chart(candlestick(data1,'1y'), use_container_width=True)
        st.plotly_chart(RSI(data1,'1y'), use_container_width=True)
    
    if chart_type=='Candle' and indicators=='MACD':
        st.plotly_chart(candlestick(data1,'1y'), use_container_width=True)
        st.plotly_chart(MACD(data1,'1y'), use_container_width=True)

    if chart_type=='Line' and indicators=='RSI':
        st.plotly_chart(close_chart(data1,'1y'), use_container_width=True)
        st.plotly_chart(RSI(data1,'1y'), use_container_width=True)

    if chart_type=='Line' and indicators=='MACD':
        st.plotly_chart(close_chart(data1,'1y'), use_container_width=True)
        st.plotly_chart(Moving_average(data1,'1y'), use_container_width=True)

    if chart_type=='Line' and indicators=='Moving Average':
        st.plotly_chart(close_chart(data1,'1y'), use_container_width=True)
        st.plotly_chart(RSI(data1,'1y'), use_container_width=True)

else:
    if chart_type=='Candle' and indicators=='RSI':
        st.plotly_chart(candlestick(new_df1, num_period), use_container_width=True)
        st.plotly_chart(RSI(new_df1, num_period), use_container_width=True)
    
    if chart_type=='Candle' and indicators=='MACD':
        st.plotly_chart(candlestick(new_df1, num_period), use_container_width=True)
        st.plotly_chart(MACD(new_df1, num_period), use_container_width=True)

    if chart_type=='Line' and indicators=='RSI':
        st.plotly_chart(close_chart(new_df1, num_period), use_container_width=True)
        st.plotly_chart(RSI(new_df1, num_period), use_container_width=True)

    if chart_type=='Line' and indicators=='MACD':
        st.plotly_chart(close_chart(new_df1, num_period), use_container_width=True)
        st.plotly_chart(Moving_average(new_df1, num_period), use_container_width=True)

    if chart_type=='Line' and indicators=='Moving Average':
        st.plotly_chart(close_chart(new_df1, num_period), use_container_width=True)
        st.plotly_chart(RSI(new_df1, num_period), use_container_width=True)
    