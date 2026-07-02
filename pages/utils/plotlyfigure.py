import plotly.graph_objects as go
import dateutil
import pandas_ta_classic as pta
import datetime


def plotly_table(dataframe):
    headerColor = 'grey'
    rowEvenColor = '#f8fafd'
    rowOddColor = '#e1efff'

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>" + str(i)[:10] + "<b>" for i in dataframe.columns],
            line_color='#0078ff',
            fill_color='#0078ff',
            align='center',
            font=dict(color='white', size=15),
            height=35,
        ),
        cells=dict(
            values=[
                ["<b>" + str(i) + "<b>" for i in dataframe.index]
            ] + [dataframe[i] for i in dataframe.columns],
            fill_color=[rowOddColor, rowEvenColor],   # partially obscured
            align='left',
            line_color=['white'],
            font=dict(color=['black'], size=15)
        )
    )])

    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0)
    )

    return fig

def filter_data(dataframe, num_period=None):
    if num_period=='1mo':
        date=dataframe.index[-1]+dateutil.relativedelta.relativedelta(months=-1)
    elif num_period=='5d':
        date=dataframe.index[-1]+dateutil.relativedelta.relativedelta(days=-5)
    elif num_period=='6mo':
        date=dataframe.index[-1]+dateutil.relativedelta.relativedelta(months=-6)
    elif num_period=='1y':
        date=dataframe.index[-1]+dateutil.relativedelta.relativedelta(years=-1)
    elif num_period=='ytd':
        date=datetime.datetime(dataframe.index[-1].year,1,1)
    elif num_period=='max':
        date=dataframe.index[0]
    else:
        date=dataframe.index[0]
    reset_df = dataframe.reset_index()
    return reset_df[reset_df['Date']>date]

def close_chart(dataframe, num_period=None):
    if num_period:
        dataframe = filter_data(dataframe, num_period)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Open'],
            mode='lines',
            name='Open',
            line=dict(width=2, color='#5ab7ff')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Close'],
            mode='lines',
            name='Close',
            line=dict(width=2, color='black')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['High'],
            mode='lines',
            name='High',
            line=dict(width=2, color='#0078ff')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Low'],
            mode='lines',
            name='Low',
            line=dict(width=2, color='red')
        )
    )

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor="top",
            xanchor="right"
        )
    )

    return fig


def candlestick(dataframe, num_period=None):
    dataframe = filter_data(dataframe, num_period)

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=dataframe['Date'],
            open=dataframe['Open'],
            high=dataframe['High'],
            low=dataframe['Low'],
            close=dataframe['Close']
        )
    )

    fig.update_layout(
        showlegend=False,
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff'
    )

    return fig
def RSI(dataframe, num_period):
    dataframe['RSI'] = pta.rsi(dataframe['Close'])
    dataframe = filter_data(dataframe, num_period)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe.RSI,
            name='RSI',
            marker_color='orange',
            line=dict(width=2, color='orange'),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=[70] * len(dataframe),
            name='Overbought',
            marker_color='red',
            line=dict(width=2, color='red', dash='dash'),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=[30] * len(dataframe),
            fill='tonexty',
            name='Oversold',
            marker_color='#79da84',
            line=dict(width=2, color='#79da84', dash='dash'),
        )
    )

    fig.update_layout(
        yaxis_range=[0, 100],
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig
def Moving_average(dataframe, num_period):

    dataframe['SMA_50'] = pta.sma(dataframe['Close'], 50)
    dataframe = filter_data(dataframe, num_period)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Open'],
            mode='lines',
            name='Open',
            line=dict(width=2, color='#5ab7ff')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Close'],
            mode='lines',
            name='Close',
            line=dict(width=2, color='black')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['High'],
            mode='lines',
            name='High',
            line=dict(width=2, color='#0078ff')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['Low'],
            mode='lines',
            name='Low',
            line=dict(width=2, color='red')
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(width=2, color='purple')
        )
    )

    fig.update_xaxes(rangeslider_visible=True)

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=20, t=20, b=0),
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        legend=dict(
            yanchor='top',
            xanchor='right'
        )
    )

    return fig

def MACD(dataframe, num_period):

    macd = pta.macd(dataframe['Close'])

    macd_line = macd.iloc[:, 0]
    macd_signal = macd.iloc[:, 1]
    macd_hist = macd.iloc[:, 2]

    dataframe['MACD'] = macd_line
    dataframe['MACD Signal'] = macd_signal
    dataframe['MACD Hist'] = macd_hist

    dataframe = filter_data(dataframe, num_period)

    c = ['red' if cl < 0 else 'green' for cl in dataframe['MACD Hist']]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['MACD'],
            name='MACD',
            marker_color='orange',
            line=dict(width=2, color='orange'),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=dataframe['Date'],
            y=dataframe['MACD Signal'],
            name='Signal',
            marker_color='red',
            line=dict(width=2, color='red', dash='dash'),
        )
    )

    fig.add_bar(
        x=dataframe['Date'],
        y=dataframe['MACD Hist'],
        marker_color=c,
        name='Histogram'
    )

    fig.update_layout(
        height=200,
        plot_bgcolor='white',
        paper_bgcolor='#e1efff',
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.02,
            xanchor='right',
            x=1,
        )
    )

    return fig
