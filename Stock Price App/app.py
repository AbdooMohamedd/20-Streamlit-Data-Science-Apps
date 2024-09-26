import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression

# App title
st.title("Advanced Stock Price App")

# Sidebar for user inputs
st.sidebar.header("User Input Options")

# Dropdown for stock symbols
stock_options = ['AAPL', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'TSLA']
tickerSymbol = st.sidebar.selectbox("Select Stock Ticker", stock_options, index=3)

# Date selection for historical data
start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-09-11'))
end_date = st.sidebar.date_input("End Date", pd.to_datetime('2024-09-11'))

# Option to select data period
period = st.sidebar.selectbox("Select Data Period", ['1d', '1mo', '3mo', '6mo', '1y', 'max'], index=1)

# Moving averages
short_window = st.sidebar.number_input("Short-term Moving Average (days)", min_value=5, max_value=100, value=20)
long_window = st.sidebar.number_input("Long-term Moving Average (days)", min_value=50, max_value=200, value=50)

# Select technical indicators
indicators = st.sidebar.multiselect("Select Technical Indicators", 
    ['RSI', 'MACD', 'OBV', 'ATR', 'ADX'], default=['RSI', 'MACD'])

# Machine learning prediction days
predict_days = st.sidebar.slider("Predict future days", 1, 30, 7)

# Fetch data from yfinance
tickerData = yf.Ticker(tickerSymbol)
tickerDf = tickerData.history(period=period, start=start_date, end=end_date)

# Compute moving averages
tickerDf['Short_MA'] = tickerDf['Close'].rolling(window=short_window, min_periods=1).mean()
tickerDf['Long_MA'] = tickerDf['Close'].rolling(window=long_window, min_periods=1).mean()

# Compute RSI
if 'RSI' in indicators:
    delta = tickerDf['Close'].diff(1)
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    tickerDf['RSI'] = 100 - (100 / (1 + rs))

# Compute MACD
if 'MACD' in indicators:
    exp12 = tickerDf['Close'].ewm(span=12, adjust=False).mean()
    exp26 = tickerDf['Close'].ewm(span=26, adjust=False).mean()
    tickerDf['MACD'] = exp12 - exp26
    tickerDf['Signal Line'] = tickerDf['MACD'].ewm(span=9, adjust=False).mean()

# Compute On-Balance Volume (OBV)
if 'OBV' in indicators:
    obv = np.where(tickerDf['Close'] > tickerDf['Close'].shift(1), 
                   tickerDf['Volume'], 
                   -tickerDf['Volume'])
    tickerDf['OBV'] = obv.cumsum()

# Compute Average True Range (ATR)
if 'ATR' in indicators:
    high_low = tickerDf['High'] - tickerDf['Low']
    high_close = np.abs(tickerDf['High'] - tickerDf['Close'].shift())
    low_close = np.abs(tickerDf['Low'] - tickerDf['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    tickerDf['ATR'] = ranges.max(axis=1).rolling(window=14).mean()

# Compute Average Directional Index (ADX)
if 'ADX' in indicators:
    plus_dm = tickerDf['High'].diff()
    minus_dm = tickerDf['Low'].diff()
    tickerDf['+DM'] = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0)
    tickerDf['-DM'] = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0)
    tickerDf['TR'] = ranges.max(axis=1)
    tickerDf['+DI'] = (tickerDf['+DM'].rolling(window=14).mean() / tickerDf['TR'].rolling(window=14).mean()) * 100
    tickerDf['-DI'] = (tickerDf['-DM'].rolling(window=14).mean() / tickerDf['TR'].rolling(window=14).mean()) * 100
    tickerDf['ADX'] = abs(tickerDf['+DI'] - tickerDf['-DI']) / (tickerDf['+DI'] + tickerDf['-DI']) * 100

# Display data statistics
st.write(f"### {tickerSymbol} Data Statistics")
st.write(tickerDf.describe())

# Candlestick chart with Plotly
st.write(f"## {tickerSymbol} Candlestick Chart")
fig = make_subplots(specs=[[{"secondary_y": True}]])
candlestick = go.Candlestick(x=tickerDf.index,
                             open=tickerDf['Open'],
                             high=tickerDf['High'],
                             low=tickerDf['Low'],
                             close=tickerDf['Close'])
fig.add_trace(candlestick)


# Add moving averages to the chart
fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['Short_MA'], mode='lines', name=f"{short_window}-Day MA"), secondary_y=False)
fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['Long_MA'], mode='lines', name=f"{long_window}-Day MA"), secondary_y=False)

# Add selected technical indicators to the chart
if 'RSI' in indicators:
    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['RSI'], mode='lines', name="RSI"), secondary_y=True)
if 'MACD' in indicators:
    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['MACD'], mode='lines', name="MACD"))
    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['Signal Line'], mode='lines', name="Signal Line"))
if 'OBV' in indicators:
    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['OBV'], mode='lines', name="OBV"))
if 'ATR' in indicators:
    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['ATR'], mode='lines', name="ATR"))
if 'ADX' in indicators:
    fig.add_trace(go.Scatter(x=tickerDf.index, y=tickerDf['ADX'], mode='lines', name="ADX"))

# Customize layout
fig.update_layout(title=f'{tickerSymbol} Stock Price', xaxis_title='Date', yaxis_title='Price')
st.plotly_chart(fig)

# Line chart for closing price
st.write(f"## {tickerSymbol} Closing Price")
st.line_chart(tickerDf['Close'])

# Volume line chart
st.write(f"## {tickerSymbol} Volume")
st.line_chart(tickerDf['Volume'])


# Machine Learning Prediction (Linear Regression)
st.write("## Price Prediction using Linear Regression")

# Filter data for the last two months
last_two_months_data = tickerDf[-60:]  

X = np.arange(len(last_two_months_data)).reshape(-1, 1) 
y = last_two_months_data['Close'].values.reshape(-1, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# User input for number of days to predict
predict_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=5)

# Predict future prices
future_X = np.arange(len(last_two_months_data), len(last_two_months_data) + predict_days).reshape(-1, 1)
predicted_prices = model.predict(future_X)

# Display predicted prices in a horizontal table
st.write(f"### Predicted prices for the next {predict_days} days:")
predicted_df = pd.DataFrame(predicted_prices, columns=['Predicted Price'], index=[f"Day {i+1}" for i in range(predict_days)])
st.table(predicted_df.T)

# Option to download data as CSV
st.sidebar.write("### Download Data")
csv = tickerDf.to_csv().encode('utf-8')
st.sidebar.download_button(label="Download Data as CSV", data=csv, file_name=f'{tickerSymbol}_data.csv', mime='text/csv')

# Adding theme toggle between Light and Dark
st.sidebar.write("### Toggle Dark Mode")
theme = st.sidebar.selectbox("Choose Theme", ["Light", "Dark"], key="theme_selectbox")

if theme == "Dark":
    st.markdown(
        """
        <style>
        body {
            background-color: #1a1a1a;
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        body {
            background-color: white;
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

