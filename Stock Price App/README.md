# Stock Price App

This is a Streamlit-based web application that allows users to visualize stock price data, apply technical indicators, and predict future stock prices using machine learning (Linear Regression).

## Features

- **User-friendly Interface**: Select popular stock symbols like AAPL, GOOGL, MSFT, and more.
- **Custom Date Range & Period**: Choose a start and end date for historical data, and select the data period (daily, monthly, yearly).
- **Technical Indicators**:
  - Add indicators such as RSI, MACD, OBV, ATR, and ADX to the charts for deeper analysis.
- **Stock Price Visualization**:
  - View interactive candlestick charts with short-term and long-term moving averages.
- **Price Prediction (Linear Regression)**:
  - Predict stock prices for the upcoming days based on the last two months of data.
- **Data Download**: Export stock data as a CSV file.
- **Dark Mode**: Toggle between light and dark themes.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/AbdooMohamedd/20-Streamlit-Data-Science-Apps.git
   ```

2. Navigate to the project folder:

   ```bash
   cd 20-Streamlit-Data-Science-Apps/Stock\ Price\ App
   ```

3. Install the required libraries:

   ```bash
   pip install streamlit yfinance pandas plotly scikit-learn
   ```

4. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

5. Open your browser and visit `http://localhost:8501` to view the app.

## Usage

1. **Select Stock Ticker**: Use the sidebar to choose from a list of popular stock symbols.
2. **Set Date Range**: Select the start and end dates for historical data.
3. **Add Indicators**: Pick technical indicators to add to the stock chart.
4. **Prediction**: Choose how many future days to predict stock prices for, using linear regression based on the past two months of data.
5. **Download Data**: You can download the stock data as a CSV file using the "Download Data as CSV" option in the sidebar.

## Machine Learning Prediction

This app uses a Linear Regression model to predict future stock prices based on historical closing prices from the last two months. The number of future days for prediction can be set using a slider in the sidebar.

## License

This project is licensed under the MIT License.

## Repository

[GitHub Repository](https://github.com/AbdooMohamedd/20-Streamlit-Data-Science-Apps)
