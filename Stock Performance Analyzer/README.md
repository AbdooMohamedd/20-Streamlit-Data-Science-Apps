# Stock Performance Analyzer

Welcome to the **Stock Performance Analyzer**! This Streamlit application provides tools to analyze and visualize stock performance for companies in the S&P 500. Users can explore company data, visualize historical stock prices, and make informed investment decisions.

![Image 1](./imgs/Market_Insights_Hub.gif)

## Features

- **Company Overview**: Quickly view essential data for selected S&P 500 companies.
- **Historical Price Visualization**: Analyze stock price trends over time with interactive charts.
- **Sector Performance Comparison**: Compare stock performance across different sectors.
- **Predictive Analytics**: Make predictions on stock prices using machine learning models.

## Technologies Used

- **Python Libraries**:
  - `Streamlit`: For building the web application.
  - `Pandas`: For data manipulation and analysis.
  - `Matplotlib` & `Seaborn`: For static and interactive data visualizations.
  - `Plotly`: For interactive graphs and charts.
  - `yfinance`: For fetching historical stock price data.
  - `Scikit-learn`: For implementing machine learning models for price predictions.

## Data Source

- **S&P 500 Companies Data**: Collected from [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies).
- **Historical Stock Prices**: Retrieved using the `yfinance` library.

## Installation

To get started with the Stock Performance Analyzer, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/AbdooMohamedd/20-Streamlit-Data-Science-Apps.git
   cd 20-Streamlit-Data-Science-Apps/Stock PerformanceAnalyzer
   ```

2. Install the required packages:
   ```bash
   pip install streamlit pandas matplotlib seaborn yfinance plotly scikit-learn
   ```

## Usage

To run the Stock Performance Analyzer app, execute the following command in your terminal:

```bash
streamlit run app.py
```

Then, open your web browser and navigate to `http://localhost:8501` to access the app.

## Contributing

Contributions are welcome! Feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License.
