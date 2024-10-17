import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import base64
import plotly.graph_objs as go
from datetime import datetime, timedelta
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression


st.set_page_config(page_title="Market Insights Hub", layout="wide")


# Streamlit app title
st.title('Market Insights Hub')

st.sidebar.header('Navigation')

# Page navigation
page = st.sidebar.radio("Go to", ["Overview Dashboard", "Stock Price Visualization", "Sector Performance Analysis", "Company Profile", "Stock Prediction and Trend Analysis"])

# Web scraping of S&P 500 data
@st.cache_data 
def load_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    html = pd.read_html(url, header=0)
    df = html[0]
    return df

df = load_data()


# If the selected page is the Overview Dashboard
if page == "Overview Dashboard":

    # Markdown description
    st.markdown("""
    This app retrieves the list of the **S&P 500** (from Wikipedia) and its corresponding **stock closing price** (year-to-date)!
    * **Python libraries:** base64, pandas, streamlit, numpy, matplotlib, seaborn
    * **Data source:** [Wikipedia](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
    """)

    # Sidebar - Sector selection
    sorted_sector_unique = sorted(df['GICS Sector'].unique())
    selected_sector = st.sidebar.multiselect('Sector', sorted_sector_unique, sorted_sector_unique)

    # Filtering data based on sector selection
    df_selected_sector = df[df['GICS Sector'].isin(selected_sector)]

    # Search bar for filtering companies by name, sector, or sub-industry
    search_term = st.sidebar.text_input('Search Company by Name, Sector, or Sub-Industry')

    # Filter companies based on search term
    if search_term:
        df_filtered = df[
            df['Security'].str.contains(search_term, case=False, na=False) |
            df['GICS Sector'].str.contains(search_term, case=False, na=False) |
            df['GICS Sub-Industry'].str.contains(search_term, case=False, na=False)
        ]
    else:
        df_filtered = df_selected_sector

    # Display filtered companies
    st.header('Display Companies in Selected Sector or Search Term')
    st.write('Data Dimension: ' + str(df_filtered.shape[0]) + ' rows and ' + str(df_filtered.shape[1]) + ' columns.')
    st.dataframe(df_filtered)

    # Download S&P 500 data
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode() 
        href = f'<a href="data:file/csv;base64,{b64}" download="SP500.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df_filtered), unsafe_allow_html=True)

    # Box Plot for Year Founded
    st.header('Distribution of Year Founded')
    df['Founded'] = pd.to_numeric(df['Founded'].str.extract('(\d+)')[0], errors='coerce')
    
    fig3, ax3 = plt.subplots(figsize=(15, 6))  
    sns.boxplot(y=df['Founded'], ax=ax3)
    ax3.set_title('Box Plot of Company Founding Years')
    ax3.set_ylabel('Year Founded')
    st.pyplot(fig3)

    # Total Companies per Sector Bar Chart
    st.header('Total Companies per Sector')
    sector_counts = df['GICS Sector'].value_counts()
    
    fig4, ax4 = plt.subplots(figsize=(16, 6))  
    sns.barplot(x=sector_counts.index, y=sector_counts.values, ax=ax4)
    ax4.set_title('Total Companies per Sector')
    ax4.set_xlabel('Sector')
    ax4.set_ylabel('Number of Companies')
    ax4.tick_params(axis='x', rotation=45)
    st.pyplot(fig4)

    # Sector Distribution Pie Chart
    st.header('Sector Distribution')
    sector_counts = df['GICS Sector'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(14, 6)) 
    ax1.pie(sector_counts, labels=sector_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  
    st.pyplot(fig1)

    # Sub-Industry Distribution Bar Chart
    st.header('Sub-Industry Distribution in Selected Sector')
    sub_industry_counts = df_selected_sector['GICS Sub-Industry'].value_counts()[:20]
    fig2, ax2 = plt.subplots(figsize=(12, 6)) 
    sns.barplot(y=sub_industry_counts.index, x=sub_industry_counts.values, ax=ax2)
    ax2.set_xlabel('Number of Companies')
    ax2.set_ylabel('Sub-Industry')
    st.pyplot(fig2)



# If the selected page is Stock Price Visualization
if page == "Stock Price Visualization":
    st.header('Stock Price Visualization')

    selected_company = st.selectbox('Select a Company for Stock Price Visualization', df['Security'])
    selected_symbol = df[df['Security'] == selected_company]['Symbol'].values[0]
    st.sidebar.header('Select Date Range')

    # Sidebar - Date range selection
    start_date = st.sidebar.date_input('Start date', datetime(2023, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.today())

    if start_date > end_date:
        st.error('Error: End date must fall after start date.')

    # Fetch stock price data for the selected company
    data = yf.download(
        tickers=selected_symbol,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=True
    )

    # Dynamic Line chart for selected company
    if not data.empty:
        st.subheader(f"Stock Price Trend for {selected_company} ({selected_symbol})")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price'))

        # Customize the layout
        fig.update_layout(
            title=f"{selected_company} Stock Price",
            xaxis_title="Date",
            yaxis_title="Closing Price (USD)",
            xaxis_rangeslider_visible=True,  
        )

        st.plotly_chart(fig)

    # Comparison Feature: Allow users to select multiple companies
    st.subheader("Compare Stock Price Trends Across Companies")

    # Set NVIDIA and Intel as default selected companies
    default_companies = ['Nvidia', 'Intel']
    selected_companies = st.multiselect(
        'Select Companies for Comparison', 
        df['Security'], 
        default=default_companies
    )

    # Fetch stock price data for the selected companies
    if selected_companies:
        symbols = df[df['Security'].isin(selected_companies)]['Symbol'].tolist()
        comparison_data = yf.download(
            tickers=symbols,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by='ticker',
            auto_adjust=True
        )

        if not comparison_data.empty:
            fig_comp = go.Figure()

            for symbol in symbols:
                if symbol in comparison_data.columns.levels[0]:
                    stock_data = comparison_data[symbol]
                    fig_comp.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name=symbol))

            # Customize layout for comparison chart
            fig_comp.update_layout(
                title="Comparison of Stock Price Trends",
                xaxis_title="Date",
                yaxis_title="Closing Price (USD)",
                xaxis_rangeslider_visible=True, 
            )

            st.plotly_chart(fig_comp)






# Define Sector Performance Analysis Page
if page == "Sector Performance Analysis":
    st.header("Sector Performance Analysis")

    # Sidebar - Date range selection for performance analysis
    st.sidebar.header('Select Date Range')
    start_date = st.sidebar.date_input('Start date', datetime(2023, 1, 1))
    end_date = st.sidebar.date_input('End date', datetime.today())

    if start_date > end_date:
        st.error('Error: End date must fall after start date.')
    all_symbols = df['Symbol'].tolist()
    data = yf.download(
        tickers=all_symbols,
        start=start_date,
        end=end_date,
        interval="1d",
        group_by='ticker',
        auto_adjust=True
    )

    # Stock performance by sector
    st.subheader("Sector Performance Heatmap")

    # Calculate sector performance
    sector_performance = {}
    for symbol in all_symbols:
        if symbol in data.columns.levels[0]:
            close_price = data[symbol]['Close']
            if len(close_price) > 0:
                percentage_change = (close_price[-1] - close_price[0]) / close_price[0] * 100
                sector = df[df['Symbol'] == symbol]['GICS Sector'].values[0]
                if sector not in sector_performance:
                    sector_performance[sector] = []
                sector_performance[sector].append(percentage_change)

    sector_avg_performance = {sector: sum(perfs) / len(perfs) for sector, perfs in sector_performance.items()}

    # Create a heatmap using Plotly Express
    performance_df = pd.DataFrame(list(sector_avg_performance.items()), columns=['Sector', 'Average Performance'])
    fig_heatmap = px.imshow([list(sector_avg_performance.values())], 
                            labels=dict(x="Sectors", y="Performance"), 
                            x=list(sector_avg_performance.keys()))
    fig_heatmap.update_layout(
        title="Sector-wise Performance Heatmap",
        xaxis_title="Sectors",
        yaxis_title="Performance",
    )

    st.plotly_chart(fig_heatmap)

    # Sector-wise Statistics
    st.subheader("Sector-wise Statistics")
    sector_stats = []

    for sector, perfs in sector_performance.items():
        if perfs: 
            avg_perf = round(sum(perfs) / len(perfs), 2)
            num_companies = len(perfs)

            # Safely get the top-performing company
            max_perf = max(perfs)
            top_performer_symbol = df[df['Symbol'] == df['Symbol'][perfs.index(max_perf)]]['Security'].values
            top_performer = top_performer_symbol[0] if len(top_performer_symbol) > 0 else "No Data"

            sector_stats.append({
                "Sector": sector,
                "Average Performance (%)": avg_perf,
                "Number of Companies": num_companies,
                "Top Performer": top_performer
            })

    # Convert list of dictionaries to DataFrame for display
    sector_stats_df = pd.DataFrame(sector_stats)
    st.dataframe(sector_stats_df)

    # Top Gainers and Losers by Sector
    st.subheader("Top Gainers and Losers by Sector")
    top_gainers = {}
    top_losers = {}

    for sector, perfs in sector_performance.items():
        if perfs: 
            max_perf = max(perfs)
            min_perf = min(perfs)
            
            # Get the index of the max and min performance
            max_index = perfs.index(max_perf)
            min_index = perfs.index(min_perf)
            
            # Get the corresponding symbols
            top_gainers[sector] = df.loc[df['Symbol'] == all_symbols[max_index], 'Security'].values[0] if max_index < len(all_symbols) else "No Data"
            top_losers[sector] = df.loc[df['Symbol'] == all_symbols[min_index], 'Security'].values[0] if min_index < len(all_symbols) else "No Data"

    gainers_df = pd.DataFrame({
        "Sector": top_gainers.keys(),
        "Top Gainer": top_gainers.values(),
        "Top Loser": top_losers.values()
    })

    st.dataframe(gainers_df)


    # Sector-wise Performance Over Time
    st.subheader("Sector-wise Performance Over Time")
    sector_trends = {}

    for sector, symbols in df.groupby('GICS Sector')['Symbol']:
        sector_close_prices = pd.DataFrame()
        for symbol in symbols:
            if symbol in data.columns.levels[0]:
                sector_close_prices[symbol] = data[symbol]['Close']
        sector_trends[sector] = sector_close_prices.mean(axis=1)

    # Plot sector performance trends
    fig_sector_trends = go.Figure()
    for sector, trend in sector_trends.items():
        fig_sector_trends.add_trace(go.Scatter(x=trend.index, y=trend, mode='lines', name=sector))

    fig_sector_trends.update_layout(
        title="Sector-wise Performance Trends",
        xaxis_title="Date",
        yaxis_title="Average Closing Price",
        xaxis_rangeslider_visible=True, 
    )

    st.plotly_chart(fig_sector_trends)



# Company Profile Page
if page == "Company Profile":
    st.header("Company Profile Page")

    # Sidebar - Company Selection
    st.sidebar.header('Select Company')
    company_options = df['Security'].tolist()
    default_company = 'Apple Inc.'
    selected_company = st.sidebar.selectbox('Select a company:', company_options, index=company_options.index(default_company))

    # Get company details from DataFrame
    company_data = df[df['Security'] == selected_company].iloc[0]
    symbol = company_data['Symbol']
    sector = company_data['GICS Sector']
    sub_industry = company_data['GICS Sub-Industry']
    headquarters = company_data['Headquarters Location']
    founding_date = company_data['Founded']  

    # Convert the founding date to a datetime object
    founding_date_dt = pd.to_datetime(founding_date)

    # Sidebar - Date Range Selection
    st.sidebar.header('Select Date Range')
    start_date = st.sidebar.date_input('Start date', value=founding_date_dt, min_value=founding_date_dt, max_value=datetime.today())
    end_date = st.sidebar.date_input('End date', value=datetime.today(), min_value=start_date)

    if start_date > end_date:
        st.error('Error: End date must fall after start date.')

    # Fetch stock data for the selected company from Yahoo Finance
    stock_data = yf.download(symbol, start=start_date, end=end_date, interval="1d")

    # Calculate percentage gains for the last year and last ten years
    last_year_gain = ((stock_data['Close'][-1] - stock_data['Close'][-252]) / stock_data['Close'][-252]) * 100 if len(stock_data) > 252 else 0
    last_ten_years_gain = ((stock_data['Close'][-1] - stock_data['Close'][0]) / stock_data['Close'][0]) * 100 if len(stock_data) > 0 else 0

    st.subheader(f"Profile for {selected_company}")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Symbol:** {symbol}")
        st.markdown(f"**Sector:** {sector}")
        st.markdown(f"**Sub-Industry:** {sub_industry}")
        st.markdown(f"**Headquarters:** {headquarters}")
        st.markdown(f"**Founding Date:** {founding_date}")

    with col2:
        st.markdown("### Performance:")
        st.markdown(f"**Gain Last Year:** {last_year_gain:.2f}%")
        st.markdown(f"**Gain Last 10 Years:** {last_ten_years_gain:.2f}%")

    # Stock Performance Over Time
    st.subheader("Stock Performance Over Time")
    st.line_chart(stock_data['Close'])



# the Stock Prediction and Trend Analysis Page
if page == "Stock Prediction and Trend Analysis":
    st.header("Stock Prediction and Trend Analysis")

    st.markdown("""
        ### Prediction Methodology
        - The model uses Linear Regression to predict future stock prices based on historical data.
        - A moving average is calculated to smooth out price fluctuations.
        - Adjust the parameters in the sidebar to explore different prediction scenarios.
    """)

    # Sidebar - Company Selection
    st.sidebar.header('Select Company')
    company_options = df['Security'].tolist()  # Assuming df contains the stock symbols
    default_company = 'Nvidia'
    selected_company = st.sidebar.selectbox('Select a company:', company_options, index=company_options.index(default_company))

    # Sidebar - Parameters for Prediction
    st.sidebar.header('Prediction Parameters')
    prediction_days = st.sidebar.slider("Days to Predict:", min_value=1, max_value=365, value=30)
    moving_average_window = st.sidebar.slider("Moving Average Window:", min_value=1, max_value=60, value=10)

    # Fetch historical stock data
    symbol = df[df['Security'] == selected_company]['Symbol'].values[0]
    stock_data = yf.download(symbol, start=(datetime.today() - timedelta(days=3650)), end=datetime.today())

    # Prepare Data for Prediction
    stock_data['Date'] = stock_data.index
    stock_data['Prediction'] = stock_data['Close'].shift(-prediction_days)

    # Create a linear regression model
    X = np.array(range(len(stock_data))).reshape(-1, 1)
    y = stock_data['Close'].values[:-prediction_days]

    model = LinearRegression()
    model.fit(X[:-prediction_days], y)

    # Predict future stock prices
    future_dates = np.array(range(len(stock_data), len(stock_data) + prediction_days)).reshape(-1, 1)
    predicted_prices = model.predict(future_dates)
    stock_data['Moving Average'] = stock_data['Close'].rolling(window=moving_average_window).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], mode='lines', name='Historical Prices'))
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Moving Average'], mode='lines', name='Moving Average'))

    # Prepare future dates for plotting
    future_dates_index = pd.date_range(start=stock_data.index[-1] + timedelta(days=1), periods=prediction_days)
    
    fig.add_trace(go.Scatter(x=future_dates_index, y=predicted_prices, mode='lines', name='Predicted Prices', line=dict(dash='dash')))
    
    fig.update_layout(
        title=f"{selected_company} Stock Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=True,
        template="plotly_dark" 
    )

    # Display the plot
    st.plotly_chart(fig)

    # Display Prediction Summary
    st.subheader("Prediction Summary")
    st.markdown(f"**Predicted prices for the next {prediction_days} days:**")

    # Create the predicted DataFrame
    predicted_df = pd.DataFrame({
        'Date': future_dates_index,
        'Predicted Price': predicted_prices
    })
    predicted_df_transposed = predicted_df.set_index('Date').T
    st.dataframe(predicted_df_transposed)
