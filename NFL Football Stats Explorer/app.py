import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64
import requests

st.title("NFL Football Stats Explorer")

st.markdown("""
This app performs simple web scraping of NFL Football player stats data, including rushing, passing, receiving, defense, and kicking.
* **Python libraries:** base64, pandas, streamlit, plotly, matplotlib, seaborn, and others
* **Data source:** [pro-football-reference.com](https://www.pro-football-reference.com/).
""")

# Sidebar
st.sidebar.header('Navigation')
page = st.sidebar.radio('Go to', ['Rushing', 'Passing', 'Receiving', 'Defense'])

# Sidebar - Year selection
st.sidebar.header('User Input Features')
selected_year = st.sidebar.selectbox('Year', list(reversed(range(1990, 2025)))) 

# Use st.cache_data for loading data
@st.cache_data
def load_data(url):
    html = pd.read_html(url, header=1)
    df = html[0]
    if 'Age' in df.columns:
        df = df.drop(df[df['Age'] == 'Age'].index)
    df = df.fillna(0)
    return df

# Function to download data
def filedownload(df, filename):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv">Download CSV File</a>'
    return href

# Page 1: Rushing Stats
if page == 'Rushing':
    st.title("NFL Rushing Stats")

    url = f"https://www.pro-football-reference.com/years/{selected_year}/rushing.htm"
    playerstats = load_data(url)

    # Common Sidebar elements
    sorted_unique_team = sorted(playerstats.Tm.unique())
    unique_pos = ['RB', 'QB', 'WR', 'FB', 'TE']

    selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)
    selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

    df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]

    st.write(f'Data Dimension: {df_selected_team.shape[0]} rows and {df_selected_team.shape[1]} columns.')
    st.dataframe(df_selected_team)
    st.markdown(filedownload(df_selected_team, "rushing_stats"), unsafe_allow_html=True)

    # Distribution of Rushing Yards
    st.header('Distribution of Rushing Yards')
    if not df_selected_team.empty:  
        fig = px.histogram(df_selected_team, x='Yds', nbins=50, title='Rushing Yards Distribution')
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected team and position.")

    # Pie Chart
    st.header('Rushing Attempts by Player (Pie Chart)')
    if not df_selected_team.empty:
        fig = px.pie(df_selected_team, values='Att', names='Player', title='Rushing Attempts by Player')
        st.plotly_chart(fig)
    else:
        st.warning("No data available for the selected team and position.")




# Define the correct column names for the receiving page
column_names = ['Rk', 'Player', 'Tm', 'Age', 'Pos', 'G', 'GS', 'Tgt', 'Rec', 'Ctch%', 'Yds', 'Y/R', 'TD', '1D', 'Succ%', 'Lng', 'Y/Tgt', 'R/G', 'Y/G', 'Fmb']

# Load the data from the URL for Receiving
def load_data_receiving(url):
    html = requests.get(url).content
    df = pd.read_html(html)[0]
    df.columns = column_names

    # Drop any rows that don't have player data 
    df = df[df['Rk'] != 'Rk']  
    df = df.dropna() 

    return df



# Page 2: Receiving Stats
if page == 'Receiving':
    st.title("NFL Receiving Stats")
    url = f"https://www.pro-football-reference.com/years/{selected_year}/receiving.htm"
    playerstats = load_data_receiving(url)

    # Check if 'Tm' (Team) column exists
    if 'Tm' in playerstats.columns:
        sorted_unique_team = sorted(playerstats['Tm'].unique())
        selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

        sorted_unique_position = sorted(playerstats['Pos'].unique())
        selected_position = st.sidebar.multiselect('Position', sorted_unique_position, sorted_unique_position)

        # Filter by team and position
        df_selected_team = playerstats[playerstats['Tm'].isin(selected_team)]
        df_selected_team = df_selected_team[df_selected_team['Pos'].isin(selected_position)]

        st.write(f"Data Dimension: {df_selected_team.shape[0]} rows and {df_selected_team.shape[1]} columns.")
        st.dataframe(df_selected_team)

        # Catch Percentage Distribution
        st.header('Catch Percentage Distribution')
        if not df_selected_team.empty:
            fig_ctch = px.histogram(df_selected_team, x='Ctch%', nbins=50, title='Catch Percentage Distribution')
            st.plotly_chart(fig_ctch)

            # Yards per Reception Distribution
            st.header('Yards per Reception Distribution')
            fig_yr = px.histogram(df_selected_team, x='Y/R', nbins=50, title='Yards per Reception Distribution')
            st.plotly_chart(fig_yr)

            # Top 5 Players by Yards
            st.header('Top 5 Players by Yards')
            top_5_yards = df_selected_team[['Player', 'Tm', 'Yds']].sort_values(by='Yds', ascending=False).head(5)
            st.dataframe(top_5_yards)

            # Longest Reception Distribution
            st.header('Longest Reception Distribution')
            fig_lng = px.histogram(df_selected_team, x='Lng', nbins=50, title='Longest Reception Distribution')
            st.plotly_chart(fig_lng)

        # File download option
        st.markdown(filedownload(df_selected_team, "receiving_stats"), unsafe_allow_html=True)
    else:
        st.error("The expected 'Tm' column is not found. Please check the data structure.")

# Load data function for Passing page
def load_data_passing(url):
    df = pd.read_html(url, header=0)[0]
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df



# Page 3: Passing Stats
if page == 'Passing':

    st.title("NFL Passing Stats")

    url = f"https://www.pro-football-reference.com/years/{selected_year}/passing.htm"
    playerstats = load_data_passing(url)

    # Check if 'Team' column is in the DataFrame
    if 'Team' not in playerstats.columns:
        st.error("The 'Team' column is missing from the dataset.")
    else:
        # Sidebar elements
        sorted_unique_team = sorted(playerstats['Team'].unique())
        selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

        df_selected_team = playerstats[playerstats['Team'].isin(selected_team)]

        st.write(f'Data Dimension: {df_selected_team.shape[0]} rows and {df_selected_team.shape[1]} columns.')
        st.dataframe(df_selected_team)

        st.markdown(filedownload(df_selected_team, "passing_stats"), unsafe_allow_html=True)

        # Distribution of Passing Yards
        st.header('Distribution of Passing Yards')
        if not df_selected_team.empty:
            fig = px.histogram(df_selected_team, x='Yds', nbins=50, title='Passing Yards Distribution')
            st.plotly_chart(fig)
        else:
            st.warning("No data available for the selected team.")

# Load data function for Defense page
def load_data_defense(url):
    df = pd.read_html(url, header=0)[0]
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df




# Page 4: Defense Stats
if page == 'Defense':
    st.title("NFL Defense Stats")

    url = f"https://www.pro-football-reference.com/years/{selected_year}/defense.htm"
    playerstats = load_data_defense(url)

    # Convert relevant columns to numeric and handle errors
    for col in playerstats.columns:
        playerstats[col] = pd.to_numeric(playerstats[col], errors='coerce').fillna(0)

    # Combine relevant columns into single features
    playerstats['Total_Def_Int'] = playerstats[['Def Interceptions', 'Def Interceptions.1', 
                                                  'Def Interceptions.2', 'Def Interceptions.3', 
                                                  'Def Interceptions.4']].sum(axis=1)
    
    playerstats['Total_Fumbles'] = playerstats[['Fumbles', 'Fumbles.1', 
                                                 'Fumbles.2', 'Fumbles.3', 
                                                 'Fumbles.4']].sum(axis=1)

    playerstats['Total_Tackles'] = playerstats[['Tackles', 'Tackles.1', 
                                                 'Tackles.2', 'Tackles.3', 
                                                 'Tackles.4']].sum(axis=1)

    # Display the first few rows to diagnose the data
    st.write("Preview of the playerstats DataFrame:")
    st.dataframe(playerstats.head())  

    # Visualizations
    st.header('Defense Statistics Overview')

    # Distribution of Total Tackles
    st.header('Distribution of Total Tackles')
    if 'Total_Tackles' in playerstats.columns:
        fig_tackles = px.histogram(playerstats, x='Total_Tackles', nbins=50, 
                                    title='Total Tackles Distribution')
        st.plotly_chart(fig_tackles)
    
    # Total Interceptions vs. Total Fumbles
    st.header('Total Interceptions vs. Total Fumbles')
    if 'Total_Def_Int' in playerstats.columns and 'Total_Fumbles' in playerstats.columns:
        fig_interceptions_fumbles = px.scatter(playerstats, 
                                                x='Total_Def_Int', 
                                                y='Total_Fumbles', 
                                                title='Total Interceptions vs Total Fumbles')
        st.plotly_chart(fig_interceptions_fumbles)
