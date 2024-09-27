import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import base64
import plotly.graph_objects as go


st.set_page_config(page_title="NBA Player Stats Explorer", layout="wide")

# 
page = st.sidebar.selectbox("Select Page", ["Home", "Advanced Visualizations"])

# Function to load data
@st.cache_data 
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == 'Age'].index)  
    raw = raw.fillna(0)
    playerstats = raw.drop(['Rk'], axis=1)
    
    playerstats.columns = playerstats.columns.str.strip()  
    
    return playerstats

# Home Page
if page == "Home":
    st.title("NBA Player Stats Explorer")

    st.markdown("""
    This app performs simple web scraping of NBA player stats data!
    * **Python libraries:** base64, pandas, streamlit, matplotlib, seaborn, plotly
    * **Data source:** [Basketball-reference.com](https://www.basketball-reference.com/).
    """)

    # Player Search
    st.header("Search for a Player")
    player_name = st.text_input("Enter Player Name:")
    if player_name:
        player_data = df_selected_pos[df_selected_pos['Player'].str.contains(player_name, case=False)]
        st.write(player_data)

    st.sidebar.header("User Input Features")
    selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2021))))

    playerstats = load_data(selected_year)

    # Sidebar - Position selection
    unique_pos = ['C', 'PF', 'SF', 'PG', 'SG']
    selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

    # Filtering data
    df_selected_pos = playerstats[playerstats['Pos'].isin(selected_pos)]
    st.write(df_selected_pos)
    st.write('Shape of dataset:', df_selected_pos.shape)

    if st.button('Show Stats Summary'):
        st.header('Player Stats Summary')
        summary_stats = df_selected_pos.describe()
        st.write(summary_stats)

    # EDA Visualizations
    st.header('Exploratory Data Analysis Visualizations')

    # Distribution of a selected statistic
    stat_to_plot = st.selectbox('Select Statistic for Distribution', df_selected_pos.columns[2:].tolist())
    if stat_to_plot:
        st.subheader(f'Distribution of {stat_to_plot}')
        
        fig_dist = plt.figure(figsize=(6, 2))  
        sns.histplot(df_selected_pos[stat_to_plot], bins=20, kde=True)
        plt.xticks(rotation=45, ha='right', fontsize=5) 
        plt.yticks(fontsize=8) 
        
        plt.title(f'Distribution of {stat_to_plot}', fontsize=10)    
        st.pyplot(fig_dist)


    # Correlation heatmap
    numeric_columns = df_selected_pos.select_dtypes(include=[np.number]).columns.tolist()

    # if st.button('Show Intercorrelation Heatmap'):
    #     st.header('Intercorrelation Matrix Heatmap')
    #     corr = df_selected_pos[numeric_columns].corr()  
    #     mask = np.zeros_like(corr)
    #     mask[np.triu_indices_from(mask)] = True
    #     with sns.axes_style("white"):
    #         f, ax = plt.subplots(figsize=(10, 8))
    #         ax = sns.heatmap(corr, mask=mask, vmax=1, square=True, annot=True, fmt='.2f', cmap='coolwarm')
    #     st.pyplot(f)

    # Download NBA player stats data
    def filedownload(df):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="playerstats.csv">Download CSV File</a>'
        return href

    st.markdown(filedownload(df_selected_pos), unsafe_allow_html=True)

    # Difference between two selected statistics
    st.header('Stat Difference Visualization')
    if numeric_columns:
        stat1 = st.selectbox('Select First Statistic', numeric_columns)
        stat2 = st.selectbox('Select Second Statistic', numeric_columns)

        if stat1 and stat2 and stat1 != stat2:
            df_selected_pos['Difference'] = df_selected_pos[stat1] - df_selected_pos[stat2]
            fig_diff = plt.figure(figsize=(6, 2))  
            sns.histplot(df_selected_pos['Difference'], bins=20, kde=True)
            plt.xticks(fontsize=5) 
            plt.yticks(fontsize=5)  
            plt.title(f'Difference between {stat1} and {stat2}', fontsize=10)          
            st.pyplot(fig_diff)


    # Bar chart for selected statistics
    if st.button('Show Bar Chart of Selected Stats'):
        st.header('Bar Chart of Selected Stats')
        stat_to_plot = st.selectbox('Select Statistic', numeric_columns)
        fig = px.bar(df_selected_pos, x='Player', y=stat_to_plot, title=f'{stat_to_plot} of Selected Players')
        st.plotly_chart(fig)

    # Box plot for statistics based on position
    if st.button('Show Box Plot by Position'):
        st.header('Box Plot of Selected Stat by Position')
        stat_for_boxplot = st.selectbox('Select Statistic for Box Plot', numeric_columns)
        fig_box = px.box(df_selected_pos, x='Pos', y=stat_for_boxplot, title=f'Box Plot of {stat_for_boxplot} by Position')
        st.plotly_chart(fig_box)



# Advanced Visualizations Page
if page == "Advanced Visualizations":
    st.title("Advanced Visualizations for NBA Player Stats")

    selected_year = st.sidebar.selectbox('Year', list(reversed(range(1950, 2021))), index=0)
    playerstats = load_data(selected_year)
    numeric_columns = playerstats.select_dtypes(include=[np.number]).columns.tolist()

    # Advanced Pair Plot
    st.header("Advanced Pair Plot of Selected Stats")
    if numeric_columns:
        selected_stats = st.sidebar.multiselect('Select Stats for Pair Plot', numeric_columns, default=numeric_columns[:2])
        
        if len(selected_stats) > 1:
            fig_pair = sns.pairplot(playerstats[selected_stats])
            fig_pair.fig.set_size_inches(10, 5)
            for ax in fig_pair.axes.flatten():
                ax.set_xlabel(ax.get_xlabel(), fontsize=5)
                ax.set_ylabel(ax.get_ylabel(), fontsize=5)
            st.pyplot(fig_pair)
        else:
            st.warning("Please select at least two statistics for the pair plot.")


    # More EDA Options
    st.header("More EDA Options")
    eda_stat = st.sidebar.selectbox('Select Statistic for More EDA', numeric_columns)
    eda_summary = playerstats[eda_stat].describe()
    # st.write(eda_summary)

    if st.button('Show EDA Visualizations for Selected Stat'):
        fig_eda = plt.figure(figsize=(6, 2))
        sns.boxplot(x='Pos', y=eda_stat, data=playerstats)
        plt.title(f'Box Plot of {eda_stat} by Position', fontsize=10)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        st.pyplot(fig_eda)

    if st.button('Show Distribution for Selected Stat'):
        fig_dist_eda = plt.figure(figsize=(6, 2))
        sns.histplot(playerstats[eda_stat], bins=20, kde=True)
        plt.title(f'Distribution of {eda_stat}', fontsize=10)
        plt.xticks(fontsize=5)
        plt.yticks(fontsize=5)
        st.pyplot(fig_dist_eda)



    # Radar Chart for Player Comparison
    st.header("Player Comparison Radar Chart")
    selected_players = st.sidebar.multiselect("Select Players", playerstats['Player'].unique())

    if len(selected_players) > 1:
        comparison_data = playerstats[playerstats['Player'].isin(selected_players)][numeric_columns]

        comparison_data = comparison_data.set_index(playerstats[playerstats['Player'].isin(selected_players)]['Player'])
        comparison_data_normalized = (comparison_data - comparison_data.min()) / (comparison_data.max() - comparison_data.min())
        fig_radar = go.Figure()

        for player in comparison_data_normalized.index:
            fig_radar.add_trace(go.Scatterpolar(
                r=comparison_data_normalized.loc[player].values,
                theta=comparison_data_normalized.columns,
                fill='toself',
                name=player
            ))

        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title="Comparison of Selected Players' Stats"
        )

        st.plotly_chart(fig_radar)
    else:
        st.warning("Please select at least two players for comparison.")