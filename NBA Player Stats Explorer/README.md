# NBA Player Stats Explorer

A Streamlit web app that scrapes and visualizes NBA player stats from Basketball Reference. This app allows users to explore player data, generate advanced visualizations, and compare player statistics.

![Image 1](./imgs/NBA_Player_Stat_App.gif)

## Features

- **Web Scraping**: Pulls NBA player statistics from [Basketball Reference](https://www.basketball-reference.com/).
- **Player Search**: Find stats for specific players.
- **Exploratory Data Analysis (EDA)**:
  - Distribution plots
  - Stat comparisons
  - Box plots and bar charts
- **Advanced Visualizations**:
  - Pair plots
  - Radar charts for player comparison

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/AbdooMohamedd/20-Streamlit-Data-Science-Apps.git
   ```
2. Navigate to the project folder:
   ```bash
   cd "20-Streamlit-Data-Science-Apps/NBA Player Stats Explorer"
   ```
3. Install the required dependencies:
   ```bash
   pip install Plotly Seaborn Matplotlib Pandas Streamlit
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Data Source

- [Basketball Reference](https://www.basketball-reference.com/)

## License

This project is licensed under the MIT License.
