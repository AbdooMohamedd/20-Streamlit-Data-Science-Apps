# NFL Football Stats Explorer

Welcome to the **NFL Football Stats Explorer** app! This application allows users to explore various statistics of NFL players through a simple web interface.

![Image 1](./imgs/app.gif)

## Overview

The app performs web scraping of NFL player statistics, including:

- Rushing
- Passing
- Receiving
- Defense

**Data Source:** [pro-football-reference.com](https://www.pro-football-reference.com/)

## Features

- Interactive data visualization with Plotly
- Filter data by year, team, and player position
- Download statistics as CSV files
- Statistical distributions and summaries for various player metrics

## Technologies Used

- **Python Libraries:**
  - `Streamlit`: For creating the web app interface.
  - `Pandas`: For data manipulation and analysis.
  - `NumPy`: For numerical computations.
  - `Plotly`: For interactive visualizations.
  - `Requests`: For making HTTP requests.

## Installation

To run the application locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/AbdooMohamedd/20-Streamlit-Data-Science-Apps.git
   cd 20-Streamlit-Data-Science-Apps/NFL Football Stats Explorer
   ```

2. Install the required packages:

   ```bash
   pip install streamlit pandas numpy plotly requests
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Select the desired year from the sidebar.
2. Navigate between different statistics pages (Rushing, Passing, Receiving, Defense) using the sidebar.
3. Filter data based on teams and positions.
4. Visualize player statistics through interactive charts and graphs.
5. Download the displayed data as a CSV file.

## License

This project is licensed under the MIT License.
