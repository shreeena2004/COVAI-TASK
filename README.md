# COVAI.CO TASK
# Public Transport Ridership Analytics Dashboard

A Streamlit-based web application for analyzing and forecasting daily public transport passenger journeys. The dashboard provides interactive visualizations, statistical insights, and predictive modeling for ridership data across various service types (e.g., Local Route, Light Rail) using historical data from 2019 to 2024. It supports forecasting for the entire year of 2025 using an XGBoost regression model.

# Predicted Output 

![WhatsApp Image 2025-06-03 at 12 00 31_aaaeed37](https://github.com/user-attachments/assets/5b8dff28-7144-429b-9d03-f2277bb127ba)

# Light Rail

![image](https://github.com/user-attachments/assets/bb09b551-fc50-42e7-bcf0-1d73d40b69d5)

# Peak Service

![image](https://github.com/user-attachments/assets/597ceb7b-e5f6-43b8-8144-859c2950cf03)

# Rapid Route

![image](https://github.com/user-attachments/assets/7bf161a8-d7b4-4f25-aed8-51f7693ec0e1)

# School 

![image](https://github.com/user-attachments/assets/e50b5c60-dd7e-46f9-ba17-24be99001ca4)



# Demo Video
https://drive.google.com/file/d/1O4rhasaGHU2P2lsqlNkI0JluLwlx6kGQ/view?usp=sharing



## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data Requirements](#data-requirements)
- [Usage](#usage)
- [Directory Structure](#directory-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
This project is designed to help transportation analysts and planners explore and predict public transport ridership patterns. The dashboard offers:
- **Data Visualization**: Interactive charts (line, area, bar, heatmap, pie) to explore trends, distributions, and correlations.
- **Statistical Insights**: Summary statistics, peak performance days, and data quality checks.
- **Forecasting**: Predict daily ridership for 2025 using historical data (2019-2024) with an XGBoost model.
- **Custom Analysis**: Comparative analysis, trend analysis, seasonal patterns, and correlation studies.

The application addresses previous issues, such as:
- Fixed the `cannot access local variable 'forecast'` error in the forecasting function.
- Mitigated high MAPE (999.99) by improving data handling and model robustness.
- Corrected a typo in `warnings.filterwarnings` to ensure the app runs without crashing.

## Features
1. **Overview Tab**:
   - Metrics: Total journeys, average daily ridership, peak day ridership, and 30-day growth rate.
   - Visualizations: Line, area, or bar charts for daily trends; pie and bar charts for service type distribution.
2. **Detailed Analysis Tab**:
   - Weekly patterns via heatmap.
   - Correlation analysis between service types.
   - Variability analysis (standard deviation).
   - Monthly trend visualization.
3. **Forecasting Tab**:
   - Predict daily ridership for 2025 (default: January 1, 2025, to December 31, 2025).
   - XGBoost regression model with lag features (1-day and 7-day lags).
   - Forecast accuracy reported via MAPE (or absolute error for zero/negative values).
   - Interactive plot showing historical (last 30 days of 2024) and forecasted (2025) data.
   - Downloadable CSV of forecast results.
4. **Data Insights Tab**:
   - Key statistics (mean, std, min, max, etc.).
   - Peak performance days per service type.
   - Data quality checks (missing values, zero values, value ranges).
   - Raw data preview.
5. **Custom Analysis Tab**:
   - Comparative analysis between two service types.
   - Trend analysis with moving average.
   - Seasonal patterns by month.
   - Custom correlation matrix with insights.
