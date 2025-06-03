# Public Transport Ridership Analytics Dashboard

A Streamlit-based web application for analyzing and forecasting daily public transport passenger journeys. The dashboard provides interactive visualizations, statistical insights, and predictive modeling for ridership data across various service types (e.g., Local Route, Light Rail) using historical data from 2019 to 2024. It supports forecasting for the entire year of 2025 using an XGBoost regression model.

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

## Requirements
- Python 3.8 or higher
- Libraries (specified in `requirements.txt`):

  streamlit==1.38.0  pandas==2.2.3  matplotlib==3.9.2  seaborn==0.13.2  xgboost==2.1.1  scikit-learn==1.5.2  numpy==2.1.1  plotly==5.24.1

## Installation
1. **Clone the Repository**:
 ```bash
 git clone <repository-url>
 cd public-transport-analytics


Create a Virtual Environment:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:pip install -r requirements.txt


Prepare Data:
Place your CSV file (2019-2024 data) in the project directory or update the default file path in app.py.
See Data Requirements for CSV format.


Run the Application:streamlit run app.py


Open your browser at http://localhost:8501.



Data Requirements

File Format: CSV
Columns:
Date: Date of ridership data (e.g., DD/MM/YYYY or YYYY-MM-DD).
Service type columns: Local Route, Light Rail, Peak Service, Rapid Route, School, Other (numeric values representing daily passenger journeys).


Sample Data:Date,Local Route,Light Rail,Peak Service,Rapid Route,School,Other
01/01/2019,5000,2000,1000,3000,1500,500
02/01/2019,5200,2100,1100,3100,1600,600
...
31/12/2024,6000,2500,1200,3500,1700,700


Notes:
Ensure dates are continuous (daily) from 2019 to 2024.
Non-numeric values are coerced to NaN and filled with 0.
Zero values may cause high MAPE; consider preprocessing (see Troubleshooting).



Usage

Launch the App:
Run streamlit run app.py and access http://localhost:8501.


Upload Data:
Use the sidebar to upload your 2019-2024 CSV or rely on the default file (Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv).


Filter Historical Data:
Select a date range in the sidebar to filter 2019-2024 data.
Choose service types (e.g., Local Route, Light Rail) for analysis.


Explore Tabs:
Overview: View metrics and trends.
Detailed Analysis: Analyze weekly patterns, correlations, and monthly trends.
Forecasting:
Select a service type.
Set the forecast range (default: January 1, 2025, to December 31, 2025).
Click "Generate Forecast" to predict 2025 ridership.
View the plot (2024 historical + 2025 forecast) and download the CSV.


Data Insights: Check statistics, peak days, and data quality.
Custom Analysis: Perform comparative, trend, seasonal, or correlation analysis.


Download Results:
Download 2025 forecast data as a CSV in the "Forecasting" tab.



Directory Structure
public-transport-analytics/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv  # Default dataset (optional)
├── README.md                       # Project documentation
└── venv/                           # Virtual environment (created after setup)

Troubleshooting

Error: module 'warnings' has no attribute 'filterWarnings':
Fixed in the latest code by using warnings.filterwarnings.
Ensure you're using the provided app.py code.


Error: cannot access local variable 'forecast':
Fixed by using a forecast_values list in generate_forecast.
Verify you're using the latest code.


High MAPE (999.99):
Likely due to zero or negative values in the service column.
Check Data:
In the "Data Insights" tab, review "Zero Values".
Add this to load_and_process_data in app.py to log zeros:for col in service_columns:
    zero_count = (df[col] == 0).sum()
    st.write(f"Zero values in {col}: {zero_count}")




Preprocess Zeros:
Replace zeros with the mean of non-zero values:for col in service_columns:
    df[col] = df[col].replace(0, df[col][df[col] > 0].mean())

Add this after pd.to_numeric in load_and_process_data.


Alternative Metrics:
Add MAE and RMSE to create_forecast_model after mape:from sklearn.metrics import mean_absolute_error, mean_squared_error
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
st.write(f"**MAE:** {mae:.2f}, **RMSE:** {rmse:.2f}")






Insufficient Data Error:
Ensure your CSV has at least 30 days of non-zero data for the selected service.
Check the "Data Insights" tab for data quality issues.


CSV Format Issues:
Verify the CSV has a Date column and service columns as specified.
Ensure dates are in a parseable format (e.g., DD/MM/YYYY).


Performance Issues:
For large datasets, forecasting 365 days (all of 2025) may be slow.
Reduce the forecast range (e.g., Q1 2025) or optimize the model (e.g., reduce n_estimators in XGBoost).



Contributing

Fork the repository.
Create a feature branch (git checkout -b feature/new-feature).
Commit changes (git commit -m "Add new feature").
Push to the branch (git push origin feature/new-feature).
Open a pull request.

Suggestions for enhancements:

Add holiday or event-based features to improve forecasting.
Implement alternative models (e.g., ARIMA, Prophet).
Enhance visualizations with zoom or filter options.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Developed by: [ASMITA G]
