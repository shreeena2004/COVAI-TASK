import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterWarnings('ignore')

st.set_page_config(
    page_title="🚌 Public Transport Analytics Dashboard",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_process_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("Daily_Public_Transport_Passenger_Journeys_by_Service_Type_20250603.csv")
    
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df['Day_of_Week'] = df['Date'].dt.day_name()
    df['Day_of_Week_Num'] = df['Date'].dt.dayofweek
    df['Is_Weekend'] = df['Day_of_Week_Num'].isin([5, 6]).astype(int)
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    service_columns = ['Local Route', 'Light Rail', 'Peak Service', 'Rapid Route', 'School', 'Other']
    
    for col in service_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    for service in service_columns[:5]:
        df[f'Lag_1_{service}'] = df[service].shift(1).astype('float64')
        df[f'Lag_7_{service}'] = df[service].shift(7).astype('float64')
    
    df = df.fillna(0)
    
    for col in df.columns:
        if col.startswith('Lag_'):
            if not pd.api.types.is_numeric_dtype(df[col]):
                st.error(f"Column {col} is not numeric: {df[col].dtype}")
                raise ValueError(f"Column {col} must be numeric")
    
    return df, service_columns

def create_forecast_model(data, service):
    lag_cols = [f'Lag_1_{service}', f'Lag_7_{service}']
    
    data_clean = data.copy()
    for col in data_clean.columns:
        if col not in ['Date', 'Day_of_Week']:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce').fillna(0).astype('float64')
    
    if all(col in data_clean.columns for col in lag_cols):
        feature_cols = ['Day_of_Week_Num', 'Is_Weekend'] + lag_cols
        X = data_clean[feature_cols].dropna()
        y = data_clean[service].iloc[len(data_clean) - len(X):]
    else:
        feature_cols = ['Day_of_Week_Num', 'Is_Weekend']
        X = data_clean[feature_cols]
        y = data_clean[service]
    
    X = X.astype('float64')
    y = y.astype('float64')
    
    if len(X) < 30:
        st.warning("Insufficient data for training (less than 30 samples).")
        return None, None, None
    
    if (y <= 0).all():
        st.warning(f"All target values for {service} are zero or negative. Cannot compute MAPE.")
        return None, None, None
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
        enable_categorical=False
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    y_test_positive = y_test[y_test > 0]
    y_pred_positive = y_pred[y_test > 0]
    
    if len(y_test_positive) > 0:
        mape = np.mean(np.abs((y_test_positive - y_pred_positive) / y_test_positive)) * 100
    else:
        mape = np.mean(np.abs(y_test - y_pred))
        st.warning(f"No positive target values for {service}. Using absolute error instead of MAPE.")
    
    mape = min(mape, 999.99)
    
    return model, mape, feature_cols

def generate_forecast(model, data, service, start_date, end_date, feature_cols):
    if model is None:
        return None
    
    forecast_days = (end_date - start_date).days + 1
    if forecast_days <= 0:
        st.error("End date must be after start date.")
        return None
    
    future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['Day_of_Week_Num'] = future_df['Date'].dt.dayofweek
    future_df['Is_Weekend'] = future_df['Day_of_Week_Num'].isin([5, 6]).astype(int)
    
    if f'Lag_1_{service}' in feature_cols:
        last_values = data[service].iloc[-7:].values
        future_features = []
        forecast_values = []
        
        for i in range(forecast_days):
            row = [future_df.iloc[i]['Day_of_Week_Num'], future_df.iloc[i]['Is_Weekend']]
            
            if i == 0:
                lag_1 = data[service].iloc[-1]
                lag_7 = data[service].iloc[-7]
            else:
                lag_1 = forecast_values[-1]
                lag_7 = last_values[i] if i < 7 else forecast_values[i-7]
            
            row.extend([lag_1, lag_7])
            future_features.append(row)
            
            future_X = pd.DataFrame([row], columns=feature_cols)
            pred = model.predict(future_X)[0]
            forecast_values.append(pred)
        
        forecast_df = pd.DataFrame({'Date': future_dates, service: forecast_values})
    else:
        future_X = future_df[['Day_of_Week_Num', 'Is_Weekend']]
        forecast_values = model.predict(future_X)
        forecast_df = pd.DataFrame({'Date': future_dates, service: forecast_values})
    
    return forecast_df

st.title("🚌 Public Transport Ridership Analytics Dashboard")
st.markdown("### Comprehensive analysis and forecasting of daily passenger journeys")

st.sidebar.header("📊 Dashboard Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=['csv'],
    help="Upload your transport data CSV file or use default data"
)

try:
    df, service_columns = load_and_process_data(uploaded_file)
    
    date_range = st.sidebar.date_input(
        "Select Historical Date Range",
        value=(df['Date'].min().date(), df['Date'].max().date()),
        min_value=df['Date'].min().date(),
        max_value=df['Date'].max().date()
    )
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
        filtered_df = df.loc[mask]
    else:
        filtered_df = df
    
    selected_services = st.sidebar.multiselect(
        "Select Service Types",
        service_columns,
        default=service_columns,
        help="Choose which transport services to analyze"
    )
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "📊 Detailed Analysis", "🔮 Forecasting", "📋 Data Insights", "⚙️ Custom Analysis"])
    
    with tab1:
        st.header("📈 Ridership Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_journeys = filtered_df[selected_services].sum().sum()
            st.metric("Total Journeys", f"{total_journeys:,.0f}")
        
        with col2:
            avg_daily = filtered_df[selected_services].sum(axis=1).mean()
            st.metric("Avg Daily Ridership", f"{avg_daily:,.0f}")
        
        with col3:
            peak_day = filtered_df[selected_services].sum(axis=1).max()
            st.metric("Peak Day Ridership", f"{peak_day:,.0f}")
        
        with col4:
            growth_rate = ((filtered_df[selected_services].sum(axis=1).iloc[-30:].mean() /
                           filtered_df[selected_services].sum(axis=1).iloc[:30].mean() - 1) * 100)
            st.metric("30-Day Growth Rate", f"{growth_rate:.1f}%")
        
        st.subheader("Daily Ridership Trends")
        
        chart_type = st.selectbox("Chart Type", ["Line Chart", "Area Chart", "Bar Chart"])
        
        if chart_type == "Line Chart":
            fig = go.Figure()
            for service in selected_services:
                fig.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df[service],
                    mode='lines',
                    name=service,
                    line=dict(width=2)
                ))
        elif chart_type == "Area Chart":
            fig = go.Figure()
            for service in selected_services:
                fig.add_trace(go.Scatter(
                    x=filtered_df['Date'],
                    y=filtered_df[service],
                    mode='lines',
                    name=service,
                    fill='tonexty' if service != selected_services[0] else 'tozeroy'
                ))
        else:
            fig = px.bar(
                filtered_df.melt(id_vars=['Date'], value_vars=selected_services),
                x='Date', y='value', color='variable',
                title="Daily Ridership by Service Type"
            )
        
        fig.update_layout(
            title="Ridership Trends Over Time",
            xaxis_title="Date",
            yaxis_title="Passenger Journeys",
            height=500,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Service Type Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            avg_ridership = filtered_df[selected_services].mean()
            fig_pie = px.pie(
                values=avg_ridership.values,
                names=avg_ridership.index,
                title="Average Daily Ridership Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            fig_bar = px.bar(
                x=avg_ridership.index,
                y=avg_ridership.values,
                title="Average Daily Ridership by Service",
                labels={'x': 'Service Type', 'y': 'Average Journeys'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        st.header("📊 Detailed Analysis")
        
        st.subheader("Weekly Patterns")
        weekly_patterns = filtered_df.groupby('Day_of_Week')[selected_services].mean()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_patterns = weekly_patterns.reindex(day_order)
        
        fig_heatmap = px.imshow(
            weekly_patterns.T,
            labels=dict(x="Day of Week", y="Service Type", color="Avg Journeys"),
            title="Average Ridership Heatmap by Day of Week"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Correlation Analysis")
            if len(selected_services) > 1:
                correlation = filtered_df[selected_services].corr()
                fig_corr = px.imshow(
                    correlation,
                    text_auto=True,
                    aspect="auto",
                    title="Service Type Correlations"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.subheader("Variability Analysis")
            variability = filtered_df[selected_services].std()
            fig_var = px.bar(
                x=variability.index,
                y=variability.values,
                title="Ridership Variability (Standard Deviation)"
            )
            st.plotly_chart(fig_var, use_container_width=True)
        
        st.subheader("Monthly Trends")
        monthly_data = filtered_df.groupby(['Year', 'Month'])[selected_services].sum().reset_index()
        monthly_data['Date'] = pd.to_datetime(monthly_data[['Year', 'Month']].assign(day=1))
        
        fig_monthly = go.Figure()
        for service in selected_services:
            fig_monthly.add_trace(go.Scatter(
                x=monthly_data['Date'],
                y=monthly_data[service],
                mode='lines+markers',
                name=service
            ))
        
        fig_monthly.update_layout(
            title="Monthly Ridership Trends",
            xaxis_title="Month",
            yaxis_title="Total Journeys",
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab3:
        st.header("🔮 Ridership Forecasting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Forecast Settings")
            forecast_service = st.selectbox("Select Service to Forecast", selected_services)
            
            last_historical_date = filtered_df['Date'].max()
            default_start_date = pd.Timestamp('2025-01-01')
            default_end_date = pd.Timestamp('2025-12-31')
            
            if last_historical_date >= default_start_date:
                default_start_date = last_historical_date + timedelta(days=1)
                default_end_date = default_start_date + timedelta(days=364)
            
            forecast_date_range = st.date_input(
                "Select Forecast Date Range (2025)",
                value=(default_start_date.date(), default_end_date.date()),
                min_value=(last_historical_date + timedelta(days=1)).date(),
                max_value=default_end_date.date()
            )
            
            if len(forecast_date_range) == 2:
                forecast_start_date, forecast_end_date = forecast_date_range
            else:
                forecast_start_date, forecast_end_date = default_start_date.date(), default_end_date.date()
            
            if st.button("Generate Forecast", type="primary"):
                with st.spinner("Training model and generating forecast..."):
                    model, mape, feature_cols = create_forecast_model(filtered_df, forecast_service)
                    
                    if model is not None:
                        st.success(f"Model trained successfully!")
                        st.write(f"**Model Accuracy (MAPE):** {mape:.4f}")
                        
                        forecast_df = generate_forecast(
                            model, filtered_df, forecast_service,
                            pd.Timestamp(forecast_start_date), pd.Timestamp(forecast_end_date),
                            feature_cols
                        )
                        
                        if forecast_df is not None:
                            st.session_state['forecast_df'] = forecast_df
                            st.session_state['forecast_service'] = forecast_service
                    else:
                        st.error("Unable to create forecast model. Insufficient data.")
        
        with col2:
            if 'forecast_df' in st.session_state:
                st.subheader(f"Forecast Results for {st.session_state['forecast_service']} (2025)")
                
                historical_recent = filtered_df[['Date', st.session_state['forecast_service']]].tail(30)
                
                fig_forecast = go.Figure()
                
                fig_forecast.add_trace(go.Scatter(
                    x=historical_recent['Date'],
                    y=historical_recent[st.session_state['forecast_service']],
                    mode='lines+markers',
                    name='Historical (2024)',
                    line=dict(color='blue')
                ))
                
                fig_forecast.add_trace(go.Scatter(
                    x=st.session_state['forecast_df']['Date'],
                    y=st.session_state['forecast_df'][st.session_state['forecast_service']],
                    mode='lines+markers',
                    name='Forecast (2025)',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_forecast.update_layout(
                    title=f"Ridership Forecast - {st.session_state['forecast_service']}",
                    xaxis_title="Date",
                    yaxis_title="Passenger Journeys",
                    height=500
                )
                st.plotly_chart(fig_forecast, use_container_width=True)
                
                st.subheader("Forecast Data")
                st.dataframe(st.session_state['forecast_df'], use_container_width=True)
                
                csv = st.session_state['forecast_df'].to_csv(index=False)
                st.download_button(
                    label="Download 2025 Forecast CSV",
                    data=csv,
                    file_name=f"forecast_{st.session_state['forecast_service'].lower().replace(' ', '_')}_2025.csv",
                    mime="text/csv"
                )
    
    with tab4:
        st.header("📋 Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Key Statistics")
            stats_df = filtered_df[selected_services].describe()
            st.dataframe(stats_df, use_container_width=True)
        
        with col2:
            st.subheader("Peak Performance Days")
            for service in selected_services:
                peak_day = filtered_df.loc[filtered_df[service].idxmax()]
                st.write(f"**{service}:**")
                st.write(f"- Peak: {peak_day[service]:,.0f} journeys")
                st.write(f"- Date: {peak_day['Date'].strftime('%Y-%m-%d')}")
                st.write(f"- Day: {peak_day['Day_of_Week']}")
                st.write("---")
        
        st.subheader("Data Quality Check")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            missing_data = filtered_df[selected_services].isnull().sum()
            st.write("**Missing Values:**")
            st.write(missing_data)
        
        with col2:
            zero_values = (filtered_df[selected_services] == 0).sum()
            st.write("**Zero Values:**")
            st.write(zero_values)
        
        with col3:
            data_range = pd.DataFrame({
                'Min': filtered_df[selected_services].min(),
                'Max': filtered_df[selected_services].max()
            })
            st.write("**Value Ranges:**")
            st.write(data_range)
        
        st.subheader("Raw Data Preview")
        st.dataframe(filtered_df[['Date', 'Day_of_Week'] + selected_services].head(20), use_container_width=True)
    
    with tab5:
        st.header("⚙️ Custom Analysis")
        
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Comparative Analysis", "Trend Analysis", "Seasonal Patterns", "Custom Correlation"]
        )
        
        if analysis_type == "Comparative Analysis":
            if len(selected_services) >= 2:
                service1 = st.selectbox("Service 1", selected_services, key="comp1")
                service2 = st.selectbox("Service 2", [s for s in selected_services if s != service1], key="comp2")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[service1], name=service1))
                fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[service2], name=service2))
                fig.update_layout(title=f"Comparison: {service1} vs {service2}")
                st.plotly_chart(fig, use_container_width=True)
                
                correlation_coef = filtered_df[service1].corr(filtered_df[service2])
                st.metric("Correlation Coefficient", f"{correlation_coef:.3f}")
        
        elif analysis_type == "Trend Analysis":
            selected_service = st.selectbox("Select Service for Trend Analysis", selected_services)
            window_size = st.slider("Moving Average Window", 3, 30, 7)
            
            filtered_df[f'{selected_service}_MA'] = filtered_df[selected_service].rolling(window=window_size).mean()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[selected_service], name='Original', opacity=0.5))
            fig.add_trace(go.Scatter(x=filtered_df['Date'], y=filtered_df[f'{selected_service}_MA'], name=f'Moving Avg ({window_size}d)'))
            fig.update_layout(title=f"Trend Analysis - {selected_service}")
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Seasonal Patterns":
            selected_service = st.selectbox("Select Service for Seasonal Analysis", selected_services)
            
            seasonal_data = filtered_df.groupby(filtered_df['Date'].dt.month)[selected_service].mean()
            
            fig = px.bar(x=seasonal_data.index, y=seasonal_data.values,
                        title=f"Monthly Average - {selected_service}",
                        labels={'x': 'Month', 'y': 'Average Journeys'})
            st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "Custom Correlation":
            if len(selected_services) >= 2:
                correlation_matrix = filtered_df[selected_services].corr()
                
                fig = px.imshow(correlation_matrix,
                               text_auto=True,
                               aspect="auto",
                               title="Custom Correlation Matrix")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Correlation Insights")
                for i, service1 in enumerate(selected_services):
                    for service2 in selected_services[i+1:]:
                        corr_val = correlation_matrix.loc[service1, service2]
                        if abs(corr_val) > 0.7:
                            st.write(f"**Strong correlation** between {service1} and {service2}: {corr_val:.3f}")
                        elif abs(corr_val) > 0.4:
                            st.write(f"**Moderate correlation** between {service1} and {service2}: {corr_val:.3f}")

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.write("Please make sure your CSV file has the correct format with Date column and service type columns.")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📝 Notes")
st.sidebar.markdown("- Upload CSV with Date and service columns (2019-2024)")
st.sidebar.markdown("- Use historical date range selector to filter data")
st.sidebar.markdown("- Select specific services for focused analysis")
st.sidebar.markdown("- Forecasting uses XGBoost to predict 2025 data")
st.markdown("---")
st.markdown("**Dashboard created with Streamlit** | Data visualization and forecasting for public transport analytics")
