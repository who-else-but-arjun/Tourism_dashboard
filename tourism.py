# Part 1: Data Preprocessing and Imputation
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(layout="wide", page_title="Tourism & Economic Analysis Dashboard")
@st.cache_data
def load_and_preprocess_data():

    df = pd.read_csv("tourism_data.csv")
    df['year'] = pd.to_datetime(df['year'], format='%Y')
    regions = {
        'USA': 'North America', 'CAN': 'North America', 'MEX': 'North America',
        'GBR': 'Europe', 'FRA': 'Europe', 'DEU': 'Europe', 'ITA': 'Europe', 'ESP': 'Europe',
        'CHN': 'Asia', 'JPN': 'Asia', 'KOR': 'Asia', 'IND': 'Asia',
        'BRA': 'South America', 'ARG': 'South America', 'CHL': 'South America',
        'AUS': 'Oceania', 'NZL': 'Oceania'
    }
    df['region'] = df['country_code'].map(regions)
    
    return df
df = load_and_preprocess_data()

def impute_missing_values(df):
    df_imputed = df.copy()
    
    for country in df_imputed['country'].unique():
        country_data = df_imputed[df_imputed['country'] == country].sort_values('year')
        numeric_cols = ['tourism_receipts', 'tourism_arrivals', 'tourism_exports', 
                       'tourism_departures', 'tourism_expenditures', 'gdp', 
                       'inflation', 'unemployment']
        
        for col in numeric_cols:
            if country_data[col].isnull().any():
                country_data[col] = country_data[col].interpolate(method='linear')
                if country_data[col].isnull().any():
                    X = country_data[country_data[col].notnull()]['year'].astype(np.int64).values.reshape(-1, 1)
                    y = country_data[country_data[col].notnull()][col].values
                    
                    if len(X) > 0:
                        reg = LinearRegression()
                        reg.fit(X, y)
                        X_missing = country_data[country_data[col].isnull()]['year'].astype(np.int64).values.reshape(-1, 1)
                        predictions = reg.predict(X_missing)

                        country_data.loc[country_data[col].isnull(), col] = predictions
                
            df_imputed.loc[country_data.index, col] = country_data[col]
    
    return df_imputed

df_imputed = impute_missing_values(df)
df_imputed['tourism_gdp_ratio'] = (df_imputed['tourism_receipts'] / df_imputed['gdp']) * 100
df_imputed['net_tourism_balance'] = df_imputed['tourism_receipts'] - df_imputed['tourism_expenditures']
df_imputed['tourism_per_capita'] = df_imputed['tourism_arrivals'] / df_imputed['gdp']

st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", 
    ["Overview", "Economic Impact", "Regional Analysis", "Time Series Analysis", "Predictive Modeling"])

# Main content
if page == "Overview":
    st.title("Tourism & Economic Analysis Dashboard")
    
    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Countries", len(df_imputed['country'].unique()))
    with col2:
        st.metric("Date Range", f"{df_imputed['year'].dt.year.min()} - {df_imputed['year'].dt.year.max()}")
    with col3:
        st.metric("Avg Tourism GDP Contribution", 
                 f"{df_imputed['tourism_gdp_ratio'].mean():.2f}%")
    with col4:
        st.metric("Total Tourism Receipts", 
                 f"${df_imputed['tourism_receipts'].sum()/1e12:.2f}T")
    
    # Global trends
    st.subheader("Global Tourism Trends")
    col1, col2 = st.columns(2)
    
    with col1:
        # Tourism arrivals trend
        yearly_arrivals = df_imputed.groupby('year')['tourism_arrivals'].sum().reset_index()
        fig = px.line(yearly_arrivals, x='year', y='tourism_arrivals',
                     title='Global Tourism Arrivals Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Tourism receipts vs GDP scatter
        fig = px.scatter(df_imputed, x='gdp', y='tourism_receipts',
                        color='region', hover_data=['country'],
                        title='Tourism Receipts vs GDP by Region',
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    # Economic impact analysis
    st.subheader("Economic Impact Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        # Tourism's contribution to GDP
        fig = px.box(df_imputed, x='region', y='tourism_gdp_ratio',
                    title="Tourism's Contribution to GDP by Region")
        st.plotly_chart(fig)

    
    with col2:
        # Tourism vs Unemployment
        fig = px.scatter(df_imputed, x='tourism_gdp_ratio', y='unemployment',
                        color='region', hover_data=['country'],
                        title='Tourism GDP Ratio vs Unemployment',
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Economic Impact":
    st.title("Economic Impact Analysis")
    
    # Country selection
    country = st.selectbox("Select Country", df_imputed['country'].unique())
    country_data = df_imputed[df_imputed['country'] == country]
    
    # Economic indicators
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(country_data, x='year', 
                     y=['tourism_receipts', 'tourism_expenditures'],
                     title='Tourism Receipts vs Expenditures')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(country_data, x='tourism_gdp_ratio', y='inflation',
                        title='Tourism GDP Ratio vs Inflation',
                        trendline="ols")
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional economic metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.line(country_data, x='year', y='unemployment',
                     title='Unemployment Rate Over Time')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.line(country_data, x='year', y='tourism_exports',
                     title='Tourism Exports Over Time')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Regional Analysis":
    st.title("Regional Analysis")
    
    # Region selection
    region = st.selectbox("Select Region", df_imputed['region'].unique())
    region_data = df_imputed[df_imputed['region'] == region]
    
    # Regional comparisons
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(region_data.groupby('country')['tourism_receipts'].mean().reset_index(),
                    x='country', y='tourism_receipts',
                    title=f'Average Tourism Receipts by Country in {region}')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(region_data, x='tourism_arrivals', y='tourism_receipts',
                        color='country', title='Tourism Arrivals vs Receipts by Country')
        st.plotly_chart(fig, use_container_width=True)
    
    # Regional trends
    col1, col2 = st.columns(2)
    
    with col1:
        yearly_region = region_data.groupby(['year', 'country'])['tourism_gdp_ratio'].mean().reset_index()
        fig = px.line(yearly_region, x='year', y='tourism_gdp_ratio',
                     color='country', title=f'Tourism GDP Ratio Trends in {region}')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(region_data, x='country', y='tourism_exports',
                    title='Distribution of Tourism Exports by Country')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Time Series Analysis":
    st.title("Time Series Analysis")
    
    # Feature selection
    feature = st.selectbox("Select Feature", 
                          ['tourism_receipts', 'tourism_arrivals', 'tourism_exports',
                           'tourism_expenditures', 'tourism_gdp_ratio'])
    
    # Global trends
    st.subheader("Global Trends")
    yearly_feature = df_imputed.groupby('year')[feature].mean().reset_index()
    fig = px.line(yearly_feature, x='year', y=feature,
                 title=f'Global {feature.replace("_", " ").title()} Trend')
    st.plotly_chart(fig, use_container_width=True)
    
    # Seasonal patterns
    col1, col2 = st.columns(2)
    
    with col1:
        # Year-over-year growth
        yearly_feature['yoy_growth'] = yearly_feature[feature].pct_change() * 100
        fig = px.bar(yearly_feature, x='year', y='yoy_growth',
                    title='Year-over-Year Growth Rate (%)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution analysis
        fig = px.box(df_imputed, x='region', y=feature,
                    title=f'Distribution of {feature.replace("_", " ").title()} by Region')
        st.plotly_chart(fig, use_container_width=True)

elif page == "Predictive Modeling":
    st.title("Predictive Modeling")
    
    # Model parameters
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.selectbox("Select Country", df_imputed['country'].unique())
        feature = st.selectbox("Select Feature to Predict",
                             ['tourism_receipts', 'tourism_arrivals', 'tourism_gdp_ratio'])
    
    with col2:
        forecast_years = st.slider("Forecast Years", 1, 5, 3)
    
    # Prepare data for forecasting
    country_data = df_imputed[df_imputed['country'] == country].sort_values('year')
    
    # LSTM forecasting
    sequence_length = 5
    series = country_data[feature].values
    X, y = [], []
    
    for i in range(len(series) - sequence_length):
        X.append(series[i:i+sequence_length])
        y.append(series[i+sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Build and train LSTM model
    model = Sequential([
        LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=50, batch_size=2, verbose=0)
    
    # Generate forecast
    last_sequence = series[-sequence_length:]
    forecast = []
    
    for _ in range(forecast_years):
        next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1))[0][0]
        forecast.append(next_pred)
        last_sequence = np.append(last_sequence[1:], next_pred)
    
    # Plot results
    col1, col2 = st.columns(2)
    
    with col1:
        # Historical and forecast plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=country_data['year'], y=country_data[feature],
                                name='Historical', mode='lines+markers'))
        
        future_dates = pd.date_range(start=country_data['year'].max(), periods=forecast_years+1, freq='Y')[1:]
        fig.add_trace(go.Scatter(x=future_dates, y=forecast,
                                name='Forecast', mode='lines+markers',
                                line=dict(dash='dash')))
        
        fig.update_layout(title=f'{feature.replace("_", " ").title()} Forecast')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Forecast summary
        st.subheader("Forecast Summary")
        forecast_df = pd.DataFrame({
            'Year': future_dates.year,
            'Forecast': forecast,
            'Change (%)': [(f/series[-1] - 1) * 100 for f in forecast]
        })
        st.dataframe(forecast_df)

st.sidebar.markdown("---")
st.sidebar.subheader("Data Quality Metrics")
st.sidebar.metric("Total Records", len(df_imputed))
completeness = (1 - df_imputed.isnull().sum().sum() / (df_imputed.shape[0] * df_imputed.shape[1])) * 100
st.sidebar.metric("Data Completeness", f"{completeness:.1f}%")