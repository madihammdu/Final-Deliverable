import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import OLS, add_constant

st.title("Starbucks Financial Analysis 1")

# Load Starbucks revenue data
@st.cache_data
def load_revenue_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
    return df

df_revenue = load_revenue_data()

# ARIMA Forecasting section
st.subheader("ARIMA Forecast")
steps = st.slider("Months to forecast", 1, 12, 4)
expected = st.number_input("Enter your expected revenue for the final forecasted month (optional)", min_value=0.0, step=100.0)

model = ARIMA(df_revenue.set_index("date")["revenue"], order=(1, 1, 1))
results = model.fit()
forecast = results.get_forecast(steps=steps)
forecast_mean = forecast.predicted_mean
forecast_index = forecast_mean.index
ci = forecast.conf_int()

# Seamless forecast line
last_actual_date = df_revenue["date"].iloc[-1]
last_actual_value = df_revenue["revenue"].iloc[-1]
forecast_x = [last_actual_date] + list(forecast_index)
forecast_y = [last_actual_value] + list(forecast_mean)

# Forecast plot using Plotly
fig2 = go.Figure()

# Observed Revenue
fig2.add_trace(go.Scatter(
    x=df_revenue["date"], y=df_revenue["revenue"],
    mode='lines',
    name='Observed Revenue',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Observed: %{y}<extra></extra>'
))

# Fitted (in-sample predicted) values
fitted_vals = results.fittedvalues
fitted_vals.index = df_revenue["date"].iloc[-len(fitted_vals):]  # align dates
fig2.add_trace(go.Scatter(
    x=fitted_vals.index,
    y=fitted_vals,
    mode='lines',
    name='Fitted (ARIMA)',
    line=dict(color='green', dash='dot'),
    hovertemplate='Date: %{x}<br>Fitted: %{y:.0f}<extra></extra>'
))

# Forecast (future values) with transition
fig2.add_trace(go.Scatter(
    x=forecast_x, y=forecast_y,
    mode='lines',
    name='Forecast',
    line=dict(color='skyblue', dash='dash'),
    hovertemplate='Date: %{x}<br>Forecast: %{y:.0f}<extra></extra>'
))

# Confidence Interval
fig2.add_trace(go.Scatter(
    x=list(forecast_index) + list(forecast_index[::-1]),
    y=list(ci.iloc[:, 0]) + list(ci.iloc[:, 1][::-1]),
    fill='toself',
    fillcolor='rgba(192,192,192,0.3)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    name='Confidence Interval'
))

# Expected Revenue Marker
if expected > 0:
    fig2.add_trace(go.Scatter(
        x=[forecast_index[-1]],
        y=[expected],
        mode='markers+text',
        name='Your Expected Revenue',
        marker=dict(color='red', size=10),
        text=[f"${expected:,.0f}"],
        textposition="top center",
        hovertemplate='Expected: %{y}<extra></extra>'
    ))

fig2.update_layout(
    title="Forecasted Revenue with Fitted Values (ARIMA)",
    xaxis_title="Date",
    yaxis_title="Revenue",
    hovermode='x unified'
)

st.plotly_chart(fig2)

# ---- ARIMAX Section ----
st.subheader("ARIMAX Model: Revenue with CPI as External Regressor")

@st.cache_data
def fetch_cpi():
    api_key = "755f96b3bf3c15588f3de4dbd65ced72"
    cpi_series_id = "CPALTT01USQ657N"
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": cpi_series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": "2018-04-01"
    }
    response = requests.get(url, params=params)
    data = response.json()["observations"]
    df_cpi = pd.DataFrame(data)[["date", "value"]]
    df_cpi.columns = ["date", "CPI"]
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_cpi["CPI"] = df_cpi["CPI"].astype(float)
    return df_cpi

df_cpi = fetch_cpi()

# Align revenue and CPI by date
df_rev_adj = df_revenue.reset_index()
df_rev_adj["date"] = df_rev_adj["date"] + pd.Ti
