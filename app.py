import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from statsmodels.tsa.arima.model import ARIMA

st.title("Starbucks Financial Analysis")

# Load Starbucks revenue data
@st.cache_data
def load_revenue_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
    return df

df_revenue = load_revenue_data()

# Historical Revenue plot with hover
st.subheader("Historical Revenue")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=df_revenue["date"], y=df_revenue["revenue"],
    mode='lines+markers',
    name='Revenue',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Revenue: %{y}<extra></extra>'
))
fig1.update_layout(
    title="Historical Revenue Over Time",
    xaxis_title="Date",
    yaxis_title="Revenue",
    hovermode='x unified'
)
st.plotly_chart(fig1)

# ARIMA Forecasting section
st.subheader("ARIMA Forecast")
steps = st.slider("Months to forecast", 1, 12, 4)
expected = st.number_input("Enter your expected revenue for the final forecasted month (optional)", min_value=0.0, step=100.0)

model = ARIMA(df_revenue.set_index("date")["revenue"], order=(1, 1, 1))
results = model.fit()
forecast = results.get_forecast(steps=steps)
forecast_index = forecast.predicted_mean.index
ci = forecast.conf_int()

# Forecast plot using Plotly
fig2 = go.Figure()

# Observed
fig2.add_trace(go.Scatter(
    x=df_revenue["date"], y=df_revenue["revenue"],
    mode='lines',
    name='Observed',
    line=dict(color='blue'),
    hovertemplate='Date: %{x}<br>Observed: %{y}<extra></extra>'
))

# Forecast
fig2.add_trace(go.Scatter(
    x=forecast_index, y=forecast.predicted_mean,
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

# Plot expected revenue if provided
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
    title="Forecasted Revenue",
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
df_rev_adj["date"] = df_rev_adj["date"] + pd.Timedelta(days=1)
df_merged = pd.merge(df_cpi, df_rev_adj, on="date", how="inner").set_index("date")

# ARIMAX modeling
exog = df_merged[["CPI"]]
arimax_model = ARIMA(df_merged["revenue"], order=(1, 1, 1), exog=exog)
arimax_results = arimax_model.fit()
df_merged["ARIMAX_Fitted"] = arimax_results.fittedvalues

# Filter out first value (which is often 0)
df_plot = df_merged[df_merged["ARIMAX_Fitted"] != 0]

# Plot with Plotly
fig_arimax = go.Figure()
fig_arimax.add_trace(go.Scatter(
    x=df_merged.index,
    y=df_merged["revenue"],
    mode="lines",
    name="Actual Revenue",
    hovertemplate="Date: %{x}<br>Revenue: %{y:.2f}<extra></extra>"
))
fig_arimax.add_trace(go.Scatter(
    x=df_plot.index,
    y=df_plot["ARIMAX_Fitted"],
    mode="lines",
    name="ARIMAX Fitted",
    line=dict(dash='dot'),
    hovertemplate="Date: %{x}<br>Fitted: %{y:.2f}<extra></extra>"
))

fig_arimax.update_layout(
    title="Starbucks Revenue: Actual vs. ARIMAX (with CPI)",
    xaxis_title="Date",
    yaxis_title="Revenue",
    hovermode="x unified"
)
st.plotly_chart(fig_arimax, use_container_width=True)
