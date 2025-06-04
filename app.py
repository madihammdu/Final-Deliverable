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
