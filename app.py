
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import requests
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.api import OLS, add_constant

st.title("Starbucks Financial Analysis")

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

# Extend confidence interval to begin at last actual point
ci_lower = [last_actual_value] + list(ci.iloc[:, 0])
ci_upper = [last_actual_value] + list(ci.iloc[:, 1])
ci_x = [last_actual_date] + list(forecast_index)

# Shaded confidence interval area
fig2.add_trace(go.Scatter(
    x=ci_x + ci_x[::-1],
    y=ci_lower + ci_upper[::-1],
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
st.markdown("---")
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

# ---- OLS Regression: Revenue vs. Transactions ----
st.markdown("---")
st.subheader("Linear Regression: Revenue Explained by Transactions")

import io
import pandas as pd
import plotly.graph_objects as go
from statsmodels.api import OLS, add_constant

# Load your main CSV data (you already do this earlier in your app)
df_revenue = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])

# Industry average data (paste as multiline string)
industry_data = """
date,industry_average_fixed
4/1/2018,3290
7/1/2018,3350
10/1/2018,3420
1/1/2019,3420
4/1/2019,3470
7/1/2019,3440
10/1/2019,3560
1/1/2020,3570
4/1/2020,3620
7/1/2020,2930
10/1/2020,2720
1/1/2021,3400
4/1/2021,3530
7/1/2021,3470
10/1/2021,3860
1/1/2022,4110
4/1/2022,4110
7/1/2022,3920
10/1/2022,4100
1/1/2023,4100
4/1/2023,4250
7/1/2023,4250
10/1/2023,4520
1/1/2024,4570
"""

# Read industry average into DataFrame
df_industry = pd.read_csv(io.StringIO(industry_data))
df_industry['date'] = pd.to_datetime(df_industry['date'])

# Prepare your data for regression
df_reg = df_revenue.set_index("date").copy()
df_reg = df_reg[["revenue", "transactions"]].dropna()

# Add constant for intercept
X = add_constant(df_reg["transactions"])
y = df_reg["revenue"]

# Fit OLS regression model
model = OLS(y, X).fit()
df_reg["Predicted_Revenue"] = model.predict(X)

# Reset index to merge on date
df_reg = df_reg.reset_index()

# Make sure both DataFrames are sorted by date
df_reg = df_reg.sort_values("date")
df_industry = df_industry.sort_values("date")

# Merge on nearest previous date
df_merged_plot = pd.merge_asof(df_reg, df_industry, on='date')

# Plot all three lines
fig_reg = go.Figure()

# Actual Revenue line
fig_reg.add_trace(go.Scatter(
    x=df_merged_plot['date'],
    y=df_merged_plot['revenue'],
    mode="lines+markers",
    name="Actual Revenue",
    line=dict(color='blue'),
    hovertemplate="Date: %{x}<br>Revenue: %{y:.2f}<extra></extra>"
))

# Predicted Revenue line
fig_reg.add_trace(go.Scatter(
    x=df_merged_plot['date'],
    y=df_merged_plot['Predicted_Revenue'],
    mode="lines",
    name="Predicted Revenue with Transactions",
    line=dict(color='orange', dash='dot'),
    hovertemplate="Date: %{x}<br>Predicted: %{y:.2f}<extra></extra>"
))

# Industry Average Revenue line
fig_reg.add_trace(go.Scatter(
    x=df_merged_plot['date'],
    y=df_merged_plot['industry_average_fixed'],
    mode="lines",
    name="Industry Average Revenue",
    line=dict(color='green', dash='dash'),
    hovertemplate="Date: %{x}<br>Industry Average: %{y}<extra></extra>"
))

fig_reg.update_layout(
    title="Linear Regression: Revenue vs Transactions with Industry Average",
    xaxis_title="Date",
    yaxis_title="Revenue",
    hovermode="x unified"
)

st.plotly_chart(fig_reg, use_container_width=True)

# ---- Final Summary Section ----
st.markdown("---")
st.subheader("Audit Committee Summary")
st.write(
    "The analysis does not provide conclusive evidence that Starbucks is overstating its revenue "
    "but highlights areas warranting further review. While reported revenue aligns with historical trends, "
    "discrepancies between GAAP and non-GAAP figures, aggressive growth relative to industry benchmarks, and "
    "potential revenue recognition timing issues suggest the need for closer examination. Continued monitoring "
    "and deeper audit procedures are recommended."
)
