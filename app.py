import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from statsmodels.tsa.arima.model import ARIMA

st.title("Starbucks Financial Analysis")

# Load Starbucks revenue data
@st.cache_data
def load_revenue_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
    return df

df_revenue = load_revenue_data()

# Historical Revenue plot with labeled axes
st.subheader("Historical Revenue")
fig1, ax1 = plt.subplots(figsize=(12, 5))
df_revenue.set_index("date")["revenue"].plot(ax=ax1)
ax1.set_title("Historical Revenue Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue")
ax1.grid(True)
st.pyplot(fig1)

# ARIMA Forecasting section
steps = st.slider("Months to forecast", 1, 12, 4)
expected = st.number_input(f"Enter your expected revenue for month {steps}", min_value=0.0, step=1.0)

model = ARIMA(df_revenue.set_index("date")["revenue"], order=(1, 1, 1))
results = model.fit()
forecast = results.get_forecast(steps=steps)
ci = forecast.conf_int()

st.subheader("ARIMA Forecast vs. Your Expectation")
fig2, ax2 = plt.subplots(figsize=(12, 5))
df_revenue.set_index("date")["revenue"].plot(label="Observed", ax=ax2)
forecast.predicted_mean.plot(label="Forecast", style="--", ax=ax2)
ax2.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="lightgray")

if expected > 0:
    expected_date = forecast.predicted_mean.index[-1]
    ax2.scatter(expected_date, expected, color='red', label="Your Expected Revenue")
    ax2.annotate(f"${expected:,.0f}", (expected_date, expected), textcoords="offset points", xytext=(0,10), ha='center')

ax2.set_xlabel("Date")
ax2.set_ylabel("Revenue")
ax2.set_title("Forecasted Revenue")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# --- New CPI Data & ARIMAX Section ---
st.subheader("ARIMAX Model: Revenue with CPI as External Regressor")

# Fetch CPI data from FRED API
api_key = "755f96b3bf3c15588f3de4dbd65ced72"
cpi_series_id = "CPALTT01USQ657N"
url = "https://api.stlouisfed.org/fred/series/observations"
params = {
    "series_id": cpi_series_id,
    "api_key": api_key,
    "file_type": "json",
    "observation_start": "2018-04-01"
}

@st.cache_data
def fetch_cpi():
    response = requests.get(url, params=params)
    data = response.json()["observations"]
    df_cpi = pd.DataFrame(data)[["date", "value"]]
    df_cpi.columns = ["date", "CPI"]
    df_cpi["date"] = pd.to_datetime(df_cpi["date"])
    df_cpi["CPI"] = df_cpi["CPI"].astype(float)
    return df_cpi

df_cpi = fetch_cpi()

# Merge CPI with revenue data (adjust dates for alignment if needed)
df_revenue_adj = df_revenue.copy()
df_revenue_adj["date"] = df_revenue_adj["date"] + pd.Timedelta(days=1)

df_merged = pd.merge(df_cpi, df_revenue_adj, on="date", how="inner").set_index("date")

# Fit ARIMAX model with CPI as exogenous variable
exog = df_merged[["CPI"]]
arimax_model = ARIMA(df_merged["revenue"], order=(1, 1, 1), exog=exog)
arimax_results = arimax_model.fit()

# Add fitted values to the dataframe
df_merged["ARIMAX_Fitted"] = arimax_results.fittedvalues

# Filter out initial zero fitted values for plotting
df_filtered = df_merged[df_merged["ARIMAX_Fitted"] != 0]

fig3, ax3 = plt.subplots(figsize=(12, 6))
ax3.plot(df_merged.index, df_merged["revenue"], label="Actual Revenue", linewidth=2)
ax3.plot(df_filtered.index, df_filtered["ARIMAX_Fitted"], label="ARIMAX Fitted", linestyle=":")
ax3.set_title("Starbucks Revenue: Actual vs. ARIMAX")
ax3.set_xlabel("Date")
ax3.set_ylabel("Revenue")
ax3.legend()
ax3.grid(True)
st.pyplot(fig3)
