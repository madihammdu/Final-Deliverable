import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import requests

# Title
st.title("Starbucks Financial Analysis")

# Load Starbucks revenue data
@st.cache_data
def load_revenue_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"])
    return df

df_revenue = load_revenue_data()

# Historical revenue plot
st.subheader("Historical Revenue")

fig1, ax1 = plt.subplots(figsize=(12, 5))
df_revenue.set_index("date")["revenue"].plot(ax=ax1)
ax1.set_title("Historical Revenue Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue")
ax1.grid(True)
st.pyplot(fig1)

# Forecast horizon input
steps = st.slider("Months to forecast", 1, 12, 4)

# Expected revenue input
expected = st.number_input(f"Enter your expected revenue for month {steps}", min_value=0.0, step=1.0)

# ARIMA model
model = ARIMA(df_revenue.set_index("date")["revenue"], order=(1, 1, 1))
results = model.fit()
forecast = results.get_forecast(steps=steps)
ci = forecast.conf_int()

# Forecast plot with expected value
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

# --- New CPI API Section ---

st.subheader("Consumer Price Index (CPI) Data from FRED API")

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

# Adjust Starbucks revenue date for merge
df_revenue["date"] = pd.to_datetime(df_revenue["date"]) + pd.Timedelta(days=1)

# Merge CPI and revenue on date
df_merged = pd.merge(df_cpi, df_revenue, on="date", how="inner")

# Show merged data table
st.write("Merged CPI and Starbucks Revenue Data")
st.dataframe(df_merged.head())

