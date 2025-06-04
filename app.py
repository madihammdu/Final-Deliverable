import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# New title
st.title("Starbucks Financial Analysis")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"], index_col="date")
    return df

df = load_data()

# Plot historical revenue with labeled axes
st.subheader("Historical Revenue")

fig1, ax1 = plt.subplots(figsize=(12, 5))
df["revenue"].plot(ax=ax1)
ax1.set_title("Historical Revenue Over Time")
ax1.set_xlabel("Date")
ax1.set_ylabel("Revenue")
ax1.grid(True)
st.pyplot(fig1)

# Forecast horizon
steps = st.slider("Months to forecast", 1, 12, 4)

# Expected revenue input
expected = st.number_input(f"Enter your expected revenue for month {steps}", min_value=0.0, step=1.0)

# ARIMA model
model = ARIMA(df["revenue"], order=(1, 1, 1))
results = model.fit()
forecast = results.get_forecast(steps=steps)
ci = forecast.conf_int()

# Plot forecast with user input
st.subheader("ARIMA Forecast vs. Your Expectation")

fig2, ax2 = plt.subplots(figsize=(12, 5))
df["revenue"].plot(label="Observed", ax=ax2)
forecast.predicted_mean.plot(label="Forecast", style="--", ax=ax2)
ax2.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="lightgray")

# Add expected point if provided
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
