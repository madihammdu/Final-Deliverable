import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Title of the app
st.title("Starbucks Monthly Revenue Forecast with ARIMA")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("starbucks_financials_expanded.csv", parse_dates=["date"], index_col="date")
    return df

df = load_data()

# Show raw data
st.subheader("Historical Revenue Data")
st.line_chart(df["revenue"])

# ARIMA model fitting
st.subheader("ARIMA Forecast")

# User input: forecast horizon
steps = st.slider("Months to forecast:", min_value=1, max_value=12, value=4)

# Fit the ARIMA model
model = ARIMA(df["revenue"], order=(1, 1, 1))
results = model.fit()

# Generate forecast
forecast = results.get_forecast(steps=steps)
ci = forecast.conf_int()

# Plotting forecast vs observed
fig, ax = plt.subplots(figsize=(12, 5))
df["revenue"].plot(label="Observed", ax=ax)
forecast.predicted_mean.plot(label="Forecast", style="--", ax=ax)
ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="lightgray")
ax.set_title("ARIMA Forecast: Sales")
ax.grid(True)
ax.legend()

# Show plot in Streamlit
st.pyplot(fig)
