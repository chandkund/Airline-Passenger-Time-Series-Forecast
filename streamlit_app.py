import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.forecast import generate_forecast
from src.evaluate import evaluate_forecast

st.set_page_config(page_title="Airline Passenger Forecast", layout="wide")

st.title("Airline Passenger Time Series Forecast")
st.markdown(
    "Use this app to load the provided AirPassengers dataset, preprocess it, train a forecasting model, and visualize future forecasts."
)

raw_data = load_data()
processed_data = preprocess_data(raw_data)

with st.expander("Raw Data"):
    st.dataframe(raw_data.head(20))

with st.expander("Processed Data"):
    st.dataframe(processed_data.head(20))

forecast_horizon = st.slider("Forecast horizon (months)", min_value=6, max_value=36, value=12, step=1)

model = train_model(processed_data)
forecast = generate_forecast(model, processed_data, periods=forecast_horizon)
metrics = evaluate_forecast(processed_data, model, forecast)

st.subheader("Forecast Results")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(processed_data.index, processed_data["Passengers"], label="Actual", marker="o")
ax.plot(forecast.index, forecast, label="Forecast", marker="o")
ax.set_xlabel("Month")
ax.set_ylabel("Passengers")
ax.set_title("Airline Passenger Forecast")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Forecast Table")
st.dataframe(forecast.to_frame(name="Forecast"))

st.subheader("Evaluation Metrics")
for name, value in metrics.items():
    st.metric(label=name, value=f"{value:.2f}")
