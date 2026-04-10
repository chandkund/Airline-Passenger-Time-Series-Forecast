# Time Series Forecasting using ARIMA and Streamlit

**GitHub Repository:** [https://github.com/chandkund/Airline-Passenger-Time-Series-Forecast](https://github.com/chandkund/Airline-Passenger-Time-Series-Forecast)

**Live Demo:** [https://chandkund-airline-passenger-time-series-fo-streamlit-app-f4njja.streamlit.app/](https://chandkund-airline-passenger-time-series-fo-streamlit-app-f4njja.streamlit.app/)

## Project Overview
This project is a simple and practical **Time Series Forecasting** application built using **Python**, **ARIMA model**, and **Streamlit**.
It is designed to forecast future values based on historical time series data.

The project uses the **Airline Passengers dataset**, which contains monthly passenger counts over time.
The main goal is to analyze the data, understand trend and seasonality, train a forecasting model, and predict future passenger values.

This project is made as a **beginner-friendly machine learning project** with proper structure and a simple deployment using **Streamlit**.

---

## Features
- Load and visualize time series dataset
- Perform basic exploratory data analysis
- Understand trend and seasonality
- Train ARIMA model on historical data
- Forecast future values
- Evaluate forecasting performance
- Interactive web app using Streamlit
- Simple and clean project structure

---

## Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Statsmodels**
- **Scikit-learn**
- **Streamlit**

---

## Dataset
The dataset used in this project is the **Airline Passengers dataset**.

It contains:
- Monthly passenger data
- Date/Month column
- Number of passengers column

This dataset is commonly used for learning and practicing time series forecasting techniques.

---

## Project Structure

```bash
Time-Series-Forecasting/
│
├── data/
│   ├── raw/
│   │   └── airline_passengers.csv
│   └── processed/
│       └── cleaned_airline_passengers.csv
│
├── notebooks/
│   └── 01_eda_and_experiment.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── forecast.py
│   ├── evaluate.py
│   └── utils.py
│
├── outputs/
│   ├── plots/
│   └── results/
│
├── streamlit_app.py
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
