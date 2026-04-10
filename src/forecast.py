import pandas as pd


def generate_forecast(model, df, periods=12):
    """Generate future forecast values."""
    forecast = model.forecast(periods)
    forecast.index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=periods, freq="MS")
    return forecast
