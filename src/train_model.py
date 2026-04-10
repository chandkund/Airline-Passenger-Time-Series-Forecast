from statsmodels.tsa.holtwinters import ExponentialSmoothing


def train_model(df):
    """Train a forecasting model on the processed data."""
    model = ExponentialSmoothing(df["Passengers"], trend="add", seasonal="add", seasonal_periods=12)
    fitted = model.fit()
    return fitted
