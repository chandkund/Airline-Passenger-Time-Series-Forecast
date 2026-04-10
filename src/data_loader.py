import pandas as pd


def load_data():
    """Load the raw airline passengers dataset."""
    return pd.read_csv("data/raw/AirPassengers.csv")
