import pandas as pd


def preprocess_data(df):
    """Preprocess the raw data and return cleaned dataframe."""
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    if "#Passengers" in df.columns:
        df = df.rename(columns={"#Passengers": "Passengers"})
    df["Month"] = pd.to_datetime(df["Month"])
    df = df.set_index("Month")
    df = df.asfreq("MS")
    df = df.ffill()
    df.to_csv("data/processed/cleaned_airline_passengers.csv")
    return df
