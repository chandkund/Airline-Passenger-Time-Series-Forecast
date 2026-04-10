from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.train_model import train_model
from src.forecast import generate_forecast
from src.evaluate import evaluate_forecast


def main():
    raw_data = load_data()
    processed = preprocess_data(raw_data)
    model = train_model(processed)
    forecast = generate_forecast(model, processed)
    evaluate_forecast(processed, model, forecast)


if __name__ == "__main__":
    main()
