from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def evaluate_forecast(df, model, forecast):
    """Evaluate the trained model and return key metrics."""
    fitted = model.fittedvalues
    actual = df["Passengers"]

    mae = mean_absolute_error(actual, fitted)
    rmse = np.sqrt(mean_squared_error(actual, fitted))
    mape = np.mean(np.abs((actual - fitted) / actual)) * 100

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    with open("outputs/results/metrics.txt", "w", encoding="utf-8") as fout:
        fout.write("MAE: {:.4f}\n".format(mae))
        fout.write("RMSE: {:.4f}\n".format(rmse))
        fout.write("MAPE: {:.2f}%\n".format(mape))

    return metrics
