import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
try:
    from preprocess import prepare_data, scale_features
except ModuleNotFoundError:
    from src.preprocess import prepare_data, scale_features
def train_and_evaluate(data_path="data/ENB2012_data_CSV.csv"):
    X, y = prepare_data(data_path)
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
  
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
 
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    print("Model training completed!")
  
    y_pred = model.predict(X_test_scaled)

    mae_y1 = mean_absolute_error(y_test["Heating_Load"], y_pred[:, 0])
    mse_y1 = mean_squared_error(y_test["Heating_Load"], y_pred[:, 0])
    rmse_y1 = np.sqrt(mse_y1)
    r2_y1 = r2_score(y_test["Heating_Load"], y_pred[:, 0])

    mae_y2 = mean_absolute_error(y_test["Cooling_Load"], y_pred[:, 1])
    mse_y2 = mean_squared_error(y_test["Cooling_Load"], y_pred[:, 1])
    rmse_y2 = np.sqrt(mse_y2)
    r2_y2 = r2_score(y_test["Cooling_Load"], y_pred[:, 1])

    print("\n--- Heating Load Metrics ---")
    print(f"MAE  : {mae_y1:,.3f}")
    print(f"MSE  : {mse_y1:,.3f}")
    print(f"RMSE : {rmse_y1:,.3f}")
    print(f"R²   : {r2_y1:.3f}")

    print("\n--- Cooling Load Metrics ---")
    print(f"MAE  : {mae_y2:,.3f}")
    print(f"MSE  : {mse_y2:,.3f}")
    print(f"RMSE : {rmse_y2:,.3f}")
    print(f"R²   : {r2_y2:.3f}")

    return model, scaler


if __name__ == "__main__":
  train_and_evaluate()
