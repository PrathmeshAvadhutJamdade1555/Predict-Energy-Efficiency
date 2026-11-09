# Libraries
import contextlib # Built-in
import os # Built-in
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/ENB2012_data_CSV.csv")
df = df.dropna(how="all")
os.makedirs("outputs/data_exploration_and_analysis", exist_ok=True)

# Dataset Exploration
with open("outputs/data_exploration_and_analysis/output.txt", "w", encoding="utf-8") as file:
    with contextlib.redirect_stdout(file):
        print("Dataset dimensions:", df.shape)
        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst five rows:")
        print(df.head())

        print("\nData types and missing-value overview:")
        df.info()

        print("\nSummary of numeric columns:")
        print(df.describe())

        print("\nMissing values in dataset:")
        print(df.isnull())

        print("\nSum of missing values in each column:")
        print(df.isnull().sum())

        # Displaying input features
        x = df[["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]]

        # Displaying target variables
        y = df[["Y1", "Y2"]]

        # Displaying training and testing data dimensions
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

        print("\nx_train shape:", x_train.shape)
        print("x_test shape:", x_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_test shape:", y_test.shape)

        # Training the Linear Regression model
        model = LinearRegression()
        model.fit(x_train, y_train)

        print("\nModel training completed successfully")

        # Making predictions on test data

        y_pred = model.predict(x_test)

        print("\ny_pred shape:", y_pred.shape)

        # Evaluating the model performance
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)

        # Displaying model evaluation results

        print("\nMean absolute error:", mae)
        print("\nMean squared error:", mse)
        print("\nRoot mean squared error:", rmse)
        print("\nR2-score:", r2)

        # Comparing actual and predicted values
        print("\nFirst five actual values:")
        print(y_test.head())

        print("\nFirst five predicted values:")
        print(y_pred[0:5])

        # Verifying model results and generated graphs

        # Heating Load Graph
        plt.figure()
        plt.plot(y_test["Y1"].values, label="Actual")
        plt.plot(y_pred[:, 0], label="Predicted")
        plt.xlabel("Sample Number")
        plt.ylabel("Heating Load")
        plt.title("Heating Load Prediction")
        plt.legend()
        plt.savefig("outputs/data_exploration_and_analysis/heating_load.png")
        plt.close()

        # Cooling Load Graph
        plt.figure()
        plt.plot(y_test["Y2"].values, label="Actual")
        plt.plot(y_pred[:, 1], label="Predicted")
        plt.xlabel("Sample Number")
        plt.ylabel("Cooling Load")
        plt.title("Cooling Load Prediction")
        plt.legend()
        plt.savefig("outputs/data_exploration_and_analysis/cooling_load.png")
        plt.close()

        print("\nBoth actual vs predicted graphs saved successfully.")

      
        # Analyzing correlation between dataset variables

        
        # Finding correlation between variables
        correlation = df.corr()

        print("\nCorrelation between variables:")
        print(correlation)

        # Predicting output for new input

        new_input = pd.DataFrame([[0.98, 514.5, 294, 110, 7, 3.5, 2, 0]],
                                 columns=x.columns)

        new_prediction = model.predict(new_input)

        print("\nPrediction for New Input:")
        print("\nHeating Load:", new_prediction[0][0])
        print("\nCooling Load:", new_prediction[0][1])

        # Project completed successfully
        
        print("\nEnergy Efficiency Prediction Project Completed Successfully!")
