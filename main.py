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