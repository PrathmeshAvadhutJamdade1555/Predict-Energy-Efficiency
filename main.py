# Libraries
import contextlib  # Built-in
import os  # Built-in

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/ENB2012_data_CSV.csv")

df.columns = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Orientation",
    "Glazing Area",
    "Glazing Area Distribution",
    "Heating Load",
    "Cooling Load",
]

exploration_folder = "outputs/data_exploration_and_analysis"
os.makedirs(exploration_folder, exist_ok=True)


# Dataset exploration
with open(
    os.path.join(exploration_folder, "data_exploration_output.txt"),
    "w",
    encoding="utf-8",
) as file:
    with contextlib.redirect_stdout(file):
        print("Dataset dimensions:", df.shape)

        print("\nColumn names:")
        print(df.columns.tolist())

        print("\nFirst five rows:")
        print(df.head())

        print("\nData types and missing values:")
        df.info()

        print("\nSummary statistics:")
        print(df.describe())

        print("\nMissing values per column:")
        print(df.isnull().sum())

        print("\nDuplicate rows:", df.duplicated().sum())

        print("\nCorrelation matrix:")
        print(df.corr().round(3))

print("Dataset dimensions:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("")
print(f"Rows with empty values found: {df.isnull().sum().max()}. Rows with duplicate values found: {df.duplicated().sum().max()}.")
print("")

# There are no duplicate values, but in case there are: (as a good practice)
df = df.dropna()
df = df.drop_duplicates().copy()