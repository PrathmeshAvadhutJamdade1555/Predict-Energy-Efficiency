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

# Save correlation matrix
corr_matrix = df.corr(numeric_only=True).round(3)

corr_matrix.to_csv(
    os.path.join(exploration_folder, "correlation_matrix.csv")
)

fig, axes = plt.subplots(4, 2, figsize=(13, 18))

for index, column in enumerate(df.columns[0:8]):
    row = index // 2
    position = index % 2

    axes[row, position].scatter(
        df[column],
        df["Heating Load"],
        alpha=0.6,
        color="darkorange",
        label="Heating Load",
    )
    axes[row, position].scatter(
        df[column],
        df["Cooling Load"],
        alpha=0.6,
        color="seagreen",
        label="Cooling Load",
    )

    axes[row, position].set_title(f"{column} vs. Loads")
    axes[row, position].set_xlabel(column)
    axes[row, position].set_ylabel("Load")
    axes[row, position].legend()

fig.suptitle("Input Features Compared with Heating and Cooling Loads")
plt.tight_layout()
plt.savefig(
    os.path.join(exploration_folder, "features_vs_targets.png"),
    dpi=150,
)
plt.close()

print("Detailed data exploration along with graph saved in:", exploration_folder)
print("\n")
