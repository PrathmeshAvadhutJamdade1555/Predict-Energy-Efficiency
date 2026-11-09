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
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/ENB2012_data_CSV.csv")
df.columns = [
    'Relative Compactness',
    'Surface Area',
    'Wall Area',
    'Roof Area',
    'Overall Height',
    'Orientation',
    'Glazing Area',
    'Glazing Area Distribution',
    'Heating Load',
    'Cooling Load'
]
os.makedirs("outputs/data_exploration_and_analysis", exist_ok=True)

# Dataset Exploration
with open("outputs/data_exploration_and_analysis/data_exploration_output.txt", "w", encoding="utf-8") as file:
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

#Dataset preprocessing

df = df.dropna().drop_duplicates().copy()
feature_columns = [
    'Relative Compactness',
    'Surface Area',
    'Wall Area',
    'Roof Area',
    'Overall Height',
    'Orientation',
    'Glazing Area',
    'Glazing Area Distribution',
]
target_columns = ['Heating Load', 'Cooling Load']

X = df[feature_columns]
y = df[target_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Preprocessing complete!")
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)

