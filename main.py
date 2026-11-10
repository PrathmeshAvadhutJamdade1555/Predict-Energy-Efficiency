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


# Feature selection: "Orientation" and "Glazing Area Distribution" removed as they do not correlate well with Heating and Cooling Loads
print("Feature selection:")
feature_columns = [
    "Relative Compactness",
    "Surface Area",
    "Wall Area",
    "Roof Area",
    "Overall Height",
    "Glazing Area",
]
target_columns = ["Heating Load", "Cooling Load"]
print("Features:", feature_columns)
print("Targets:", target_columns)
print("")


# Preprocessing
df = df.dropna(subset = target_columns)
df = df.drop_duplicates().copy()
X = df[feature_columns]
y = df[target_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

scaler = StandardScaler()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("numeric", scaler)
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop='first', handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
    ("numeric", numeric_transformer, numeric_features),
    ("categorical", categorical_transformer, categorical_features)
    ]
)

regression_model = LinearRegression()

model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("regression", regression_model),
    ]
)

model.fit(X_train, y_train)
print("Model training complete.")
test_predictions = model.predict(X_test)

