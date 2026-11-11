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
print("Dataset has been loaded sucessfully from data/ENB2012_data_CSV.csv")
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
print("Column names:")
print(df.columns.tolist())
print(f"Rows with empty values found: {df.isnull().sum().max()}. Rows with duplicate values found: {df.duplicated().sum().max()}.")
print("")
print(f"The detailed Data Exploration has been saved as \033[94m{exploration_folder}/data_exploration_output.txt\033[0m")

# Save correlation matrix
corr_matrix = df.corr(numeric_only=True).round(3)

corr_matrix.to_csv(
    os.path.join(exploration_folder, "correlation_matrix.csv")
)
print(f"The detailed Correlation Matrix has been saved as \033[94m{exploration_folder}/correlation_matrix.csv\033[0m")

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

print(f"The features vs targets plots have been saved as \033[94m{exploration_folder}/features_vs_targets.png\033[0m")
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


# Model evaluation
prediction_folder = "outputs/prediction_results"
os.makedirs(prediction_folder, exist_ok=True)

heating_mae = mean_absolute_error(
    y_test["Heating Load"],
    test_predictions[:, 0],
)
heating_mse = mean_squared_error(
        y_test["Heating Load"],
        test_predictions[:, 0],
)
heating_rmse = np.sqrt(heating_mse)
heating_r2 = r2_score(
    y_test["Heating Load"],
    test_predictions[:, 0],
)

cooling_mae = mean_absolute_error(
    y_test["Cooling Load"],
    test_predictions[:, 1],
)
cooling_mse =  mean_squared_error(
    y_test["Cooling Load"],
        test_predictions[:, 1],
)
cooling_rmse = np.sqrt(cooling_mse)
cooling_r2 = r2_score(
    y_test["Cooling Load"],
    test_predictions[:, 1],
)

prediction_results = pd.DataFrame(
    test_predictions,
    columns=["Predicted Heating Load", "Predicted Cooling Load"],
    index=y_test.index,
)

prediction_results["Actual Heating Load"] = y_test["Heating Load"]
prediction_results["Actual Cooling Load"] = y_test["Cooling Load"]

prediction_results["Heating Residual"] = (
    prediction_results["Actual Heating Load"]
    - prediction_results["Predicted Heating Load"]
)

prediction_results["Cooling Residual"] = (
    prediction_results["Actual Cooling Load"]
    - prediction_results["Predicted Cooling Load"]
)

prediction_results.to_csv(
    os.path.join(prediction_folder, "test_predictions.csv"),
    index_label="Original Row",
)
print(f"The test predictions have been saved as \033[94m{prediction_folder}/test_predictions.csv\033[0m")


with open(
    os.path.join(prediction_folder, "evaluation_metrics.txt"),
    "w",
    encoding="utf-8",
) as file:
    file.write("Test-set evaluation results\n")
    file.write(f"Test rows: {len(y_test)}\n\n")

    file.write("Heating Load\n")
    file.write(f"MAE:  {heating_mae:.3f}\n")
    file.write(f"MSE:  {heating_mse:.3f}\n")
    file.write(f"RMSE: {heating_rmse:.3f}\n")
    file.write(f"R2:   {heating_r2:.3f}\n\n")

    file.write("Cooling Load\n")
    file.write(f"MAE:  {cooling_mae:.3f}\n")
    file.write(f"MSE:  {cooling_mse:.3f}\n")
    file.write(f"RMSE: {cooling_rmse:.3f}\n")
    file.write(f"R2:   {cooling_r2:.3f}\n")


print("")
print("Testing Results for Heating Load:")
print(f"MAE  : {heating_mae:,.3f}")
print(f"MSE  : {heating_mse:,.3f}")
print(f"RMSE : {heating_rmse:,.3f}")
print(f"R²   : {heating_r2:.3f}")
print("")
print("Testing Results for Cooling Load:")
print(f"MAE  : {cooling_mae:,.3f}")
print(f"MSE  : {cooling_mse:,.3f}")
print(f"RMSE : {cooling_rmse:,.3f}")
print(f"R²   : {cooling_r2:.3f}")

print("")
print(f"The evaluation of the model using test data has been saved as \033[94m{prediction_folder}/evaluation_metrics.txt\033[0m")
plt.figure(figsize=(9, 7))

all_values = np.concatenate([
    y_test["Heating Load"].values,
    y_test["Cooling Load"].values,
    test_predictions[:, 0],
    test_predictions[:, 1]
])

min_val = all_values.min()
max_val = all_values.max()

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    color="black",
    linestyle="--",
    linewidth=2,
    label="Ideal Prediction (y = x)"
)

plt.scatter(
    y_test["Heating Load"],
    test_predictions[:, 0],
    alpha=0.7,
    color="darkorange",
    label="Heating Load",
)

plt.scatter(
    y_test["Cooling Load"],
    test_predictions[:, 1],
    alpha=0.7,
    color="seagreen",
    label="Cooling Load",
)

plt.title("Actual vs Predicted Values")
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(prediction_folder, "actual_vs_predicted.png"),
    dpi=150,
)
plt.close()

print(f"The actual value vs predicted value plot has been saved as \033[94m{prediction_folder}/actual_vs_predicted.png\033[0m")
