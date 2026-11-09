import pandas as pd
from sklearn.preprocessing import StandardScaler

COLUMN_NAMES = {
    "X1": "Relative_Compactness",
    "X2": "Surface_Area",
    "X3": "Wall_Area",
    "X4": "Roof_Area",
    "X5": "Overall_Height",
    "X6": "Orientation",
    "X7": "Glazing_Area",
    "X8": "Glazing_Area_Distribution",
    "Y1": "Heating_Load",
    "Y2": "Cooling_Load",
}


def prepare_data(data_path="data/ENB2012_data_CSV.csv"):
    df = pd.read_csv(data_path)
    df = df.rename(columns=COLUMN_NAMES)

    # Drop missing/NaN rows and duplicate entries (Lab 7 style)
    df = df.dropna().drop_duplicates().copy()

    feature_columns = [
        "Relative_Compactness",
        "Surface_Area",
        "Wall_Area",
        "Roof_Area",
        "Overall_Height",
        "Orientation",
        "Glazing_Area",
        "Glazing_Area_Distribution",
    ]
    target_columns = ["Heating_Load", "Cooling_Load"]

    X = df[feature_columns]
    y = df[target_columns]

    return X, y


def scale_features(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Feature scaling completed.")
    return X_train_scaled, X_test_scaled, scaler
