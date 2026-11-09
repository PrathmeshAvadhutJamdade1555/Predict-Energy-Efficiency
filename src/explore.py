import matplotlib.pyplot as plt
import pandas as pd

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


def load_and_inspect(data_path="data/ENB2012_data.xlsx"):
    df = pd.read_excel(data_path)
    df = df.rename(columns=COLUMN_NAMES)

    print("Dataset dimensions:", df.shape)
    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst five rows:")
    print(df.head())

    print("\nData types and missing-value overview:")
    df.info()

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nSummary of numeric columns:")
    print(df.describe())

    return df


def plot_visualizations(df):
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.hist(df["Heating_Load"].dropna(), bins=20, edgecolor="black")
    plt.title("Distribution of Heating Load")
    plt.xlabel("Heating Load")
    plt.ylabel("Number of buildings")
    plt.grid(alpha=0.25)

    plt.subplot(1, 2, 2)
    plt.hist(df["Cooling_Load"].dropna(), bins=20, edgecolor="black")
    plt.title("Distribution of Cooling Load")
    plt.xlabel("Cooling Load")
    plt.ylabel("Number of buildings")
    plt.grid(alpha=0.25)

    plt.tight_layout()
    plt.show()


df = load_and_inspect("data/ENB2012_data.xlsx")
plot_visualizations(df) 
