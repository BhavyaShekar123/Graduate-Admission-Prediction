import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_data():
    csv_path = Path(__file__).resolve().parent.parent / "data" / "admission.csv"

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Place admission.csv inside the data folder."
        )

    return pd.read_csv(csv_path)


def analyze_data(df):
    print("\nFIRST FIVE ROWS")
    print(df.head())

    print("\nDATASET INFORMATION")
    df.info()

    print("\nSTATISTICAL SUMMARY")
    print(df.describe())

    print("\nMISSING VALUES")
    print(df.isnull().sum())

    df.hist(figsize=(10, 8), bins=10, edgecolor="black")
    plt.suptitle("Graduate Admission Dataset - Histograms")
    plt.tight_layout()
    plt.show()
