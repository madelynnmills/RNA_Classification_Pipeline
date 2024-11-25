import matplotlib.pyplot as plt
import pandas as pd

def save_to_csv(df, filename):
    """
    Saves DataFrame to CSV file.
    """
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")


def load_test_dataset(filepath):
    """
    Loads a test dataset from a CSV file.

    Parameters:
        filepath (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded test dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Test dataset loaded from {filepath}")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()

def plot_data_distribution(df):
    """
    Plots the distribution of RNA types and motif frequencies.
    """
    # Plot RNA_Type counts
    plt.figure(figsize=(6, 4))
    df["RNA_Type"].value_counts().plot(kind="bar", title="RNA Type Distribution")
    plt.ylabel("Count")
    plt.xlabel("RNA Type")
    plt.show()

    # Plot motif frequencies
    motifs = [col for col in df.columns if "_Frequency" in col]
    df[motifs].hist(bins=10, figsize=(10, 8), layout=(2, 2))
    plt.suptitle("Motif Frequency Distribution")
    plt.tight_layout()
    plt.show()


def plot_feature_importance(model, feature_names):
    """
    Plots feature importance from the trained model.
    """
    importance = model.feature_importances_
    sorted_idx = importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_idx], importance[sorted_idx])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()


# New functions for motif calculations
def calculate_motif_frequency(sequence, motif):
    """
    Calculate the frequency of a specific motif in a sequence.
    """
    return sequence.count(motif)


def find_motif_positions(sequence, motif):
    """
    Find the starting positions of a motif within a sequence.
    """
    positions = []
    start = sequence.find(motif)
    while start != -1:
        positions.append(start)
        start = sequence.find(motif, start + 1)
    return positions
