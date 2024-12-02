import matplotlib.pyplot as plt
import pandas as pd
import os


def save_to_csv(df, filename, output_dir="./output"):
    """
    Saves DataFrame to a CSV file in a designated output directory.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        filename (str): Name of the CSV file.
        output_dir (str): Directory to save the file in (default: "./output").

    Returns:
        str: Path to the saved file, or None if saving failed.
    """
    if df.empty:
        print(f"Warning: Attempted to save an empty DataFrame to {filename}.")
        return None

    try:
        # Ensure output directory exists
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Save file
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"Data successfully saved to: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error saving DataFrame to {filename}: {e}")
        return None


def load_test_dataset(filepath):
    """
    Loads a test dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The loaded dataset as a DataFrame, or an empty DataFrame if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}. Returning empty DataFrame.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(filepath)
        if df.empty:
            print(f"Warning: Loaded an empty DataFrame from {filepath}.")
        else:
            print(f"Test dataset successfully loaded from {filepath}. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset from {filepath}: {e}. Returning empty DataFrame.")
        return pd.DataFrame()


def plot_data_distribution(df):
    """
    Plots the distribution of RNA types and motif frequencies.

    Parameters:
        df (pd.DataFrame): Input DataFrame with "RNA_Type" and motif columns.

    Returns:
        None
    """
    if "RNA_Type" not in df.columns:
        print("Error: 'RNA_Type' column not found in the DataFrame.")
        return

    try:
        # Plot RNA_Type counts
        plt.figure(figsize=(6, 4))
        df["RNA_Type"].value_counts().plot(kind="bar", title="RNA Type Distribution")
        plt.ylabel("Count")
        plt.xlabel("RNA Type")
        plt.show()

        # Plot motif frequencies
        motifs = [col for col in df.columns if "_Frequency" in col]
        if motifs:
            df[motifs].hist(bins=10, figsize=(10, 8), layout=(2, 2))
            plt.suptitle("Motif Frequency Distribution")
            plt.tight_layout()
            plt.show()
        else:
            print("No motif frequency columns found for plotting.")
    except Exception as e:
        print(f"Error during plotting data distribution: {e}")


def plot_feature_importance(model, feature_names):
    """
    Plots feature importance from the trained model.

    Parameters:
        model: Trained model with `feature_importances_`.
        feature_names (list or pd.Index): List of feature names.

    Returns:
        None
    """
    if not hasattr(model, "feature_importances_"):
        print("Error: Model does not have feature importances.")
        return

    try:
        importance = model.feature_importances_
        sorted_idx = importance.argsort()

        plt.figure(figsize=(10, 6))
        plt.barh(feature_names[sorted_idx], importance[sorted_idx])
        plt.title("Feature Importance")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.show()
    except Exception as e:
        print(f"Error during plotting feature importance: {e}")


def calculate_motif_frequency(sequence, motif):
    """
    Calculate the frequency of a specific motif in a sequence.

    Parameters:
        sequence (str): RNA sequence.
        motif (str): Motif to count.

    Returns:
        int: Frequency of the motif in the sequence.
    """
    return sequence.count(motif)


def find_motif_positions(sequence, motif):
    """
    Find the starting positions of a motif within a sequence.

    Parameters:
        sequence (str): RNA sequence.
        motif (str): Motif to locate.

    Returns:
        list: List of starting positions of the motif.
    """
    positions = []
    start = sequence.find(motif)
    while start != -1:
        positions.append(start)
        start = sequence.find(motif, start + 1)
    return positions
