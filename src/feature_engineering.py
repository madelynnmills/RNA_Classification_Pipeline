import pandas as pd

def calculate_motif_frequency(sequence, motif):
    """
    Calculates the frequency of a motif in a sequence.
    """
    return sequence.count(motif) / len(sequence)

def add_motif_features(df, motifs):
    """
    Adds motif frequency features to the DataFrame.
    """
    for motif in motifs:
        df[f"{motif}_Frequency"] = df["Sequence"].apply(lambda x: calculate_motif_frequency(x, motif))
    return df
