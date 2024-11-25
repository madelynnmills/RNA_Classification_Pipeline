import pandas as pd
from Bio import Entrez, SeqIO
from src.utils import calculate_motif_frequency, find_motif_positions


def fetch_sequences_to_dataframe(query, email, motifs=None, retmax=20):
    """
    Fetch RNA sequences from NCBI using Biopython's Entrez API and calculate motif features if provided.

    Parameters:
        query (str): Search query for RNA sequences.
        email (str): Email address for NCBI API identification.
        motifs (list, optional): List of motifs to calculate frequencies and positions.
        retmax (int): Maximum number of sequences to fetch.

    Returns:
        pd.DataFrame: DataFrame containing sequence data and optional motif features.
    """
    Entrez.email = email

    sequences = []

    try:
        # Fetch IDs matching the query
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=retmax)
        record = Entrez.read(handle)
        ids = record["IdList"]

        # Fetch and parse sequences
        for seq_id in ids:
            try:
                fetch_handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
                for record in SeqIO.parse(fetch_handle, "fasta"):
                    sequences.append({"ID": record.id, "Sequence": str(record.seq)})
                fetch_handle.close()
            except Exception as e:
                print(f"Error fetching sequence for ID {seq_id}: {e}")
                continue

        # Create DataFrame
        if not sequences:
            print(f"No sequences fetched for query: {query}")
            return pd.DataFrame(columns=["ID", "Sequence", "Length", "GC_Content"] +
                                [f"{motif}_Frequency" for motif in (motifs or [])] +
                                [f"{motif}_Positions" for motif in (motifs or [])])

        df = pd.DataFrame(sequences)
        df["Length"] = df["Sequence"].apply(len)
        df["GC_Content"] = df["Sequence"].apply(lambda seq: (seq.count("G") + seq.count("C")) / len(seq) * 100)

        # Add motif features if motifs are provided
        if motifs:
            df = add_motif_features(df, motifs)

        if df.empty:
            print(f"No data returned for query: {query}")
            return pd.DataFrame(columns=["ID", "Sequence", "Length", "GC_Content", "RNA_Type"])

        return df

    except Exception as e:
        print(f"Error during sequence fetching: {e}")
        return pd.DataFrame(columns=["ID", "Sequence", "Length", "GC_Content"] +
                            [f"{motif}_Frequency" for motif in (motifs or [])] +
                            [f"{motif}_Positions" for motif in (motifs or [])])


def add_motif_features(df, motifs):
    """
    Adds motif-related features (frequency and positions) to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with sequences.
        motifs (list): List of motifs to calculate features for.

    Returns:
        pd.DataFrame: Updated DataFrame with motif features.
    """
    for motif in motifs:
        print(f"Calculating features for motif: {motif}")
        df[f"{motif}_Frequency"] = df["Sequence"].apply(lambda seq: calculate_motif_frequency(seq, motif))
        df[f"{motif}_Positions"] = df["Sequence"].apply(lambda seq: find_motif_positions(seq, motif))
    return df

__all__ = ["fetch_sequences_to_dataframe", "add_motif_features"]
