import pandas as pd         #provides tools for creating and malnipulating DataFrames
from Bio import Entrez, SeqIO           #used to interact with the NCBI database for fetching sequence data, parses sequences fetched from NCBI into usable formats
from src.utils import calculate_motif_frequency, find_motif_positions           #utility function from src.utils to calculate the frequency of specific motifs in a sequence
                                                                                #utility function from src.utils to locate specific motif positions within a sequence

def fetch_sequences_to_dataframe(query, email, motifs=None, retmax=20):         #purpose of function: fetches RNA sequences from the NCBI database based on a query, process them into a DataFrame, and optionally calculate motif features
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
    Entrez.email = email        #sets the email for NCBI API compliance

    sequences = []      #an empty list to store sequence data fetched from NCBI

    try:
        # Fetch IDs matching the query
        handle = Entrez.esearch(db="nucleotide", term=query, retmax=retmax)       #queries the NCBI databaase for sequence IDs matching the RNA type
        record = Entrez.read(handle)
        ids = record["IdList"]        #a list of IDs for sequences matching the query

        # Fetch and parse sequences
        for seq_id in ids:
            try:
                fetch_handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")       #fetrieves full sequence data for each ID
                for record in SeqIO.parse(fetch_handle, "fasta"):       #parses the fetched data in FASTA format
                    sequences.append({"ID": record.id, "Sequence": str(record.seq)})        #adds a dictionary containing the sequene ID and sequence string to the list
                fetch_handle.close()
            except Exception as e:
                print(f"Error fetching sequence for ID {seq_id}: {e}")
                continue

        # Create DataFrame
        if not sequences:       #if no sequences were fetched, returns an empty DataFrame with predefined columns
            print(f"No sequences fetched for query: {query}")
            return pd.DataFrame(columns=["ID", "Sequence", "Length", "GC_Content"] +
                                [f"{motif}_Frequency" for motif in (motifs or [])] +
                                [f"{motif}_Positions" for motif in (motifs or [])])

        df = pd.DataFrame(sequences)        #converts the list of dictionaries into a Dataframe
        df["Length"] = df["Sequence"].apply(len)
        df["GC_Content"] = df["Sequence"].apply(lambda seq: (seq.count("G") + seq.count("C")) / len(seq) * 100)

        # Add motif features if motifs are provided
        if motifs:
            df = add_motif_features(df, motifs)        #if motifs is provided, calculates motif frequencies and positions and appends them as columns

        if df.empty:
            print(f"No data returned for query: {query}")
            return pd.DataFrame(columns=["ID", "Sequence", "Length", "GC_Content", "RNA_Type"])

        return df

    except Exception as e:      #prints the error message and returns an empty DataFrame with appropriate columns
        print(f"Error during sequence fetching: {e}")
        return pd.DataFrame(columns=["ID", "Sequence", "Length", "GC_Content"] +
                            [f"{motif}_Frequency" for motif in (motifs or [])] +
                            [f"{motif}_Positions" for motif in (motifs or [])])


def add_motif_features(df, motifs):     #function purpose: adds motif-related features to the DataFrame, including frequencies and positions
    """
    Adds motif-related features (frequency and positions) to the DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame with sequences.
        motifs (list): List of motifs to calculate features for.

    Returns:
        pd.DataFrame: Updated DataFrame with motif features.
    """
    for motif in motifs:        #loop through motifsL iterates over the list of motifs to add features for each one
        print(f"Calculating features for motif: {motif}")
        df[f"{motif}_Frequency"] = df["Sequence"].apply(lambda seq: calculate_motif_frequency(seq, motif))      #calculates the number of times a motif appears in a sequence
        df[f"{motif}_Positions"] = df["Sequence"].apply(lambda seq: find_motif_positions(seq, motif))       #identifies the starting positions of a motif in a sequence
    return df       #the DataFrame now contains additional columns for motif frequencies and positions

__all__ = ["fetch_sequences_to_dataframe", "add_motif_features"]        #module export: specifies the functions that will be imported when from src.preprocesing import* is used
