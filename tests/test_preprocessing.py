import unittest
from src.preprocessing import fetch_sequences_to_dataframe, add_motif_features, add_kmer_features
import pandas as pd

class TestPreprocessing(unittest.TestCase):
    def test_fetch_sequences(self):
        """
        Test the basic functionality of fetch_sequences_to_dataframe.
        """
        df = fetch_sequences_to_dataframe("mRNA[Filter] AND Homo sapiens[Organism]", email="maddymills2012@gmail.com", retmax=5)
        self.assertFalse(df.empty, "The DataFrame should not be empty.")
        self.assertIn("ID", df.columns, "The 'ID' column is missing.")
        self.assertIn("Sequence", df.columns, "The 'Sequence' column is missing.")

    def test_add_motif_features(self):
        """
        Test the add_motif_features function.
        """
        # Create a sample DataFrame
        data = {
            "ID": ["seq1", "seq2"],
            "Sequence": ["ATGTTTGGGATG", "GGGATGTTT"]
        }
        df = pd.DataFrame(data)

        # Define motifs to test
        motifs = ["ATG", "TTT", "GGG"]

        # Apply the function
        updated_df = add_motif_features(df, motifs)

        # Assert motif-related columns exist
        for motif in motifs:
            self.assertIn(f"{motif}_Frequency", updated_df.columns, f"{motif}_Frequency column is missing.")
            self.assertIn(f"{motif}_Positions", updated_df.columns, f"{motif}_Positions column is missing.")

        # Assert calculated values for frequencies
        self.assertEqual(updated_df["ATG_Frequency"].iloc[0], 2, "ATG frequency for seq1 is incorrect.")
        self.assertEqual(updated_df["TTT_Frequency"].iloc[1], 1, "TTT frequency for seq2 is incorrect.")
        self.assertEqual(updated_df["GGG_Frequency"].iloc[0], 1, "GGG frequency for seq1 is incorrect.")

    def test_add_kmer_features(self):
        """
        Test the add_kmer_features function.
        """
        # Create a sample DataFrame
        df = pd.DataFrame({"Sequence": ["ATGCGT", "GCGTAT"]})

        # Apply the function with k=3
        df = add_kmer_features(df, k=3)

        # Assert k-mer columns exist
        self.assertIn("ATG", df.columns, "ATG k-mer column is missing.")
        self.assertIn("CGT", df.columns, "CGT k-mer column is missing.")

        # Assert k-mer counts are calculated correctly
        self.assertEqual(df.at[0, "ATG"], 1, "ATG count for seq1 is incorrect.")
        self.assertEqual(df.at[1, "ATG"], 0, "ATG count for seq2 is incorrect.")

if __name__ == "__main__":
    unittest.main()
