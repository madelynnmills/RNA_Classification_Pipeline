import unittest
from src.feature_engineering import calculate_motif_frequency, add_motif_features
import pandas as pd

class TestFeatureEngineering(unittest.TestCase):
    def test_calculate_motif_frequency(self):
        frequency = calculate_motif_frequency("ATGGTGG", "ATG")
        self.assertAlmostEqual(frequency, 1 / 7)

    def test_add_motif_features(self):
        df = pd.DataFrame({"Sequence": ["ATGGTGG", "CCAATG"]})
        motifs = ["ATG"]
        df = add_motif_features(df, motifs)
        self.assertIn("ATG_Frequency", df.columns)
        self.assertAlmostEqual(df.loc[0, "ATG_Frequency"], 1 / 7)

if __name__ == "__main__":
    unittest.main()
