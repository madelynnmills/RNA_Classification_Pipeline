import unittest
from src.preprocessing import fetch_sequences_to_dataframe

class TestPreprocessing(unittest.TestCase):
    def test_fetch_sequences(self):
        df = fetch_sequences_to_dataframe("mRNA[Filter] AND Homo sapiens[Organism]", email="maddymills2012@gmail.com", retmax=5)
        self.assertFalse(df.empty)
        self.assertIn("ID", df.columns)
        self.assertIn("Sequence", df.columns)

if __name__ == "__main__":
    unittest.main()
