import unittest
from src.utils import find_motif_positions, save_to_csv
import pandas as pd
import os

class TestUtils(unittest.TestCase):
    def test_find_motif_positions(self):
        sequence = "ATGGTGGATG"
        motif = "ATG"
        positions = find_motif_positions(sequence, motif)
        self.assertEqual(positions, [0, 7])

    def test_save_to_csv(self):
        data = {"ID": [1, 2], "Sequence": ["ATGGTGG", "CCAATG"]}
        df = pd.DataFrame(data)
        test_path = "test.csv"
        save_to_csv(df, test_path)
        self.assertTrue(os.path.exists(test_path))
        os.remove(test_path)

if __name__ == "__main__":
    unittest.main()
