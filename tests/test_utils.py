import unittest
from src.utils import find_motif_positions, save_to_csv
import pandas as pd
import os

class TestUtils(unittest.TestCase):
    def test_save_to_csv(self):
        """
        Test the save_to_csv function by saving a test DataFrame and checking its existence.
        """
        # Create a test DataFrame
        data = {"ID": ["seq1", "seq2"], "RNA_Type": ["mRNA", "tRNA"]}
        df = pd.DataFrame(data)

        # Define test output path
        test_filename = "test.csv"
        test_path = save_to_csv(df, test_filename)  # Use the updated function

        # Assert file existence
        self.assertTrue(os.path.exists(test_path), f"File {test_path} does not exist.")
        print(f"Test file successfully created at {test_path}")

        # Clean up after the test
        if os.path.exists(test_path):
            os.remove(test_path)

if __name__ == "__main__":
    unittest.main()