import unittest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.model import evaluate_model

class TestModel(unittest.TestCase):
    def test_evaluate_model(self):
        """
        Test the evaluate_model function.
        """
        # Mock data
        model = RandomForestClassifier(random_state=42)
        X = np.random.rand(100, 5)  # Random features
        y = np.random.randint(0, 3, size=100)  # Random labels (3 classes)
        
        # Train the model
        model.fit(X, y)

        # Create a label encoder for decoding labels
        label_encoder = LabelEncoder().fit(["mRNA", "tRNA", "rRNA", "miRNA", "siRNA"])

        # Call the evaluate_model function
        try:
            evaluate_model(model, X, y, label_encoder)
        except Exception as e:
            self.fail(f"evaluate_model raised an exception: {e}")

if __name__ == "__main__":
    unittest.main()
