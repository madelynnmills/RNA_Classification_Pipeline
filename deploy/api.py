import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from src.preprocessing import add_motif_features

app = Flask(__name__)

# Path to the trained model
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, "rb") as f:
    model = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for RNA type prediction.
    """
    try:
        data = request.json
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' field"}), 400

        sequence = data["sequence"]
        motifs = ["ATG", "TTT", "GGG"]

        # Add motif features
        sequence_df = pd.DataFrame([{"ID": "input", "Sequence": sequence}])
        sequence_df = add_motif_features(sequence_df, motifs)
        features = sequence_df.drop(columns=["ID", "Sequence"])

        prediction = model.predict(features)[0]
        return jsonify({"RNA_Type": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
