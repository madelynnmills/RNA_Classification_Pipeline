import os
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from src.preprocessing import add_motif_features, add_kmer_features
import tracemalloc

app = Flask(__name__)

# Path to the trained model
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

# Check if model file exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model and label encoder
with open(model_path, "rb") as f:
    model, label_encoder = pickle.load(f)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict RNA type from input sequence via POST request.
    """
    try:
        tracemalloc.start()  # Start memory tracking

        data = request.json
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' in the input data"}), 400

        sequence = data["sequence"]

        # Prepare the sequence DataFrame
        sequence_df = pd.DataFrame([{"ID": "input", "Sequence": sequence}])
        sequence_df["Length"] = sequence_df["Sequence"].apply(len)
        sequence_df["GC_Content"] = sequence_df["Sequence"].apply(
            lambda seq: (seq.count("G") + seq.count("C")) / len(seq) * 100
        )

        # Add motif features
        motifs = ["ATG", "TTT", "GGG"]
        sequence_df = add_motif_features(sequence_df, motifs)

        # Add k-mer features with a max_kmers limit
        sequence_df = add_kmer_features(sequence_df, k=3, max_kmers=100)

        # Save intermediate DataFrame for debugging
        sequence_df.to_csv("intermediate_api_sequence.csv", index=False)

        # Prepare features for prediction
        features = sequence_df.drop(columns=["ID", "Sequence"])

        # Predict RNA type
        prediction = model.predict(features)[0]
        predicted_rna_type = label_encoder.inverse_transform([prediction])[0]

        # Track memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("[Memory Usage]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()

        return jsonify({"RNA_Type": predicted_rna_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict RNA types from a batch of input sequences via POST request.
    """
    try:
        tracemalloc.start()  # Start memory tracking

        data = request.json
        if "sequences" not in data or not isinstance(data["sequences"], list):
            return jsonify({"error": "Missing 'sequences' or invalid format"}), 400

        sequences = data["sequences"]
        if not sequences:
            return jsonify({"error": "Empty sequence list"}), 400

        # Prepare the sequence DataFrame
        sequence_df = pd.DataFrame([{"ID": f"input_{i}", "Sequence": seq} for i, seq in enumerate(sequences)])
        sequence_df["Length"] = sequence_df["Sequence"].apply(len)
        sequence_df["GC_Content"] = sequence_df["Sequence"].apply(
            lambda seq: (seq.count("G") + seq.count("C")) / len(seq) * 100
        )

        # Add motif features
        motifs = ["ATG", "TTT", "GGG"]
        sequence_df = add_motif_features(sequence_df, motifs)

        # Add k-mer features with a max_kmers limit
        sequence_df = add_kmer_features(sequence_df, k=3, max_kmers=100)

        # Save intermediate DataFrame for debugging
        sequence_df.to_csv("intermediate_api_batch_sequence.csv", index=False)

        # Prepare features for prediction
        features = sequence_df.drop(columns=["ID", "Sequence"])

        # Predict RNA types
        predictions = model.predict(features)
        predicted_rna_types = label_encoder.inverse_transform(predictions)

        # Track memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("[Memory Usage]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()

        return jsonify({"Predictions": [{"ID": row["ID"], "RNA_Type": rna_type}
                                        for row, rna_type in zip(sequence_df.to_dict(orient="records"), predicted_rna_types)]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
