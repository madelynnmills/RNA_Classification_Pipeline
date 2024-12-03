import os       #used for file operations and loading the trained model and label encoder
import pickle       #used for file operations and loading the trained model and label encoder
import pandas as pd         #handles data manipulation and sequence generation
from flask import Flask, request, jsonify       #enables the RESTful API framework for serving predictions via HTTP endpoints
from src.preprocessing import add_motif_features, add_kmer_features         #functions from preprocessing module, essntial for feature engineering on input sequences
import tracemalloc      #tracks memory usage for debugging and optimization

app = Flask(__name__)

# Path to the trained model
model_path = os.path.join(os.path.dirname(__file__), "trained_model.pkl")

# Check if model file exists
if not os.path.exists(model_path):      #ensures the model file exists, raising an error if it doesn't
    raise FileNotFoundError(f"Model file not found at {model_path}")

# Load the trained model and label encoder
with open(model_path, "rb") as f:       #deserializes the model and label encoder for use in predictions
    model, label_encoder = pickle.load(f)


@app.route('/predict', methods=['POST'])        #handles single-sequence predictions by validating and processing the input, generating features for the input sequence, using the trained model to predict the RNA type
def predict():
    """
    Predict RNA type from input sequence via POST request.
    """
    try:
        tracemalloc.start()  # Start memory tracking

        data = request.json         #input validation: ensures teh sequence field exists in the POST request's JSON payload, responds to 400 bad requests
        if "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' in the input data"}), 400

        sequence = data["sequence"]

        # Prepare the sequence DataFrame
        sequence_df = pd.DataFrame([{"ID": "input", "Sequence": sequence}])     #converts the input sequence into a structured DataFrame for processing
        sequence_df["Length"] = sequence_df["Sequence"].apply(len)      #calculates basic features for the input sequence
        sequence_df["GC_Content"] = sequence_df["Sequence"].apply(
            lambda seq: (seq.count("G") + seq.count("C")) / len(seq) * 100
        )

        # Add motif features
        motifs = ["ATG", "TTT", "GGG"]
        sequence_df = add_motif_features(sequence_df, motifs)       #calls the add_motif_features to compute motif frequencies and positions

        # Add k-mer features with a max_kmers limit
        sequence_df = add_kmer_features(sequence_df, k=3, max_kmers=100)        #adds kmer frequencies as features, with a max_kmers limit

        # Save intermediate DataFrame for debugging
        sequence_df.to_csv("intermediate_api_sequence.csv", index=False)

        # Prepare features for prediction
        features = sequence_df.drop(columns=["ID", "Sequence"])     #drops non-feature columns (ID, Sequence) to prepare the input for the model

        # Predict RNA type
        prediction = model.predict(features)[0]     #uses teh trained model to predict the RNA type and decodes it using the label encoder
        predicted_rna_type = label_encoder.inverse_transform([prediction])[0]

        # Track memory usage
        snapshot = tracemalloc.take_snapshot()      #captures memory usage for debugging
        top_stats = snapshot.statistics("lineno")
        print("[Memory Usage]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()

        return jsonify({"RNA_Type": predicted_rna_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500         #sends the predicted RNA type as a JSON response


@app.route('/predict_batch', methods=['POST'])      #handles batch predictions by: validating multiple sequences, processing and predicting RNA types for each sequence in the batch
def predict_batch():
    """
    Predict RNA types from a batch of input sequences via POST request.
    """
    try:
        tracemalloc.start()  # Start memory tracking

        data = request.json         #ensures the sequences field exists and is a non-empty list
        if "sequences" not in data or not isinstance(data["sequences"], list):
            return jsonify({"error": "Missing 'sequences' or invalid format"}), 400

        sequences = data["sequences"]
        if not sequences:
            return jsonify({"error": "Empty sequence list"}), 400

        # Prepare the sequence DataFrame
        sequence_df = pd.DataFrame([{"ID": f"input_{i}", "Sequence": seq} for i, seq in enumerate(sequences)])      #processes each sequence in the batch to generate features as done in the /predict endpoint
        sequence_df["Length"] = sequence_df["Sequence"].apply(len)
        sequence_df["GC_Content"] = sequence_df["Sequence"].apply(
            lambda seq: (seq.count("G") + seq.count("C")) / len(seq) * 100
        )

        # Add motif features
        motifs = ["ATG", "TTT", "GGG"]
        sequence_df = add_motif_features(sequence_df, motifs)       #adds basic motif, and k-mer features for all input sequences

        # Add k-mer features with a max_kmers limit
        sequence_df = add_kmer_features(sequence_df, k=3, max_kmers=100)

        # Save intermediate DataFrame for debugging
        sequence_df.to_csv("intermediate_api_batch_sequence.csv", index=False)

        # Prepare features for prediction
        features = sequence_df.drop(columns=["ID", "Sequence"])         #uses the trained model to predict RNA types for all sequences in the batch

        # Predict RNA types
        predictions = model.predict(features)       #decodes teh prediction into human-readable RNA types
        predicted_rna_types = label_encoder.inverse_transform(predictions)

        # Track memory usage
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics("lineno")
        print("[Memory Usage]")
        for stat in top_stats[:10]:
            print(stat)
        tracemalloc.stop()

        return jsonify({"Predictions": [{"ID": row["ID"], "RNA_Type": rna_type}     #constructs JSON response containing the predicted RNA types for each input sequence
                                        for row, rna_type in zip(sequence_df.to_dict(orient="records"), predicted_rna_types)]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":      #runs teh flask application in debug mode for development purposes 
    app.run(debug=True)
