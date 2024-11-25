import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("../models/rna_classifier.pkl")

st.title("RNA Type Classifier")
sequence = st.text_area("Enter RNA Sequence")

if st.button("Predict"):
    features = {
        "Length": len(sequence),
        "GC_Content": (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
    }
    df = pd.DataFrame([features])
    prediction = model.predict(df)
    st.write(f"Predicted RNA Type: {prediction[0]}")
