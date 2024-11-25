from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the trained model and prints metrics.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, predictions, target_names=label_encoder.classes_))
