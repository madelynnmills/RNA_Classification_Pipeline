import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def train_rna_classifier(X, y):
    """
    Trains a RandomForestClassifier to predict RNA types.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.

    Returns:
        model: Trained RandomForestClassifier.
        label_encoder: LabelEncoder used for encoding labels.
    """
    try:
        print("Feature Matrix Shape Before Validation:", X.shape)
        print("Label Vector Shape Before Validation:", y.shape)

        # Align X and y to have consistent indices
        X, y = X.align(y, join="inner", axis=0)

        # Drop rows with missing values in X or y
        X = X.dropna()
        y = y.loc[X.index]

        print("Validated Feature Matrix Shape:", X.shape)
        print("Validated Label Vector Shape:", y.shape)

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Handle class imbalance
        print("Checking for class imbalance...")
        class_counts = pd.Series(y_encoded).value_counts()
        print("Class distribution before balancing:", class_counts.to_dict())

        if class_counts.min() / class_counts.max() < 0.8:
            print("Balancing data...")
            dfs = []
            for cls in np.unique(y_encoded):
                cls_indices = y_encoded == cls
                X_cls = X.loc[cls_indices]
                y_cls = np.full(len(X_cls), cls)

                X_resampled, y_resampled = resample(
                    X_cls,
                    y_cls,
                    replace=True,
                    n_samples=class_counts.max(),
                    random_state=42
                )
                dfs.append((X_resampled, y_resampled))

            X = pd.concat([df[0] for df in dfs])
            y_encoded = np.concatenate([df[1] for df in dfs])
            print("Data balanced.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        print("Train Features Shape:", X_train.shape)
        print("Test Features Shape:", X_test.shape)
        print("Train Labels Shape:", y_train.shape)
        print("Test Labels Shape:", y_test.shape)

        # Define RandomForest model and hyperparameters
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }

        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)

        # Fit model
        print("Training model with GridSearchCV...")
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate model
        accuracy = best_model.score(X_test, y_test)
        print(f"Test accuracy: {accuracy:.2f}")

        # Classification report and confusion matrix
        predictions = best_model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=label_encoder.classes_))
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

        return best_model, label_encoder

    except Exception as e:
        print(f"Error in training classifier: {e}")
        return None, None



def evaluate_model(model, X_test, y_test, label_encoder):
    """
    Evaluates the performance of the trained model.

    Parameters:
        model: Trained RandomForestClassifier.
        X_test (pd.DataFrame): Test feature matrix.
        y_test (pd.Series or np.ndarray): True labels for the test set.
        label_encoder (LabelEncoder): Encoder for decoding labels.

    Returns:
        None: Displays the evaluation metrics and confusion matrix plot.
    """
    try:
        print("Evaluating model performance...")
        predictions = model.predict(X_test)

        # Ensure y_test and predictions are numpy arrays
        y_test = np.array(y_test)
        predictions = np.array(predictions)

        # Decode labels if using LabelEncoder
        # Decode labels safely during evaluation
        decoded_y_test = label_encoder.inverse_transform(np.array(y_test, dtype=int))
        decoded_predictions = label_encoder.inverse_transform(np.array(predictions, dtype=int))


        # Classification report
        print("\nClassification Report:")
        print(classification_report(decoded_y_test, decoded_predictions))

        # Confusion matrix
        cm = confusion_matrix(decoded_y_test, decoded_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.show()

    except Exception as e:
        print(f"Error during model evaluation: {e}")


__all__ = ["train_rna_classifier", "evaluate_model"]
