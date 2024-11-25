import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder

def train_rna_classifier(df):
    """
    Trains a RandomForestClassifier to predict RNA types.
    """
    try:
        # Prepare features and labels
        X = df.drop(columns=["ID", "Sequence", "RNA_Type"], errors="ignore")
        y = df["RNA_Type"]

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Validate and clean X
        X_train = X_train.map(lambda x: np.nan if isinstance(x, list) else x)  # Stay as DataFrame
        X_train = X_train.dropna()  # Drop rows with NaN values
        X_test = X_test.map(lambda x: np.nan if isinstance(x, list) else x)  # Clean test set
        X_test = X_test.dropna()

        # Convert to NumPy arrays for model training
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()

        # Define RandomForest model and hyperparameters
        # src/model.py
        param_grid = {
            "n_estimators": [50, 100, 150],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
        }

        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(model, param_grid, cv=3)

        # Fit model
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Evaluate model
        accuracy = best_model.score(X_test, y_test)
        print(f"Test accuracy: {accuracy:.2f}")

        from sklearn.utils import resample

        # Inside train_rna_classifier function:
        # Balancing data if classes are imbalanced
        if y_train.value_counts().min() / y_train.value_counts().max() < 0.8:
            print("Balancing data...")
            dfs = []
            for label in y_train.unique():
                df_class = X_train[y_train == label]
                df_resampled = resample(
                    df_class,
                    replace=True,
                    n_samples=y_train.value_counts().max(),
                    random_state=42
                )
                dfs.append(df_resampled)
            X_train = pd.concat(dfs)
            y_train = pd.Series([label for label in y_train.unique() for _ in range(y_train.value_counts().max())])

        return best_model, label_encoder

    except Exception as e:
        print(f"Error in training classifier: {e}")
        return None, None
