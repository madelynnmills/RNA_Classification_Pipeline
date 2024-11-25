import matplotlib.pyplot as plt
import seaborn as sns

def plot_class_distribution(df):
    """
    Plots the class distribution of RNA types.
    """
    class_counts = df["RNA_Type"].value_counts()
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind="bar", color="skyblue", title="RNA Class Distribution")
    plt.ylabel("Count")
    plt.xlabel("RNA Type")
    plt.show()

def plot_feature_importance(model, feature_names):
    """
    Plots feature importance from the trained model.
    """
    importance = model.feature_importances_
    sorted_idx = importance.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names[sorted_idx], importance[sorted_idx], color="green")
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

def plot_pairplot(df):
    """
    Creates a pair plot for selected numerical features.

    Parameters:
        df (pd.DataFrame): DataFrame containing features.
    """
    features = [col for col in df.columns if "_Frequency" in col] + ["GC_Content", "Length"]
    sns.pairplot(df[features])
    plt.suptitle("Feature Relationships", y=1.02)
    plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Plots a confusion matrix for model predictions.

    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        labels (list): Label names for the classes.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
