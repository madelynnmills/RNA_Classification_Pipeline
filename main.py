import pandas as pd
from src.preprocessing import fetch_sequences_to_dataframe
from src.model import train_rna_classifier
from src.visualizations import plot_class_distribution, plot_feature_importance
from src.utils import save_to_csv

def main():
    rna_data = []
    rna_types = {
        "mRNA": "mRNA[Filter]",
        "tRNA": "tRNA[Filter]",
        "rRNA": "rRNA[Filter]",
        "miRNA": "microRNA",
        "siRNA": "small interfering RNA"
    }

    print("Fetching RNA data...")
    for rna_type, query in rna_types.items():
        print(f"Fetching {rna_type} data...")
        df = fetch_sequences_to_dataframe(query=query, email="maddymills2012@gmail.com", retmax=50)
        if df.empty:
            print(f"No sequences fetched for {rna_type}. Skipping...")
        else:
            df["RNA_Type"] = rna_type
            rna_data.append(df)

    if not rna_data:
        print("No RNA data fetched. Exiting pipeline.")
        return

    print("Combining RNA data...")
    combined_df = pd.concat(rna_data, ignore_index=True)
    print(combined_df.head())
    print("Columns in combined_df:", combined_df.columns)

    print("Adding motif features...")
    motifs = ["ATG", "TTT", "GGG"]
    for motif in motifs:
        combined_df[f"{motif}_Frequency"] = combined_df["Sequence"].apply(lambda seq: seq.count(motif))

    print("Visualizing data distribution...")
    plot_class_distribution(combined_df)

    print("Training RNA classifier...")
    model, label_encoder = train_rna_classifier(combined_df)

    if model:
        print("Visualizing feature importance...")
        plot_feature_importance(model, combined_df.columns.drop(["ID", "Sequence", "RNA_Type"], errors="ignore"))

        print("Saving processed data...")
        save_to_csv(combined_df, "rna_analysis_results.csv")

    print("Pipeline complete!")

if __name__ == "__main__":
    main()
