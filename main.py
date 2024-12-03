import pandas as pd     #Used for creating and manipulating DataFrames (primary data structure for this pipeline)
from src.preprocessing import fetch_sequences_to_dataframe      #A utility from src.preprocesing that queries the NCBI database and converts RNA sequence data into a DataFrame
from src.model import train_rna_classifier      #A function from src.model to train a machine learning classifier to predict RNA types
from src.visualizations import plot_class_distribution, plot_feature_importance         #A visualization function to display class distributions of RNA types and which features contributed to the model's predictions 
from src.utils import save_to_csv       #A utility from src.utils to save the procesed data to a CSV file 

def main():
    rna_data = []       #A list to store DataFrames for each RNA type
    rna_types = {       #A dictionary mapping RNA types (keys) to their corresponding NCBI database search queries (values)
        "mRNA": "mRNA[Filter]",
        "tRNA": "tRNA[Filter]",
        "rRNA": "rRNA[Filter]",
        "miRNA": "microRNA",
        "siRNA": "small interfering RNA"
    }

    print("Fetching RNA data...")
    for rna_type, query in rna_types.items():
        print(f"Fetching {rna_type} data...")
        df = fetch_sequences_to_dataframe(query=query, email="maddymills2012@gmail.com", retmax=50)     #fetches sequences from NCBI based on the query and returns them in a DataFrame, query passes the specific RNA type 
        if df.empty:        #checks if no data was fetched for the RNA type
            print(f"No sequences fetched for {rna_type}. Skipping...")
        else:
            df["RNA_Type"] = rna_type       #adds a column to label the RNA type in the DataFrame
            rna_data.append(df)         #appends the DataFrame to the list for later concatenation

    if not rna_data:        #Exits the pipeline if no data was getched, ensuring no further steps are executed unnecessarily
        print("No RNA data fetched. Exiting pipeline.")
        return

    print("Combining RNA data...")
    combined_df = pd.concat(rna_data, ignore_index=True)        #pd.concat() combines all RNA type DataFrames into a single DataFrame for further processing 
    print(combined_df.head())          #prints the first few rows for inspection
    print("Columns in combined_df:", combined_df.columns)       #displays column names to verify the structure of the DataFrame

    print("Adding motif features...")
    motifs = ["ATG", "TTT", "GGG"]         #A list of motifs to analyze within sequences
    for motif in motifs:
        combined_df[f"{motif}_Frequency"] = combined_df["Sequence"].apply(lambda seq: seq.count(motif))        #calculates teh frequency of each motif in a sequence and adds it as a new column (e.g., "ATG_Frequency")

    print("Visualizing data distribution...")
    plot_class_distribution(combined_df)         #displays the distribution of RNA types, helping identify any class imbalances  

    X = combined_df.drop(columns=["ID", "Sequence", "RNA_Type"], errors="ignore")       #X: the feature matrix, excluding clumns irrelevant for prediction ("ID", "Sequence", "RNA_Type")
    y = combined_df["RNA_Type"]         #y: the target vector containing RNA type labels 


    print("Training RNA classifier...")
    model, label_encoder = train_rna_classifier(X, y)       #trains a classifier using X (features) and y(labels), model: the trained machine learning model, label_encoder: encodes string RNA labels into numeric form for model compatibility

    if model:
        print("Visualizing feature importance...")
        plot_feature_importance(model, combined_df.columns.drop(["ID", "Sequence", "RNA_Type"], errors="ignore"))       #highlights the most influential features in the model's predictions, offering interpretability

        print("Saving processed data...")
        save_to_csv(combined_df, "rna_analysis_results.csv")        #saves the combined DataFrame with added features and labeels the CSV file for later analysis

    print("Pipeline complete!")

if __name__ == "__main__":          #ensures the script runs only when executed directly, not when imported as a module
    main()
