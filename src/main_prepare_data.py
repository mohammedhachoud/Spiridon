import os
import argparse
from . import config
from .data_loader import load_all_dataframes
from .preprocess import translate_dataframes, process_tech_cat_names
from .feature_engineering import create_ceramic_summary, generate_one_hot_embeddings
from .graph_utils import (
    build_category_hierarchy_and_map_ceramics,
    calculate_completeness_score,
    extract_triplets_for_selection
)
from .data_preparation.format_rgcn_link_prediction_data import (
    prepare_study_datasets_hybrid_embeddings,
    save_rgcn_study_data
)
from .data_preparation.format_rgcn_classification_data import (
    process_and_save_classification_data
)
from .utils import seed_everything


def main(data_source_path):
    """
    Main data preparation pipeline for Ceramic Ontology GNN project.
    
    Processes raw ceramic data through the complete pipeline:
    1. Load and preprocess dataframes
    2. Create ceramic summaries with embeddings
    3. Prepare link prediction datasets
    4. Prepare classification datasets
    
    Args:
        data_source_path (str): Path to directory containing input CSV files
    """
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Using data source path: {data_source_path}")

    # Load and validate dataframes
    dfs = load_all_dataframes(data_base_path=data_source_path)
    if not any(not df.empty for df in dfs.values()):
        print("ERROR: No dataframes were loaded. Exiting.")
        return

    # Preprocess dataframes
    dfs = translate_dataframes(dfs)
    dfs = process_tech_cat_names(dfs)

    # Create ceramic summary
    ceramic_summary_df = create_ceramic_summary(dfs)
    if ceramic_summary_df.empty:
        print("ERROR: ceramic_summary_df is empty. Exiting.")
        return
    
    # Save ceramic summary
    output_summary_path = os.path.join(config.OUTPUT_BASE_DIR, "ceramic_summary_prepared.csv")
    try:
        os.makedirs(config.OUTPUT_BASE_DIR, exist_ok=True)
        ceramic_summary_df.to_csv(output_summary_path, index=False)
        print(f"Ceramic Summary saved to: {output_summary_path}")
    except Exception as e:
        print(f"ERROR saving ceramic_summary_df: {e}")

    # Generate embeddings and add to dataframes
    ceramic_summary_df_with_ohe = generate_one_hot_embeddings(ceramic_summary_df.copy())
    dfs["ceramic_summary"] = ceramic_summary_df_with_ohe

    # Prepare link prediction datasets
    print("Preparing Link Prediction Study Datasets...")
    study_results_hybrid = prepare_study_datasets_hybrid_embeddings(
        dfs,
        bert_model_name=config.BERT_MODEL_NAME
    )
    
    if study_results_hybrid:
        print("Saving link prediction study datasets...")
        save_rgcn_study_data(study_results_hybrid, base_output_dir=config.RGCN_STUDY_DATASETS_DIR)
    else:
        print("WARNING: Hybrid study dataset preparation failed or returned None.")

    # Prepare classification datasets
    print("Preparing Classification Study Datasets...")
    studies_to_process_clf = ['etude1', 'etude1_prime', 'etude2']
    
    for study_name_clf in studies_to_process_clf:
        if study_name_clf in study_results_hybrid and study_results_hybrid[study_name_clf] is not None:
            process_and_save_classification_data(
                study_name_clf, 
                study_results_hybrid,
                dfs,
                base_output_dir=config.CLASSIFICATION_DATA_DIR
            )
        else:
            print(f"Skipping classification data prep for '{study_name_clf}' - RGCN data missing.")
    
    print("Data Preparation Script Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare all data for Ceramic Ontology GNN project.")
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.LOCAL_DATA_PATH,
        help=f"Path to the directory containing input CSV files. Default: {config.LOCAL_DATA_PATH}"
    )
    args = parser.parse_args()
    main(args.data_path)