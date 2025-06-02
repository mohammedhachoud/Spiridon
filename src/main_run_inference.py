import argparse
import os
from . import config
from .data_loader import load_all_dataframes
from .preprocess import translate_dataframes, process_tech_cat_names
from .feature_engineering import create_ceramic_summary, generate_one_hot_embeddings
from .inference.link_prediction_inference import predict_selected_ceramics_with_saved_model
from .inference.inductive_inference import predict_inductive_ceramics_with_dynamic_feature_rels
from .utils import seed_everything

def main():
    parser = argparse.ArgumentParser(description="Run inference tasks for Ceramic Ontology GNN.")
    parser.add_argument(
        "task",
        choices=['link_pred_test', 'inductive_test'],
        help="Which inference task to run: 'link_pred_test' or 'inductive_test'."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default='etude1', # Default study model to load
        choices=['etude1', 'etude1_prime', 'etude2'],
        help="Name of the study whose trained model will be used."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=config.LOCAL_DATA_PATH,
        help=f"Path to the input CSV data. Default: {config.LOCAL_DATA_PATH}"
    )
    parser.add_argument(
        "--num_test_items",
        type=int,
        default=10,
        help="Number of items for testing (used by both tasks)."
    )
    parser.add_argument(
        "--items_per_root",
        type=int,
        default=2,
        help="Items per root category for link_pred_test sampling."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Optional: Direct path to a .ckpt model file to override default loading."
    )

    args = parser.parse_args()
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if args.task == 'link_pred_test':
        print(f"\n>>> Running Link Prediction Model Test for Study: {args.study_name} <<<")
        results, _ = predict_selected_ceramics_with_saved_model(
            STUDY_NAME=args.study_name,
            num_test_ceramics=args.num_test_items,
            ceramics_per_root_category=args.items_per_root,
            checkpoint_path_override=args.checkpoint_path
        )
        if results:
            print(f"\nDetailed results for {args.study_name} link prediction test available.")
            # Optionally save `results` to a file
    
    elif args.task == 'inductive_test':
        print(f"\n>>> Running Inductive Prediction Test using model from Study: {args.study_name} <<<")
        print(f"Loading all dataframes from: {args.data_path}")
        dfs = load_all_dataframes(data_base_path=args.data_path)
        
        # Crucial: Ensure dfs["ceramic_summary"] exists and is processed
        # This assumes main_prepare_data.py has been run or dfs contains it.
        # If not, it needs to be generated here.
        if "ceramic_summary" not in dfs or dfs["ceramic_summary"].empty:
            print("  'ceramic_summary' not found. Attempting to generate it...")
            dfs = translate_dataframes(dfs)
            dfs = process_tech_cat_names(dfs)
            ceramic_summary_df = create_ceramic_summary(dfs)
            if ceramic_summary_df.empty:
                print("ERROR: Failed to create ceramic_summary_df. Cannot run inductive test.")
                return
            ceramic_summary_df_with_ohe = generate_one_hot_embeddings(ceramic_summary_df.copy())
            dfs["ceramic_summary"] = ceramic_summary_df_with_ohe
            print("  'ceramic_summary' generated.")


        inductive_results = predict_inductive_ceramics_with_dynamic_feature_rels(
            STUDY_NAME=args.study_name,
            dfs=dfs,
            num_new_ceramics_to_test=args.num_test_items,
            checkpoint_path_override=args.checkpoint_path
        )
        if inductive_results:
            print("\n--- Inductive Prediction Detailed Results ---")
            for item in inductive_results:
                print(item)
    
    print(f"\n--- {args.task.replace('_', ' ').title()} for {args.study_name} Finished ---")

if __name__ == "__main__":
    main()