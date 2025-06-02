# src/main_mlp_classification.py
import argparse
import os
from . import config
from .data_loader import load_all_dataframes
from .preprocess import translate_dataframes, process_tech_cat_names
from .feature_engineering import create_ceramic_summary
from .data_preparation.format_mlp_classification_data import create_mlp_input_data
from .models.mlp_classifier_sklearn import train_and_evaluate_mlp_model
from .utils import seed_everything
import numpy as np
import json
from sklearn.model_selection import train_test_split

def run_mlp_classification_scenarios(dfs, study_name="etude1"):
    """
    Run multiple MLP classification scenarios with different configurations.
    Now using multi-label one-hot encoding instead of multi-slot encoding.
    
    Args:
        dfs (dict): Dictionary containing all dataframes
        study_name (str): Name of the study for organizing outputs (etude1, etude1_prime, or etude2)
    
    Returns:
        dict: Results from each scenario with test accuracies
    """
    print(f"\n{'='*60}")
    print(f"STARTING MLP CLASSIFICATION SCENARIOS FOR: {study_name}")
    print(f"{'='*60}")
    
    results = {}
    
    scenarios = {
        "baseline_architecture": {
            "mlp_params": {
                'hidden_layer_sizes': (32, 16),
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.15,
                'alpha': 0.001
            }
        }
    }
    
    # Prepare MLP input data once
    print(f"\nPreparing MLP input data with multi-label encoding...")
    """
    Type 0: Only ceramic attributes (origin, color, context, source, reuse, production_fail)
    Type 1: Only functions + features (original behavior)
    Type 2: Combined ceramic attributes + functions + features
    """
    X, y, root_category_names, function_map, feature_map = create_mlp_input_data(dfs, study_name="etude2", embedding_type=2)
    
    if X is None or y is None:
        print(f"ERROR: Data preparation failed for study {study_name}")
        return {}
    
    print(f"Data prepared successfully:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Unique classes: {len(np.unique(y))}")
    
    # Split data into train/test once
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.15, 
        random_state=42, 
        stratify=y
    )
    
    print(f"Train/Test split completed:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")

    for scenario_name, scenario_config in scenarios.items():
        try:
            print(f"\n{'='*50}")
            print(f"RUNNING SCENARIO: {scenario_name}")
            print(f"Architecture: {scenario_config['mlp_params']['hidden_layer_sizes']}")
            print(f"{'='*50}")
            
            if X is None or y is None:
                print(f"ERROR: Data preparation failed for scenario {scenario_name}")
                results[scenario_name] = None
                continue
            
            # Train and evaluate MLP model
            mlp_model, scaler, test_accuracy = train_and_evaluate_mlp_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                scenario_name=f"{study_name}_{scenario_name}",
                root_category_names_map=root_category_names,
                mlp_params=scenario_config['mlp_params']
            )
            
            results[scenario_name] = test_accuracy
            
            # Save model and scaler for this scenario
            scenario_output_dir = os.path.join(
                config.OUTPUT_BASE_DIR, 
                "mlp_classification_results", 
                study_name, 
                scenario_name
            )
            os.makedirs(scenario_output_dir, exist_ok=True)
            
            # Save model using joblib (sklearn standard)
            import joblib
            joblib.dump(mlp_model, os.path.join(scenario_output_dir, "trained_mlp_model.pkl"))
            joblib.dump(scaler, os.path.join(scenario_output_dir, "feature_scaler.pkl"))
            
            # Save scenario metadata
            metadata = {
                "scenario_name": scenario_name,
                "study_name": study_name,
                "test_accuracy": float(test_accuracy),
                "config": scenario_config,
                "data_shapes": {
                    "X_train": X_train.shape,
                    "X_test": X_test.shape,
                    "total_features": int(X.shape[1]),
                    "num_classes": int(len(np.unique(y))),
                    "function_embedding_size": len(function_map),
                    "feature_embedding_size": len(feature_map)
                }
            }
            
            with open(os.path.join(scenario_output_dir, "scenario_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=4)
            
            print(f"Model and results saved to: {scenario_output_dir}")
            print(f"Test Accuracy for {scenario_name}: {test_accuracy:.4f}")
            
        except Exception as e:
            print(f"ERROR in scenario {scenario_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[scenario_name] = None
    
    return results


def main(data_source_path, study_name="etude1"):
    """
    Main function for MLP classification pipeline.
    
    Args:
        data_source_path (str): Path to the data directory
        study_name (str): Study configuration name (etude1, etude1_prime, or etude2)
    """
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"{'='*60}")
    print(f"MLP CLASSIFICATION PIPELINE")
    print(f"{'='*60}")
    print(f"Using data source path: {data_source_path}")
    print(f"Study configuration: {study_name}")

    # Validate study name
    valid_studies = ["etude1", "etude1_prime", "etude2"]
    if study_name not in valid_studies:
        print(f"ERROR: Invalid study_name '{study_name}'. Must be one of {valid_studies}")
        print("Using default: etude1")
        study_name = "etude1"

    # 1. Load and preprocess data
    print("\n1. Loading dataframes...")
    dfs = load_all_dataframes(data_base_path=data_source_path)
    
    if not any(not df.empty for df in dfs.values()):
        print("ERROR: No dataframes were loaded for MLP. Exiting.")
        return

    print("2. Translating dataframes...")
    dfs = translate_dataframes(dfs)
    
    print("3. Processing technical category names...")
    dfs = process_tech_cat_names(dfs)

    print("4. Creating ceramic summary...")
    ceramic_summary_df = create_ceramic_summary(dfs)
    
    if ceramic_summary_df.empty:
        print("ERROR: ceramic_summary_df is empty. Exiting MLP process.")
        return
    
    dfs["ceramic_summary"] = ceramic_summary_df
    print(f"Ceramic summary created with {len(ceramic_summary_df)} entries")

    print(f"5. Running MLP classification scenarios for study: {study_name}")
    
    # 3. Run MLP Classification Scenarios
    mlp_results = run_mlp_classification_scenarios(dfs, study_name=study_name)

    # 4. Summary of results
    print(f"\n{'='*60}")
    print("MLP CLASSIFICATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    if mlp_results:
        successful_scenarios = 0
        total_scenarios = len(mlp_results)
        
        for scenario, accuracy in mlp_results.items():
            if accuracy is not None:
                print(f"✓ {scenario}: {accuracy:.4f}")
                successful_scenarios += 1
            else:
                print(f"✗ {scenario}: FAILED")
        
        print(f"\nSuccessful scenarios: {successful_scenarios}/{total_scenarios}")
        
        if successful_scenarios > 0:
            valid_accuracies = [acc for acc in mlp_results.values() if acc is not None]
            best_accuracy = max(valid_accuracies)
            best_scenario = [k for k, v in mlp_results.items() if v == best_accuracy][0]
            print(f"Best performing scenario: {best_scenario} ({best_accuracy:.4f})")
    else:
        print("No results to display. All scenarios failed.")

    print(f"\nMLP Classification pipeline completed.")
    print(f"Results saved in: {os.path.join(config.OUTPUT_BASE_DIR, 'mlp_classification_results', study_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLP Classification pipeline.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory containing CSV files."
    )
    parser.add_argument(
        "--study_name",
        type=str,
        default="etude1",
        choices=["etude1", "etude1_prime", "etude2"],
        help="Study configuration: etude1 (138 per class), etude1_prime (276 per class), etude2 (remove min class, 950 from others). Default: etude1"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.data_path):
        print(f"ERROR: Data path '{args.data_path}' does not exist.")
        exit(1)
    
    main(args.data_path, args.study_name)