import argparse
import os
from . import config
from .training.classification_pipeline import train_and_test_classification_model
from .utils import seed_everything
import torch

# Add this to your main training script for debugging


def main(study_name, custom_hyperparams_dict=None, custom_trainer_params_dict=None):
    """
    Run RGCN classification experiment for a specified study.
    
    Args:
        study_name (str): Name of the study to run classification on
        custom_hyperparams_dict (dict, optional): Custom hyperparameters to override defaults
        custom_trainer_params_dict (dict, optional): Custom trainer parameters to override defaults
    """
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"Running Classification Experiment for Study: {study_name}")

    # Default classification hyperparameters from config
    hyperparams = {
        'rgcn_hidden_dim': config.CLF_RGCN_HIDDEN_DIM,
        'learning_rate': config.CLF_LEARNING_RATE,
        'patience': config.CLF_PATIENCE_EARLY_STOPPING,
        'dropout': config.CLF_DROPOUT,
        'l2_reg': config.CLF_L2_REG
    }
    
    if custom_hyperparams_dict:
        hyperparams.update(custom_hyperparams_dict)

    # Default trainer parameters
    trainer_params = {
        'max_epochs': config.CLF_NUM_EPOCHS,
        'accelerator': 'auto',
        'devices': 'auto',
        'enable_progress_bar': True
    }
    
    if custom_trainer_params_dict:
        trainer_params.update(custom_trainer_params_dict)

    # Run classification experiment
    results = train_and_test_classification_model(
        study_name=study_name,
        base_data_dir=config.CLASSIFICATION_DATA_DIR,
        hyperparameters=hyperparams,
        trainer_params=trainer_params,
        plot_history=True
    )
    
    if results:
        print(f"Classification Experiment for {study_name} Completed Successfully")
        print(f"Log Directory: {results.get('log_dir', 'N/A')}")
        print(f"Best Checkpoint: {results.get('best_checkpoint_path', 'N/A')}")
        print(f"Test Metrics: {results.get('test_metrics', 'N/A')}")
    else:
        print(f"Classification experiment for {study_name} failed or was skipped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RGCN Classification experiment.")
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        choices=['etude1', 'etude1_prime', 'etude2'],
        help="Name of the classification study to run."
    )
    args = parser.parse_args()
    main(args.study_name)