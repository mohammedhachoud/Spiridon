import argparse
import os
from . import config
from .training.link_prediction_pipeline import run_rgcn_experiment
from .utils import seed_everything

def main(study_name, custom_hyperparams_dict=None, custom_trainer_params_dict=None):
    seed_everything(42)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print(f"\n>>> Running Link Prediction Experiment for Study: {study_name} <<<")

    run_rgcn_experiment(
        STUDY_NAME=study_name,
        custom_hyperparameters=custom_hyperparams_dict,
        custom_trainer_params=custom_trainer_params_dict
    )
    print(f"\n--- Link Prediction Experiment for {study_name} Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RGCN Link Prediction experiment.")
    parser.add_argument(
        "--study_name",
        type=str,
        required=True,
        choices=['etude1', 'etude1_prime', 'etude2'], 
        help="Name of the study to run (e.g., etude1)."
    )
    args = parser.parse_args()
    
  
    # hyperparams = {}
    # if args.study_name == "etude1":
    #     hyperparams = {'learning_rate': 1e-3}
        
    main(args.study_name) 