# Spiridon

This project implements Graph Neural Network (RGCN) models for tasks related to a ceramic ontology dataset. It includes:
1.  **Link Prediction:** Predicting the root category a ceramic object belongs to.
2.  **Node Classification:** Classifying ceramic objects into their root categories.
3.  **Inductive Prediction:** Predicting categories for new, unseen ceramic objects.


## Directory Structure

ceramic_ontology_gnn/
├── data/
├── output/                   # For models, logs, predictions
│   ├── rgcn_study_datasets/
│   ├── classification_data/
│   ├── lightning_models/
│   ├── lightning_logs/
│   └── predictions/
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Configurations, paths, hyperparameters
│   ├── data_loader.py        # Loading raw CSVs
│   ├── preprocess.py         # Data cleaning and translations
│   ├── feature_engineering.py # Creation of ceramic_summary, one-hot embeddings
│   ├── graph_utils.py        # General graph utilities (hierarchy, triplet extraction)
│   ├── data_preparation/     # NEW: For specific data formatting pipelines
│   │   ├── __init__.py
│   │   ├── format_rgcn_link_prediction_data.py  # Was part of graph_utils
│   │   └── format_rgcn_classification_data.py # Was part of graph_utils
│   ├── models/               # PyTorch Lightning model definitions
│   │   ├── __init__.py
│   │   ├── link_predictor.py
│   │   └── classifier.py
│   ├── training/             # Training pipeline logic
│   │   ├── __init__.py
│   │   ├── link_prediction_pipeline.py
│   │   └── classification_pipeline.py
│   ├── inference/            # Inference scripts (from previous detailed response)
│   │   ├── __init__.py
│   │   ├── link_prediction_inference.py
│   │   └── inductive_inference.py
│   ├── utils.py              # Utility functions (seeding, plotting)
│   ├── main_prepare_data.py  # Script to run all data preparation steps
│   ├── main_link_prediction.py # Script to run link prediction studies
│   ├── main_classification.py  # Script to run classification studies
│   └── main_run_inference.py          # Script for inference tasks
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore                # Git ignore file

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone <url>
    cd spiridon
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    # .venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note for `torch-geometric`: Depending on your CUDA version, you might need to install it separately following instructions from [PyTorch Geometric documentation](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).*

4.  **Install system dependencies (for `pygraphviz`):**
    ```bash
    # On Debian/Ubuntu
    sudo apt-get update
    sudo apt-get install -y graphviz libgraphviz-dev
    # For other OS, refer to pygraphviz installation instructions.
    ```

## Running the Project

Ensure your virtual environment is activated. All scripts are run from the project root directory (`ceramic_ontology_gnn/`).

1.  **Prepare all datasets:**
    This step processes raw data, creates embeddings, and generates specific datasets for link prediction and classification studies.
    ```bash
    python -m src.main_prepare_data --data_path ./data
    ```
    Outputs will be saved in the `output/rgcn_study_datasets/` and `output/classification_data/` directories.

2.  **Run Link Prediction Training/Evaluation:**
    Specify the study name (`etude1`, `etude1_prime`, or `etude2`).
    ```bash
    python -m src.main_link_prediction --study_name etude1
    # python -m src.main_link_prediction --study_name etude1_prime
    # python -m src.main_link_prediction --study_name etude2
    ```
    Models, logs, and predictions will be saved under the `output/` directory.

3.  **Run Classification Training/Evaluation:**
    Specify the study name (based on the prepared classification data).
    ```bash
    python -m src.main_classification --study_name etude1
    # python -m src.main_classification --study_name etude1_prime
    # python -m src.main_classification --study_name etude2
    ```
    Models and logs will be saved under `output/classification_data/<study_name>_root_classification_data/` and `output/lightning_logs/<study_name>_classification/` (adjust paths in script if needed).

4.  **Run Inductive Prediction Test:**
    This uses a pre-trained link prediction model to predict categories for "new" ceramics.
    ```bash
    python -m src.main_inductive_test --study_name etude1 --data_path ./data --num_test_ceramics 10
    ```
    Replace `etude1` with the study whose model you want to use for the inductive test.

## Outputs

*   **`output/rgcn_study_datasets/`**: Contains preprocessed graph data (node mappings, relation mappings, triplets, embeddings) for each link prediction study.
*   **`output/classification_data/`**: Contains preprocessed graph data and labels for each node classification study.
*   **`output/lightning_models/`**: Stores PyTorch Lightning model checkpoints for link prediction studies.
*   **`output/classification_data/<study_name>_root_classification_data/checkpoints/`**: Stores PyTorch Lightning model checkpoints for classification studies.
*   **`output/lightning_logs/`**: Contains logs (e.g., `metrics.csv`) from PyTorch Lightning training runs for both link prediction and classification.
*   **`output/predictions/`**: Stores CSV files with detailed test set predictions for link prediction studies.