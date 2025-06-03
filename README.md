# Spiridon

This project implements Graph Neural Network (RGCN) and Multi-Layer Perceptron (MLP) models for tasks related to a ceramic ontology dataset. It includes:
1.  **Node Classification:** Classifying ceramic objects into their root categories using GNNs + MLP.
2.  **MLP Classification:** Classifying ceramic objects into their root categories using an MLP.

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

Ensure your virtual environment is activated. All scripts are run from the project root directory (`spiridon/`).

1.  **Prepare all datasets:**
    This step processes raw data, creates embeddings, and generates specific datasets for link prediction and classification studies.
    ```bash
    python -m src.main_prepare_data --data_path ./data
    ```
    Outputs will be saved in the `output/rgcn_study_datasets/` and `output/classification_data/` directories. This step is also a prerequisite for MLP classification if it relies on processed data from these directories.


2.  **Run Node Classification Training/Evaluation (RGCN):**
    Specify the study name (based on the prepared classification data).
    ```bash
    python -m src.main_rgcn_mlp_classification --study_name etude1
    # python -m src.main_rgcn_mlp_classification --study_name etude1_prime
    # python -m src.main_rgcn_mlp_classification --study_name etude2
    ```
    Models and logs will be saved under `output/classification_data/<study_name>_root_classification_data/` and `output/lightning_logs/<study_name>_classification/`.

3.  **Run MLP Classification Training/Evaluation:**
    Specify the study name (e.g., `etude2`). The `--data_path` should point to the directory containing the raw or initially processed data needed by `create_mlp_input_data`.
    ```bash
    python -m src.main_mlp_classification --data_path ./data --study_name etude2
    ```
    Models and logs will likely be saved under a path similar to GNN classification (e.g., `output/classification_data/<study_name>_mlp_root_classification_data/` and `output/lightning_logs/<study_name>_mlp_classification/`) or a dedicated `output/mlp_outputs/` directory, depending on the script's implementation.

    *Note on MLP input data types:*
    The `main_mlp_classification.py` script internally uses a function `create_mlp_input_data` to prepare features for the MLP. This function can be configured (e.g., via an `embedding_type` parameter) to use different sets of input features:
    *   **Type 0:** Only ceramic attributes (origin, color, context, source, reuse, production_fail).
    *   **Type 1:** Only functions + features.
    *   **Type 2:** Combined ceramic attributes + functions + features.
    The provided example command `python -m src.main_mlp_classification --data_path ./data --study_name etude2` will use the feature type as configured within the `main_mlp_classification.py` script (e.g., `embedding_type=2` is used in the development snippet for `etude2`). To experiment with different types, you may need to modify the script or check if it supports a command-line argument to control this behavior.

## Outputs

*   **`output/rgcn_study_datasets/`**: Contains preprocessed graph data (node mappings, relation mappings, triplets, embeddings) for each link prediction study.
*   **`output/classification_data/`**: Contains preprocessed graph data and labels for each node classification study (GNN and potentially MLP if it uses the same structure).
*   **`output/lightning_models/`**: Stores PyTorch Lightning model checkpoints for link prediction studies.
*   **`output/classification_data/<study_name>_root_classification_data/checkpoints/`**: Stores PyTorch Lightning model checkpoints for GNN classification studies.
*   **`output/classification_data/<study_name>_mlp_root_classification_data/checkpoints/` (Example Path)**: May store MLP model checkpoints (actual path might vary).
*   **`output/lightning_logs/`**: Contains logs (e.g., `metrics.csv`), GNN classification, and MLP classification (e.g., under subdirectories like `<study_name>_classification/` or `<study_name>_mlp_classification/`).
*   **`output/predictions/`**: Stores CSV files with detailed test set predictions for link prediction studies.
*   MLP-specific outputs (e.g., trained models, evaluation results) will also be saved, typically within the `output/` directory, possibly under `output/mlp_outputs/` or following a pattern similar to the GNN classification outputs.