import os
import json
import traceback
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
from collections import defaultdict
import warnings

from ..models.link_predictor import RGCNLinkPredictor # Relative import
from .. import config # For paths and default hyperparameters
from ..utils import plot_training_history

# --- Modified: Function signature and docstring/print statement ---
def prepare_lightning_data_btc_focused(rgcn_data, evaluation_relation_name="BELONGS_TO_CATEGORY",
                                       train_ratio=0.7, val_ratio=0.15):
    test_ratio = 1.0 - train_ratio - val_ratio
    if not (0 < train_ratio < 1 and 0 <= val_ratio < 1 and 0 < test_ratio < 1 and abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9):
        raise ValueError(f"Train/Val/Test ratios must be valid and sum to 1. Got train: {train_ratio}, val: {val_ratio}, test: {test_ratio}")

    print(f"--- Preparing Data (BTC-Focused Loss, WITH Validation): Target '{evaluation_relation_name}' for DistMult loss. "
          f"RGCN graph from non-BTC training data. BTC split: {train_ratio*100:.0f}% Train / {val_ratio*100:.0f}% Val / {test_ratio*100:.0f}% Test. ---")
    try:
        num_nodes = rgcn_data['num_nodes']
        relation_to_idx = rgcn_data['relation_to_idx']
        idx_to_node = rgcn_data['idx_to_node']
        cat_idx_to_root_idx_map = rgcn_data.get('cat_idx_to_root_idx_map', {})
        root_category_node_indices_for_sampling = rgcn_data.get('root_category_node_indices_in_graph', [])

        eval_relation_id = relation_to_idx.get(evaluation_relation_name) # Use relation_to_idx
        if eval_relation_id is None:
             print(f"ERROR: Eval relation '{evaluation_relation_name}' not found in relation_to_idx: {relation_to_idx}."); return (None,) * 12 # Modified count

        non_btc_triplets_for_graph = []
        if 'training_triplets' in rgcn_data and rgcn_data['training_triplets']:
            for h, r, t in rgcn_data['training_triplets']:
                if r != eval_relation_id: # Use the dynamically found eval_relation_id
                    non_btc_triplets_for_graph.append((h, r, t))
        non_btc_triplets_for_graph = sorted(list(set(non_btc_triplets_for_graph))) # Deduplicate
        print(f"Triplets for RGCN graph construction (non-BTC from 'training_triplets'): {len(non_btc_triplets_for_graph)}")
        if not non_btc_triplets_for_graph:
            print("Warning: No non-BTC triplets found in 'training_triplets' to form the GNN graph.")
            graph_edge_index = torch.empty((2, 0), dtype=torch.long)
            graph_edge_type = torch.empty((0), dtype=torch.long)
        else:
            graph_triplets_tensor = torch.tensor(non_btc_triplets_for_graph, dtype=torch.long).t()
            graph_edge_index = graph_triplets_tensor[[0, 2], :]
            graph_edge_type = graph_triplets_tensor[1, :]
        print(f"RGCN Graph structure created with {graph_edge_index.shape[1]} edges.")

        all_btc_triplets_original = []
        temp_all_triplets_for_btc_pool = []
        if 'training_triplets' in rgcn_data: temp_all_triplets_for_btc_pool.extend(rgcn_data['training_triplets'])
        if 'evaluation_triplets' in rgcn_data: temp_all_triplets_for_btc_pool.extend(rgcn_data['evaluation_triplets'])
        
        for h, r, t in temp_all_triplets_for_btc_pool:
            if r == eval_relation_id: # Use the dynamically found eval_relation_id
                all_btc_triplets_original.append((h,r,t))
        all_btc_triplets_original = sorted(list(set(all_btc_triplets_original))) # Deduplicate
        print(f"Total unique BTC triplets found across all input data (for relation '{evaluation_relation_name}'): {len(all_btc_triplets_original)}")

        # --- Modified: Split BTC triplets into train, validation, and test ---
        btc_train_positives_for_distmult = []
        btc_val_positives_for_distmult = [] # New list for validation
        btc_test_set_for_distmult = []

        if not cat_idx_to_root_idx_map and all_btc_triplets_original:
            warnings.warn("cat_idx_to_root_idx_map is empty! Using random split for BTC triplets (Train/Val/Test).")
            random.shuffle(all_btc_triplets_original)
            train_end_idx = int(len(all_btc_triplets_original) * train_ratio)
            val_end_idx = train_end_idx + int(len(all_btc_triplets_original) * val_ratio)
            
            btc_train_positives_for_distmult = all_btc_triplets_original[:train_end_idx]
            btc_val_positives_for_distmult = all_btc_triplets_original[train_end_idx:val_end_idx]
            btc_test_set_for_distmult = all_btc_triplets_original[val_end_idx:]
        elif all_btc_triplets_original:
            print("Using balanced split (by root category) for BTC triplets (Train/Val/Test).")
            target_triplets_by_root = defaultdict(list)
            unmapped_target_triplets = []

            for h, r, t_specific in all_btc_triplets_original:
                root_idx = cat_idx_to_root_idx_map.get(t_specific)
                if root_idx is not None: 
                    target_triplets_by_root[root_idx].append((h, r, t_specific))
                else: 
                    unmapped_target_triplets.append((h, r, t_specific))
            
            if unmapped_target_triplets:
                warnings.warn(f"{len(unmapped_target_triplets)} BTC triplets unmapped to root. Splitting them randomly.")
                random.shuffle(unmapped_target_triplets)
                train_end_idx_unmapped = int(len(unmapped_target_triplets) * train_ratio)
                val_end_idx_unmapped = train_end_idx_unmapped + int(len(unmapped_target_triplets) * val_ratio)

                btc_train_positives_for_distmult.extend(unmapped_target_triplets[:train_end_idx_unmapped])
                btc_val_positives_for_distmult.extend(unmapped_target_triplets[train_end_idx_unmapped:val_end_idx_unmapped])
                btc_test_set_for_distmult.extend(unmapped_target_triplets[val_end_idx_unmapped:])

            for root_idx, triplets_in_root in target_triplets_by_root.items():
                random.shuffle(triplets_in_root)
                train_end_point = int(train_ratio * len(triplets_in_root))
                val_end_point = train_end_point + int(val_ratio * len(triplets_in_root))
                
                btc_train_positives_for_distmult.extend(triplets_in_root[:train_end_point])
                btc_val_positives_for_distmult.extend(triplets_in_root[train_end_point:val_end_point])
                btc_test_set_for_distmult.extend(triplets_in_root[val_end_point:])
        
        print(f"BTC triplets for DistMult training (positives): {len(btc_train_positives_for_distmult)}")
        print(f"BTC triplets for DistMult validation: {len(btc_val_positives_for_distmult)}") # New print
        print(f"BTC triplets for DistMult testing: {len(btc_test_set_for_distmult)}")

        train_dataset_distmult_btc = TensorDataset(torch.tensor(btc_train_positives_for_distmult, dtype=torch.long) if btc_train_positives_for_distmult else torch.empty((0,3), dtype=torch.long))
        # --- New: Create validation TensorDataset ---
        val_dataset_distmult_btc = TensorDataset(torch.tensor(btc_val_positives_for_distmult, dtype=torch.long) if btc_val_positives_for_distmult else torch.empty((0,3), dtype=torch.long))
        test_dataset_distmult_btc = TensorDataset(torch.tensor(btc_test_set_for_distmult, dtype=torch.long) if btc_test_set_for_distmult else torch.empty((0, 3), dtype=torch.long))

        all_specific_category_node_indices = {idx for idx, s in idx_to_node.items() if isinstance(s, str) and s.startswith('Cat_')}
        print(f"Found {len(all_specific_category_node_indices)} specific category node indices for eval ranking.")
        
        # This dictionary is used for filtered ranking, so it should contain ALL true BTC triplets.
        all_triplets_dict_for_filtering_btc = defaultdict(lambda: defaultdict(set))
        for h, r, t in all_btc_triplets_original: # Iterate over all original BTC triplets
            if r == eval_relation_id: 
                all_triplets_dict_for_filtering_btc[h][r].add(t)
        print(f"Built BTC triplets dictionary for eval filtering: {sum(len(v) for i in all_triplets_dict_for_filtering_btc.values() for v in i.values())} facts (using all original BTC triplets).")

        # --- Modified: Return tuple now includes val_dataset_distmult_btc ---
        return (graph_edge_index, graph_edge_type,
                train_dataset_distmult_btc,
                val_dataset_distmult_btc, # Added
                test_dataset_distmult_btc,
                list(all_specific_category_node_indices),
                root_category_node_indices_for_sampling,
                cat_idx_to_root_idx_map,
                all_triplets_dict_for_filtering_btc,
                eval_relation_id,
                num_nodes, len(relation_to_idx))

    except KeyError as e:
        print(f"Error: Missing key in rgcn_data: {e}"); traceback.print_exc(); return (None,) * 12
    except Exception as e:
        print(f"Error in Lightning data prep: {e}"); traceback.print_exc(); return (None,) * 12 
    
def run_rgcn_experiment(STUDY_NAME, custom_hyperparameters=None, custom_trainer_params=None):
    BASE_DATA_DIR = '/kaggle/working/rgcn_study_datasets' # Example path
    MODEL_SAVE_DIR = '/kaggle/working/lightning_models'
    LOG_DIR = '/kaggle/working/lightning_logs'
    PREDICTIONS_DIR = '/kaggle/working/predictions'

    os.makedirs(os.path.join(MODEL_SAVE_DIR, STUDY_NAME), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, STUDY_NAME), exist_ok=True)
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    DEFAULT_BATCH_SIZE = 16
    DEFAULT_LEARNING_RATE = 5e-4
    DEFAULT_NUM_EPOCHS = 200
    DEFAULT_RGCN_HIDDEN_DIM = 32
    DEFAULT_NUM_RGCN_LAYERS = 2
    DEFAULT_NUM_BASES = 4
    DEFAULT_DROPOUT = 0.2
    DEFAULT_L2_REG = 0.2
    EVALUATION_RELATION_NAME = "BELONGS_TO_CATEGORY"
    DEFAULT_PATIENCE_EARLY_STOPPING = 10
    DEFAULT_NEG_SAMPLES_BTC = 4
    DEFAULT_USE_SIMILARITY_REL_INIT = False
    
    # --- New: Define default split ratios ---
    DEFAULT_TRAIN_RATIO_BTC = 0.7
    DEFAULT_VAL_RATIO_BTC = 0.15
    # Test ratio will be 1.0 - train_ratio - val_ratio

    # --- Hyperparameter Setup ---
    if custom_hyperparameters is None:
        custom_hyperparameters = {}

    batch_size = custom_hyperparameters.get('BATCH_SIZE', DEFAULT_BATCH_SIZE)
    num_epochs = custom_trainer_params.get('max_epochs', DEFAULT_NUM_EPOCHS) if custom_trainer_params else DEFAULT_NUM_EPOCHS
    patience_early_stopping = custom_hyperparameters.get('patience_early_stopping', DEFAULT_PATIENCE_EARLY_STOPPING)
    use_similarity_rel_init = custom_hyperparameters.get('use_similarity_based_relation_init', DEFAULT_USE_SIMILARITY_REL_INIT)
    
    # --- New: Get split ratios from custom_hyperparameters or use defaults ---
    train_ratio_btc = custom_hyperparameters.get('TRAIN_RATIO_BTC', DEFAULT_TRAIN_RATIO_BTC)
    val_ratio_btc = custom_hyperparameters.get('VAL_RATIO_BTC', DEFAULT_VAL_RATIO_BTC)


    print(f"--- Loading Raw Formatted Data for {STUDY_NAME} ---")
    try:
        study_dir = os.path.join(BASE_DATA_DIR, STUDY_NAME)
        node_embeddings_path = os.path.join(study_dir, 'node_embeddings.npy')
        node_mapping_path = os.path.join(study_dir, 'node_mapping.csv')
        relation_mapping_path = os.path.join(study_dir, 'relation_mapping.csv')
        training_triplets_df_path = os.path.join(study_dir, 'training_triplets.csv')
        evaluation_triplets_df_path = os.path.join(study_dir, 'evaluation_triplets.csv')
        root_cat_indices_path = os.path.join(study_dir, 'root_category_node_indices_in_graph.json')
        cat_to_root_map_path = os.path.join(study_dir, 'cat_idx_to_root_idx_map.json')

        required_files = [node_embeddings_path, node_mapping_path, relation_mapping_path,
                          training_triplets_df_path, evaluation_triplets_df_path,
                          root_cat_indices_path, cat_to_root_map_path]

        if not all(os.path.exists(p) for p in required_files):
             missing = [p for p in required_files if not os.path.exists(p)]
             print(f"[ERROR] Missing data files in {study_dir}: {missing}"); return

        node_embeddings = np.load(node_embeddings_path)
        node_mapping_df = pd.read_csv(node_mapping_path)
        relation_mapping_df = pd.read_csv(relation_mapping_path)
        training_triplets_df = pd.read_csv(training_triplets_df_path)
        evaluation_triplets_df = pd.read_csv(evaluation_triplets_df_path)
        with open(root_cat_indices_path, 'r') as f: root_category_node_indices_in_graph = json.load(f)
        with open(cat_to_root_map_path, 'r') as f: cat_idx_to_root_idx_map = {int(k): v for k,v in json.load(f).items()}
        
        training_triplets_list = training_triplets_df[['SourceIndex', 'RelationIndex', 'TargetIndex']].values.tolist()
        evaluation_triplets_list = evaluation_triplets_df[['SourceIndex', 'RelationIndex', 'TargetIndex']].values.tolist()
        
        all_triples_for_init_list = None 
        if use_similarity_rel_init:
            print(f"Combining training and evaluation triplets for similarity-based relation initialization.")
            combined_triplets = training_triplets_list + evaluation_triplets_list
            unique_triplets_set = set()
            for triplet in combined_triplets:
                unique_triplets_set.add(tuple(triplet)) 
            all_triples_for_init_list = [list(t) for t in unique_triplets_set] 
            if not all_triples_for_init_list and combined_triplets: 
                 print("[WARNING] No triples available for similarity initialization after combining and deduplicating, though input lists were not empty.")
            elif not all_triples_for_init_list:
                 print("[INFO] No training or evaluation triplets provided, so no triples for similarity initialization.")
            print(f"Generated {len(all_triples_for_init_list if all_triples_for_init_list else [])} unique triples for similarity initialization.")

        rgcn_data = {
            "num_nodes": node_embeddings.shape[0],
            "num_relations": len(relation_mapping_df),
            "embedding_dim": node_embeddings.shape[1],
            "node_to_idx": pd.Series(node_mapping_df.NodeIndex.values, index=node_mapping_df.NodeIdentifier.astype(str)).to_dict(),
            "relation_to_idx": pd.Series(relation_mapping_df.RelationIndex.values, index=relation_mapping_df.RelationName.astype(str)).to_dict(),
            "idx_to_node": pd.Series(node_mapping_df.NodeIdentifier.values, index=node_mapping_df.NodeIndex).to_dict(),
            "idx_to_relation": pd.Series(relation_mapping_df.RelationName.values, index=relation_mapping_df.RelationIndex).to_dict(),
            "training_triplets": training_triplets_list, # Use list directly
            "evaluation_triplets": evaluation_triplets_list, # Use list directly
            "initial_node_embeddings": torch.tensor(node_embeddings, dtype=torch.float),
            "root_category_node_indices_in_graph": root_category_node_indices_in_graph,
            "cat_idx_to_root_idx_map": cat_idx_to_root_idx_map,
            "all_triples_for_relation_init_list": all_triples_for_init_list 
        }
        EMBEDDING_DIM = rgcn_data['embedding_dim']
        print("emb dim",EMBEDDING_DIM )
    except Exception as e:
        print(f"[ERROR] Failed to load data for {STUDY_NAME}: {e}"); traceback.print_exc(); return

    # --- Data Preparation for Lightning ---
    print(f"--- Preparing Lightning Data for {STUDY_NAME} ---")
    data_prep_output = prepare_lightning_data_btc_focused(
        rgcn_data, 
        evaluation_relation_name=EVALUATION_RELATION_NAME,
        # --- Modified: Pass new split ratios ---
        train_ratio=train_ratio_btc, 
        val_ratio=val_ratio_btc
    )
    if data_prep_output is None or data_prep_output[0] is None: 
        print(f"Data preparation failed for {STUDY_NAME}. Exiting."); return

    # --- Modified: Unpack validation dataset ---
    (graph_edge_index, graph_edge_type,
     train_dataset_btc, val_dataset_btc, test_dataset_btc, # val_dataset_btc added
     specific_category_node_indices_for_eval,
     root_category_node_indices_for_sampling,
     cat_idx_to_root_idx_map_for_model,
     all_triplets_dict_for_filtering_btc,
     eval_relation_id_for_model,
     num_nodes_from_prep, num_relations_from_prep) = data_prep_output

    if eval_relation_id_for_model is None:
         print(f"[ERROR] Eval relation ID is None after data preparation. Cannot proceed."); return

    # --- Modified: Create DataLoader for validation set ---
    train_loader = DataLoader(train_dataset_btc, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=True if torch.cuda.is_available() and batch_size > 1 else False)
    val_loader = DataLoader(val_dataset_btc, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=True if torch.cuda.is_available() and batch_size > 1 else False)
    test_loader = DataLoader(test_dataset_btc, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available(), persistent_workers=True if torch.cuda.is_available() and batch_size > 1 else False)

    if len(train_dataset_btc) == 0: print(f"Warning: Training dataset for {STUDY_NAME} is empty.")
    # --- New: Warning for empty validation set ---
    if len(val_dataset_btc) == 0: print(f"Warning: Validation dataset for {STUDY_NAME} is empty. Validation callbacks might not function as expected.")
    if len(test_dataset_btc) == 0: print(f"Warning: Test dataset for {STUDY_NAME} is empty.")


    # --- Model Instantiation ---
    # ... (rest of model instantiation is largely the same, ensure all_triples_for_relation_init is passed)
    print(f"--- Initializing RGCNLinkPredictor for {STUDY_NAME} ---")
    rgcn_hidden_dim_val = custom_hyperparameters.get('rgcn_hidden_dim', DEFAULT_RGCN_HIDDEN_DIM)
    
    if use_similarity_rel_init and EMBEDDING_DIM != rgcn_hidden_dim_val:
        print(f"[WARNING] `use_similarity_based_relation_init` is True, but node embedding_dim ({EMBEDDING_DIM}) "
              f"!= rgcn_hidden_dim ({rgcn_hidden_dim_val}). Adjusting rgcn_hidden_dim to match embedding_dim.")
        rgcn_hidden_dim_val = EMBEDDING_DIM

    model_params = {
        'num_nodes': num_nodes_from_prep,
        'num_relations': num_relations_from_prep,
        'embedding_dim': EMBEDDING_DIM, 
        'rgcn_hidden_dim': rgcn_hidden_dim_val, 
        'num_rgcn_layers': custom_hyperparameters.get('num_rgcn_layers', DEFAULT_NUM_RGCN_LAYERS),
        'num_bases': custom_hyperparameters.get('num_bases', DEFAULT_NUM_BASES),
        'dropout': custom_hyperparameters.get('dropout', DEFAULT_DROPOUT),
        'learning_rate': custom_hyperparameters.get('learning_rate', DEFAULT_LEARNING_RATE),
        'l2_reg': custom_hyperparameters.get('l2_reg', DEFAULT_L2_REG),
        'neg_samples_btc_relation': custom_hyperparameters.get('neg_samples_btc_relation', DEFAULT_NEG_SAMPLES_BTC),
        'initial_node_embeddings': rgcn_data['initial_node_embeddings'],
        'evaluation_relation_id': eval_relation_id_for_model,
        'category_node_indices': specific_category_node_indices_for_eval,
        'root_category_node_indices': root_category_node_indices_for_sampling,
        'cat_idx_to_root_idx_map': cat_idx_to_root_idx_map_for_model,
        'all_triplets_dict_btc': all_triplets_dict_for_filtering_btc,
        'idx_to_node': rgcn_data['idx_to_node'],
        'use_similarity_based_relation_init': use_similarity_rel_init,
        'all_triples_for_relation_init': rgcn_data['all_triples_for_relation_init_list'] 
    }

    model = RGCNLinkPredictor(**model_params)
    print(ModelSummary(model, max_depth=1)) 
    model.setup_graph(graph_edge_index, graph_edge_type)

    # --- Trainer Setup ---
    print(f"--- Configuring Trainer for {STUDY_NAME} ---")
    effective_trainer_params = {
        'max_epochs': num_epochs,
        'accelerator': "auto",
        'devices': "auto", 
        'logger': CSVLogger(save_dir=LOG_DIR, name=STUDY_NAME, version=None), 
        'log_every_n_steps': 10, 
        'deterministic': True, 
    }
    if custom_trainer_params:
        for key, value in custom_trainer_params.items():
            if key not in ['logger', 'callbacks', 'max_epochs']: 
                effective_trainer_params[key] = value

    # --- Modified: Callbacks monitor validation metrics ---
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss', # Monitor validation loss
        mode='min',
        dirpath=os.path.join(MODEL_SAVE_DIR, STUDY_NAME),
        filename=f'{STUDY_NAME}-best-{{epoch:02d}}-{{val_loss:.4f}}', # Filename reflects val_loss
        save_top_k=1,
        verbose=True,
        save_last=True 
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience_early_stopping,
        mode='min',
        verbose=True
    )
    callbacks = [checkpoint_callback, early_stop_callback]
    effective_trainer_params['callbacks'] = callbacks
    
    # --- Modified: Set limit_val_batches and limit_test_batches based on dataset sizes ---
    effective_trainer_params['limit_val_batches'] = 1.0 if len(val_dataset_btc) > 0 else 0.0
    effective_trainer_params['limit_test_batches'] = 1.0 if len(test_dataset_btc) > 0 else 0.0


    trainer = pl.Trainer(**effective_trainer_params)

    # --- Training ---
    # --- Modified: Pass validation dataloader to trainer.fit ---
    print(f"\n--- Training {STUDY_NAME} with Validation ---")
    if len(train_loader) > 0:
        if len(val_loader) > 0 :
            trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        else:
            print(f"Validation loader for {STUDY_NAME} is empty. Training without validation loop during fit.")
            trainer.fit(model, train_dataloaders=train_loader) # Fit without val_dataloaders
    else:
        print(f"Skipping training for {STUDY_NAME} as train_loader is empty.")


    # --- Testing ---
    # ... (Testing section remains largely the same, best_model_path will now be based on val_loss)
    print(f"\n--- Testing Best Model for {STUDY_NAME} (selected based on validation performance) ---")
    if len(test_dataset_btc) > 0:
        best_model_path_to_test = checkpoint_callback.best_model_path
        
        if not (best_model_path_to_test and os.path.exists(best_model_path_to_test)):
            print(f"Best model path '{best_model_path_to_test}' not found or invalid.")
            if hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path and \
               os.path.exists(checkpoint_callback.last_model_path):
                 print(f"Using last saved model: {checkpoint_callback.last_model_path}")
                 best_model_path_to_test = checkpoint_callback.last_model_path
            else:
                last_ckpt_path = os.path.join(MODEL_SAVE_DIR, STUDY_NAME, "last.ckpt")
                if os.path.exists(last_ckpt_path):
                    print(f"Using last.ckpt: {last_ckpt_path}")
                    best_model_path_to_test = last_ckpt_path
                else:
                    print(f"No valid best or last checkpoint found. Will test current model state (if training occurred).")
                    best_model_path_to_test = None 

        if best_model_path_to_test and os.path.exists(best_model_path_to_test):
             print(f"Loading model from: {best_model_path_to_test} for testing.")
             test_results = trainer.test(dataloaders=test_loader, ckpt_path=best_model_path_to_test)
        elif len(train_loader) > 0 : 
             print(f"No valid checkpoint found. Testing current model state after training.")
             test_results = trainer.test(model, dataloaders=test_loader)
        else:
            print(f"Skipping testing as no checkpoint found and no training was performed.")
            test_results = None

        if test_results:
            print(f"--- Test Metrics for {STUDY_NAME} ---")
            print(test_results[0]) 

            if model.test_predictions:
                print(f"\n--- Saving Test Predictions for {STUDY_NAME} ({len(model.test_predictions)} entries) ---")
                predictions_df = pd.DataFrame(model.test_predictions)
                predictions_df['head_identifier'] = predictions_df['head_idx'].apply(lambda x: rgcn_data['idx_to_node'].get(x, f'UnknownNode_{x}'))
                predictions_df['relation_identifier'] = predictions_df['relation_idx'].apply(lambda x: rgcn_data['idx_to_relation'].get(x, f'UnknownRel_{x}'))
                
                ordered_columns = [
                    'head_idx', 'head_identifier', 'relation_idx', 'relation_identifier',
                    'true_tail_idx', 'true_tail_identifier_specific', 
                    'predicted_tail_idx', 'predicted_tail_identifier_specific', 
                    'rank', 'true_score', 'predicted_score',
                    'true_root_idx', 'true_root_identifier', 
                    'predicted_root_idx', 'predicted_root_identifier', 
                    'same_root_correct'
                ]
                final_columns = [col for col in ordered_columns if col in predictions_df.columns]
                predictions_df = predictions_df[final_columns]
                
                predictions_csv_path = os.path.join(PREDICTIONS_DIR, f'{STUDY_NAME}_test_predictions.csv')
                predictions_df.to_csv(predictions_csv_path, index=False)
                print(f"Test predictions for {STUDY_NAME} saved to: {predictions_csv_path}")
            else:
                print(f"No test predictions collected for {STUDY_NAME} (model.test_predictions is empty).")
        else:
            print(f"No test results generated for {STUDY_NAME}.")
    else:
        print(f"Skipping testing for {STUDY_NAME} (empty test dataset).")

    # --- Conclusion ---
    print(f"\n--- {STUDY_NAME} Done ---")
    # ... (Conclusion section remains the same)
    if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path and os.path.exists(checkpoint_callback.best_model_path):
         print(f"Best Checkpoint (monitored on {checkpoint_callback.monitor}) for {STUDY_NAME}: {checkpoint_callback.best_model_path}")
    elif hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path and os.path.exists(checkpoint_callback.last_model_path):
         print(f"Last Checkpoint for {STUDY_NAME}: {checkpoint_callback.last_model_path}")
    else:
        manual_last_ckpt_path = os.path.join(MODEL_SAVE_DIR, STUDY_NAME, "last.ckpt")
        if os.path.exists(manual_last_ckpt_path):
            print(f"Last Checkpoint (last.ckpt) for {STUDY_NAME}: {manual_last_ckpt_path}")
        else:
            print(f"No model checkpoint seems to have been saved for {STUDY_NAME}.")