import os
import json
import traceback
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.utilities.model_summary import ModelSummary
import warnings

from ..models.classifier import RGCNClassifier
from .. import config
from ..utils import plot_training_history


def load_classification_data(output_dir):
    """
    Loads classification data components from a directory.
    
    Args:
        output_dir (str): Directory containing classification data files
        
    Returns:
        dict: Dictionary containing loaded data and hyperparameters, or None if loading fails
    """
    try:
        # Load data files
        node_embeddings = np.load(os.path.join(output_dir, 'node_embeddings.npy'))
        node_mapping_df = pd.read_csv(os.path.join(output_dir, 'node_mapping.csv'))
        relation_mapping_df = pd.read_csv(os.path.join(output_dir, 'relation_mapping.csv'))
        training_triplets_df = pd.read_csv(os.path.join(output_dir, 'training_triplets.csv'))
        ceramic_labels_df = pd.read_csv(os.path.join(output_dir, 'ceramic_labels.csv'))
        label_mapping_df = pd.read_csv(os.path.join(output_dir, 'label_mapping.csv'))
        
        with open(os.path.join(output_dir, 'classification_stats.json'), 'r') as f:
            stats = json.load(f)

        # Create mapping dictionaries
        node_to_idx = dict(zip(node_mapping_df['node_identifier'], node_mapping_df['new_node_idx']))
        idx_to_node = dict(zip(node_mapping_df['new_node_idx'], node_mapping_df['node_identifier']))
        relation_to_idx = dict(zip(relation_mapping_df['relation_name'], relation_mapping_df['relation_idx']))
        idx_to_relation = dict(zip(relation_mapping_df['relation_idx'], relation_mapping_df['relation_name']))

        # Prepare graph structure
        edge_index = torch.tensor(training_triplets_df[['head_idx', 'tail_idx']].values.T, dtype=torch.long)
        edge_type = torch.tensor(training_triplets_df['relation_idx'].values, dtype=torch.long)
        labeled_nodes_and_labels = torch.tensor(ceramic_labels_df[['new_ceramic_node_idx', 'label_id']].values, dtype=torch.long)

        # NEW: Compute node types based on connectivity patterns
        num_nodes = stats['total_nodes_in_graph']
        node_types = compute_node_types(edge_index, num_nodes)

        # Extract hyperparameters with defaults
        hyperparams = {
            'num_nodes': stats['total_nodes_in_graph'],
            'num_relations': stats['total_relations'],
            'embedding_dim': stats['final_embedding_dim'],
            'num_classes': stats['classification_num_classes'],
            'rgcn_hidden_dim': stats.get('rgcn_hidden_dim', 64),
            'num_bases': stats.get('num_bases', 6),
            'num_rgcn_layers': stats.get('num_rgcn_layers', 2),
            'dropout': stats.get('dropout', 0.2),
            'learning_rate': stats.get('learning_rate', 5e-5),
            'patience': stats.get('patience', 20),
            'l2_reg': stats.get('l2_reg', 1e-4)
        }

        print(f"Loaded data: {hyperparams['num_nodes']} nodes, {hyperparams['num_relations']} relations, "
              f"{edge_index.shape[1]} edges, {len(labeled_nodes_and_labels)} labeled nodes, "
              f"{hyperparams['num_classes']} classes.")

        return {
            **hyperparams,
            'initial_node_embeddings': node_embeddings,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'node_types': node_types,  # NEW: Add node types
            'labeled_nodes_and_labels': labeled_nodes_and_labels,
            'label_to_category_id': dict(zip(label_mapping_df['label_id'], label_mapping_df['original_root_category_id'])),
            'label_to_category_name': dict(zip(label_mapping_df['label_id'], label_mapping_df['root_category_name'])),
            'stats': stats
        }

    except Exception as e:
        print(f"❌ Error loading classification data from {output_dir}: {e}")
        traceback.print_exc()
        return None


def compute_node_types(edge_index, num_nodes):
    """
    Compute node types based on connectivity patterns.
    
    Args:
        edge_index: Edge index tensor of shape [2, num_edges]
        num_nodes: Total number of nodes
        
    Returns:
        torch.Tensor: Node types tensor of shape [num_nodes]
            0: nodes with only outgoing edges
            1: nodes with only incoming edges  
            2: nodes with both incoming and outgoing edges
            3: isolated nodes (no edges)
    """
    # Count incoming and outgoing edges for each node
    out_degree = torch.bincount(edge_index[0], minlength=num_nodes)
    in_degree = torch.bincount(edge_index[1], minlength=num_nodes)
    
    # Determine node types
    node_types = torch.zeros(num_nodes, dtype=torch.long)
    
    # Type 0: Only outgoing edges
    node_types[(out_degree > 0) & (in_degree == 0)] = 0
    
    # Type 1: Only incoming edges
    node_types[(out_degree == 0) & (in_degree > 0)] = 1
    
    # Type 2: Both incoming and outgoing edges
    node_types[(out_degree > 0) & (in_degree > 0)] = 2
    
    # Type 3: Isolated nodes (no edges)
    node_types[(out_degree == 0) & (in_degree == 0)] = 3
    
    # Print node type distribution
    unique_types, counts = torch.unique(node_types, return_counts=True)
    print(f"\nNode type distribution:")
    type_names = ["Only outgoing", "Only incoming", "Bidirectional", "Isolated"]
    for node_type, count in zip(unique_types, counts):
        print(f"  Type {node_type} ({type_names[node_type]}): {count}")
    
    return node_types


def _create_data_loaders(labeled_nodes_and_labels, study_name, train_ratio=0.7, val_ratio=0.15, 
                        test_ratio=0.15, batch_size=512, random_state=42):
    """
    Create proper train/val/test data loaders with no data leakage
    """
    print(f"Creating data loaders for {study_name}")
    
    if labeled_nodes_and_labels is None or len(labeled_nodes_and_labels) == 0:
        print("No labeled data available")
        return None, None, None
    
    # Convert to tensor if needed
    if isinstance(labeled_nodes_and_labels, np.ndarray):
        labeled_nodes_and_labels = torch.tensor(labeled_nodes_and_labels, dtype=torch.long)
    
    total_samples = labeled_nodes_and_labels.shape[0]
    print(f"Total labeled samples: {total_samples}")
    
    # Check if we have enough samples for splitting
    if total_samples < 10:
        print("Warning: Very few samples available. Using all for training.")
        train_dataset = TensorDataset(labeled_nodes_and_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        return train_loader, None, None
    
    # First split: separate test set
    train_val_data, test_data = train_test_split(
        labeled_nodes_and_labels, 
        test_size=test_ratio, 
        random_state=random_state,
        stratify=labeled_nodes_and_labels[:, 1] if total_samples > labeled_nodes_and_labels[:, 1].unique().shape[0] else None
    )
    
    # Second split: separate train and validation
    if train_val_data.shape[0] > 5:  # Only split if we have enough samples
        train_data, val_data = train_test_split(
            train_val_data,
            test_size=val_ratio / (train_ratio + val_ratio),  # Adjust ratio for remaining data
            random_state=random_state,
            stratify=train_val_data[:, 1] if train_val_data.shape[0] > train_val_data[:, 1].unique().shape[0] else None
        )
    else:
        train_data = train_val_data
        val_data = None
    
    print(f"Data split - Train: {train_data.shape[0]}, Val: {val_data.shape[0] if val_data is not None else 0}, Test: {test_data.shape[0]}")
    
    # Create datasets and loaders
    train_dataset = TensorDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_loader = None
    if val_data is not None and val_data.shape[0] > 0:
        val_dataset = TensorDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    test_loader = None
    if test_data is not None and test_data.shape[0] > 0:
        test_dataset = TensorDataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader


def _setup_trainer(study_name, base_data_dir, patience, trainer_params=None):
    """Setup PyTorch Lightning trainer with proper callbacks"""
    
    # Create log directory
    log_dir = os.path.join(base_data_dir, "lightning_logs", study_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=base_data_dir,
        name="lightning_logs",
        version=study_name
    )
    
    # Callbacks
    callbacks = []
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        verbose=True,
        mode='min',
        strict=True
    )
    callbacks.append(early_stop_callback)
    
    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename='{epoch}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )
    callbacks.append(checkpoint_callback)
    
    # Default trainer parameters
    default_trainer_params = {
        'max_epochs': 100,
        'accelerator': 'gpu' if torch.cuda.is_available() else 'cpu',
        'devices': 1,
        'logger': logger,
        'callbacks': callbacks,
        'enable_progress_bar': True,
        'log_every_n_steps': 10,
        'check_val_every_n_epoch': 1,
        'gradient_clip_val': 1.0,  # Add gradient clipping
        'deterministic': True,  # For reproducibility
    }
    
    # Override with user parameters
    if trainer_params:
        default_trainer_params.update(trainer_params)
    
    return pl.Trainer(**default_trainer_params)

def debug_model_and_data(model, train_dataloader):
    """Quick debugging function to check model and data"""
    
    print("=== MODEL DEBUG ===")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check parameter initialization
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}")
    
    print("\n=== DATA DEBUG ===")
    # Check a few batches
    for i, batch in enumerate(train_dataloader):
        if i >= 3:  # Only check first 3 batches
            break
            
        if isinstance(batch, list):
            batch = batch[0]
            
        node_indices = batch[:, 0]
        labels = batch[:, 1]
        
        print(f"Batch {i}:")
        print(f"  Node indices range: {node_indices.min()}-{node_indices.max()}")
        print(f"  Labels: {torch.unique(labels)}")
        print(f"  Batch size: {len(batch)}")
        
        # Check if indices are valid
        if node_indices.max() >= model.num_nodes:
            print(f"  WARNING: Node index {node_indices.max()} >= num_nodes {model.num_nodes}")
    
    print("\n=== FORWARD PASS DEBUG ===")
    model.eval()
    with torch.no_grad():
        try:
            embeddings = model()
            print(f"Forward pass successful. Output shape: {embeddings.shape}")
            print(f"Output stats: mean={embeddings.mean():.4f}, std={embeddings.std():.4f}")
            
            # Test classification head with first batch
            batch = next(iter(train_dataloader))
            if isinstance(batch, list):
                batch = batch[0]
            
            node_indices = batch[:5, 0]  # First 5 samples
            sample_embeddings = embeddings[node_indices]
            logits = model.classification_head(sample_embeddings)
            
            print(f"Classification head test:")
            print(f"  Input shape: {sample_embeddings.shape}")
            print(f"  Output shape: {logits.shape}")
            print(f"  Logits: {logits}")
            
        except Exception as e:
            print(f"Forward pass failed: {e}")
    
    model.train()

# Usage in your training script:


def train_and_test_classification_model(study_name, base_data_dir="/kaggle/working/", 
                                      hyperparameters=None, trainer_params=None, plot_history=True):
    """
    Train and test an RGCN classification model for a given study.
    
    Args:
        study_name (str): Name of the study to process
        base_data_dir (str): Base directory containing study data
        hyperparameters (dict, optional): Model hyperparameters to override defaults
        trainer_params (dict, optional): PyTorch Lightning trainer parameters
        plot_history (bool): Whether to plot training history
        
    Returns:
        dict: Results including test metrics, log directory, and checkpoint path, or None if training fails
    """
    print(f"\n>>> Training and Testing Classification Model for study: {study_name} <<<")

    # Load classification data
    data_dir_path = os.path.join(base_data_dir, f"{study_name}_root_classification_data")
    print(f"Loading classification data from: {data_dir_path}")
    
    classification_data = load_classification_data(data_dir_path)
    if classification_data is None:
        print(f"❌ Failed to load classification data for '{study_name}'.")
        return None

    # Merge hyperparameters
    params = {k: v for k, v in classification_data.items() 
              if k in ['num_nodes', 'num_relations', 'embedding_dim', 'num_classes', 
                      'rgcn_hidden_dim', 'num_rgcn_layers', 'num_bases', 'dropout', 
                      'learning_rate', 'patience', 'l2_reg']}
    
    if hyperparameters:
        print("Overriding hyperparameters:")
        for k, v in hyperparameters.items():
            if k in params:
                print(f"  {k}: {params[k]} -> {v}")
                params[k] = v
            else:
                # Allow adding new parameters
                params[k] = v
                print(f"  Adding {k}: {v}")

    # Ensure num_bases is properly converted
    if params.get('num_bases') is not None:
        try:
            params['num_bases'] = int(params['num_bases'])
        except (ValueError, TypeError):
            print(f"Warning: num_bases conversion failed, setting to None.")
            params['num_bases'] = None

    # Create data loaders with proper splitting
    train_loader, val_loader, test_loader = _create_data_loaders(
        classification_data['labeled_nodes_and_labels'], 
        study_name,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=512,
        random_state=42
    )
    
    if train_loader is None:
        print(f"Could not create training data loader for '{study_name}'.")
        return None

    # Initialize model
    print("Instantiating RGCNClassifier model...")
    try:
        model = RGCNClassifier(
            num_nodes=params['num_nodes'],
            num_relations=params['num_relations'],
            embedding_dim=params['embedding_dim'],
            num_classes=params['num_classes'],
            num_bases=params['num_bases'],
            rgcn_hidden_dim=params['rgcn_hidden_dim'],
            num_rgcn_layers=params['num_rgcn_layers'],
            dropout=params['dropout'],
            learning_rate=params['learning_rate'],
            patience=params['patience'],
            l2_reg=params['l2_reg'],
            node_types=classification_data.get('node_types'),
            initial_node_embeddings=classification_data['initial_node_embeddings'],
            edge_index=classification_data['edge_index'],
            edge_type=classification_data['edge_type'],
            add_inverse_relations=params.get('add_inverse_relations', True)  # Default to True
        )
        print("Model instantiated successfully.")
    except Exception as e:
        print(f"❌ Error instantiating RGCNClassifier for '{study_name}': {e}")
        traceback.print_exc()
        return None

    # Setup trainer
    trainer = _setup_trainer(study_name, base_data_dir, params['patience'], trainer_params)
    print("PyTorch Lightning Trainer initialized.")

    # Train model
    print("Starting model training...")
    try:
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        print("Training completed.")
        
        best_model_path = getattr(trainer.checkpoint_callback, 'best_model_path', None)
        if best_model_path:
            print(f"Best model checkpoint: {best_model_path}")
    except Exception as e:
        print(f"❌ Error during training for '{study_name}': {e}")
        traceback.print_exc()
        return None

    # Test model
    final_test_metrics = {}
    if test_loader:
        print("Starting model testing...")
        try:
            if best_model_path and os.path.exists(best_model_path):
                print(f"Loading best model from checkpoint for testing.")
                test_results = trainer.test(ckpt_path='best', dataloaders=test_loader)
            else:
                print("Testing current model state.")
                test_results = trainer.test(model, dataloaders=test_loader)

            if test_results and isinstance(test_results, list) and len(test_results) > 0:
                final_test_metrics = test_results[0]
                print("Final Test Metrics:")
                for metric, value in final_test_metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            print("Testing completed.")
        except Exception as e:
            print(f"❌ Error during testing for '{study_name}': {e}")
            traceback.print_exc()
    else:
        print("No test data available, skipping testing.")

    # Plot training history
    if plot_history:
        print("Plotting training history...")
        try:
            plot_training_history(trainer.logger.log_dir, study_name)
        except Exception as e:
            print(f"Warning: Could not plot training history: {e}")

    # Return results
    results_info = {
        'study_name': study_name,
        'test_metrics': final_test_metrics,
        'log_dir': trainer.logger.log_dir,
        'best_checkpoint_path': best_model_path,
        'model': model  # Include model for further analysis if needed
    }

    print(f"\n>>> Finished processing study: {study_name} <<<")
    return results_info

