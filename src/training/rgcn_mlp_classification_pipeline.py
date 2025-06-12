import os
import json
import traceback
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
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


def _create_data_loaders(labeled_nodes_and_labels, study_name, train_size=0.7, val_size=0.15, test_size=0.15):
    """
    Creates train, validation, and test data loaders with balanced sampling from each class.
    Takes the specified proportion from EACH CLASS rather than from the total dataset.
    
    Args:
        labeled_nodes_and_labels: Tensor of shape (N, 2) with node indices and labels
        study_name: Name of the study for logging purposes
        train_size, val_size, test_size: Split ratios PER CLASS (should sum to 1.0)
        
    Returns:
        tuple: (train_loader, val_loader, test_loader) or (None, None, None) if split fails
    """
    if labeled_nodes_and_labels.numel() == 0:
        print(f"No labeled nodes found for '{study_name}'.")
        return None, None, None

    labeled_indices = labeled_nodes_and_labels[:, 0].numpy()
    labels = labeled_nodes_and_labels[:, 1].numpy()
    
    # Verify split ratios sum to 1
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        print(f"⚠️  Warning: Split ratios don't sum to 1.0 for '{study_name}' "
              f"(train={train_size}, val={val_size}, test={test_size})")

    try:
        # Get unique classes and their counts
        unique_classes, class_counts = np.unique(labels, return_counts=True)
        print(f"\nClass distribution for '{study_name}':")
        for cls, count in zip(unique_classes, class_counts):
            print(f"  Class {cls}: {count} samples")
        
        # Check if any class has too few samples for the split
        min_samples_needed = max(1, int(1 / min(train_size, val_size, test_size)))
        insufficient_classes = class_counts < min_samples_needed
        if np.any(insufficient_classes):
            problematic_classes = unique_classes[insufficient_classes]
            print(f"❌ Classes {problematic_classes} have insufficient samples for the requested split.")
            return None, None, None

        # Initialize lists for each split
        train_indices, train_labels = [], []
        val_indices, val_labels = [], []
        test_indices, test_labels = [], []

        # Process each class separately
        for class_label in unique_classes:
            # Get all samples for this class
            class_mask = (labels == class_label)
            class_indices = labeled_indices[class_mask]
            class_labels = labels[class_mask]
            
            n_samples = len(class_indices)
            
            # Calculate split sizes for this class
            n_train = max(1, int(n_samples * train_size))
            n_val = max(1, int(n_samples * val_size))
            n_test = n_samples - n_train - n_val  # Remaining samples go to test
            
            # Ensure we don't exceed available samples
            if n_train + n_val > n_samples:
                n_val = max(0, n_samples - n_train)
                n_test = 0
            
            print(f"  Class {class_label}: {n_train} train, {n_val} val, {n_test} test")
            
            # Randomly shuffle indices for this class
            np.random.seed(0)  # For reproducibility
            shuffled_idx = np.random.permutation(len(class_indices))
            
            # Split the shuffled indices
            train_idx = shuffled_idx[:n_train]
            val_idx = shuffled_idx[n_train:n_train + n_val]
            test_idx = shuffled_idx[n_train + n_val:]
            
            # Add to respective splits
            train_indices.extend(class_indices[train_idx])
            train_labels.extend(class_labels[train_idx])
            
            if len(val_idx) > 0:
                val_indices.extend(class_indices[val_idx])
                val_labels.extend(class_labels[val_idx])
            
            if len(test_idx) > 0:
                test_indices.extend(class_indices[test_idx])
                test_labels.extend(class_labels[test_idx])

        # Convert to numpy arrays
        train_indices, train_labels = np.array(train_indices), np.array(train_labels)
        val_indices, val_labels = np.array(val_indices), np.array(val_labels)
        test_indices, test_labels = np.array(test_indices), np.array(test_labels)

        print(f"\n📊 Final Split Summary for '{study_name}':")
        print(f"  Train: {len(train_indices)} nodes")
        print(f"  Validation: {len(val_indices)} nodes")
        print(f"  Test: {len(test_indices)} nodes")
        
        # Show class distribution in each split
        for split_name, split_labels in [('Train', train_labels), ('Val', val_labels), ('Test', test_labels)]:
            if len(split_labels) > 0:
                unique, counts = np.unique(split_labels, return_counts=True)
                class_dist = {cls: count for cls, count in zip(unique, counts)}
                print(f"  {split_name} class distribution: {class_dist}")

        # Create data loaders
        loaders = []
        for indices, labels_arr, name in [(train_indices, train_labels, 'Train'), 
                                         (val_indices, val_labels, 'Validation'), 
                                         (test_indices, test_labels, 'Test')]:
            if len(indices) > 0:
                dataset = TensorDataset(torch.tensor(np.vstack([indices, labels_arr]).T, dtype=torch.long))
                loader = DataLoader(dataset, batch_size=len(indices), shuffle=(name == 'Train'))
                loaders.append(loader)
            else:
                loaders.append(None)
                if name != 'Test':  # Test loader being None is less critical
                    warnings.warn(f"{name} split resulted in 0 nodes for '{study_name}'.", UserWarning)

        return tuple(loaders)

    except Exception as e:
        print(f"❌ Unexpected error during balanced node split for '{study_name}': {e}")
        traceback.print_exc()
        return None, None, None

def _setup_trainer(study_name, base_data_dir, patience, trainer_params=None):
    """
    Sets up PyTorch Lightning trainer with callbacks and logging.
    
    Args:
        study_name: Name of the study
        base_data_dir: Base directory for saving logs and checkpoints
        patience: Early stopping patience
        trainer_params: Optional custom trainer parameters
        
    Returns:
        Trainer: Configured PyTorch Lightning trainer
    """
    log_save_dir = os.path.join(base_data_dir, f'{study_name}_logs')
    os.makedirs(log_save_dir, exist_ok=True)

    default_params = {
        'max_epochs': 100,
        'accelerator': 'auto',
        'devices': 'auto',
        'enable_progress_bar': True,
        'logger': CSVLogger(save_dir=log_save_dir, name=""),
        'check_val_every_n_epoch': 1,
        'gradient_clip_val': 1.0,  # Add gradient clipping
        'deterministic': True,  # For reproducibility
    }

    if trainer_params:
        print("Overriding trainer parameters:")
        for k, v in trainer_params.items():
            print(f"  {k}: {default_params.get(k, 'N/A')} -> {v}")
        default_params.update(trainer_params)

    # Setup callbacks if not provided
    if 'callbacks' not in default_params or default_params['callbacks'] is None:
        print("Using default EarlyStopping and ModelCheckpoint callbacks.")
        
        checkpoint_dir = os.path.join(base_data_dir, f"{study_name}_root_classification_data", 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.0001,
                patience=patience * 2,
                verbose=True,
                mode='min'
            ),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=checkpoint_dir,
                filename='best-checkpoint',
                save_top_k=1,
                mode='min',
                verbose=True
            )
        ]
        default_params['callbacks'] = callbacks
    else:
        # Ensure ModelCheckpoint exists in custom callbacks
        custom_callbacks = default_params['callbacks']
        if not any(isinstance(cb, ModelCheckpoint) for cb in custom_callbacks):
            print("Adding default ModelCheckpoint as none was provided.")
            checkpoint_dir = os.path.join(base_data_dir, f"{study_name}_root_classification_data", 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint_callback = ModelCheckpoint(
                monitor='val_loss',
                dirpath=checkpoint_dir,
                filename='best-checkpoint',
                save_top_k=1,
                mode='min',
                verbose=True
            )
            custom_callbacks.append(checkpoint_callback)

    return Trainer(**default_params)


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

    # Ensure num_bases is properly converted
    if params.get('num_bases') is not None:
        try:
            params['num_bases'] = int(params['num_bases'])
        except (ValueError, TypeError):
            print(f"Warning: num_bases conversion failed, setting to None.")
            params['num_bases'] = None

    # Create data loaders
    train_loader, val_loader, test_loader = _create_data_loaders(
        classification_data['labeled_nodes_and_labels'], 
        study_name
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
            node_types=classification_data.get('node_types'),  # NEW: Add node_types
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
        #debug_model_and_data(model, train_loader)
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
        plot_training_history(trainer.logger.log_dir, study_name)

    # Return results
    results_info = {
        'study_name': study_name,
        'test_metrics': final_test_metrics,
        'log_dir': trainer.logger.log_dir,
        'best_checkpoint_path': best_model_path
    }

    print(f"\n>>> Finished processing study: {study_name} <<<")
    return results_info