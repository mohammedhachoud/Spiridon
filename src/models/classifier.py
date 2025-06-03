import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torch_geometric.nn import RGCNConv
import numpy as np
from torchmetrics import Accuracy, F1Score, Precision, Recall
import warnings
from ..utils import seed_everything


class RGCNClassifier(pl.LightningModule):
    def __init__(self, num_nodes, num_relations, embedding_dim, num_classes, num_bases,
         rgcn_hidden_dim, num_rgcn_layers=3, dropout=0.2, learning_rate=1e-3, patience=20,
         initial_node_embeddings=None, l2_reg=0, edge_index=None, edge_type=None):

        super().__init__()
        self.save_hyperparameters(ignore=['initial_node_embeddings', 'edge_index', 'edge_type'])
        self.num_nodes = self.hparams.num_nodes
        self.num_relations = num_relations  
        self.total_relations = num_relations * 2  # Forward + Inverse relations
        
        self.embedding_dim = self.hparams.embedding_dim
        self.num_classes = self.hparams.num_classes
        self.rgcn_hidden_dim = self.hparams.rgcn_hidden_dim
        self.num_rgcn_layers = self.hparams.num_rgcn_layers
        self.num_bases = self.hparams.num_bases
        self.dropout = self.hparams.dropout
        self.learning_rate = self.hparams.learning_rate
        self.l2_reg = self.hparams.l2_reg
        self.patience = self.hparams.patience
        
        print(f"Initializing RGCNClassifier with {self.num_nodes} nodes, {self.num_relations} base relations ({self.total_relations} total with inverses), embedding dim {self.embedding_dim}, and {self.num_classes} classes.")
        if edge_index is not None:
            print(f"Graph stats - Nodes: {self.num_nodes}, Edges: {edge_index.shape[1]}")
            print(f"Edge types: {torch.unique(edge_type)}")
            print(f"Node degree stats: {torch.bincount(edge_index[0]).float().mean():.2f}")
        
        # Metrics for classification
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.train_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.val_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_f1 = F1Score(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_precision = Precision(task="multiclass", num_classes=self.num_classes, average='macro')
        self.test_recall = Recall(task="multiclass", num_classes=self.num_classes, average='macro')

        # Node Embeddings 
        if initial_node_embeddings is not None:
            if initial_node_embeddings.shape[0] != self.num_nodes:
                raise ValueError(f"Initial node embedding rows ({initial_node_embeddings.shape[0]}) != specified num_nodes ({self.num_nodes})")
            if initial_node_embeddings.shape[1] != self.embedding_dim:
                raise ValueError(f"Initial node embedding dim ({initial_node_embeddings.shape[1]}) != specified embedding_dim ({self.embedding_dim})")
            self.node_emb = nn.Embedding.from_pretrained(torch.tensor(initial_node_embeddings, dtype=torch.float32), freeze=False)
        else:
            print(f"Initializing node embeddings with Xavier uniform distribution for {self.num_nodes} nodes and embedding dimension {self.embedding_dim}.")
            self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim)
            nn.init.xavier_uniform_(self.node_emb.weight, gain=1.0)

        self.rgcn_layers = nn.ModuleList()

        input_dim = self.embedding_dim
        for i in range(self.num_rgcn_layers):
            if i == self.num_rgcn_layers - 1:
                output_dim = self.rgcn_hidden_dim
            else:
                output_dim = max(self.rgcn_hidden_dim, input_dim // 2) if input_dim > self.rgcn_hidden_dim else self.rgcn_hidden_dim
            
            print(f"RGCN Layer {i}: {input_dim} → {output_dim}")
            # Use total_relations (doubled) for RGCN layers
            self.rgcn_layers.append(RGCNConv(input_dim, output_dim, self.total_relations, num_bases=self.num_bases))
            input_dim = output_dim

        self.dropout_layer = nn.Dropout(self.dropout)

        final_rgcn_dim = self.rgcn_hidden_dim
        
        self.classification_head = nn.Sequential(
            nn.LayerNorm(final_rgcn_dim),  
            nn.Linear(final_rgcn_dim, final_rgcn_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_rgcn_dim // 2, self.num_classes)
        )

        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        
        self.loss_fn = nn.CrossEntropyLoss()

        # Store graph data temporarily
        self._initial_edge_index = edge_index
        self._initial_edge_type = edge_type
        
        # Initialize graph buffers as None
        self.register_buffer('graph_edge_index', None)
        self.register_buffer('graph_edge_type', None)

    def setup(self, stage=None):
        """Setup method called by PyTorch Lightning with stage parameter"""
        if self._initial_edge_index is not None and self._initial_edge_type is not None:
            # Move graph data to device and register as buffers
            self.graph_edge_index = self._initial_edge_index.to(self.device)
            self.graph_edge_type = self._initial_edge_type.to(self.device)
            
            # Clear temporary storage
            self._initial_edge_index = None
            self._initial_edge_type = None

    def forward(self, edge_index=None, edge_type=None):
        """Computes node embeddings using RGCN layers for all nodes in the graph."""
        
        # Ensure setup has been called and buffers exist
        if not hasattr(self, 'graph_edge_index') or self.graph_edge_index is None:
            # If setup hasn't been called yet, call it now with default stage
            if hasattr(self, '_initial_edge_index') and self._initial_edge_index is not None:
                self.setup(stage='fit')  # Provide default stage
            else:
                raise RuntimeError("Graph structure not available. No edge_index provided during initialization.")
        
        if not hasattr(self, 'graph_edge_type') or self.graph_edge_type is None:
            raise RuntimeError("Graph structure not set up properly. graph_edge_type is None.")

        x = self.node_emb.weight
        
        # Use stored graph buffers
        original_edge_index = self.graph_edge_index
        original_edge_type = self.graph_edge_type
        
        # Create reverse edges with inverse relation types
        reverse_edge_index = torch.stack([original_edge_index[1], original_edge_index[0]])
        
        # Create inverse relations: r -> r + num_relations (r-1)
        # relations 0 to num_relations-1 are forward relations
        # relations num_relations to 2*num_relations-1 are their inverses
        reverse_edge_type = original_edge_type + self.num_relations
        
        # Combine original and reverse edges
        undirected_edge_index = torch.cat([original_edge_index, reverse_edge_index], dim=1)
        undirected_edge_type = torch.cat([original_edge_type, reverse_edge_type])
        
        # Apply RGCN layers with undirected edges and inverse relations
        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, undirected_edge_index, undirected_edge_type)
            
            if i < len(self.rgcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)

        return x
    
    def training_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], torch.Tensor):
            batch = batch[0]
        elif not isinstance(batch, torch.Tensor):
             raise ValueError(f"Unexpected batch type in training_step: {type(batch)}")

        if batch.numel() == 0:
             warnings.warn(f"Training batch {batch_idx} is empty. Skipping.", UserWarning)
             return None

        labeled_node_indices = batch[:, 0].long()
        labels = batch[:, 1].long()

        # Get node embeddings from RGCN
        try:
            final_node_embeddings = self()
        except Exception as e:
            warnings.warn(f"Error in forward pass: {e}")
            return None

        # Select embeddings for labeled nodes in this batch
        try:
            labeled_node_embeddings = final_node_embeddings[labeled_node_indices]
        except IndexError as e:
            warnings.warn(f"IndexError in training_step batch {batch_idx}: {e}. Skipping batch.", UserWarning)
            return None
        except RuntimeError as e:
             warnings.warn(f"Runtime error getting final embeddings for batch {batch_idx}: {e}. Skipping batch.", UserWarning)
             return None

        # Pass embeddings through MLP classification head
        logits = self.classification_head(labeled_node_embeddings)

        # Calculate classification loss
        loss = self.loss_fn(logits, labels)

        # Apply L2 regularization if specified
        if self.l2_reg > 0:
            l2_loss = 0
            for param in self.parameters():
                if param.requires_grad:
                    l2_loss += torch.norm(param, p=2)**2
            total_loss = loss + self.l2_reg * l2_loss
        else:
            total_loss = loss

        # Logging
        batch_size = labeled_node_indices.size(0)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train_ce_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.train_f1(preds, labels)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return total_loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], torch.Tensor):
            batch = batch[0]
        elif not isinstance(batch, torch.Tensor):
             raise ValueError(f"Unexpected batch type in validation_step: {type(batch)}")

        if batch.numel() == 0:
             warnings.warn(f"Validation batch {batch_idx} is empty. Skipping.", UserWarning)
             return None

        labeled_node_indices = batch[:, 0].long()
        labels = batch[:, 1].long()

        # Get node embeddings from RGCN
        final_node_embeddings = self()

        # Select embeddings for labeled nodes in this batch
        try:
             labeled_node_embeddings = final_node_embeddings[labeled_node_indices]
        except IndexError as e:
             warnings.warn(f"IndexError in validation_step batch {batch_idx}: {e}. Skipping batch.", UserWarning)
             return None
        except RuntimeError as e:
             warnings.warn(f"Runtime error getting final embeddings for batch {batch_idx}: {e}. Skipping batch.", UserWarning)
             return None

        # Pass through classification head
        logits = self.classification_head(labeled_node_embeddings)
        loss = self.loss_fn(logits, labels)

        # Logging
        batch_size = labeled_node_indices.size(0)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss

    def test_step(self, batch, batch_idx):
        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], torch.Tensor):
            batch = batch[0]
        elif not isinstance(batch, torch.Tensor):
             raise ValueError(f"Unexpected batch type in test_step: {type(batch)}")

        if batch.numel() == 0:
             warnings.warn(f"Test batch {batch_idx} is empty. Skipping.", UserWarning)
             return None

        labeled_node_indices = batch[:, 0].long()
        labels = batch[:, 1].long()

        # Get node embeddings from RGCN
        final_node_embeddings = self()

        # Select embeddings for labeled nodes in this batch
        try:
             labeled_node_embeddings = final_node_embeddings[labeled_node_indices]
        except IndexError as e:
             warnings.warn(f"IndexError in test_step batch {batch_idx}: {e}. Skipping batch.", UserWarning)
             return None
        except RuntimeError as e:
             warnings.warn(f"Runtime error getting final embeddings for batch {batch_idx}: {e}. Skipping batch.", UserWarning)
             return None

        # Pass through classification head
        logits = self.classification_head(labeled_node_embeddings)
        loss = self.loss_fn(logits, labels)

        # Logging
        batch_size = labeled_node_indices.size(0)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Metrics
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.test_precision(preds, labels)
        self.test_recall(preds, labels)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('test_precision', self.test_precision, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('test_recall', self.test_recall, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self):
        # Use AdamW with weight decay instead of manual L2
        if self.l2_reg > 0:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            
        # Monitor validation loss for learning rate scheduling
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min',
            factor=0.5, 
            patience=self.patience // 2,
            verbose=True,
            min_lr=1e-6
        ) 
        return {
            "optimizer": optimizer, 
            "lr_scheduler": {
                "scheduler": scheduler, 
                "monitor": "val_loss",
                "frequency": 1
            }
        }