import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import RGCNConv
from torchmetrics import Accuracy, F1Score, Precision, Recall
import torch.optim as optim
import warnings

class RGCNClassifier(pl.LightningModule):
    def __init__(self, num_nodes, num_relations, embedding_dim, num_classes, num_bases,
                 rgcn_hidden_dim, num_rgcn_layers=2, dropout=0.2, patience=10, l2_reg=0.0001, 
                 learning_rate=1e-3, node_types=None, initial_node_embeddings=None, 
                 edge_index=None, edge_type=None, add_inverse_relations=True):
        """
        RGCN with proper inverse relation handling
        
        Args:
            add_inverse_relations: If True, automatically adds inverse relations
        """
        super().__init__()
        self.save_hyperparameters(ignore=['node_types', 'edge_index', 'edge_type'])
        
        # Store node types
        self.node_types = node_types
        self.add_inverse_relations = add_inverse_relations
        
        # If adding inverse relations, double the number of relations
        self.num_base_relations = num_relations
        self.total_relations = num_relations * 2 if add_inverse_relations else num_relations
        
        self.num_nodes = self.hparams.num_nodes
        self.embedding_dim = self.hparams.embedding_dim
        self.num_classes = self.hparams.num_classes
        self.rgcn_hidden_dim = self.hparams.rgcn_hidden_dim
        self.num_rgcn_layers = self.hparams.num_rgcn_layers
        self.num_bases = self.hparams.num_bases
        self.dropout = self.hparams.dropout
        self.learning_rate = self.hparams.learning_rate
        self.l2_reg = self.hparams.l2_reg
        self.patience = self.hparams.patience
        
        print(f"Initializing RGCNClassifier:")
        print(f"  - Nodes: {self.num_nodes}")
        print(f"  - Base relations: {self.num_base_relations}")
        print(f"  - Total relations (with inverses): {self.total_relations}")
        print(f"  - Embedding dim: {self.embedding_dim}")
        print(f"  - Classes: {self.num_classes}")
        
        if edge_index is not None:
            print(f"  - Original edges: {edge_index.shape[1]}")
            print(f"  - Edge types: {torch.unique(edge_type).tolist()}")
        
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
            print(f"Initializing node embeddings with Xavier uniform distribution")
            self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim)
            nn.init.xavier_uniform_(self.node_emb.weight, gain=1.0)

        # Build RGCN layers
        self.rgcn_layers = nn.ModuleList()
        self._build_rgcn_layers()

        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Classification head
        final_rgcn_dim = self.rgcn_hidden_dim
        self.classification_head = nn.Sequential(
            nn.LayerNorm(final_rgcn_dim),  
            nn.Linear(final_rgcn_dim, final_rgcn_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_rgcn_dim // 2, self.num_classes)
        )

        # Initialize classification head
        for module in self.classification_head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        
        self.loss_fn = nn.CrossEntropyLoss()

        # Process and store graph data
        if edge_index is not None and edge_type is not None:
            processed_edge_index, processed_edge_type = self._process_graph_data(edge_index, edge_type)
            self._initial_edge_index = processed_edge_index
            self._initial_edge_type = processed_edge_type
        else:
            self._initial_edge_index = None
            self._initial_edge_type = None
        
        # Initialize graph buffers as None
        self.register_buffer('graph_edge_index', None)
        self.register_buffer('graph_edge_type', None)

    def _process_graph_data(self, edge_index, edge_type):
        """Process graph data to add inverse relations if needed"""
        if not self.add_inverse_relations:
            return edge_index, edge_type
        
        print("Adding inverse relations...")
        
        # Original edges
        original_edges = edge_index
        original_types = edge_type
        
        # Create inverse edges (swap source and target)
        inverse_edges = torch.stack([original_edges[1], original_edges[0]], dim=0)
        # Inverse relation types: add num_base_relations to original types
        inverse_types = original_types + self.num_base_relations
        
        # Combine original and inverse
        combined_edge_index = torch.cat([original_edges, inverse_edges], dim=1)
        combined_edge_type = torch.cat([original_types, inverse_types], dim=0)
        
        print(f"  - Original edges: {original_edges.shape[1]}")
        print(f"  - After adding inverses: {combined_edge_index.shape[1]}")
        print(f"  - Original relation types: {torch.unique(original_types).tolist()}")
        print(f"  - All relation types: {torch.unique(combined_edge_type).tolist()}")
        
        return combined_edge_index, combined_edge_type

    def _build_rgcn_layers(self):
        """Build standard RGCN layers"""
        input_dim = self.embedding_dim
        
        for i in range(self.num_rgcn_layers):
            if i == self.num_rgcn_layers - 1:
                output_dim = self.rgcn_hidden_dim
            else:
                # Gradually reduce dimensions
                output_dim = max(self.rgcn_hidden_dim, input_dim // 2)
            
            self.rgcn_layers.append(
                RGCNConv(
                    input_dim,
                    output_dim, 
                    self.total_relations,
                    num_bases=self.num_bases
                )
            )
            print(f"RGCN Layer {i}: {input_dim} → {output_dim} (relations: {self.total_relations})")
            input_dim = output_dim

    def forward(self, edge_index=None, edge_type=None):
        """Forward pass"""
        # Setup graph buffers if needed
        if not hasattr(self, 'graph_edge_index') or self.graph_edge_index is None:
            if hasattr(self, '_initial_edge_index') and self._initial_edge_index is not None:
                self.setup(stage='fit')
            else:
                raise RuntimeError("Graph structure not available.")
        
        x = self.node_emb.weight
        
        # Apply RGCN layers
        for i, layer in enumerate(self.rgcn_layers):
            x = layer(x, self.graph_edge_index, self.graph_edge_type)
            
            if i < len(self.rgcn_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layer(x)
        
        return x
    
    def setup(self, stage=None):
        """Setup method called by PyTorch Lightning"""
        if self._initial_edge_index is not None and self._initial_edge_type is not None:
            self.graph_edge_index = self._initial_edge_index.to(self.device)
            self.graph_edge_type = self._initial_edge_type.to(self.device)
            
            # Move node types to device if available
            if self.node_types is not None:
                self.node_types = self.node_types.to(self.device)
            
            # Clear initial data to save memory
            self._initial_edge_index = None
            self._initial_edge_type = None
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'test')
    
    def _shared_step(self, batch, batch_idx, stage):
        """Shared step logic for train/val/test"""
        # Handle batch format
        if isinstance(batch, list) and len(batch) == 1 and isinstance(batch[0], torch.Tensor):
            batch = batch[0]
        elif not isinstance(batch, torch.Tensor):
            raise ValueError(f"Unexpected batch type in {stage}_step: {type(batch)}")

        if batch.numel() == 0:
            warnings.warn(f"{stage.capitalize()} batch {batch_idx} is empty. Skipping.", UserWarning)
            return None

        labeled_node_indices = batch[:, 0].long()
        labels = batch[:, 1].long()

        # Forward pass
        try:
            final_node_embeddings = self()
            labeled_node_embeddings = final_node_embeddings[labeled_node_indices]
            logits = self.classification_head(labeled_node_embeddings)
            loss = self.loss_fn(logits, labels)
        except Exception as e:
            warnings.warn(f"Error in {stage}_step batch {batch_idx}: {e}. Skipping batch.", UserWarning)
            return None

        # Logging and metrics
        batch_size = labeled_node_indices.size(0)
        self.log(f'{stage}_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        # Calculate metrics
        preds = torch.argmax(logits, dim=1)
        
        if stage == 'train':
            self.train_acc(preds, labels)
            self.train_f1(preds, labels)
            self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        elif stage == 'val':
            self.val_acc(preds, labels)
            self.val_f1(preds, labels)
            self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        elif stage == 'test':
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
        if self.l2_reg > 0:
            optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.l2_reg)
            print(f"Using AdamW optimizer with weight_decay={self.l2_reg}")
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
            print("Using Adam optimizer without regularization")
            
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