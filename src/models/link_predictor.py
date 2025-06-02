import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import RGCNConv
import numpy as np
from torchmetrics import Accuracy, F1Score 
import warnings
from ..utils import seed_everything

class RGCNLinkPredictor(pl.LightningModule):
    def __init__(self, num_nodes, num_relations, embedding_dim, num_bases,
                 category_node_indices,
                 root_category_node_indices,
                 cat_idx_to_root_idx_map,
                 all_triplets_dict_btc,
                 evaluation_relation_id,
                 rgcn_hidden_dim, idx_to_node, num_rgcn_layers=1,
                 dropout=0.4, learning_rate=1e-5,
                 initial_node_embeddings=None,
                 l2_reg=0.4,
                 neg_samples_btc_relation=4,
                 # New parameters for similarity-based relation initialization
                 use_similarity_based_relation_init: bool = False,
                 all_triples_for_relation_init: list = None):

        super().__init__()
        
        ignore_list = ['initial_node_embeddings', 'all_triplets_dict_btc',
                       'cat_idx_to_root_idx_map', 'idx_to_node',
                       'category_node_indices', 'root_category_node_indices',
                       'all_triples_for_relation_init'] 
        self.save_hyperparameters(ignore=ignore_list)

        self.idx_to_node = idx_to_node
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim # For node embeddings
        self.rgcn_hidden_dim = rgcn_hidden_dim # For final RGCN layer output and DistMult relation embeddings
        self.num_rgcn_layers = num_rgcn_layers
        self.l2_reg = l2_reg
        self.evaluation_relation_id = evaluation_relation_id
        self.cat_idx_to_root_idx_map = cat_idx_to_root_idx_map if cat_idx_to_root_idx_map else {}
        self.all_triplets_dict_btc = all_triplets_dict_btc if all_triplets_dict_btc else defaultdict(lambda: defaultdict(set))

        if category_node_indices is not None and ( (isinstance(category_node_indices, torch.Tensor) and category_node_indices.numel() > 0) or \
                                            (not isinstance(category_node_indices, torch.Tensor) and category_node_indices) ):
            if not isinstance(category_node_indices, torch.Tensor):
                self.category_node_indices = torch.tensor(list(category_node_indices), dtype=torch.long)
            else:
                self.category_node_indices = category_node_indices.long() # Ensure it's a long tensor
        else:
            self.category_node_indices = torch.empty(0, dtype=torch.long)
        
        # Handle root_category_node_indices
        if root_category_node_indices is not None and ( (isinstance(root_category_node_indices, torch.Tensor) and root_category_node_indices.numel() > 0) or \
                                                        (not isinstance(root_category_node_indices, torch.Tensor) and root_category_node_indices) ):
            if not isinstance(root_category_node_indices, torch.Tensor):
                self.root_category_node_indices = torch.tensor(list(root_category_node_indices), dtype=torch.long)
            else:
                self.root_category_node_indices = root_category_node_indices.long() # Ensure it's a long tensor
        else:
            self.root_category_node_indices = torch.empty(0, dtype=torch.long)

        
        self.test_predictions = []

        # (Your existing warnings about BTC, categories, etc.)
        if self.evaluation_relation_id is None:
            warnings.warn("evaluation_relation_id (BTC relation) is None. Specific BTC logic might fail or be inactive.")
        if not self.category_node_indices.numel() > 0 :
            warnings.warn("category_node_indices is empty. Evaluation against specific categories might not be possible.")
        if not self.root_category_node_indices.numel() > 0 or not self.cat_idx_to_root_idx_map:
            warnings.warn("Root category information (root_category_node_indices or cat_idx_to_root_idx_map) is missing or empty. Negative sampling for BTC and root-level evaluation might be affected.")


        # Node Embedding Initialization
        if initial_node_embeddings is not None:
            print("Initializing node embeddings from pre-computed values.")
            if not isinstance(initial_node_embeddings, torch.Tensor):
                initial_node_embeddings = torch.tensor(initial_node_embeddings, dtype=torch.float)
            if initial_node_embeddings.shape[0] != self.num_nodes:
                 raise ValueError(f"Initial node embeddings count mismatch: expected {self.num_nodes}, got {initial_node_embeddings.shape[0]}")
            if initial_node_embeddings.shape[1] != self.embedding_dim:
                 raise ValueError(f"Initial node embeddings dimension mismatch: expected {self.embedding_dim}, got {initial_node_embeddings.shape[1]}")
            self.node_emb = nn.Embedding.from_pretrained(initial_node_embeddings, freeze=False)
        else:
            print(f"No pre-computed node embeddings provided. Initializing node embeddings randomly (dim: {self.embedding_dim}).")
            self.node_emb = nn.Embedding(self.num_nodes, self.embedding_dim)
            nn.init.xavier_uniform_(self.node_emb.weight) 

        # Relation Embedding Initialization
        if self.hparams.use_similarity_based_relation_init:
            print("Attempting Technique 2: Similarity-Based Initialization for relation embeddings.")
            if self.embedding_dim != self.rgcn_hidden_dim:
                raise ValueError(
                    f"For similarity-based relation initialization, node embedding_dim ({self.embedding_dim}) "
                    f"must be equal to relation embedding_dim (rgcn_hidden_dim: {self.rgcn_hidden_dim})."
                )
            if all_triples_for_relation_init is None:
                raise ValueError("`all_triples_for_relation_init` (a list of (s,r,o) tuples) must be provided "
                                 "when `use_similarity_based_relation_init` is True.")
            
            initial_relation_weights = self._similarity_based_relation_init(
                self.node_emb.weight.detach(), 
                all_triples_for_relation_init
            )
            self.relation_emb = nn.Embedding.from_pretrained(initial_relation_weights, freeze=False)
            print(f"Successfully initialized relation embeddings using Technique 2. Dimension: {self.rgcn_hidden_dim}")
        else:
            print(f"Initializing relation embeddings for DistMult with Kaiming normal. Dimension: {self.rgcn_hidden_dim}")
            self.relation_emb = nn.Embedding(self.num_relations, self.rgcn_hidden_dim)
            nn.init.kaiming_normal_(self.relation_emb.weight)

        print(f"Creating {self.num_rgcn_layers} RGCN layers. Initial input: {self.embedding_dim}, Final output: {self.rgcn_hidden_dim}")
        self.rgcn_layers = nn.ModuleList()
        current_in_dim = self.embedding_dim 

        if self.num_rgcn_layers == 1:
            self.rgcn_layers.append(RGCNConv(current_in_dim, self.rgcn_hidden_dim, self.num_relations, num_bases=self.hparams.num_bases))
            print(f"  RGCN Layer 1: {current_in_dim} -> {self.rgcn_hidden_dim}")
        elif self.num_rgcn_layers == 2:
            intermediate_dim_l1 = 64 
            self.rgcn_layers.append(RGCNConv(current_in_dim, intermediate_dim_l1, self.num_relations, num_bases=self.hparams.num_bases))
            print(f"  RGCN Layer 1: {current_in_dim} -> {intermediate_dim_l1}")
            current_in_dim = intermediate_dim_l1
            
            self.rgcn_layers.append(RGCNConv(current_in_dim, self.rgcn_hidden_dim, self.num_relations, num_bases=self.hparams.num_bases))
            print(f"  RGCN Layer 2: {current_in_dim} -> {self.rgcn_hidden_dim}")
        else: # For num_rgcn_layers == 0 or > 2, use original logic where all layers output rgcn_hidden_dim
            if self.num_rgcn_layers > 0 :
                 print(f"  Using generic RGCN layer structure for {self.num_rgcn_layers} layers. All intermediate and final layers outputting {self.rgcn_hidden_dim}.")
            for i in range(self.num_rgcn_layers):
                out_dim = self.rgcn_hidden_dim 
                self.rgcn_layers.append(RGCNConv(current_in_dim, out_dim, self.num_relations, num_bases=self.hparams.num_bases))
                print(f"  RGCN Layer {i+1}: {current_in_dim} -> {out_dim}")
                current_in_dim = out_dim

        self.dropout_layer = nn.Dropout(self.hparams.dropout)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.register_buffer("graph_edge_index", None, persistent=False)
        self.register_buffer("graph_edge_type", None, persistent=False)

    def _similarity_based_relation_init(self, node_embeddings_tensor, triples_list):
        """
        Technique 2: Similarity-Based Initialization (adapted for vector embeddings)
        
        Initializes relation vectors based on cosine similarity patterns
        between entities that participate in each relation.
        Assumes node_embeddings_tensor.shape[1] == self.rgcn_hidden_dim (enforced by a check before calling).
        """
        node_emb_dim = node_embeddings_tensor.shape[1]

        relation_triples_map = defaultdict(list)
        for s, r, o in triples_list:
            relation_triples_map[r].append((s, o))
        
        relation_vectors = []
        
        for rel_id in range(self.num_relations):
            if rel_id in relation_triples_map:
                pairs = relation_triples_map[rel_id]
                element_similarities = []
                
                for s, o in pairs:
                    if not (0 <= s < node_embeddings_tensor.shape[0] and \
                            0 <= o < node_embeddings_tensor.shape[0]):
                        warnings.warn(f"SimilarityInit: Invalid node index in triple ({s}, {rel_id}, {o}). Max index: {node_embeddings_tensor.shape[0]-1}. Skipping this pair.")
                        continue

                    s_emb = node_embeddings_tensor[s]
                    o_emb = node_embeddings_tensor[o]
                    
                    norm_s = torch.norm(s_emb)
                    norm_o = torch.norm(o_emb)
                    
                    if norm_s < 1e-8 or norm_o < 1e-8: 
                        element_sim = torch.zeros(node_emb_dim, device=node_embeddings_tensor.device)
                    else:
                        element_sim = torch.abs(s_emb * o_emb) / (norm_s * norm_o + 1e-8)
                    
                    element_similarities.append(element_sim)
                
                if element_similarities:
                    avg_similarities = torch.stack(element_similarities).mean(dim=0)
                else: 
                    warnings.warn(f"SimilarityInit: No valid pairs for relation {rel_id}. Initializing with scaled ones.")
                    avg_similarities = torch.ones(node_emb_dim, device=node_embeddings_tensor.device) / \
                                       torch.sqrt(torch.tensor(node_emb_dim, dtype=torch.float, device=node_embeddings_tensor.device))
            else: 
                avg_similarities = torch.ones(node_emb_dim, device=node_embeddings_tensor.device) / \
                                   torch.sqrt(torch.tensor(node_emb_dim, dtype=torch.float, device=node_embeddings_tensor.device))
            
            relation_vectors.append(avg_similarities)
        
        if not relation_vectors and self.num_relations > 0:
            raise RuntimeError("SimilarityInit: Failed to generate any relation vectors, but num_relations > 0.")
        if not relation_vectors and self.num_relations == 0:
             return torch.empty((0, self.rgcn_hidden_dim), device=node_embeddings_tensor.device)

        return torch.stack(relation_vectors)

    def setup_graph(self, edge_index, edge_type):
         if not isinstance(edge_index, torch.Tensor): edge_index = torch.tensor(edge_index, dtype=torch.long)
         if not isinstance(edge_type, torch.Tensor): edge_type = torch.tensor(edge_type, dtype=torch.long)
         self.graph_edge_index = edge_index
         self.graph_edge_type = edge_type
         print(f"RGCN Graph (EXCLUDING target relation {self.evaluation_relation_id}) setup with {self.graph_edge_index.shape[1]} edges.")
         if self.graph_edge_index is not None and self.graph_edge_index.numel() == 0 and self.graph_edge_index.shape[0] == 2 and self.graph_edge_index.shape[1] == 0:
             print("RGCN Graph is empty (0 edges). Model will run, but no GNN message passing will occur.")
         elif self.graph_edge_index is None or self.graph_edge_type is None:
             raise ValueError("Graph edge_index or edge_type is None after setup_graph call.")

    def forward(self):
         if self.graph_edge_index is None or self.graph_edge_type is None :
             raise RuntimeError("RGCN Graph structure (edge_index, edge_type) not set up. Call setup_graph with graph data first.")

         graph_edge_index = self.graph_edge_index.to(self.device)
         graph_edge_type = self.graph_edge_type.to(self.device)
         
         x = self.node_emb.weight 

         for i, layer in enumerate(self.rgcn_layers):
             x = layer(x, graph_edge_index, graph_edge_type)
             if i < len(self.rgcn_layers) - 1: # Apply ReLU and Dropout to all but the last layer
                 x = F.relu(x)
                 x = self.dropout_layer(x)
         return x 

    def _distmult_score(self, h_emb, r_emb, t_emb):
        return torch.sum(h_emb * r_emb * t_emb, dim=-1)

    def _calculate_btc_scores(self, head_indices, tail_indices, final_node_embeddings):
         if head_indices.numel() == 0 or tail_indices.numel() == 0:
              return torch.empty(0, device=self.device)
         
         max_node_idx = self.num_nodes -1
         if head_indices.max() > max_node_idx or tail_indices.max() > max_node_idx or \
            head_indices.min() < 0 or tail_indices.min() < 0:
              raise IndexError(f"Node index out of bounds during BTC scoring. Head max: {head_indices.max()}, Tail max: {tail_indices.max()}, Num nodes: {self.num_nodes}")

         h_emb = final_node_embeddings[head_indices]
         
         r_btc_emb_single = self.relation_emb(torch.tensor(self.evaluation_relation_id, device=self.device, dtype=torch.long))
         r_btc_emb_batch = r_btc_emb_single.unsqueeze(0).expand(h_emb.size(0), -1)
         
         t_emb = final_node_embeddings[tail_indices]
         
         return self._distmult_score(h_emb, r_btc_emb_batch, t_emb)

    def _sample_negatives_for_target_relation(self, pos_h_target_rel, pos_r_target_rel, pos_t_target_rel_true_specific, num_neg_samples):
        if pos_h_target_rel.numel() == 0:
            return (torch.empty(0, dtype=torch.long, device=self.device),) * 3, []

        num_target_triplets_in_batch = pos_h_target_rel.size(0)
        neg_h_list, neg_r_list, neg_t_sampled_roots_list = [], [], []
        neg_counts_per_positive = []
        
        if self.root_category_node_indices.numel() == 0 or not self.cat_idx_to_root_idx_map or num_neg_samples == 0:
            neg_counts_per_positive = [0] * num_target_triplets_in_batch
            return (torch.empty(0, dtype=torch.long, device=self.device),) * 3, neg_counts_per_positive

        root_candidates_device = self.root_category_node_indices.to(self.device)

        for i in range(num_target_triplets_in_batch):
            current_h_idx = pos_h_target_rel[i].item()
            true_specific_cat_idx = pos_t_target_rel_true_specific[i].item()
            true_root_idx_val = self.cat_idx_to_root_idx_map.get(true_specific_cat_idx)
            
            actual_num_neg_sampled_for_this_h = 0
            if true_root_idx_val is not None:
                true_root_idx_tensor = torch.tensor(true_root_idx_val, device=self.device, dtype=torch.long)
                
                possible_neg_roots = root_candidates_device[root_candidates_device != true_root_idx_tensor]
                
                if possible_neg_roots.numel() > 0:
                    actual_num_neg_sampled_for_this_h = min(num_neg_samples, possible_neg_roots.numel())
                    perm = torch.randperm(possible_neg_roots.size(0), device=self.device)
                    sampled_neg_roots_for_this_h = possible_neg_roots[perm[:actual_num_neg_sampled_for_this_h]]

                    neg_t_sampled_roots_list.extend(sampled_neg_roots_for_this_h.tolist())
                    neg_h_list.extend([current_h_idx] * actual_num_neg_sampled_for_this_h)
                    neg_r_list.extend([self.evaluation_relation_id] * actual_num_neg_sampled_for_this_h)
            
            neg_counts_per_positive.append(actual_num_neg_sampled_for_this_h)

        if not neg_t_sampled_roots_list:
            return (torch.empty(0, dtype=torch.long, device=self.device),) * 3, neg_counts_per_positive
            
        return (torch.tensor(neg_h_list, dtype=torch.long, device=self.device),
                torch.tensor(neg_r_list, dtype=torch.long, device=self.device),
                torch.tensor(neg_t_sampled_roots_list, dtype=torch.long, device=self.device),
                neg_counts_per_positive)

    def training_step(self, batch, batch_idx):
        if isinstance(batch, list): pos_btc_triplets_tensor = batch[0]
        elif isinstance(batch, torch.Tensor): pos_btc_triplets_tensor = batch
        else: raise ValueError(f"Unexpected batch type: {type(batch)}")
        
        if pos_btc_triplets_tensor.numel() == 0: 
            current_batch_size = 1 
            self.log('train_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            self.log('train_accuracy', torch.tensor(float('nan'), device=self.device), on_step=False, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            return None 
    
        pos_h_btc, pos_r_btc, pos_t_btc_specific = (
            pos_btc_triplets_tensor[:, 0], pos_btc_triplets_tensor[:, 1], pos_btc_triplets_tensor[:, 2]
        )
        current_batch_size = pos_h_btc.size(0) if pos_h_btc.numel() > 0 else 1

        final_node_embeddings = self() 
        
        pos_scores_btc = torch.empty(0, device=self.device)
        if pos_h_btc.numel() > 0:
            pos_scores_btc = self._calculate_btc_scores(pos_h_btc, pos_t_btc_specific, final_node_embeddings)
    
        neg_h_btc_root, _, neg_t_btc_root, neg_counts_per_pos = self._sample_negatives_for_target_relation(
            pos_h_btc, pos_r_btc, pos_t_btc_specific, self.hparams.neg_samples_btc_relation
        )
        
        neg_scores_btc_root = torch.empty(0, device=self.device)
        if neg_h_btc_root.numel() > 0:
            neg_scores_btc_root = self._calculate_btc_scores(neg_h_btc_root, neg_t_btc_root, final_node_embeddings)
    
        if pos_scores_btc.numel() == 0 and neg_scores_btc_root.numel() == 0:
            self.log('train_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            self.log('train_accuracy', torch.tensor(float('nan'), device=self.device), on_step=False, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            return None 
            
        scores_for_loss = torch.cat([pos_scores_btc, neg_scores_btc_root])
        labels_for_loss = torch.cat([torch.ones_like(pos_scores_btc), torch.zeros_like(neg_scores_btc_root)])
        
        if scores_for_loss.numel() == 0: 
            self.log('train_loss', 0.0, on_step=True, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            self.log('train_accuracy', torch.tensor(float('nan'), device=self.device), on_step=False, on_epoch=True, prog_bar=True, batch_size=current_batch_size)
            return None

        loss = self.loss_fn(scores_for_loss, labels_for_loss)
        total_loss = loss
        
        train_accuracy_local_hits1 = torch.tensor(float('nan'), device=self.device)
        if pos_scores_btc.numel() > 0 and self.hparams.neg_samples_btc_relation > 0 and neg_counts_per_pos:
            num_hits_train = 0
            num_comparisons_made = 0
            
            current_neg_score_idx = 0
            for i in range(pos_scores_btc.size(0)):
                num_neg_for_this_pos = neg_counts_per_pos[i]
                
                if num_neg_for_this_pos > 0: 
                    num_comparisons_made += 1
                    current_pos_score = pos_scores_btc[i]
                    
                    if neg_scores_btc_root.numel() > 0 :
                        end_idx = current_neg_score_idx + num_neg_for_this_pos
                        if end_idx <= neg_scores_btc_root.size(0): 
                            current_neg_block = neg_scores_btc_root[current_neg_score_idx : end_idx]
                            if current_neg_block.numel() > 0: 
                                if current_pos_score > current_neg_block.max():
                                    num_hits_train += 1
                    
                    current_neg_score_idx += num_neg_for_this_pos
            
            if num_comparisons_made > 0:
                train_accuracy_local_hits1 = torch.tensor(num_hits_train / num_comparisons_made, device=self.device)
        
        if self.l2_reg > 0 and scores_for_loss.numel() > 0: 
            l2_penalty_val = torch.tensor(0.0, device=self.device)
            try:
                involved_node_indices_list = []
                if pos_h_btc.numel() > 0: involved_node_indices_list.extend(pos_h_btc.tolist() + pos_t_btc_specific.tolist())
                if neg_h_btc_root.numel() > 0: involved_node_indices_list.extend(neg_h_btc_root.tolist() + neg_t_btc_root.tolist())
                
                if involved_node_indices_list:
                    unique_involved_node_indices = torch.tensor(list(set(involved_node_indices_list)), dtype=torch.long, device=self.device)
                    valid_node_mask = (unique_involved_node_indices >= 0) & (unique_involved_node_indices < self.num_nodes)
                    if valid_node_mask.any():
                         l2_penalty_val += torch.norm(final_node_embeddings[unique_involved_node_indices[valid_node_mask]], p=2)**2
    
                btc_rel_id_tensor = torch.tensor([self.evaluation_relation_id], dtype=torch.long, device=self.device)
                l2_penalty_val += torch.norm(self.relation_emb(btc_rel_id_tensor), p=2)**2
                
                total_loss = total_loss + (self.l2_reg * l2_penalty_val) / scores_for_loss.size(0) 
            except IndexError as ie: 
                warnings.warn(f"IndexError during L2 regularization calculation: {ie}. Using BCE loss only for this step."); total_loss = loss
            except Exception as e: 
                warnings.warn(f"Generic error during L2 regularization calculation: {e}. Using BCE loss only for this step."); total_loss = loss
    
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=current_batch_size)
        self.log('train_accuracy', train_accuracy_local_hits1, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=current_batch_size)
        self.log('train_bce_loss_btc', loss, on_step=False, on_epoch=True, logger=True, batch_size=current_batch_size) 
        self.log('num_pos_btc_train', float(pos_scores_btc.numel()), on_step=False, on_epoch=True, batch_size=current_batch_size)
        self.log('num_neg_btc_train', float(neg_scores_btc_root.numel()), on_step=False, on_epoch=True, batch_size=current_batch_size)
        
        return total_loss

    def validation_step(self, batch, batch_idx):
        if isinstance(batch, list): pos_btc_triplets_tensor = batch[0]
        elif isinstance(batch, torch.Tensor): pos_btc_triplets_tensor = batch
        else: raise ValueError(f"Unexpected batch type in validation_step: {type(batch)}")

        current_batch_size = pos_btc_triplets_tensor.size(0) if pos_btc_triplets_tensor.numel() > 0 else 1
        
        val_loss_value = torch.tensor(0.0, device=self.device)
        num_pos_val = 0
        num_neg_val = 0

        if pos_btc_triplets_tensor.numel() > 0:
            pos_h_btc, pos_r_btc, pos_t_btc_specific = (
                pos_btc_triplets_tensor[:, 0], pos_btc_triplets_tensor[:, 1], pos_btc_triplets_tensor[:, 2]
            )
            num_pos_val = pos_h_btc.size(0)

            final_node_embeddings = self() 
            
            pos_scores_btc_val = torch.empty(0, device=self.device)
            if pos_h_btc.numel() > 0:
                 pos_scores_btc_val = self._calculate_btc_scores(pos_h_btc, pos_t_btc_specific, final_node_embeddings)
        
            neg_h_btc_root_val, _, neg_t_btc_root_val, _ = self._sample_negatives_for_target_relation(
                pos_h_btc, pos_r_btc, pos_t_btc_specific, self.hparams.neg_samples_btc_relation
            )
            num_neg_val = neg_h_btc_root_val.size(0)
            
            neg_scores_btc_root_val = torch.empty(0, device=self.device)
            if neg_h_btc_root_val.numel() > 0:
                neg_scores_btc_root_val = self._calculate_btc_scores(neg_h_btc_root_val, neg_t_btc_root_val, final_node_embeddings)
        
            if pos_scores_btc_val.numel() > 0 or neg_scores_btc_root_val.numel() > 0:
                scores_for_loss_val = torch.cat([pos_scores_btc_val, neg_scores_btc_root_val])
                labels_for_loss_val = torch.cat([torch.ones_like(pos_scores_btc_val), torch.zeros_like(neg_scores_btc_root_val)])
                
                if scores_for_loss_val.numel() > 0: 
                    val_loss_value = self.loss_fn(scores_for_loss_val, labels_for_loss_val)
            else: 
                val_loss_value = torch.tensor(0.0, device=self.device) 
        
        self.log('val_loss', val_loss_value, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=current_batch_size, sync_dist=True)
        self.log('num_pos_btc_val', float(num_pos_val), on_step=False, on_epoch=True, logger=True, batch_size=current_batch_size, sync_dist=True)
        self.log('num_neg_btc_val', float(num_neg_val), on_step=False, on_epoch=True, logger=True, batch_size=current_batch_size, sync_dist=True)

        self._evaluate_step(batch, batch_idx, step_name="val")

    def _evaluate_step(self, batch, batch_idx, step_name="test"):
        if isinstance(batch, list): triplets_tensor = batch[0]
        elif isinstance(batch, torch.Tensor): triplets_tensor = batch
        else: raise ValueError(f"Unexpected batch type in _evaluate_step: {type(batch)}")
    
        y_true_specific_tails = []
        y_pred_top1_specific_tails = []
        ranks_for_mrr = []
        batch_results_for_csv = []
    
        nan_metrics = {
            f'{step_name}_accuracy': torch.tensor(float('nan'), device=self.device),
            f'{step_name}_precision_macro': torch.tensor(float('nan'), device=self.device),
            f'{step_name}_recall_macro': torch.tensor(float('nan'), device=self.device),
            f'{step_name}_f1_macro': torch.tensor(float('nan'), device=self.device),
            f'{step_name}_mrr': torch.tensor(float('nan'), device=self.device)
        }
    
        if triplets_tensor.numel() == 0:
            self.log_dict(nan_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return nan_metrics, []
    
        eval_h_all, eval_r_all, eval_t_true_specific_all = triplets_tensor[:, 0], triplets_tensor[:, 1], triplets_tensor[:, 2]
        
        mask_target_relation = (eval_r_all == self.evaluation_relation_id)
        eval_h = eval_h_all[mask_target_relation]
        eval_t_true_specific = eval_t_true_specific_all[mask_target_relation]
    
        if eval_h.numel() == 0:
            self.log_dict(nan_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return nan_metrics, []
            
        if self.category_node_indices.numel() == 0:
            self.log_dict(nan_metrics, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            return nan_metrics, []
    
        final_node_embeddings = self() 
        
        candidate_specific_tails_device = self.category_node_indices.to(self.device)
        num_candidate_specific_tails = candidate_specific_tails_device.size(0)
    
        for i in range(eval_h.size(0)):
            h_idx = eval_h[i]
            t_true_idx = eval_t_true_specific[i]
            
            h_idx_repeated = h_idx.expand(num_candidate_specific_tails) 
            
            try:
                all_candidate_scores = self._calculate_btc_scores(
                    h_idx_repeated, candidate_specific_tails_device, final_node_embeddings
                )
            except Exception as e:
                warnings.warn(f"Evaluation scoring error for h_idx {h_idx.item()}: {e}. Skipping this item for evaluation."); 
                continue
    
            true_tail_in_candidates_mask = (candidate_specific_tails_device == t_true_idx)
            if not true_tail_in_candidates_mask.any():
                 warnings.warn(f"True tail {t_true_idx.item()} for head {h_idx.item()} not found in candidate_specific_tails_device. Skipping this item for ranking."); 
                 continue
    
            score_of_true_tail = all_candidate_scores[true_tail_in_candidates_mask].squeeze() 
            
            scores_for_ranking = all_candidate_scores.clone() 
            
            if self.all_triplets_dict_btc:
                 known_true_tails_for_h_and_r = self.all_triplets_dict_btc.get(h_idx.item(), {}).get(self.evaluation_relation_id, set())
                 
                 if known_true_tails_for_h_and_r:
                      indices_to_filter_out = [
                          pos for pos, candidate_id_val in enumerate(candidate_specific_tails_device.tolist())
                          if candidate_id_val in known_true_tails_for_h_and_r and candidate_id_val != t_true_idx.item()
                      ]
                      if indices_to_filter_out:
                           scores_for_ranking[torch.tensor(indices_to_filter_out, device=self.device)] = -float('Inf')
    
            current_rank = (scores_for_ranking >= score_of_true_tail).sum().item()
            ranks_for_mrr.append(current_rank)
    
            predicted_top_1_relative_idx = torch.argmax(scores_for_ranking).item()
            predicted_top_1_specific_cat_idx = candidate_specific_tails_device[predicted_top_1_relative_idx].item()
    
            y_true_specific_tails.append(t_true_idx.item())
            y_pred_top1_specific_tails.append(predicted_top_1_specific_cat_idx)
    
            if step_name == "test": 
                true_root_idx_val = self.cat_idx_to_root_idx_map.get(t_true_idx.item())
                pred_root_idx_val = self.cat_idx_to_root_idx_map.get(predicted_top_1_specific_cat_idx)
                
                is_same_root_correct = False
                if true_root_idx_val is not None and pred_root_idx_val is not None and true_root_idx_val == pred_root_idx_val:
                    is_same_root_correct = True
                
                true_node_identifier = self.idx_to_node.get(t_true_idx.item(), f"UnknownNode_{t_true_idx.item()}")
                pred_node_identifier = self.idx_to_node.get(predicted_top_1_specific_cat_idx, f"UnknownNode_{predicted_top_1_specific_cat_idx}")
                true_root_identifier = self.idx_to_node.get(true_root_idx_val, f"UnknownRoot_{true_root_idx_val}") if true_root_idx_val is not None else "N/A"
                pred_root_identifier = self.idx_to_node.get(pred_root_idx_val, f"UnknownRoot_{pred_root_idx_val}") if pred_root_idx_val is not None else "N/A"

                batch_results_for_csv.append({
                     'head_idx': h_idx.item(), 
                     'relation_idx': self.evaluation_relation_id, 
                     'true_tail_idx': t_true_idx.item(),
                     'predicted_tail_idx': predicted_top_1_specific_cat_idx,
                     'rank': current_rank,
                     'true_score': score_of_true_tail.item() if torch.is_tensor(score_of_true_tail) else score_of_true_tail,
                     'predicted_score': scores_for_ranking[predicted_top_1_relative_idx].item(),
                     'true_root_idx': true_root_idx_val,
                     'predicted_root_idx': pred_root_idx_val,
                     'same_root_correct': is_same_root_correct,
                     'true_tail_identifier_specific': true_node_identifier,
                     'predicted_tail_identifier_specific': pred_node_identifier,
                     'true_root_identifier': true_root_identifier,
                     'predicted_root_identifier': pred_root_identifier
                 })
        
        metrics_to_log = {}
        
        if y_true_specific_tails: 
            hits1_count = sum(1 for true_label, pred_label in zip(y_true_specific_tails, y_pred_top1_specific_tails) if true_label == pred_label)
            accuracy_hits1 = hits1_count / len(y_true_specific_tails)
            metrics_to_log[f'{step_name}_accuracy'] = torch.tensor(accuracy_hits1, device=self.device) 
            
            metrics_to_log[f'{step_name}_precision_macro'] = torch.tensor(float('nan'), device=self.device)
            metrics_to_log[f'{step_name}_recall_macro'] = torch.tensor(float('nan'), device=self.device)
            metrics_to_log[f'{step_name}_f1_macro'] = torch.tensor(float('nan'), device=self.device)
    
            if ranks_for_mrr:
                ranks_tensor = torch.tensor(ranks_for_mrr, dtype=torch.float, device=self.device)
                mrr_value = (1.0 / ranks_tensor).mean()
                metrics_to_log[f'{step_name}_mrr'] = mrr_value
            else: 
                metrics_to_log[f'{step_name}_mrr'] = torch.tensor(float('nan'), device=self.device)
        else:
            metrics_to_log.update(nan_metrics) 
    
        for key_template_base in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'mrr']:
            full_key = f'{step_name}_{key_template_base}'
            if full_key not in metrics_to_log: 
                metrics_to_log[full_key] = torch.tensor(float('nan'), device=self.device)
    
        self.log_dict(metrics_to_log, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return metrics_to_log, batch_results_for_csv 

    def test_step(self, batch, batch_idx):
        metrics, batch_csv_data = self._evaluate_step(batch, batch_idx, "test")
        if batch_csv_data: 
             self.test_predictions.extend(batch_csv_data)
      
    def on_test_start(self):
        self.test_predictions = [] 

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
    def predict_category_for_ceramic(self, ceramic_node_id: int, true_category_node_id: int):
        """
        Predicts the category for a given ceramic node ID and compares it against a true category.

        This function assumes the model is already trained and uses the "belongs_to_category"
        relation (self.evaluation_relation_id) for prediction. The prediction is made
        by scoring the ceramic against all candidate categories specified in 
        self.category_node_indices.
        """
        if not hasattr(self, 'graph_edge_index') or self.graph_edge_index is None:
            # This check implies the model might not have its graph structure properly set up,
            # which is essential for the self() call (forward pass through GNN layers).
            raise RuntimeError("RGCN Graph structure (edge_index, edge_type) not set up. "
                               "Call setup_graph() or ensure the model is loaded with its state.")

        if self.category_node_indices is None or self.category_node_indices.numel() == 0:
            raise ValueError("self.category_node_indices is not set or is empty. "
                             "Cannot make category predictions.")

        if self.evaluation_relation_id is None:
            raise ValueError("self.evaluation_relation_id is not set. "
                             "Cannot determine the relation for category prediction.")

        # Ensure model is in evaluation mode for inference
        original_training_state = self.training
        self.eval()

        result_dict = {
            'ceramic_id': ceramic_node_id,
            'ceramic_name': self.idx_to_node.get(ceramic_node_id, str(ceramic_node_id)),
            'true_category_id': true_category_node_id,
            'true_category_name': self.idx_to_node.get(true_category_node_id, str(true_category_node_id)),
            'predicted_category_id': None,
            'predicted_category_name': "N/A",
            'predicted_score': None,
            'score_for_true_category': None,
            'is_correct': None,
            'candidates_count': 0
        }

        with torch.no_grad(): # Disable gradient calculations for inference
            try:
                # 1. Get GNN-processed node embeddings
                final_node_embeddings = self() # Calls the model's forward() method

                # 2. Prepare candidate tails (categories)
                candidate_tails = self.category_node_indices.to(self.device)
                num_candidates = candidate_tails.size(0)
                result_dict['candidates_count'] = num_candidates

                if num_candidates == 0:
                    warnings.warn("No candidate categories available for prediction.")
                    result_dict['error'] = "No candidate categories available."
                    if original_training_state: self.train() # Restore mode
                    return result_dict

                # 3. Prepare head input for scoring
                #    We need to score (ceramic_node_id, evaluation_relation_id, candidate_tail_i) for all i
                head_scalar_tensor = torch.tensor(ceramic_node_id, device=self.device, dtype=torch.long)
                head_indices_for_scoring = head_scalar_tensor.expand(num_candidates) # Repeats head for each candidate

                # 4. Calculate scores using the BTC scoring mechanism
                scores = self._calculate_btc_scores(head_indices_for_scoring, 
                                                    candidate_tails, 
                                                    final_node_embeddings)

                if scores.numel() == 0: # Should not happen if num_candidates > 0
                    warnings.warn(f"Scoring resulted in an empty tensor for ceramic {ceramic_node_id}.")
                    result_dict['error'] = "Scoring returned empty tensor."
                    if original_training_state: self.train() # Restore mode
                    return result_dict
                    
                # 5. Determine predicted category (the one with the highest score)
                best_score_idx = torch.argmax(scores).item()
                predicted_category_id = candidate_tails[best_score_idx].item()
                predicted_score = scores[best_score_idx].item()

                result_dict['predicted_category_id'] = predicted_category_id
                result_dict['predicted_category_name'] = self.idx_to_node.get(predicted_category_id, str(predicted_category_id))
                result_dict['predicted_score'] = predicted_score

                # 6. Compare with the true category
                is_correct = (predicted_category_id == true_category_node_id)
                result_dict['is_correct'] = is_correct

                # 7. Get the score for the provided true category (if it was among candidates)
                true_category_mask = (candidate_tails == true_category_node_id)
                if true_category_mask.any():
                    score_for_true_category = scores[true_category_mask].item()
                    result_dict['score_for_true_category'] = score_for_true_category
                else:
                    # This is an important warning: if the true category isn't even a choice,
                    # the model can't possibly predict it.
                    warnings.warn(f"True category ID {true_category_node_id} for ceramic {ceramic_node_id} "
                                  f"was not found in self.category_node_indices (the set of candidate categories). "
                                  "The 'is_correct' field might be misleading if the true answer wasn't an option.")

            except IndexError as e:
                warnings.warn(f"IndexError during prediction for ceramic {ceramic_node_id}: {e}. "
                              "This might indicate an invalid node_id or an issue with embedding shapes.")
                result_dict['error'] = f"Indexing failed during scoring: {e}"
            except Exception as e: # Catch any other unexpected errors
                warnings.warn(f"An unexpected error occurred during prediction for ceramic {ceramic_node_id}: {e}")
                result_dict['error'] = f"Unexpected error: {e}"

        # Restore the original training state of the model
        if original_training_state:
            self.train()
            
        return result_dict
