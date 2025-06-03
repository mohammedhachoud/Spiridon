# src/data_preparation/format_rgcn_classification_data.py

import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
from collections import defaultdict
from .. import config

# --- Helper: get_level1_root_group ---
def get_level1_root_group(node_str):
    if not isinstance(node_str, str) or not node_str.startswith('Cat_'):
        return None 
    
    if "_Root_" not in node_str:
        parts = node_str.split('_')
        if len(parts) == 2 and parts[0] == 'Cat':
             try:
                  _ = int(float(parts[1])) 
             except (ValueError, TypeError):
                  warnings.warn(f"Could not parse category ID from node string '{node_str}' lacking _Root_.")
                  return None
        else:
             warnings.warn(f"Unexpected category node format lacking _Root_: '{node_str}'")
             return None

    parts = node_str.split("_Root_")
    if len(parts) != 2:
        warnings.warn(f"Unexpected category node format with _Root_: '{node_str}'")
        return None

    try:
        specific_id_part = parts[0] # e.g., "Cat_135"
        root_id_str_from_node = parts[1] # e.g., "135"      
        if specific_id_part.split('_')[1] == root_id_str_from_node:
            return node_str 
        else:
            warnings.warn(f"Node string '{node_str}' does not seem to be a canonical root identifier. Returning as is for label key.")
            return node_str
            
    except Exception as e:
        warnings.warn(f"Error parsing node string '{node_str}' for root group: {e}")
        return None

def adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs):
    if rgcn_data is None:
        print("❌ Input rgcn_data is None. Cannot adapt for classification.")
        return None

    print("\n--- Adapting RGCN Data for Ceramic Classification (Root Labels) ---")

    original_node_to_idx = rgcn_data['node_to_idx']
    original_idx_to_node = rgcn_data['idx_to_node']
    original_num_nodes = rgcn_data['num_nodes']

    # FIX: Build the mapping properly - keep all nodes for classification
    original_idx_to_new_idx = {}
    new_node_to_idx = {}
    
    # Create a 1:1 mapping (no filtering for classification task)
    for original_idx in range(original_num_nodes):
        if original_idx in original_idx_to_node:
            node_identifier = original_idx_to_node[original_idx]
            original_idx_to_new_idx[original_idx] = original_idx
            new_node_to_idx[node_identifier] = original_idx

    print(f"  Original nodes: {original_num_nodes}")
    print(f"  Nodes kept for classification: {len(new_node_to_idx)}")

    original_embeddings = rgcn_data['node_embeddings']
    embedding_dim = rgcn_data['embedding_dim']

    original_training_triplets = rgcn_data['training_triplets']
    relation_to_idx = rgcn_data['relation_to_idx']
    idx_to_relation = rgcn_data['idx_to_relation']
    
    # --- Label Extraction Logic ---
    ceramic_labels = {} 
    root_node_identifier_to_label_id = {} 
    label_id_to_category_id = {} 
    label_id_to_category_name = {}
    label_counter = 0

    # Use ALL "BELONGS_TO_CATEGORY" links for label creation
    all_link_pred_triplets = rgcn_data.get('training_triplets', []) + rgcn_data.get('evaluation_triplets', [])
    
    # FIX: Use the correct evaluation relation name
    EVALUATION_RELATION_NAME = "BELONGS_TO_CATEGORY"
    
    belongs_to_category_rel_id_link_pred = None
    if 'relation_to_idx' in rgcn_data and EVALUATION_RELATION_NAME in rgcn_data['relation_to_idx']:
        belongs_to_category_rel_id_link_pred = rgcn_data['relation_to_idx'][EVALUATION_RELATION_NAME]
    else:
        print(f"  CRITICAL ERROR: Relation '{EVALUATION_RELATION_NAME}' not found. Cannot create labels.")
        print(f"  Available relations: {list(rgcn_data['relation_to_idx'].keys())}")
        return None

    # This map should be: {original_specific_cat_id (int): original_root_cat_id (int)}
    cat_to_root_map_original_ids_str = rgcn_data['stats'].get('cat_to_root_map_original_ids')
    if not cat_to_root_map_original_ids_str:
        print("  CRITICAL ERROR: cat_to_root_map_original_ids not in stats. Cannot create labels.")
        return None
    cat_to_root_map_original_ids = {int(k): int(v) for k,v in cat_to_root_map_original_ids_str.items()}

    tech_cat_name_col = 'cat_name_processed' if 'cat_name_processed' in dfs['tech_cat'].columns else 'cat_name'
    tech_cat_lookup = {}
    if 'tech_cat' in dfs and not dfs['tech_cat'].empty:
        try:
             tech_cat_lookup = dfs['tech_cat'].set_index('id').to_dict('index')
        except Exception as e:
             print(f"❌ Error with tech_cat for names: {e}.")

    print(f"  Extracting root category labels from {len(all_link_pred_triplets)} available triplets...")
    print(f"  Looking for relation ID {belongs_to_category_rel_id_link_pred} ('{EVALUATION_RELATION_NAME}')")
    
    for h_orig_idx, r_orig_idx, t_orig_idx_specific_cat_graph in all_link_pred_triplets:
        if r_orig_idx != belongs_to_category_rel_id_link_pred:
            continue

        h_node_identifier_ceramic = original_idx_to_node.get(h_orig_idx)
        t_node_identifier_specific_cat = original_idx_to_node.get(t_orig_idx_specific_cat_graph)

        if h_node_identifier_ceramic and t_node_identifier_specific_cat and \
           h_node_identifier_ceramic.startswith("Ceramic_") and t_node_identifier_specific_cat.startswith("Cat_"):
            
            if h_orig_idx in original_idx_to_new_idx:
                h_new_idx_clf = original_idx_to_new_idx[h_orig_idx]
                
                original_specific_cat_id_from_node_name = None
                try:
                    # Node ID is like "Cat_SpecificID_Root_RootID" or "Cat_RootID_Root_RootID"
                    parts = t_node_identifier_specific_cat.split('_')
                    if len(parts) >= 2 and parts[0] == 'Cat':
                        original_specific_cat_id_from_node_name = int(parts[1])
                except (ValueError, IndexError, TypeError):
                    print(f"  ⚠️ Warn: Could not parse specific cat ID from '{t_node_identifier_specific_cat}'.")
                    continue
                
                if original_specific_cat_id_from_node_name is None: 
                    continue

                authoritative_root_id_original = cat_to_root_map_original_ids.get(original_specific_cat_id_from_node_name)

                if authoritative_root_id_original is None:
                    print(f"  ⚠️ Warn: Original root not found for specific_cat_id {original_specific_cat_id_from_node_name} (from node '{t_node_identifier_specific_cat}').")
                    continue
                
                # Use the original root ID to create a consistent string key for labels
                root_label_key = f"RootCatOriginal_{authoritative_root_id_original}"

                if root_label_key not in root_node_identifier_to_label_id:
                    label_id = label_counter
                    root_node_identifier_to_label_id[root_label_key] = label_id
                    label_id_to_category_id[label_id] = authoritative_root_id_original
                    
                    cat_info = tech_cat_lookup.get(authoritative_root_id_original, {})
                    root_cat_name = cat_info.get(tech_cat_name_col, f"Root_{authoritative_root_id_original}")
                    label_id_to_category_name[label_id] = root_cat_name
                    label_counter += 1
                else:
                    label_id = root_node_identifier_to_label_id[root_label_key]
                
                ceramic_labels[h_new_idx_clf] = label_id
    
    print(f"  Extracted labels for {len(ceramic_labels)} ceramic nodes. Found {label_counter} unique root labels.")
    if label_counter > 0:
        print(f"  Label mapping ({label_counter} labels):")
        for lid, name in label_id_to_category_name.items():
            orig_id = label_id_to_category_id.get(lid, 'N/A')
            count = sum(1 for label in ceramic_labels.values() if label == lid)
            print(f"    Label {lid}: '{name}' (Original Root ID: {orig_id}, {count} ceramics)")

    ceramic_node_indices_clf = sorted([
        new_idx for original_idx, new_idx in original_idx_to_new_idx.items()
        if original_idx in original_idx_to_node and original_idx_to_node[original_idx].startswith("Ceramic_")
    ])
    print(f"  Identified {len(ceramic_node_indices_clf)} ceramic nodes in the new mapping for classification.")

    updated_stats = rgcn_data['stats'].copy()
    updated_stats['classification_num_nodes'] = original_num_nodes
    updated_stats['classification_nodes_removed_category'] = 0
    updated_stats['classification_training_triplets_filtered'] = len(original_training_triplets)
    updated_stats['classification_ceramic_nodes_with_labels'] = len(ceramic_labels)
    updated_stats['classification_num_classes'] = label_counter
    updated_stats['classification_ceramic_node_indices'] = ceramic_node_indices_clf
    updated_stats['classification_labeling_strategy'] = 'root_category_from_link_pred_data'

    classification_data = {
        "num_nodes": original_num_nodes,
        "num_relations": len(relation_to_idx), 
        "node_to_idx": new_node_to_idx,
        "idx_to_node": {v: k for k, v in new_node_to_idx.items()}, 
        "relation_to_idx": relation_to_idx,
        "idx_to_relation": idx_to_relation,
        "training_triplets": original_training_triplets,
        "node_embeddings": original_embeddings,
        "embedding_dim": embedding_dim,
        "ceramic_node_indices": ceramic_node_indices_clf,
        "ceramic_labels": ceramic_labels,
        "label_to_category_id": label_id_to_category_id,
        "label_to_category_name": label_id_to_category_name,
        "stats": updated_stats
    }

    print("\n--- Adaptation for Classification Complete ---")
    print(f"  Final ceramic nodes with labels: {len(ceramic_labels)}")
    print(f"  Total classes: {label_counter}")
    return classification_data

# --- Helper: convert_numpy_to_python_native---
def convert_numpy_to_python_native(data):
    if isinstance(data, dict):
        return {k: convert_numpy_to_python_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python_native(i) for i in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray): 
        return convert_numpy_to_python_native(data.tolist())
    return data

def process_and_save_classification_data(study_name, study_datasets_dict, dfs_dict, base_output_dir="./"):
    print(f"\n>>> Processing and saving classification data for study: {study_name} <<<")

    if study_name not in study_datasets_dict or study_datasets_dict[study_name] is None:
        print(f"❌ Study '{study_name}' data not found or is None. Skipping.")
        return None

    rgcn_data_from_link_pred = study_datasets_dict[study_name]

    try:
        classification_data = adapt_rgcn_data_for_ceramic_classification(rgcn_data_from_link_pred, dfs_dict)
    except Exception as e:
        print(f"❌ Error during adaptation of '{study_name}' data: {e}"); traceback.print_exc(); return None

    if classification_data is None:
        print(f"❌ Adaptation of '{study_name}' data returned None. Skipping saving.")
        return None

    output_dir = os.path.join(base_output_dir, f"{study_name}_root_classification_data")
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory created/ensured: {output_dir}")
    except Exception as e:
        print(f"❌ Error creating output directory {output_dir}: {e}"); return None

    print("  Saving data components...")
    try:
        np.save(os.path.join(output_dir, 'node_embeddings.npy'), classification_data['node_embeddings'])
        print("    Saved node_embeddings.npy")
        
        pd.DataFrame(classification_data['node_to_idx'].items(), columns=['node_identifier', 'new_node_idx']).to_csv(os.path.join(output_dir, 'node_mapping.csv'), index=False)
        print("    Saved node_mapping.csv")
        
        pd.DataFrame(classification_data['relation_to_idx'].items(), columns=['relation_name', 'relation_idx']).to_csv(os.path.join(output_dir, 'relation_mapping.csv'), index=False)
        print("    Saved relation_mapping.csv")
        
        pd.DataFrame(classification_data['training_triplets'], columns=['head_idx', 'relation_idx', 'tail_idx']).to_csv(os.path.join(output_dir, 'training_triplets.csv'), index=False)
        print("    Saved training_triplets.csv")
        
        ceramic_labels_data = []
        for node_idx, label_id in classification_data['ceramic_labels'].items():
            try:
                # Ensure both values are integers
                node_idx_int = int(node_idx)
                label_id_int = int(label_id)
                ceramic_labels_data.append([node_idx_int, label_id_int])
            except (ValueError, TypeError) as e:
                print(f"    ⚠️ Warning: Skipping invalid ceramic label entry: node_idx={node_idx} (type: {type(node_idx)}), label_id={label_id} (type: {type(label_id)}). Error: {e}")
                continue
        
        if not ceramic_labels_data:
            print(f"    ❌ No valid ceramic labels found for {study_name}")
            return None
            
        ceramic_labels_df = pd.DataFrame(ceramic_labels_data, columns=['new_ceramic_node_idx', 'label_id'])
        
        # Explicitly set data types to ensure they're integers
        ceramic_labels_df['new_ceramic_node_idx'] = ceramic_labels_df['new_ceramic_node_idx'].astype('int64')
        ceramic_labels_df['label_id'] = ceramic_labels_df['label_id'].astype('int64')
        
        # Print ceramic labels info
        print(f"    Ceramic labels DataFrame shape: {ceramic_labels_df.shape}")
        print(f"    Ceramic labels DataFrame dtypes:\n{ceramic_labels_df.dtypes}")
        print(f"    Ceramic labels DataFrame head:\n{ceramic_labels_df.head()}")
        
        ceramic_labels_df.to_csv(os.path.join(output_dir, 'ceramic_labels.csv'), index=False)
        print("    Saved ceramic_labels.csv")
        
        label_map_data = []
        for label_id in classification_data['label_to_category_id'].keys():
            try:
                label_id_int = int(label_id)
                category_id_int = int(classification_data['label_to_category_id'][label_id])
                category_name = str(classification_data['label_to_category_name'][label_id])
                label_map_data.append([label_id_int, category_id_int, category_name])
            except (ValueError, TypeError) as e:
                print(f"    ⚠️ Warning: Skipping invalid label mapping entry: label_id={label_id}. Error: {e}")
                continue
        
        label_map_df = pd.DataFrame(label_map_data, columns=[
            'label_id', 'original_root_category_id', 'root_category_name'
        ])
        
        # Explicitly set data types
        label_map_df['label_id'] = label_map_df['label_id'].astype('int64')
        label_map_df['original_root_category_id'] = label_map_df['original_root_category_id'].astype('int64')
        
        label_map_df.to_csv(os.path.join(output_dir, 'label_mapping.csv'), index=False)
        print("    Saved label_mapping.csv")

        # ** APPLY CONVERSION BEFORE SAVING STATS **
        stats_to_save = convert_numpy_to_python_native(classification_data['stats'])
        with open(os.path.join(output_dir, 'classification_stats.json'), 'w') as f:
            json.dump(stats_to_save, f, indent=4)
        print("    Saved classification_stats.json")

    except Exception as e:
        print(f"❌ Error during saving data components for '{study_name}': {e}"); traceback.print_exc(); return None

    print(f"  ✅ Successfully processed and saved classification data for '{study_name}' to {output_dir}")
    return output_dir