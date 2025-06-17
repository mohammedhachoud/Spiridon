import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
import torch
from sklearn.decomposition import PCA
import gc
from collections import Counter, defaultdict
import pandas as pd

from graph_utils import (
    build_category_hierarchy_and_map_ceramics,
    calculate_completeness_score,
    extract_triplets_for_selection
)
from utils import get_feature_parent_relation_label, CategoryHierarchy, export_hierarchy_to_csv, demonstrate_hierarchy_usage

def format_rgcn_data_with_hybrid_embeddings(dfs, triplets_for_study, study_name,
                                            bert_model_name="paraphrase-multilingual-mpnet-base-v2"):
    """
    Formats data for RGCN with ALL nodes using BERT embeddings:
    - Ceramic Nodes: Uses BERT embeddings from 'description' column
    - Function Nodes: Uses BERT embeddings from 'function_name_fr' column
    - Feature Nodes: Uses BERT embeddings from 'Nom complet' column  
    - Category Nodes: Uses BERT embeddings from 'cat_name' column
    - All embeddings have the same dimensionality (no PCA needed)
    - Category nodes in the graph are ONLY authoritative root categories.
    - BELONGS_TO_CATEGORY links Ceramic_Node -> Authoritative_Root_Category_Node.
    - Category-to-feature/function links originate from Authoritative_Root_Category_Node.
    - No IS_A hierarchy for categories is built into the graph for BELONGS_TO_CATEGORY evaluation.

    Returns:
        dict: RGCN formatted data including all-BERT embeddings, or None on failure.
    """
    print(f"  Formatting {study_name} data for RGCN with ALL-BERT Embeddings (BERT: {bert_model_name})...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")

    bert_model = None
    bert_embedding_dim_native = -1
    try:
        from sentence_transformers import SentenceTransformer
        bert_model = SentenceTransformer(bert_model_name, device=device)
        bert_embedding_dim_native = bert_model.get_sentence_embedding_dimension()
        print(f"    Loaded Sentence-BERT model '{bert_model_name}' (Native Dim: {bert_embedding_dim_native})")
    except Exception as e:
        print(f"    ❌ Error loading Sentence-BERT model '{bert_model_name}': {e}")
        traceback.print_exc()
        return None

    def get_bert_embedding_for_node(text, level=None, node_type=None):
        if not text or (isinstance(text, str) and not text.strip()): 
            return None
        if isinstance(text, (int, float)) and pd.isna(text):
            return None
        combined_text = str(text)
        if level is not None and pd.notna(level):
             combined_text += f" [Level {level}]"
        if node_type is not None:
             combined_text += f" [{node_type}]"
        try:
            embedding_vector = bert_model.encode(combined_text, convert_to_numpy=True, show_progress_bar=False)
            return embedding_vector
        except Exception as e:
            return None

    final_embedding_dim = bert_embedding_dim_native
    
    required_dfs = [
        'ceramic_summary', 'object_function', 'tech_cat', 'Features_Ontology',
        'tech_cat_function_attrib', 'tech_cat_feature_attrib'
    ]
    if not all(req_df in dfs and isinstance(dfs[req_df], pd.DataFrame) and not dfs[req_df].empty for req_df in required_dfs):
        missing = [req for req in required_dfs if req not in dfs or not isinstance(dfs[req], pd.DataFrame) or dfs[req].empty]
        print(f"    ❌ Error: Required DataFrames missing or empty: {missing}")
        if bert_model: del bert_model; gc.collect(); torch.cuda.empty_cache()
        return None

    ceramic_summary = dfs['ceramic_summary'].copy()
    object_function = dfs['object_function'].copy()
    tech_cat = dfs['tech_cat'].copy()
    Features_Ontology_df = dfs['Features_Ontology'].copy()
    tech_cat_func_attrib = dfs['tech_cat_function_attrib'].copy()
    tech_cat_feat_attrib = dfs['tech_cat_feature_attrib'].copy()

    print("    Building category hierarchy to get root mapping (using original category IDs)...")
    cat_to_root_map_original_ids = {}
    try:
        tech_cat_for_map = tech_cat.copy()
        tech_cat_for_map['id'] = pd.to_numeric(tech_cat_for_map['id'], errors='coerce')
        tech_cat_for_map['inherit_from'] = pd.to_numeric(tech_cat_for_map['inherit_from'], errors='coerce')
        tech_cat_for_map = tech_cat_for_map.dropna(subset=['id'])
        tech_cat_for_map['id'] = tech_cat_for_map['id'].astype(int)

        temp_parent_map = tech_cat_for_map.set_index('id')['inherit_from'].to_dict()
        memo_find_root = {}

        def find_root_original(cat_id_int_orig):
            if cat_id_int_orig in memo_find_root: return memo_find_root[cat_id_int_orig]
            path = {cat_id_int_orig}
            current = cat_id_int_orig
            while True:
                parent_orig = temp_parent_map.get(current)
                if parent_orig is None or pd.isna(parent_orig) or parent_orig == 0 or parent_orig == current or int(parent_orig) == -1:
                    root = current; break
                parent_orig = int(parent_orig)
                if parent_orig in path:
                    root = current
                    warnings.warn(f"Cycle detected in category hierarchy at original ID {parent_orig} (path from {cat_id_int_orig}). Using current node {current} as root for this path.", UserWarning)
                    break
                path.add(parent_orig)
                current = parent_orig
                if len(path) > 50:
                    root = current
                    warnings.warn(f"Max depth (50) reached in category hierarchy starting from original ID {cat_id_int_orig}. Using current node {current} as root.", UserWarning)
                    break
            memo_find_root[cat_id_int_orig] = root
            return root

        for cat_id_int_val_orig in tech_cat_for_map['id'].unique():
            if pd.notna(cat_id_int_val_orig):
                 root_orig = find_root_original(int(cat_id_int_val_orig))
                 if root_orig is not None:
                     cat_to_root_map_original_ids[int(cat_id_int_val_orig)] = root_orig
        print(f"    Obtained mapping for {len(cat_to_root_map_original_ids)} original categories to their original root IDs.")
        if not cat_to_root_map_original_ids:
             print("    ⚠️ Warning: Original category to original root ID map is empty! This will affect BELONGS_TO_CATEGORY links.")
    except Exception as e:
        print(f"    ❌ Error building original category root ID map: {e}.")
        traceback.print_exc()

    try:
        ceramic_summary['ceramic_id'] = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce').dropna().astype(int)
        ceramic_summary['tech_cat_id'] = pd.to_numeric(ceramic_summary['tech_cat_id'], errors='coerce')
        if 'description' not in ceramic_summary.columns: 
            raise ValueError("Missing 'description' column in ceramic_summary for BERT embeddings")

        object_function['id'] = pd.to_numeric(object_function['id'], errors='coerce').dropna().astype(int)
        if 'function_name_fr' not in object_function.columns: 
            raise ValueError("Missing 'function_name_fr' in object_function for BERT embeddings")
        if 'function_level' not in object_function.columns: object_function['function_level'] = None
        if 'function_parent' not in object_function.columns: object_function['function_parent'] = None

        tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
        if 'cat_name' not in tech_cat.columns: 
            raise ValueError("Missing 'cat_name' in tech_cat for BERT embeddings of root categories")
        if 'inherit_from' not in tech_cat.columns: tech_cat['inherit_from'] = None

        Features_Ontology_df['id'] = pd.to_numeric(Features_Ontology_df['id'], errors='coerce').dropna().astype(float).astype(int).astype(str)
        Features_Ontology_df['feature_parent'] = pd.to_numeric(Features_Ontology_df['feature_parent'], errors='coerce').fillna('-1').astype(int).astype(str)
        if 'Nom complet' not in Features_Ontology_df.columns: 
            raise ValueError("Missing 'Nom complet' in Features_Ontology for BERT embeddings")
        if 'tree_depth' not in Features_Ontology_df.columns: Features_Ontology_df['tree_depth'] = None
        for col_opt in ['Ceramic_Relation', 'Relation', 'Nom abrégé avec la relation']:
            if col_opt not in Features_Ontology_df.columns: Features_Ontology_df[col_opt] = None

        tech_cat_func_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_func_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
        tech_cat_func_attrib['function_id'] = pd.to_numeric(tech_cat_func_attrib['function_id'], errors='coerce').dropna().astype(int)

        tech_cat_feat_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_feat_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
        tech_cat_feat_attrib['feature_id'] = pd.to_numeric(tech_cat_feat_attrib['feature_id'], errors='coerce').dropna().astype(float).astype(int).astype(str)
        if 'relation_def' not in tech_cat_feat_attrib.columns: tech_cat_feat_attrib['relation_def'] = None

    except Exception as e:
        print(f"    ❌ Error standardizing IDs or checking required columns: {e}")
        traceback.print_exc()
        if bert_model: del bert_model; gc.collect(); torch.cuda.empty_cache()
        return None

    node_to_idx = {}
    relation_to_idx = {}
    idx_counter = 0
    rel_idx_counter = 0

    # Replace this section in your code (around line 140-160):

    print("    Identifying all unique nodes in the sampled data (Ceramics, Functions, Features, and ONLY ROOT Categories)...")
    nodes_in_sample = set()
    for entry in triplets_for_study:
        raw_cid = entry.get("ceramic_id")
        try:
            cid_val = int(float(raw_cid))
            nodes_in_sample.add(f"Ceramic_{cid_val}")
        except (TypeError, ValueError, IndexError):
            continue

        def add_node_str_id_to_sample_set(prefix, item_id_val):
            if item_id_val is not None and pd.notna(item_id_val):
                nodes_in_sample.add(f"{prefix}{str(item_id_val)}")

        categories_in_entry = entry.get("categories", [])
        if categories_in_entry:
            most_specific_cat_original_id = categories_in_entry[0].get('category_id')
            if most_specific_cat_original_id is not None and pd.notna(most_specific_cat_original_id):
                try:
                    authoritative_root_id_original = cat_to_root_map_original_ids.get(int(most_specific_cat_original_id))
                    if authoritative_root_id_original is not None:
                        nodes_in_sample.add(f"Cat_{authoritative_root_id_original}_Root_{authoritative_root_id_original}")
                except ValueError:
                    pass
        
        # FIX: Add ALL function nodes (main function + all its parents)
        for fid, parents_original_ids in entry.get("functions", []):
            # Add the main function
            add_node_str_id_to_sample_set("Func_", fid)
            # Add ALL parent functions
            for parent_fid in parents_original_ids:
                add_node_str_id_to_sample_set("Func_", parent_fid)

        # FIX: Add ALL feature nodes (main feature + all its parents)
        for feat_id_original, parents_original_str_ids in entry.get("features", []):
            # Add the main feature
            add_node_str_id_to_sample_set("Feat_", str(feat_id_original))
            # Add ALL parent features
            for parent_feat_id in parents_original_str_ids:
                add_node_str_id_to_sample_set("Feat_", str(parent_feat_id))

        all_root_ids_from_attribute_tables = set()
        for _, row in tech_cat_func_attrib.iterrows():
            cat_id_original = int(row['tech_cat_id'])
            root_id_original = cat_to_root_map_original_ids.get(cat_id_original)
            if root_id_original is not None:
                all_root_ids_from_attribute_tables.add(root_id_original)
        for _, row in tech_cat_feat_attrib.iterrows():
            cat_id_original = int(row['tech_cat_id'])
            root_id_original = cat_to_root_map_original_ids.get(cat_id_original)
            if root_id_original is not None:
                all_root_ids_from_attribute_tables.add(root_id_original)
        
        for root_id_orig in all_root_ids_from_attribute_tables:
            nodes_in_sample.add(f"Cat_{root_id_orig}_Root_{root_id_orig}")

    print(f"    Found {len(nodes_in_sample)} unique node identifiers in the sample (ceramics, functions, features, and ROOT categories).")
    if not nodes_in_sample:
        print("    ❌ Error: No nodes identified in the sample. Cannot proceed.")
        if bert_model: del bert_model; gc.collect(); torch.cuda.empty_cache()
        return None

    print("    Assigning graph indices and generating ALL-BERT embeddings...")
    try:
        ceramic_summary.set_index('ceramic_id', inplace=True, drop=False)
        object_function.set_index('id', inplace=True, drop=False)
        tech_cat.set_index('id', inplace=True, drop=False)
        Features_Ontology_df.set_index('id', inplace=True, drop=False)
    except KeyError as e:
        print(f"    ❌ Error setting DataFrame index for lookup: {e}.")
        if bert_model: del bert_model; gc.collect(); torch.cuda.empty_cache()
        return None

    # Statistics tracking
    num_bert_generated = 0
    num_bert_failed = 0
    examples_by_type = {
        'Ceramic': {'node_id': None, 'text': None, 'embedding': None},
        'Function': {'node_id': None, 'text': None, 'embedding': None},
        'Category': {'node_id': None, 'text': None, 'embedding': None},
        'Feature': {'node_id': None, 'text': None, 'embedding': None}
    }
    
    estimated_max_nodes = len(nodes_in_sample) + 50
    embedding_matrix_temp = np.zeros((estimated_max_nodes, final_embedding_dim), dtype=np.float32)

    for node_identifier in sorted(list(nodes_in_sample)):
        if node_identifier not in node_to_idx:
            node_idx_val = idx_counter
            node_to_idx[node_identifier] = node_idx_val
            idx_counter += 1

            if node_idx_val >= embedding_matrix_temp.shape[0]:
                new_rows_needed = node_idx_val - embedding_matrix_temp.shape[0] + 1
                buffer_to_add = max(100, int(embedding_matrix_temp.shape[0] * 0.2))
                padding = np.zeros((max(new_rows_needed, buffer_to_add), final_embedding_dim), dtype=np.float32)
                embedding_matrix_temp = np.vstack((embedding_matrix_temp, padding))
            
            source_text_for_embedding = None
            level_info_for_embedding = None
            node_type_for_embedding = None
            
            try:
                if node_identifier.startswith("Ceramic_"):
                    ceramic_id_val = int(float(node_identifier.split('_')[-1]))
                    if ceramic_id_val in ceramic_summary.index:
                        source_text_for_embedding = ceramic_summary.loc[ceramic_id_val, 'description']
                        level_info_for_embedding = ""
                        node_type_for_embedding = "Ceramic"
                        
                elif node_identifier.startswith("Func_"):
                    func_id_val = int(float(node_identifier.split('_')[-1]))
                    if func_id_val in object_function.index:
                        source_text_for_embedding = object_function.loc[func_id_val, 'function_name_fr']
                        level_info_for_embedding = object_function.loc[func_id_val].get('function_level')
                        node_type_for_embedding = "Function"
                        
                elif node_identifier.startswith("Cat_") and "_Root_" in node_identifier:
                    parts = node_identifier.split('_')
                    if len(parts) == 4 and parts[0] == "Cat" and parts[2] == "Root" and parts[1] == parts[3]:
                        root_id_val = int(float(parts[1]))
                        if root_id_val in tech_cat.index:
                            source_text_for_embedding = tech_cat.loc[root_id_val, 'cat_name']
                            level_info_for_embedding = "Root Category"
                            node_type_for_embedding = "Category"
                            
                elif node_identifier.startswith("Feat_"):
                    feat_id_str_from_node = node_identifier.split('_')[-1]
                    if feat_id_str_from_node in Features_Ontology_df.index:
                        source_text_for_embedding = Features_Ontology_df.loc[feat_id_str_from_node, 'Nom complet']
                        level_info_for_embedding = Features_Ontology_df.loc[feat_id_str_from_node].get('tree_depth')
                        node_type_for_embedding = "Feature"
                
                # Generate BERT embedding for ALL node types
                if source_text_for_embedding is not None:
                    bert_emb_vector = get_bert_embedding_for_node(
                        source_text_for_embedding, 
                        level_info_for_embedding, 
                        node_type_for_embedding
                    )
                    if bert_emb_vector is not None:
                        if bert_emb_vector.ndim > 1: 
                            bert_emb_vector = bert_emb_vector.flatten()
                        if bert_emb_vector.shape[0] == bert_embedding_dim_native:
                            embedding_matrix_temp[node_idx_val] = bert_emb_vector
                            num_bert_generated += 1
                            
                            # Store example for each node type (first occurrence)
                            if (node_type_for_embedding in examples_by_type and 
                                examples_by_type[node_type_for_embedding]['node_id'] is None):
                                examples_by_type[node_type_for_embedding]['node_id'] = node_identifier
                                examples_by_type[node_type_for_embedding]['text'] = source_text_for_embedding
                                examples_by_type[node_type_for_embedding]['embedding'] = bert_emb_vector[:5]  # First 5 dimensions
                                
                        else:
                            print(f"    🚨 BERT embedding for {node_identifier} has unexpected dimension! Expected {bert_embedding_dim_native}, got {bert_emb_vector.shape[0]}.")
                            num_bert_failed += 1
                    else:
                        num_bert_failed += 1
                else:
                    num_bert_failed += 1
                    
            except Exception as e_bert_node_proc:
                print(f"    ❌ Error processing node {node_identifier} for BERT: {e_bert_node_proc}")
                num_bert_failed += 1

    print(f"    ALL-BERT Embeddings - Generated: {num_bert_generated}, Failed/Missing Text: {num_bert_failed}")
    
    # Print examples for each node type
    print("\n    --- Node Type Examples with BERT Embeddings ---")
    for node_type, example_data in examples_by_type.items():
        if example_data['node_id'] is not None:
            print(f"    {node_type} Node Example:")
            print(f"      Node ID: {example_data['node_id']}")
            print(f"      Text: '{example_data['text']}'")
            print(f"      Embedding (first 5 dims): {example_data['embedding']}")
            print()
        else:
            print(f"    {node_type} Node: No examples found in this dataset")
    print("    --- End Node Type Examples ---\n")

    num_nodes_final = idx_counter
    final_embedding_matrix = embedding_matrix_temp[:num_nodes_final, :]

    nodes_receiving_final_embedding = np.sum(np.any(final_embedding_matrix != 0, axis=1))
    print(f"    Populated final ALL-BERT embedding matrix with shape ({final_embedding_matrix.shape[0]} nodes, {final_embedding_matrix.shape[1]} dim). "
          f"{nodes_receiving_final_embedding} nodes have non-zero embeddings.")

    idx_to_node = {v: k for k, v in node_to_idx.items()}
    
    print("    Identifying root category node indices in the final graph map (all 'Cat_' nodes are roots)...")
    root_category_node_indices_in_graph = []
    for node_idx_val_current, node_str_current in idx_to_node.items():
        if node_str_current.startswith("Cat_") and "_Root_" in node_str_current:
            parts = node_str_current.split('_')
            if len(parts) == 4 and parts[0] == "Cat" and parts[2] == "Root" and parts[1] == parts[3]:
                root_category_node_indices_in_graph.append(node_idx_val_current)
    root_category_node_indices_in_graph = sorted(list(set(root_category_node_indices_in_graph)))
    print(f"    Identified {len(root_category_node_indices_in_graph)} unique root category nodes in the graph's node_to_idx map.")

    cat_idx_to_root_idx_map = {root_cat_node_idx: root_cat_node_idx for root_cat_node_idx in root_category_node_indices_in_graph}
    print(f"    Created cat_idx_to_root_idx_map with {len(cat_idx_to_root_idx_map)} entries (each root maps to itself).")

    print("    Processing triplets using pre-assigned graph indices...")
    training_triplets_final = []
    evaluation_triplets_final = []

    def get_or_assign_relation_idx(relation_name): 
        nonlocal rel_idx_counter 
        relation_name_str = str(relation_name).strip().replace(" ", "_") if pd.notna(relation_name) else "UNKNOWN_RELATION"
        if not relation_name_str: relation_name_str = "UNKNOWN_RELATION" 
        if relation_name_str not in relation_to_idx:
            relation_to_idx[relation_name_str] = rel_idx_counter
            rel_idx_counter += 1
        return relation_to_idx[relation_name_str]

    # Define core relation types and get their indices
    BELONGS_TO_REL = get_or_assign_relation_idx("BELONGS_TO_CATEGORY")
    HAS_FUNCTION_REL = get_or_assign_relation_idx("HAS_FUNCTION")
    IS_A_REL = get_or_assign_relation_idx("IS_A")

    # For quick lookup of feature ontology data during triplet creation
    feature_id_to_data = Features_Ontology_df.set_index('id').to_dict('index') if 'id' in Features_Ontology_df.columns else {}


    for entry_idx, entry in enumerate(triplets_for_study): 
        raw_cid = entry.get("ceramic_id")
        
        ceramic_idx = None 
        try:
            cid_val = int(float(raw_cid))
            ceramic_node_identifier = f"Ceramic_{cid_val}"
            ceramic_idx = node_to_idx.get(ceramic_node_identifier) 
            if ceramic_idx is None:
                # print(f"    ⚠️ Warning: Ceramic node '{ceramic_node_identifier}' not in node_to_idx. Skipping its triplets.")
                continue
        except (TypeError, ValueError, IndexError): 
            # print(f"    ⚠️ Warning: Invalid ceramic_id '{raw_cid}' in entry {entry_idx}. Skipping.")
            continue

        # --- BELONGS_TO_CATEGORY Triplets (Ceramic -> Authoritative Root Category) ---
        categories_in_entry = entry.get("categories", [])
        if categories_in_entry: 
            most_specific_cat_original_id = categories_in_entry[0].get('category_id') 
            if most_specific_cat_original_id is not None and pd.notna(most_specific_cat_original_id):
                try:
                    # Find the authoritative root for this specific category using the precomputed map
                    authoritative_root_id_original = cat_to_root_map_original_ids.get(int(most_specific_cat_original_id))
                    if authoritative_root_id_original is not None:
                        authoritative_root_node_identifier = f"Cat_{authoritative_root_id_original}_Root_{authoritative_root_id_original}"
                        authoritative_root_graph_idx = node_to_idx.get(authoritative_root_node_identifier)
                        if authoritative_root_graph_idx is not None:
                            evaluation_triplets_final.append((ceramic_idx, BELONGS_TO_REL, authoritative_root_graph_idx))
                            # training_triplets_final.append((ceramic_idx, BELONGS_TO_REL, authoritative_root_graph_idx)) # Also add to training
                        else: print(f"    ⚠️ Root category node '{authoritative_root_node_identifier}' for ceramic {cid_val} not in node_to_idx.")
                except ValueError:
                    pass

        # --- Ceramic -> Function Triplets & Function IS_A Hierarchy ---entry.get("functions", []):
        for func_id_original, parents_original_ids_func in entry.get("functions", []): 
            func_node_identifier = f"Func_{func_id_original}"
            func_graph_idx = node_to_idx.get(func_node_identifier)
            if func_graph_idx is not None: 
                training_triplets_final.append((ceramic_idx, HAS_FUNCTION_REL, func_graph_idx))
                current_child_func_graph_idx = func_graph_idx
                for p_id_original_func in parents_original_ids_func:
                    parent_func_node_identifier = f"Func_{p_id_original_func}"
                    parent_func_graph_idx = node_to_idx.get(parent_func_node_identifier)
                    if parent_func_graph_idx is not None:
                        training_triplets_final.append((current_child_func_graph_idx, IS_A_REL, parent_func_graph_idx))
                        current_child_func_graph_idx = parent_func_graph_idx
        
        # --- Ceramic -> Feature Triplets & Feature IS_A Hierarchy --- 
        for feat_id_original_int, parents_original_str_ids_feat in entry.get("features", []):
            feat_id_original_str = str(feat_id_original_int) 
            feat_node_identifier = f"Feat_{feat_id_original_str}"
            feat_graph_idx = node_to_idx.get(feat_node_identifier)
            if feat_graph_idx is None: continue

            # Determine relation type for Ceramic -> Feature based on ontology
            feat_ontology_data = feature_id_to_data.get(feat_id_original_str, {})
            ceramic_relation_name_raw = feat_ontology_data.get('Ceramic_Relation')
            # Use specific relation if defined, else default to HAS_FEATURE
            
            if pd.notna(ceramic_relation_name_raw) and str(ceramic_relation_name_raw).strip() :
                rel_idx_ceramic_to_feature = get_or_assign_relation_idx(ceramic_relation_name_raw)
                                         
            training_triplets_final.append((ceramic_idx, rel_idx_ceramic_to_feature, feat_graph_idx))

            # Add IS_A links for feature hierarchy (using 'Relation' field from ontology for parent link type)
            current_child_feat_graph_idx = feat_graph_idx
            child_feat_data_for_relation_lookup = feat_ontology_data # Current feature's ontology entry
            for p_id_original_str_feat in parents_original_str_ids_feat:
                parent_feat_node_identifier = f"Feat_{p_id_original_str_feat}"
                parent_feat_graph_idx = node_to_idx.get(parent_feat_node_identifier)
                if parent_feat_graph_idx is None: continue
                
                # Get the relation label for Child_Feature -> Parent_Feature from child's ontology entry
                relation_child_to_parent_name_raw = get_feature_parent_relation_label(child_feat_data_for_relation_lookup)
                rel_idx_feat_to_parent = get_or_assign_relation_idx(relation_child_to_parent_name_raw)
                training_triplets_final.append((current_child_feat_graph_idx, rel_idx_feat_to_parent, parent_feat_graph_idx))
                
                current_child_feat_graph_idx = parent_feat_graph_idx # Traverse up
                child_feat_data_for_relation_lookup = feature_id_to_data.get(p_id_original_str_feat, {}) # Get next child's data

    # Only add direct functions and features for root categories themselves
# Do NOT propagate child category attributes to root

    added_rootcat_func_triplets = 0
    for _, row in tech_cat_func_attrib.iterrows():
        try:
            cat_id_original = int(row['tech_cat_id'])
            func_id_original = int(row['function_id'])
            
            # Only process if this category IS a root category (not mapping children to root)
            if cat_id_original in cat_to_root_map_original_ids and cat_to_root_map_original_ids[cat_id_original] == cat_id_original:
                # This is a root category, add its direct functions
                cat_node_identifier_root = f"Cat_{cat_id_original}_Root_{cat_id_original}"
                func_node_identifier = f"Func_{func_id_original}"
                cat_graph_idx_root = node_to_idx.get(cat_node_identifier_root)
                func_graph_idx = node_to_idx.get(func_node_identifier)
                
                if cat_graph_idx_root is not None and func_graph_idx is not None:
                    training_triplets_final.append((cat_graph_idx_root, HAS_FUNCTION_REL, func_graph_idx))
                    added_rootcat_func_triplets += 1
                    
        except Exception as e_attr_func:
            # print(f"    ⚠️ Warn: Error in tech_cat_func_attrib processing for row {row.to_dict()}: {e_attr_func}")
            continue

    added_rootcat_feat_triplets = 0
    for _, row in tech_cat_feat_attrib.iterrows():
        try:
            cat_id_original = int(row['tech_cat_id'])
            feat_id_original_str = str(row['feature_id'])
            
            # Only process if this category IS a root category (not mapping children to root)
            if cat_id_original in cat_to_root_map_original_ids and cat_to_root_map_original_ids[cat_id_original] == cat_id_original:
                # This is a root category, add its direct features
                cat_node_identifier_root = f"Cat_{cat_id_original}_Root_{cat_id_original}"
                feat_node_identifier = f"Feat_{feat_id_original_str}"
                cat_graph_idx_root = node_to_idx.get(cat_node_identifier_root)
                feat_graph_idx = node_to_idx.get(feat_node_identifier)
                
                if cat_graph_idx_root is not None and feat_graph_idx is not None:
                    feat_data_from_ontology = feature_id_to_data.get(feat_id_original_str, {})
                    relation_name_cat_to_feat_raw = feat_data_from_ontology.get('Ceramic_Relation')
                    
                    if pd.notna(relation_name_cat_to_feat_raw) and str(relation_name_cat_to_feat_raw).strip():
                        rel_idx_cat_to_feat = get_or_assign_relation_idx(relation_name_cat_to_feat_raw)
                        training_triplets_final.append((cat_graph_idx_root, rel_idx_cat_to_feat, feat_graph_idx))
                        added_rootcat_feat_triplets += 1
                        
        except Exception as e_attr_feat:
            # print(f"    ⚠️ Warn: Error in tech_cat_feat_attrib processing for row {row.to_dict()}: {e_attr_feat}")
            continue

    print(f"    Added {added_rootcat_func_triplets} RootCat->Function and {added_rootcat_feat_triplets} RootCat->Feature triplets (direct connections only).")

    idx_to_relation = {v: k for k, v in relation_to_idx.items()} # Inverse map for relation index to name

    # Deduplicate and sort triplet lists
    initial_train_count = len(training_triplets_final)
    training_triplets_final = sorted(list(set(map(tuple, training_triplets_final))))
    print(f"    Removed {initial_train_count - len(training_triplets_final)} duplicate training triplets.")
    
    initial_eval_count = len(evaluation_triplets_final)
    evaluation_triplets_final = sorted(list(set(map(tuple, evaluation_triplets_final))))
    print(f"    Removed {initial_eval_count - len(evaluation_triplets_final)} duplicate evaluation triplets (Ceramic->RootCat).")


    # --- Final RGCN Data Dictionary Construction ---
    rgcn_data = {
        "num_nodes": num_nodes_final, # Actual number of nodes in the graph
        "num_relations": len(relation_to_idx), # Actual number of unique relations
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "relation_to_idx": relation_to_idx,
        "idx_to_relation": idx_to_relation,
        "training_triplets": training_triplets_final,
        "evaluation_triplets": evaluation_triplets_final,
        "node_embeddings": final_embedding_matrix,
        "embedding_dim": final_embedding_dim,    
        "root_category_node_indices_in_graph": root_category_node_indices_in_graph, 
        "cat_idx_to_root_idx_map": cat_idx_to_root_idx_map, 
        "stats": { 
            "study_name": study_name,
            "bert_model_used": bert_model_name,
            "native_bert_dim": bert_embedding_dim_native,
            "final_embedding_dim": final_embedding_dim,
            "target_dim_for_pca_and_ohe": final_embedding_dim,
            "total_nodes_in_graph": num_nodes_final,
            "bert_nodes_generated_pre_pca": num_bert_generated,
            "bert_nodes_failed_or_missing_text": num_bert_failed,
            "nodes_in_final_matrix_with_embedding": nodes_receiving_final_embedding,
            "total_relations": len(relation_to_idx),
            "total_training_triplets_for_gnn": len(training_triplets_final), 
            "total_evaluation_triplets_for_btc_task": len(evaluation_triplets_final), 
             "cat_to_root_map_original_ids": {str(k): str(v) for k,v in cat_to_root_map_original_ids.items()},
        }
    }

    # Print a summary of the generated data
    print("\n    --- RGCN Data Summary (Hybrid Embeddings with PCA) ---")
    stats = rgcn_data['stats']
    print(f"    Study: {stats['study_name']}")
    print(f"    BERT Model for Non-Ceramics: {stats['bert_model_used']} (Native Dim: {stats['native_bert_dim']})")
    print(f"    Ceramic Embeddings Source: OHE from 'Embedding' column (Expected Dim: {stats['target_dim_for_pca_and_ohe']})")
    print(f"    Final Hybrid Embedding Dimension: {stats['final_embedding_dim']}")
    print(f"    Total Nodes in Graph: {stats['total_nodes_in_graph']}")
    print(f"    Nodes with Non-Zero Embedding in Final Matrix: {stats['nodes_in_final_matrix_with_embedding']}")
    print(f"    Total Relation Types in Graph: {stats['total_relations']}")
    print(f"    Training Triplets (for GNN structure): {stats['total_training_triplets_for_gnn']}")
    print(f"    Evaluation Triplets (Ceramic->RootCat, for BTC task): {stats['total_evaluation_triplets_for_btc_task']}")
    print(f"    Number of Root Category Nodes in Graph (for neg. sampling): {len(rgcn_data.get('root_category_node_indices_in_graph', []))}")
    print("    --- End Summary ---")

    # --- Cleanup ---
    if bert_model: del bert_model; gc.collect(); torch.cuda.empty_cache()
   
    # Delete DataFrame copies to free memory
    del ceramic_summary, object_function, tech_cat, Features_Ontology_df
    del tech_cat_func_attrib, tech_cat_feat_attrib
    # Delete temporary mapping structures if they exist
    if 'cat_to_root_map_original_ids' in locals(): del cat_to_root_map_original_ids
    if 'temp_parent_map' in locals(): del temp_parent_map
    if 'memo_find_root' in locals(): del memo_find_root
    gc.collect() # Explicit garbage collection

    return rgcn_data


def prepare_study_datasets_hybrid_embeddings(dfs, bert_model_name="paraphrase-multilingual-mpnet-base-v2"):
    """
    Prepares study datasets including one using ALL ceramics from ceramic_summary.
    Uses the extract_triplets_for_selection to include hierarchies.
    """
    # --- 0. Pre-checks and Get ALL Ceramic IDs ---
    if not isinstance(dfs, dict):
        print("❌ Error: dfs parameter must be a dictionary.")
        return None
    if 'ceramic_summary' not in dfs or not isinstance(dfs['ceramic_summary'], pd.DataFrame) or dfs['ceramic_summary'].empty:
        print("❌ Error: 'ceramic_summary' DataFrame is missing, not a DataFrame, or empty in dfs.")
        return None

    print("Identifying all valid ceramic IDs from ceramic_summary...")
    all_ceramic_ids_in_summary = []
    try:
        summary_copy = dfs['ceramic_summary'].copy()
        summary_copy['ceramic_id'] = pd.to_numeric(summary_copy['ceramic_id'], errors='coerce')
        summary_copy = summary_copy.dropna(subset=['ceramic_id'])
        all_ceramic_ids_in_summary = sorted(list(summary_copy['ceramic_id'].astype(int).unique()))
        if not all_ceramic_ids_in_summary:
            print("❌ Error: No valid ceramic IDs found in ceramic_summary after processing.")
            return None
        print(f"Found {len(all_ceramic_ids_in_summary)} unique valid ceramic IDs in summary.")
    except Exception as e:
        print(f"❌ Error processing ceramic_summary for IDs: {e}")
        return None

    # --- 1. Build Hierarchy & Map Ceramics (for sampling info) ---
    print("Building category hierarchy and mapping ceramics (for sampling info)...\n")
    hierarchy_results = build_category_hierarchy_and_map_ceramics(dfs)
    cat_to_root_map, ceramic_to_root_map, root_category_counts = {}, {}, Counter()
    if hierarchy_results is None:
        print("⚠️ Warning: Failed to build hierarchy or map ceramics. Sampling for Etude 1, 1', 2 will be affected or skipped.")
    else:
        try:
            if isinstance(hierarchy_results, tuple) and len(hierarchy_results) == 3:
                cat_to_root_map, ceramic_to_root_map, root_category_counts = hierarchy_results
                if not (isinstance(ceramic_to_root_map, dict) and 
                        isinstance(root_category_counts, Counter) and 
                        isinstance(cat_to_root_map, dict)):
                    print("⚠️ Warning: Invalid data types from hierarchy build. Resetting."); cat_to_root_map, ceramic_to_root_map, root_category_counts = {}, {}, Counter()
            else: print("⚠️ Warning: Unexpected structure from hierarchy build. Resetting."); cat_to_root_map, ceramic_to_root_map, root_category_counts = {}, {}, Counter()
        except ValueError: print("⚠️ Warning: Could not unpack hierarchy results. Resetting."); cat_to_root_map, ceramic_to_root_map, root_category_counts = {}, {}, Counter()

    ceramics_by_root = defaultdict(list)
    if ceramic_to_root_map: 
        for ceramic_id, root_cat_id in ceramic_to_root_map.items():
            try:
                int_ceramic_id = int(ceramic_id) 
                if int_ceramic_id in all_ceramic_ids_in_summary:
                    ceramics_by_root[int(root_cat_id)].append(int_ceramic_id)
            except (ValueError, TypeError): continue

    study_datasets = {}
    valid_counts = {int(k): v for k, v in root_category_counts.items() if isinstance(v, int) and v > 0 and pd.notna(k)}
    min_count = min(valid_counts.values()) if valid_counts else 0
    target_size_etude1 = min_count
    target_size_etude1_prime = 276 
    counts_without_140 = {k: v for k, v in valid_counts.items() if k != 140} # Assuming 140 is an original root ID
    target_size_etude2 = min(counts_without_140.values()) if counts_without_140 else 0


    print(f"\n--- Target Sample Sizes ---")
    print(f"Etude 1 (Min Count among roots: {min_count}): Sample Size = {target_size_etude1 if target_size_etude1 > 0 else 'N/A'}")
    print(f"Etude 1' (Fixed): Sample Size = {target_size_etude1_prime}")
    min_rem_count_etude2 = min(counts_without_140.values()) if counts_without_140 else 'N/A'
    print(f"Etude 2 (Exclude root 140, Min Rem Count: {min_rem_count_etude2}): Sample Size = {target_size_etude2 if target_size_etude2 > 0 else 'N/A'}")
    print(f"Etude All: Using all {len(all_ceramic_ids_in_summary)} ceramics from summary")

    study_definitions = [
        ('etude1', target_size_etude1, None, False),
        ('etude1_prime', target_size_etude1_prime, None, False),
        ('etude2', target_size_etude2, 140, False), # 140 is an example original_root_id to exclude
        #('etude_all', -1, None, True) 
    ]

    for study_name, target_size, excluded_root_id, use_all_ceramics_flag in study_definitions:
        print(f"\n--- Preparing Dataset: {study_name} ---")
        selected_ceramics_for_study = []

        if use_all_ceramics_flag:
            print(f"  Using all {len(all_ceramic_ids_in_summary)} ceramics.")
            selected_ceramics_for_study = all_ceramic_ids_in_summary
        else:
            if hierarchy_results is None or target_size <= 0:
                print(f"  Sampling skipped for {study_name} (target size: {target_size}).")
                study_datasets[study_name] = None
                continue

            current_roots_to_sample_from = set(valid_counts.keys())
            if excluded_root_id is not None:
                current_roots_to_sample_from.discard(int(excluded_root_id)) # Ensure excluded_root_id is int

            if not current_roots_to_sample_from:
                print(f"  No valid root categories for sampling in {study_name}. Skipping.")
                study_datasets[study_name] = None
                continue

            print(f"  Sampling {target_size} ceramics per root from roots: {current_roots_to_sample_from}")
            for root_id_int in sorted(list(current_roots_to_sample_from)): 
                candidate_ceramic_ids_for_root = ceramics_by_root.get(root_id_int, [])
                if not candidate_ceramic_ids_for_root: continue

                if len(candidate_ceramic_ids_for_root) > target_size:
                    scores = calculate_completeness_score(candidate_ceramic_ids_for_root, dfs)
                    sorted_candidates = sorted(candidate_ceramic_ids_for_root, key=lambda cid: (-scores.get(cid, 0), cid))
                else:
                    sorted_candidates = sorted(candidate_ceramic_ids_for_root)

                actual_sample_size_for_root = min(len(sorted_candidates), target_size)
                if actual_sample_size_for_root == 0: continue
                selected_ceramics_for_study.extend(sorted_candidates[:actual_sample_size_for_root])
        
        unique_selected_ceramics = sorted(list(set(selected_ceramics_for_study)))
        if not unique_selected_ceramics:
            print(f"  ❌ Error: No ceramics selected for {study_name}. Skipping.")
            study_datasets[study_name] = None
            continue
        print(f"  Total unique ceramics selected for {study_name}: {len(unique_selected_ceramics)}")

        print(f"  Extracting triplets for {len(unique_selected_ceramics)} ceramics in {study_name}...")
        triplets_for_study = extract_triplets_for_selection(unique_selected_ceramics, dfs)
        if triplets_for_study is None or not triplets_for_study : 
            print(f"  ❌ Error: Failed to extract triplets for {study_name}.")
            study_datasets[study_name] = None
            continue
        print(f"  Extracted {len(triplets_for_study)} primary entries for triplet generation.")

        print(f"  Formatting data for RGCN model for {study_name}...")
        rgcn_formatted_data = format_rgcn_data_with_hybrid_embeddings(
            dfs=dfs,
            triplets_for_study=triplets_for_study,
            study_name=study_name,
            bert_model_name=bert_model_name
        )

        if rgcn_formatted_data is None:
            print(f"  ❌ Error: RGCN data formatting failed for {study_name}.")
        study_datasets[study_name] = rgcn_formatted_data # Store None if failed
        if rgcn_formatted_data:
             print(f"  ✅ Successfully prepared and formatted dataset for {study_name}.")


    print("\n--- Finished Preparing All Study Datasets ---")
    return study_datasets


def save_rgcn_study_data(study_results_dict, base_output_dir="rgcn_study_datasets"):
    """
    Saves the RGCN data for each study (etude1, etude1_prime, etude2)
    into separate subdirectories. """
    
    if not study_results_dict:
        print("Error: Study results dictionary is empty or None. Nothing to save.")
        return

    print(f"\n--- Saving Study Datasets to subdirectories under: '{base_output_dir}' ---")

    for study_name, rgcn_data in study_results_dict.items():
        if rgcn_data is None:
            print(f"--- Skipping study '{study_name}' (data is None) ---")
            continue

        print(f"\n--- Processing study: '{study_name}' ---")
        study_output_dir = os.path.join(base_output_dir, study_name)
        save_single_rgcn_dataset(rgcn_data, study_output_dir, dataset_label=study_name)

    print(f"\n--- Finished saving all study datasets under '{base_output_dir}' ---")


def save_single_rgcn_dataset(rgcn_data, output_dir="dataaa", dataset_label="dataset"):
    """
    Saves the components of a single RGCN dataset (mappings, triplets, embeddings, and new JSON maps).
    Helper function for save_rgcn_study_data. """
    
    if not rgcn_data:
        print(f"Error: RGCN data for '{dataset_label}' is None or empty. Cannot save.")
        return

    print(f"Saving '{dataset_label}' data to directory: '{output_dir}'")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory '{output_dir}' ensured.")

        saved_files = []

        # 1. Save Node Mappings
        if 'node_to_idx' in rgcn_data and rgcn_data['node_to_idx']:
            try:
                pd.DataFrame(list(rgcn_data['node_to_idx'].items()), columns=['NodeIdentifier', 'NodeIndex']).to_csv(os.path.join(output_dir, "node_mapping.csv"), index=False)
                saved_files.append("node_mapping.csv")
            except Exception as e: print(f"  ❌ Error saving node mapping for {dataset_label}: {e}")
        else: print(f"  ⚠️ Node mapping data missing for {dataset_label}.")

        # 2. Save Relation Mappings
        if 'relation_to_idx' in rgcn_data and rgcn_data['relation_to_idx']:
            try:
                pd.DataFrame(list(rgcn_data['relation_to_idx'].items()), columns=['RelationName', 'RelationIndex']).to_csv(os.path.join(output_dir, "relation_mapping.csv"), index=False)
                saved_files.append("relation_mapping.csv")
            except Exception as e: print(f"  ❌ Error saving relation mapping for {dataset_label}: {e}")
        else: print(f"  ⚠️ Relation mapping data missing for {dataset_label}.")

        # 3. Save Training Triplets
        if 'training_triplets' in rgcn_data and rgcn_data['training_triplets']: # These now include BELONGS_TO_CATEGORY for neg sampling guidance
             try:
                pd.DataFrame(rgcn_data['training_triplets'], columns=['SourceIndex', 'RelationIndex', 'TargetIndex']).to_csv(os.path.join(output_dir, "training_triplets.csv"), index=False)
                saved_files.append("training_triplets.csv")
             except Exception as e: print(f"  ❌ Error saving training triplets for {dataset_label}: {e}")
        else: print(f"  ⚠️ Training triplets data missing for {dataset_label}.")

        # 4. Save Evaluation Triplets
        if 'evaluation_triplets' in rgcn_data and rgcn_data['evaluation_triplets']: # These are the 30% BELONGS_TO_CATEGORY for val/test
             try:
                pd.DataFrame(rgcn_data['evaluation_triplets'], columns=['SourceIndex', 'RelationIndex', 'TargetIndex']).to_csv(os.path.join(output_dir, "evaluation_triplets.csv"), index=False)
                saved_files.append("evaluation_triplets.csv")
             except Exception as e: print(f"  ❌ Error saving evaluation triplets for {dataset_label}: {e}")
        else: print(f"  ⚠️ Evaluation triplets data missing for {dataset_label}.")

        # 5. Save Embeddings 
        if 'node_embeddings' in rgcn_data and isinstance(rgcn_data['node_embeddings'], np.ndarray):
             try:
                embedding_path = os.path.join(output_dir, "node_embeddings.npy")
                np.save(embedding_path, rgcn_data['node_embeddings'])
                print(f"    Node embeddings saved to '{embedding_path}' (Shape: {rgcn_data['node_embeddings'].shape})")
                saved_files.append("node_embeddings.npy")
             except Exception as e: print(f"  ❌ Error saving node embeddings for {dataset_label}: {e}")
        
        # 6. Save root_category_node_indices_in_graph
        if 'root_category_node_indices_in_graph' in rgcn_data and rgcn_data['root_category_node_indices_in_graph'] is not None:
            try:
                with open(os.path.join(output_dir, "root_category_node_indices_in_graph.json"), 'w') as f:
                    json.dump(rgcn_data['root_category_node_indices_in_graph'], f) # It's a list
                saved_files.append("root_category_node_indices_in_graph.json")
            except Exception as e: print(f"  ❌ Error saving root_category_node_indices_in_graph for {dataset_label}: {e}")
        else: print(f"  ⚠️ root_category_node_indices_in_graph data missing or None for {dataset_label}.")

        # 7. Save cat_idx_to_root_idx_map (NEW)
        if 'cat_idx_to_root_idx_map' in rgcn_data and rgcn_data['cat_idx_to_root_idx_map'] is not None:
            try:
                # Convert integer keys to strings for JSON compatibility if they aren't already
                map_to_save = {str(k): v for k, v in rgcn_data['cat_idx_to_root_idx_map'].items()}
                with open(os.path.join(output_dir, "cat_idx_to_root_idx_map.json"), 'w') as f:
                    json.dump(map_to_save, f)
                saved_files.append("cat_idx_to_root_idx_map.json")
            except Exception as e: print(f"  ❌ Error saving cat_idx_to_root_idx_map for {dataset_label}: {e}")
        else: print(f"  ⚠️ cat_idx_to_root_idx_map data missing or None for {dataset_label}.")


        print(f"  Finished saving for '{dataset_label}'. {len(saved_files)} files created/attempted in '{output_dir}'.")

    except Exception as e:
        print(f"An unexpected error occurred saving data for {dataset_label}: {e}")
        traceback.print_exc()



# Bert Embedding For RGCN + MLP : Including Ontology 

def format_rgcn_data_for_study(dfs, 
                               triplets_for_study, 
                               study_name,
                               category_hierarchy: CategoryHierarchy,
                               target_connection_level: int,
                               bert_model_name="all-MiniLM-L6-v2"):
    """
    Formats data for RGCN with ALL nodes using BERT embeddings, with flexible
    category linking.

    - All nodes (Ceramic, Function, Feature, Category) use BERT embeddings.
    - BELONGS_TO_CATEGORY links are established from Ceramic_Node -> Category_Node
      at the specified `target_connection_level`.
    - Category-to-feature/function links originate ONLY from Root Category Nodes.
    
    Args:
        ...
        category_hierarchy (CategoryHierarchy): An initialized hierarchy manager object.
        target_connection_level (int): The hierarchy level (0=root, 1, 2, etc.) to
                                       link ceramics to.
    Returns:
        dict: RGCN formatted data, or None on failure.
    """
    print(f"\n  🔄 Formatting {study_name} data for RGCN...")
    print(f"    🎯 Target Ceramic->Category Connection Level: {target_connection_level}")
    print(f"    🤖 BERT Model: {bert_model_name}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = None
    try:
        from sentence_transformers import SentenceTransformer
        bert_model = SentenceTransformer(bert_model_name, device=device)
        bert_embedding_dim_native = bert_model.get_sentence_embedding_dimension()
        print(f"    📊 BERT Embedding Dimension: {bert_embedding_dim_native}D")
    except Exception as e:
        print(f"    ❌ Error loading Sentence-BERT model '{bert_model_name}': {e}")
        return None
        
    def get_bert_embedding_for_node(text, level=None, node_type=None):
        if not text or (isinstance(text, str) and not text.strip()): return None
        if isinstance(text, (int, float)) and pd.isna(text): return None
        combined_text = str(text)
        if level is not None and pd.notna(level): combined_text += f" [Level {level}]"
        if node_type is not None: combined_text += f" [{node_type}]"
        try:
            return bert_model.encode(combined_text, convert_to_numpy=True, show_progress_bar=False)
        except Exception: return None
    
    final_embedding_dim = bert_embedding_dim_native

    ceramic_summary = dfs['ceramic_summary'].copy()
    object_function = dfs['object_function'].copy()
    tech_cat = dfs['tech_cat'].copy()
    Features_Ontology_df = dfs['Features_Ontology'].copy()
    tech_cat_func_attrib = dfs['tech_cat_function_attrib'].copy()
    tech_cat_feat_attrib = dfs['tech_cat_feature_attrib'].copy()
    
    # Standardize types (simplified for brevity, use your original robust code)
    ceramic_summary['ceramic_id'] = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce').dropna().astype(int)
    ceramic_summary['tech_cat_id'] = pd.to_numeric(ceramic_summary['tech_cat_id'], errors='coerce')
    object_function['id'] = pd.to_numeric(object_function['id'], errors='coerce').dropna().astype(int)
    tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
    Features_Ontology_df['id'] = Features_Ontology_df['id'].astype(str)
    tech_cat_func_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_func_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
    tech_cat_func_attrib['function_id'] = pd.to_numeric(tech_cat_func_attrib['function_id'], errors='coerce').dropna().astype(int)
    tech_cat_feat_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_feat_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
    tech_cat_feat_attrib['feature_id'] = tech_cat_feat_attrib['feature_id'].astype(str)
    
    # --- NEW: Identify all required nodes for the graph ---
    print("    🔍 Identifying all unique nodes in the sampled data...")
    nodes_in_sample = set()
    
    # Add all functions, features, and their parents from the sample
    for entry in triplets_for_study:
        # Add Ceramic node
        try:
            nodes_in_sample.add(f"Ceramic_{int(float(entry.get('ceramic_id')))}")
        except (TypeError, ValueError):
            continue

        # Add Function nodes and their parents
        for fid, parents_ids in entry.get("functions", []):
            nodes_in_sample.add(f"Func_{fid}")
            for pid in parents_ids:
                nodes_in_sample.add(f"Func_{pid}")

        # Add Feature nodes and their parents
        for fid, parents_ids in entry.get("features", []):
            nodes_in_sample.add(f"Feat_{str(fid)}")
            for pid in parents_ids:
                nodes_in_sample.add(f"Feat_{str(pid)}")
        
        # Add the specific TARGET CATEGORY nodes for the ceramics in the sample
        categories_in_entry = entry.get("categories", [])
        if categories_in_entry:
            most_specific_cat_id = categories_in_entry[0].get('category_id')
            if most_specific_cat_id is not None and pd.notna(most_specific_cat_id):
                target_ancestor_id = category_hierarchy.get_ancestor_at_level(most_specific_cat_id, target_connection_level)
                if target_ancestor_id is not None:
                    nodes_in_sample.add(f"Cat_{target_ancestor_id}")

    # Add ALL ROOT categories to the graph. They are needed for the Category->Function/Feature links.
    for root_id in category_hierarchy.roots:
        nodes_in_sample.add(f"Cat_{root_id}")

    print(f"    📋 Found {len(nodes_in_sample)} unique node identifiers to include in the graph.")

    # --- Indexing and Embedding Generation (Largely Unchanged) ---
    node_to_idx = {}
    idx_counter = 0
    
    ceramic_summary.set_index('ceramic_id', inplace=True, drop=False)
    object_function.set_index('id', inplace=True, drop=False)
    tech_cat.set_index('id', inplace=True, drop=False)
    Features_Ontology_df.set_index('id', inplace=True, drop=False)
    
    embedding_matrix = np.zeros((len(nodes_in_sample), final_embedding_dim), dtype=np.float32)
    
    print("    🎯 Assigning graph indices and generating ALL-BERT embeddings...")
    for node_identifier in sorted(list(nodes_in_sample)):
        if node_identifier not in node_to_idx:
            node_idx_val = idx_counter
            node_to_idx[node_identifier] = node_idx_val
            idx_counter += 1
            
            source_text, level_info, node_type = None, None, None
            try:
                if node_identifier.startswith("Ceramic_"):
                    # Ceramic embedding logic...
                    ceramic_id = int(node_identifier.split('_')[1])
                    source_text = ceramic_summary.loc[ceramic_id, 'description']
                    node_type = "Ceramic"
                elif node_identifier.startswith("Func_"):
                    # Function embedding logic...
                    func_id = int(node_identifier.split('_')[1])
                    source_text = object_function.loc[func_id, 'function_name_fr']
                    node_type = "Function"
                elif node_identifier.startswith("Feat_"):
                    # Feature embedding logic...
                    feat_id = node_identifier.split('_')[1]
                    source_text = Features_Ontology_df.loc[feat_id, 'Nom complet']
                    node_type = "Feature"
                elif node_identifier.startswith("Cat_"):
                    # Category embedding logic...
                    cat_id = int(node_identifier.split('_')[1])
                    source_text = category_hierarchy.cat_names.get(cat_id, f"Unknown Category {cat_id}")
                    level_info = category_hierarchy.get_level(cat_id)
                    node_type = "Category"

                if source_text:
                    emb = get_bert_embedding_for_node(source_text, level_info, node_type)
                    if emb is not None:
                        embedding_matrix[node_idx_val] = emb.flatten()
            except (KeyError, IndexError) as e:
                # print(f"      Warning: Could not find data for node {node_identifier}. Embedding will be zero. Error: {e}")
                pass
            except Exception as e:
                print(f"      ERROR processing node {node_identifier}: {e}")
    
    # --- Triplet Generation (Modified) ---
    print("    🔗 Processing triplets using pre-assigned graph indices...")
    training_triplets = []
    evaluation_triplets = []
    relation_to_idx = {}
    rel_idx_counter = 0

    def get_or_assign_relation_idx(name):
        nonlocal rel_idx_counter
        if name not in relation_to_idx:
            relation_to_idx[name] = rel_idx_counter
            rel_idx_counter += 1
        return relation_to_idx[name]

    BELONGS_TO_REL = get_or_assign_relation_idx("BELONGS_TO_CATEGORY")
    HAS_FUNCTION_REL = get_or_assign_relation_idx("HAS_FUNCTION")
    IS_A_REL = get_or_assign_relation_idx("IS_A")
    HAS_FEATURE_REL = get_or_assign_relation_idx("HAS_FEATURE") 
    feature_id_to_data = Features_Ontology_df.to_dict('index')

    for entry in triplets_for_study:
        try:
            cid_val = int(float(entry.get("ceramic_id")))
            ceramic_node_id = f"Ceramic_{cid_val}"
            ceramic_idx = node_to_idx.get(ceramic_node_id)
            if ceramic_idx is None: continue
        except (TypeError, ValueError):
            continue

        # --- MODIFIED: BELONGS_TO_CATEGORY Triplets ---
        categories_in_entry = entry.get("categories", [])
        if categories_in_entry:
            most_specific_cat_id = categories_in_entry[0].get('category_id')
            if most_specific_cat_id is not None and pd.notna(most_specific_cat_id):
                target_ancestor_id = category_hierarchy.get_ancestor_at_level(most_specific_cat_id, target_connection_level)
                if target_ancestor_id is not None:
                    target_cat_node_id = f"Cat_{target_ancestor_id}"
                    target_cat_idx = node_to_idx.get(target_cat_node_id)
                    if target_cat_idx is not None:
                        evaluation_triplets.append((ceramic_idx, BELONGS_TO_REL, target_cat_idx))

        # Ceramic -> Function & Function IS_A Hierarchy
        for func_id, parents_ids in entry.get("functions", []):
            func_idx = node_to_idx.get(f"Func_{func_id}")
            if func_idx is not None:
                training_triplets.append((ceramic_idx, HAS_FUNCTION_REL, func_idx))
                # Add IS-A links
                child_idx = func_idx
                for p_id in parents_ids:
                    parent_idx = node_to_idx.get(f"Func_{p_id}")
                    if parent_idx is not None:
                        training_triplets.append((child_idx, IS_A_REL, parent_idx))
                        child_idx = parent_idx
        
        # Ceramic -> Feature & Feature IS_A Hierarchy
        for feat_id, parents_ids in entry.get("features", []):
            feat_id_str = str(feat_id)
            feat_idx = node_to_idx.get(f"Feat_{feat_id_str}")
            if feat_idx is not None:
                # Use specific relation if available, otherwise default
                relation_name = feature_id_to_data.get(feat_id_str, {}).get('Ceramic_Relation', 'HAS_FEATURE')
                if pd.isna(relation_name) or not str(relation_name).strip(): relation_name = 'HAS_FEATURE'
                rel_idx = get_or_assign_relation_idx(relation_name)
                training_triplets.append((ceramic_idx, rel_idx, feat_idx))
                # Add IS-A links for features
                child_idx = feat_idx
                child_data = feature_id_to_data.get(feat_id_str, {})
                for p_id in parents_ids:
                    p_id_str = str(p_id)
                    parent_idx = node_to_idx.get(f"Feat_{p_id_str}")
                    if parent_idx is not None:
                        # Get relation from child to parent
                        parent_rel_name = get_feature_parent_relation_label(child_data)
                        parent_rel_idx = get_or_assign_relation_idx(parent_rel_name)
                        training_triplets.append((child_idx, parent_rel_idx, parent_idx))
                        child_idx = parent_idx
                        child_data = feature_id_to_data.get(p_id_str, {})

    # --- Category to Function/Feature links ---
    # These links ALWAYS originate from the ROOT categories.
    print("    🌳 Adding RootCategory->Function/Feature triplets...")
    for root_id in category_hierarchy.roots:
        root_node_id = f"Cat_{root_id}"
        root_idx = node_to_idx.get(root_node_id)
        if root_idx is None: continue
        
        # Root -> Functions
        linked_funcs = tech_cat_func_attrib[tech_cat_func_attrib['tech_cat_id'] == root_id]
        for _, row in linked_funcs.iterrows():
            func_idx = node_to_idx.get(f"Func_{int(row['function_id'])}")
            if func_idx is not None:
                training_triplets.append((root_idx, HAS_FUNCTION_REL, func_idx))

        # Root -> Features
        linked_feats = tech_cat_feat_attrib[tech_cat_feat_attrib['tech_cat_id'] == root_id]
        for _, row in linked_feats.iterrows():
            feat_id_str = str(row['feature_id'])
            feat_idx = node_to_idx.get(f"Feat_{feat_id_str}")
            if feat_idx is not None:
                relation_name = feature_id_to_data.get(feat_id_str, {}).get('Ceramic_Relation', 'HAS_FEATURE')
                if pd.isna(relation_name) or not str(relation_name).strip(): relation_name = 'HAS_FEATURE'
                rel_idx = get_or_assign_relation_idx(relation_name)
                training_triplets.append((root_idx, rel_idx, feat_idx))

    # --- Final RGCN Data Dictionary Construction ---
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    
    # Identify all category nodes at the target level in the graph
    target_level_category_indices = []
    for idx, node_id in idx_to_node.items():
        if node_id.startswith("Cat_"):
            cat_id = int(node_id.split("_")[1])
            if category_hierarchy.get_level(cat_id) == target_connection_level:
                target_level_category_indices.append(idx)

    # Clean up and build final dict
    if bert_model: del bert_model; gc.collect(); torch.cuda.empty_cache()
    
    rgcn_data = {
        "num_nodes": len(node_to_idx),
        "num_relations": len(relation_to_idx),
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "relation_to_idx": relation_to_idx,
        "idx_to_relation": {v: k for k, v in relation_to_idx.items()},
        "training_triplets": sorted(list(set(map(tuple, training_triplets)))),
        "evaluation_triplets": sorted(list(set(map(tuple, evaluation_triplets)))),
        "node_embeddings": embedding_matrix,
        "embedding_dim": final_embedding_dim,    
        "target_category_node_indices": sorted(list(set(target_level_category_indices))),
        "stats": { 
            "study_name": study_name,
            "target_connection_level": target_connection_level,
            "bert_model_used": bert_model_name,
            "native_bert_dim": bert_embedding_dim_native,
            "final_embedding_dim": final_embedding_dim,
            "target_dim_for_pca_and_ohe": final_embedding_dim,
            "total_nodes_in_graph": len(node_to_idx),
            "total_relations": len(relation_to_idx)
        }
    }
    
    print(f"    ✅ Formatted data for {study_name}. "
          f"Train triplets: {len(rgcn_data['training_triplets'])}, "
          f"Eval triplets: {len(rgcn_data['evaluation_triplets'])}. "
          f"Target categories: {len(rgcn_data['target_category_node_indices'])}")

    return rgcn_data


def prepare_all_level_based_studies(dfs, bert_model_name="all-MiniLM-L6-v2", auto_save=True, base_output_dir="output/rgcn_data/ontology"):
    """
    Main orchestrator for the comparative study with DYNAMIC ROOT DISCOVERY.

    This version implements the corrected logic:
    1.  **Dynamically discovers root categories** (those with no parent).
    2.  **Selects ONE master set of ceramics**: All those belonging to categories
        at Level 2 or deeper.
    3.  **Samples this master set THREE times**: Once for each "étude".
    4.  **Generates NINE datasets**: By taking each of the three sampled sets and
        linking them to their ancestors at Level 2, Level 1, and Level 0.
    
    Args:
        dfs: Dictionary of DataFrames
        bert_model_name: Name of the BERT model to use
        auto_save: If True, automatically saves the datasets
        base_output_dir: Directory to save the datasets
    """
    print("======================================================")
    print("=== STARTING DATA PREPARATION (Dynamic Root Discovery) ===")
    print("======================================================")
    print(f"🤖 BERT Model: {bert_model_name}")

    # Load BERT model to get embedding info
    try:
        from sentence_transformers import SentenceTransformer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bert_model_temp = SentenceTransformer(bert_model_name, device=device)
        bert_embedding_dim = bert_model_temp.get_sentence_embedding_dimension()
        print(f"📊 BERT Embedding Dimension: {bert_embedding_dim}")
        del bert_model_temp  # Clean up
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"❌ Error loading BERT model for info: {e}")
        bert_embedding_dim = "Unknown"

    # --- 0. Initialize Hierarchy and DYNAMICALLY Discover Roots ---
    try:
        print("Initializing CategoryHierarchy to discover roots from data...")
       
        # Demonstrate the hierarchy
        hierarchy = demonstrate_hierarchy_usage(dfs['tech_cat'])
        
        # Export to CSV
        export_df = export_hierarchy_to_csv(hierarchy, "category_hierarchy_paths.csv")
        # **DYNAMIC DISCOVERY**: Use the roots found by the hierarchy class.
        discovered_root_ids = hierarchy.roots
        print(f"✅ Dynamically discovered {len(discovered_root_ids)} root categories: {discovered_root_ids}\n")

        # Map all ceramics to their info, including their discovered root.
        ceramic_to_info = {}
        for _, row in dfs['ceramic_summary'].iterrows():
            cid = int(row['ceramic_id'])
            cat_id = row.get('tech_cat_id')
            if pd.notna(cat_id):
                cat_id = int(cat_id)
                root = hierarchy.get_root(cat_id)
                # If a root exists, the ceramic is part of a valid tree.
                if root is not None:
                    level = hierarchy.get_level(cat_id)
                    if level is not None:
                         ceramic_to_info[cid] = {'cat_id': cat_id, 'level': level, 'root': root}

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to initialize hierarchy or map ceramics: {e}")
        traceback.print_exc()
        return None

    # --- 1. Select the Master Set of Candidate Ceramics (Fixed Rule) ---
    print("--- STEP 1: Selecting Master Set of Ceramics (Level >= 2) ---")
    candidate_ceramics_by_root = defaultdict(list)
    for cid, info in ceramic_to_info.items():
        # The ceramic's root must be one of the dynamically discovered ones.
        if info['root'] in discovered_root_ids and info['level'] >= 2:
            candidate_ceramics_by_root[info['root']].append(cid)
    
    candidate_counts = {root: len(cids) for root, cids in candidate_ceramics_by_root.items()}
    print(f"  Master candidate pool (ceramics from L2+) counts per discovered root: {candidate_counts}")
    if not candidate_counts:
        print("❌ No candidate ceramics found at Level 2 or deeper. Aborting.")
        return None

    # --- 2. Perform Sampling for Each Étude on the Master Set ---
    print("\n--- STEP 2: Sampling Master Set for Each Étude ---")
    
    # Note: The exclusion of root 140 is an experimental parameter. It is kept here
    # as per the study design, even though roots are discovered dynamically.
    counts_for_etude_3 = {r: c for r, c in candidate_counts.items() if r != 140}
    min_count_etude_3 = min(counts_for_etude_3.values()) if counts_for_etude_3 else 0

    etude_definitions = {
        'etude1': {'target_size': 126, 'exclude_root': None},
        'etude1_prime': {'target_size': 252, 'exclude_root': None},
        'etude2': {'target_size': min_count_etude_3, 'exclude_root': 140}
    }

    sampled_ceramics_per_etude = {}
    for etude_name, params in etude_definitions.items():
        print(f"  Sampling for {etude_name} (size={params['target_size']}, exclude_root={params['exclude_root']})...")
        if params['target_size'] <= 0:
            print(f"    SKIPPING {etude_name}: Target size is {params['target_size']}.")
            sampled_ceramics_per_etude[etude_name] = []
            continue

        selected_ceramics = []
        # Use the dynamically discovered roots for sampling.
        roots_to_sample_from = set(candidate_ceramics_by_root.keys())
        if params['exclude_root'] is not None:
            roots_to_sample_from.discard(params['exclude_root'])
        
        for root_id in sorted(list(roots_to_sample_from)):
            candidates = candidate_ceramics_by_root.get(root_id, [])
            if not candidates: continue
            
            scores = calculate_completeness_score(candidates, dfs)
            sorted_candidates = sorted(candidates, key=lambda cid: (-scores.get(cid, 0), cid))
            num_to_sample = min(len(sorted_candidates), params['target_size'])
            selected_ceramics.extend(sorted_candidates[:num_to_sample])
        
        unique_ceramics = sorted(list(set(selected_ceramics)))
        sampled_ceramics_per_etude[etude_name] = unique_ceramics
        print(f"    -> Selected {len(unique_ceramics)} unique ceramics for {etude_name}.")

    # --- 3. Generate All 9 Datasets by Looping Through Linking Levels and Études ---
    print("\n--- STEP 3: Generating All 9 Datasets with Variable Linking ---")
    all_study_datasets = {}

    for linking_level in [2, 1, 0]:
        repo_name = f"level_{linking_level}_connections"
        all_study_datasets[repo_name] = {}
        print(f"\n======================================================\n--- Generating Repo: {repo_name} (Linking to Level {linking_level}) ---")

        for etude_name, ceramic_ids in sampled_ceramics_per_etude.items():
            print(f"  --- Processing: {etude_name} ---")

            if not ceramic_ids:
                print(f"    SKIPPING: No ceramics were sampled for {etude_name}.")
                all_study_datasets[repo_name][etude_name] = None
                continue
            
            print(f"    Using {len(ceramic_ids)} pre-sampled ceramics.")
            print(f"    Extracting triplets...")
            triplets_for_study = extract_triplets_for_selection(ceramic_ids, dfs)
            if not triplets_for_study:
                print(f"    ❌ Error: Failed to extract triplets for {etude_name}. Skipping.")
                all_study_datasets[repo_name][etude_name] = None
                continue

            rgcn_data = format_rgcn_data_for_study(
                dfs=dfs,
                triplets_for_study=triplets_for_study,
                study_name=f"{repo_name}_{etude_name}",
                category_hierarchy=hierarchy,
                target_connection_level=linking_level,
                bert_model_name=bert_model_name
            )
            
            all_study_datasets[repo_name][etude_name] = rgcn_data
            if rgcn_data:
                print(f"    ✅ Successfully prepared dataset for {repo_name} / {etude_name}")
                print(f"    📊 Embedding Info: {bert_model_name} -> {bert_embedding_dim}D")
            else:
                print(f"    ❌ FAILED to prepare dataset for {repo_name} / {etude_name}")

    print("\n======================================================")
    print("=== FINISHED ALL COMPARATIVE STUDY PREPARATION     ===")
    print("======================================================")
    
    # --- AUTO-SAVE if requested ---
    if auto_save:
        print(f"\n🔄 AUTO-SAVING datasets to '{base_output_dir}'...")
        save_all_study_datasets(all_study_datasets, base_output_dir)
        print(f"✅ All datasets saved successfully!")
    else:
        print(f"\n⚠️  Auto-save disabled. Call save_all_study_datasets() manually to save.")
    
    return all_study_datasets


def save_all_study_datasets(all_study_results, base_output_dir="output/rgcn_data"):
    """
    Saves the RGCN data for each repo and etude into a nested directory structure.
    e.g., output/rgcn_data/level_2_connections/etude_1/
    """
    if not all_study_results:
        print("Error: Study results dictionary is empty. Nothing to save.")
        return

    print(f"\n💾 --- Saving All Study Datasets to: '{base_output_dir}' ---")
    
    # Create base directory if it doesn't exist
    os.makedirs(base_output_dir, exist_ok=True)

    for repo_name, etudes_dict in all_study_results.items():
        print(f"\n📁 --- Saving Repo: {repo_name} ---")
        for etude_name, rgcn_data in etudes_dict.items():
            if rgcn_data is None:
                print(f"  ⚠️  Skipping '{etude_name}' (data is None).")
                continue
            
            # Create a nested output directory
            output_dir = os.path.join(base_output_dir, repo_name, etude_name)
            print(f"  💾 Saving '{etude_name}' data to '{output_dir}'...")
            
            # Reuse your single dataset saving logic
            # You might need to adapt save_single_rgcn_dataset to handle the new fields
            # like 'target_category_node_indices' instead of 'root_category_node_indices_in_graph'
            save_single_rgcn_dataset(rgcn_data, output_dir, dataset_label=f"{repo_name}/{etude_name}")

    print(f"\n✅ --- Finished saving all study datasets to '{base_output_dir}' ---")


