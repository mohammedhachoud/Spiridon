import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
import torch
from sklearn.decomposition import PCA
import gc
import ast
from collections import Counter, defaultdict
import pandas as pd

from graph_utils import (
    build_category_hierarchy_and_map_ceramics,
    calculate_completeness_score,
    extract_triplets_for_selection
)
from utils import get_feature_parent_relation_label, CategoryHierarchy, export_hierarchy_to_csv, demonstrate_hierarchy_usage


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
        
        # 6. Save target_category_node_indices (updated from root_category_node_indices_in_graph)
        if 'target_category_node_indices' in rgcn_data and rgcn_data['target_category_node_indices'] is not None:
            try:
                with open(os.path.join(output_dir, "target_category_node_indices.json"), 'w') as f:
                    json.dump(rgcn_data['target_category_node_indices'], f) # It's a list
                saved_files.append("target_category_node_indices.json")
            except Exception as e: print(f"  ❌ Error saving target_category_node_indices for {dataset_label}: {e}")
        else: print(f"  ⚠️ target_category_node_indices data missing or None for {dataset_label}.")

        # 7. Save cat_idx_to_root_idx_map (if present - may not be needed in new version)
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


def _generate_ceramic_attribute_embeddings(df):
    """Generate attribute-based embeddings for ceramic nodes"""
    print("    Generating ceramic attribute embeddings...")
    origin_values = sorted(df['origin'].dropna().unique())
    all_colors = set()
    for color_list in df['color_name_list'].dropna():
        try: 
            color_list = ast.literal_eval(color_list) if isinstance(color_list, str) else color_list
        except: 
            continue
        if isinstance(color_list, list): 
            all_colors.update([str(c).strip() for c in color_list if pd.notna(c)])
    color_values = sorted(list(all_colors))
    context_values = sorted(df['context_type_name'].dropna().unique())
    source_values = sorted(df['identifier_origin_source_name'].dropna().unique())
    
    origin_map = {v: i for i, v in enumerate(origin_values)}
    color_map = {v: i for i, v in enumerate(color_values)}
    context_map = {v: i for i, v in enumerate(context_values)}
    source_map = {v: i for i, v in enumerate(source_values)}

    def get_ceramic_embedding(row):
        origin_vector = [0] * len(origin_map)
        color_vector = [0] * len(color_map)
        context_vector = [0] * len(context_map)
        source_vector = [0] * len(source_map)
        
        if pd.notna(row['origin']) and row['origin'] in origin_map: 
            origin_vector[origin_map[row['origin']]] = 1
            
        color_list = row['color_name_list']
        try: 
            color_list = ast.literal_eval(color_list) if isinstance(color_list, str) else color_list
        except: 
            color_list = []
        if isinstance(color_list, list):
            for c_val in color_list:
                c_str = str(c_val).strip()
                if c_str in color_map: 
                    color_vector[color_map[c_str]] = 1
                    
        if pd.notna(row['context_type_name']) and row['context_type_name'] in context_map: 
            context_vector[context_map[row['context_type_name']]] = 1
            
        if pd.notna(row['identifier_origin_source_name']) and row['identifier_origin_source_name'] in source_map: 
            source_vector[source_map[row['identifier_origin_source_name']]] = 1
            
        reuse_emb = [0, 0] if pd.isna(row['reuse']) else ([0, 1] if row['reuse'] else [1, 0])
        prod_fail_emb = [0, 0] if pd.isna(row['production_fail']) else ([0, 1] if row['production_fail'] else [1, 0])
        
        return origin_vector + color_vector + context_vector + source_vector + reuse_emb + prod_fail_emb
    
    df['ceramic_attribute_embedding'] = df.apply(get_ceramic_embedding, axis=1)
    info = {'total_dim': len(origin_map) + len(color_map) + len(context_map) + len(source_map) + 4}
    maps = {'origin_map': origin_map, 'color_map': color_map, 'context_map': context_map, 'source_map': source_map}
    print(f"    Attribute embedding generated. Length: {info['total_dim']}")
    return df, info, maps


def debug_missing_ceramics_detailed(ceramic_summary, ceramic_ids_in_sample):
    """
    Enhanced debugging to find where ceramic IDs are being lost
    """
    print("🔍 DETAILED CERAMIC ID DEBUGGING:")
    
    # Check original data types and values
    print(f"📊 Original ceramic_summary['ceramic_id'] dtype: {ceramic_summary['ceramic_id'].dtype}")
    print(f"📊 Sample of original ceramic_id values: {ceramic_summary['ceramic_id'].head(10).tolist()}")
    
    # Check for any NaN values
    nan_count = ceramic_summary['ceramic_id'].isna().sum()
    print(f"📊 NaN values in ceramic_id column: {nan_count}")
    
    # Check the conversion process step by step
    print("\n🔄 STEP-BY-STEP CONVERSION DEBUGGING:")
    
    # Step 1: to_numeric conversion
    numeric_converted = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce')
    print(f"Step 1 - After to_numeric: {numeric_converted.isna().sum()} NaN values")
    
    # Step 2: dropna
    after_dropna = numeric_converted.dropna()
    print(f"Step 2 - After dropna: {len(after_dropna)} remaining values")
    print(f"         Lost in dropna: {len(numeric_converted) - len(after_dropna)} values")
    
    # Step 3: astype(int)
    try:
        final_converted = after_dropna.astype(int)
        print(f"Step 3 - After astype(int): {len(final_converted)} values")
    except Exception as e:
        print(f"Step 3 - ERROR in astype(int): {e}")
        return
    
    # Check what IDs were lost in conversion
    original_ids = set(ceramic_summary.index)  # Assuming index is set to ceramic_id
    converted_ids = set(final_converted.values)
    lost_in_conversion = original_ids - converted_ids
    
    if lost_in_conversion:
        print(f"\n❌ IDs LOST IN CONVERSION: {len(lost_in_conversion)}")
        print(f"   Sample lost IDs: {list(lost_in_conversion)[:10]}")
    
    # Check missing IDs from sample
    missing_ids = ceramic_ids_in_sample - converted_ids
    if missing_ids:
        print(f"\n❌ MISSING IDs FROM SAMPLE: {len(missing_ids)}")
        sample_missing = list(missing_ids)[:10]
        print(f"   Sample missing: {sample_missing}")
        
        # Check if these exist in original data
        for missing_id in sample_missing:
            if missing_id in ceramic_summary['ceramic_id'].values:
                print(f"   ✅ ID {missing_id} EXISTS in original ceramic_summary")
                # Check why it was lost
                original_value = ceramic_summary[ceramic_summary['ceramic_id'] == missing_id]['ceramic_id'].iloc[0]
                print(f"      Original value: {original_value} (type: {type(original_value)})")
                try:
                    converted = pd.to_numeric(original_value, errors='coerce')
                    print(f"      to_numeric result: {converted} (type: {type(converted)})")
                except Exception as e:
                    print(f"      to_numeric ERROR: {e}")
            else:
                print(f"   ❌ ID {missing_id} NOT FOUND in original ceramic_summary")


def fix_ceramic_id_processing(ceramic_summary):
    """
    Fixed version of ceramic ID processing that preserves all valid IDs
    """
    print("🔧 APPLYING FIXED CERAMIC ID PROCESSING:")
    
    # Make a copy to avoid modifying original
    df = ceramic_summary.copy()
    
    # More careful conversion process
    print(f"📊 Starting with {len(df)} rows")
    
    # First, ensure ceramic_id is string for consistent processing
    df['ceramic_id'] = df['ceramic_id'].astype(str)
    
    # Remove obvious non-numeric values but be more careful
    # Only remove NaN, empty strings, and clearly non-numeric values
    valid_mask = df['ceramic_id'].notna() & (df['ceramic_id'] != '') & (df['ceramic_id'] != 'nan')
    df = df[valid_mask]
    print(f"📊 After removing NaN/empty: {len(df)} rows")
    
    # Try to convert to numeric, but handle errors more carefully
    def safe_convert_to_int(value):
        try:
            # Handle different formats
            if isinstance(value, str):
                # Remove any whitespace
                value = value.strip()
                # Try to convert to float first (handles scientific notation)
                float_val = float(value)
                # Then to int
                return int(float_val)
            else:
                return int(float(value))
        except (ValueError, TypeError):
            return None
    
    df['ceramic_id_converted'] = df['ceramic_id'].apply(safe_convert_to_int)
    
    # Remove rows where conversion failed
    conversion_failed = df['ceramic_id_converted'].isna()
    if conversion_failed.sum() > 0:
        print(f"📊 Conversion failed for {conversion_failed.sum()} rows")
        failed_values = df[conversion_failed]['ceramic_id'].head(10).tolist()
        print(f"   Sample failed values: {failed_values}")
    
    df = df[df['ceramic_id_converted'].notna()]
    print(f"📊 After conversion: {len(df)} rows")
    
    # Replace the original ceramic_id with converted values
    df['ceramic_id'] = df['ceramic_id_converted'].astype(int)
    df = df.drop('ceramic_id_converted', axis=1)
    
    # Also convert tech_cat_id more carefully
    if 'tech_cat_id' in df.columns:
        df['tech_cat_id'] = pd.to_numeric(df['tech_cat_id'], errors='coerce')
    
    print(f"✅ Final processed DataFrame: {len(df)} rows")
    print(f"   ceramic_id range: {df['ceramic_id'].min()} to {df['ceramic_id'].max()}")
    
    return df


def format_rgcn_data_for_study_onehot_fixed(dfs, 
                                      triplets_for_study, 
                                      study_name,
                                      category_hierarchy,
                                      target_connection_level: int):
    """
    FIXED version - the issue is index management after data conversion
    """
    print(f"\n  🔄 Formatting {study_name} data for RGCN with TRUE ONE-HOT ENCODINGS (FIXED)...")
    print(f"    🎯 Target Ceramic->Category Connection Level: {target_connection_level}")

    ceramic_summary = dfs['ceramic_summary'].copy()
    object_function = dfs['object_function'].copy()
    tech_cat = dfs['tech_cat'].copy()
    Features_Ontology_df = dfs['Features_Ontology'].copy()
    tech_cat_func_attrib = dfs['tech_cat_function_attrib'].copy()
    tech_cat_feat_attrib = dfs['tech_cat_feature_attrib'].copy()
    
    # FIXED: Handle data conversion more carefully
    print("🔧 FIXED: Converting ceramic_id with proper index management...")
    
    # Convert ceramic_id column
    ceramic_summary['ceramic_id'] = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce').dropna().astype(int)
    ceramic_summary['tech_cat_id'] = pd.to_numeric(ceramic_summary['tech_cat_id'], errors='coerce')
    
    # Remove rows where ceramic_id conversion failed
    ceramic_summary = ceramic_summary.dropna(subset=['ceramic_id'])
    
    print(f"📊 After ceramic_id conversion: {len(ceramic_summary)} rows")
    print(f"📊 Ceramic ID range: {ceramic_summary['ceramic_id'].min()} to {ceramic_summary['ceramic_id'].max()}")
    
    # Convert other columns
    object_function['id'] = pd.to_numeric(object_function['id'], errors='coerce').dropna().astype(int)
    tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
    Features_Ontology_df['id'] = Features_Ontology_df['id'].astype(str)
    tech_cat_func_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_func_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
    tech_cat_func_attrib['function_id'] = pd.to_numeric(tech_cat_func_attrib['function_id'], errors='coerce').dropna().astype(int)
    tech_cat_feat_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_feat_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
    tech_cat_feat_attrib['feature_id'] = tech_cat_feat_attrib['feature_id'].astype(str)
    
    # --- Build FULL VOCABULARY from entire database ---
    print("    🌍 Building full vocabulary from entire database...")
    
    # All ceramics from database
    all_ceramic_ids = sorted(ceramic_summary['ceramic_id'].unique())
    
    # All functions from database  
    all_function_ids = sorted(object_function['id'].unique())
    
    # All features from database
    all_feature_ids = sorted(Features_Ontology_df['id'].unique())
    
    # All categories from database
    all_category_ids = sorted(list(category_hierarchy.parent_map.keys()))
    
    print(f"    📊 Full vocabulary sizes:")
    print(f"      🏺 All Ceramics: {len(all_ceramic_ids)}")
    print(f"      ⚙️  All Functions: {len(all_function_ids)}")
    print(f"      🔧 All Features: {len(all_feature_ids)}")
    print(f"      📂 All Categories: {len(all_category_ids)}")
    
    # Create position mappings
    function_id_to_pos = {fid: i for i, fid in enumerate(all_function_ids)}
    feature_id_to_pos = {fid: i for i, fid in enumerate(all_feature_ids)}
    category_id_to_pos = {cid: i for i, cid in enumerate(all_category_ids)}
    
    # --- Identify nodes that actually appear in the sampled data ---
    print("    🔍 Identifying nodes present in sampled data...")
    nodes_in_sample = set()
    ceramic_ids_in_sample = set()
    function_ids_in_sample = set()
    feature_ids_in_sample = set()
    category_ids_in_sample = set()
    
    # Collect from triplets
    for entry in triplets_for_study:
        # Add Ceramic node
        try:
            cid = int(float(entry.get('ceramic_id')))
            nodes_in_sample.add(f"Ceramic_{cid}")
            ceramic_ids_in_sample.add(cid)
        except (TypeError, ValueError):
            continue

        # Add Function nodes and their parents
        for fid, parents_ids in entry.get("functions", []):
            nodes_in_sample.add(f"Func_{fid}")
            function_ids_in_sample.add(fid)
            for pid in parents_ids:
                nodes_in_sample.add(f"Func_{pid}")
                function_ids_in_sample.add(pid)

        # Add Feature nodes and their parents  
        for fid, parents_ids in entry.get("features", []):
            fid_str = str(fid)
            nodes_in_sample.add(f"Feat_{fid_str}")
            feature_ids_in_sample.add(fid_str)
            for pid in parents_ids:
                pid_str = str(pid)
                nodes_in_sample.add(f"Feat_{pid_str}")
                feature_ids_in_sample.add(pid_str)
        
        # Add target category nodes for ceramics
        categories_in_entry = entry.get("categories", [])
        if categories_in_entry:
            most_specific_cat_id = categories_in_entry[0].get('category_id')
            if most_specific_cat_id is not None and pd.notna(most_specific_cat_id):
                target_ancestor_id = category_hierarchy.get_ancestor_at_level(most_specific_cat_id, target_connection_level)
                if target_ancestor_id is not None:
                    nodes_in_sample.add(f"Cat_{target_ancestor_id}")
                    category_ids_in_sample.add(target_ancestor_id)

    # Add ALL ROOT categories
    for root_id in category_hierarchy.roots:
        nodes_in_sample.add(f"Cat_{root_id}")
        category_ids_in_sample.add(root_id)

    print(f"    📋 Nodes in sample: {len(nodes_in_sample)}")
    print(f"      🏺 Ceramics in sample: {len(ceramic_ids_in_sample)}")
    print(f"      ⚙️  Functions in sample: {len(function_ids_in_sample)}")
    print(f"      🔧 Features in sample: {len(feature_ids_in_sample)}")
    print(f"      📂 Categories in sample: {len(category_ids_in_sample)}")

    # Create node ordering and global index mapping
    all_nodes_ordered = sorted(list(nodes_in_sample))
    node_to_idx = {node: i for i, node in enumerate(all_nodes_ordered)}

    # FIXED: Debug missing ceramics properly
    print("🔍 DEBUGGING MISSING CERAMICS (FIXED):")
    available_ceramic_ids = set(ceramic_summary['ceramic_id'].values)  # Use .values, not index
    missing_ceramics = ceramic_ids_in_sample - available_ceramic_ids
    
    print(f"📊 Available ceramic IDs: {len(available_ceramic_ids)}")
    print(f"📊 Ceramic IDs in sample: {len(ceramic_ids_in_sample)}")
    print(f"❌ Missing ceramics: {len(missing_ceramics)}")
    
    if missing_ceramics:
        print(f"   Sample missing IDs: {sorted(list(missing_ceramics))[:10]}")

    # FIXED: Generate ceramic attribute embeddings with proper indexing
    print("    🏺 Generating ceramic attribute embeddings...")
    ceramic_summary_with_embeddings, ceramic_embed_info, ceramic_maps = _generate_ceramic_attribute_embeddings(ceramic_summary)
    ceramic_attr_dim = ceramic_embed_info['total_dim']
    
    # CRITICAL FIX: Set index AFTER generating embeddings, and ensure we use ceramic_id values
    print("🔧 CRITICAL FIX: Setting up ceramic lookup properly...")
    ceramic_summary_with_embeddings.set_index('ceramic_id', inplace=True, drop=False)
    print(f"📊 Ceramic lookup index range: {ceramic_summary_with_embeddings.index.min()} to {ceramic_summary_with_embeddings.index.max()}")
    
    # Update total embedding dimension
    total_embedding_dim = ceramic_attr_dim + len(all_function_ids) + len(all_feature_ids) + len(all_category_ids)
    ceramic_offset = 0
    function_offset = ceramic_attr_dim
    feature_offset = ceramic_attr_dim + len(all_function_ids)
    category_offset = ceramic_attr_dim + len(all_function_ids) + len(all_feature_ids)
    
    print(f"    🎯 Total embedding dimension: {total_embedding_dim}")
    print(f"      🏺 Ceramic attributes: {ceramic_attr_dim}")
    print(f"      ⚙️  Function one-hot: {len(all_function_ids)}")
    print(f"      🔧 Feature one-hot: {len(all_feature_ids)}")
    print(f"      📂 Category one-hot: {len(all_category_ids)}")
    
    # --- Generate embeddings ---
    print("    🎯 Generating mixed embeddings (attributes + one-hot)...")
    embedding_matrix = np.zeros((len(nodes_in_sample), total_embedding_dim), dtype=np.float32)
    
    missing_ceramic_count = 0
    for node_identifier in all_nodes_ordered:
        node_idx = node_to_idx[node_identifier]
        
        if node_identifier.startswith("Ceramic_"):
            ceramic_id = int(node_identifier.split("_")[1])
            # FIXED: Check if ceramic exists in our processed data
            if ceramic_id in ceramic_summary_with_embeddings.index:
                ceramic_attr_embedding = ceramic_summary_with_embeddings.loc[ceramic_id, 'ceramic_attribute_embedding']
                embedding_matrix[node_idx, ceramic_offset:ceramic_offset + ceramic_attr_dim] = ceramic_attr_embedding
            else:
                missing_ceramic_count += 1
                if missing_ceramic_count <= 10:  # Only show first 10 warnings
                    print(f"      ⚠️  Warning: Ceramic {ceramic_id} not found in ceramic_summary")
                elif missing_ceramic_count == 11:
                    print(f"      ⚠️  ... (suppressing further ceramic warnings)")
            
        elif node_identifier.startswith("Func_"):
            function_id = int(node_identifier.split("_")[1])
            pos = function_offset + function_id_to_pos[function_id]
            embedding_matrix[node_idx, pos] = 1.0
            
        elif node_identifier.startswith("Feat_"):
            feature_id = node_identifier.split("_")[1]
            pos = feature_offset + feature_id_to_pos[feature_id]
            embedding_matrix[node_idx, pos] = 1.0
            
        elif node_identifier.startswith("Cat_"):
            category_id = int(node_identifier.split("_")[1])
            pos = category_offset + category_id_to_pos[category_id]
            embedding_matrix[node_idx, pos] = 1.0

    if missing_ceramic_count > 0:
        print(f"    ❌ TOTAL MISSING CERAMICS: {missing_ceramic_count}")

    # Debug: Show embedding structure for first node of each type
    print("    🔍 EMBEDDING STRUCTURE DEBUG:")
    for node_type, prefix in [("Ceramic", "Ceramic_"), ("Function", "Func_"), 
                              ("Feature", "Feat_"), ("Category", "Cat_")]:
        sample_nodes = [n for n in all_nodes_ordered if n.startswith(prefix)]
        if sample_nodes:
            first_node = sample_nodes[0]
            node_idx = node_to_idx[first_node]
            embedding = embedding_matrix[node_idx]
            
            if node_type == "Ceramic":
                # For ceramics, show attribute embedding structure
                non_zero_positions = np.where(embedding != 0)[0]
                print(f"      {node_type} {first_node} (idx {node_idx}): {len(non_zero_positions)} non-zero positions in attribute embedding")
                print(f"        Ceramic attribute embedding range: [0:{ceramic_attr_dim}], sum: {embedding[:ceramic_attr_dim].sum()}")
            else:
                # For others, show one-hot bit position
                non_zero_positions = np.where(embedding == 1.0)[0]
                print(f"      {node_type} {first_node} (idx {node_idx}): bit at position {non_zero_positions[0] if len(non_zero_positions) > 0 else 'None'}")
                print(f"        Embedding shape: {embedding.shape}, sum: {embedding.sum()}")
    
    # Set dataframe indices for lookups
    ceramic_summary.set_index('ceramic_id', inplace=True, drop=False)
    object_function.set_index('id', inplace=True, drop=False)
    tech_cat.set_index('id', inplace=True, drop=False)
    Features_Ontology_df.set_index('id', inplace=True, drop=False)
    
    # --- Triplet Generation (Same logic as before) ---
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

        # --- BELONGS_TO_CATEGORY Triplets ---
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
        "embedding_dim": total_embedding_dim,    
        "target_category_node_indices": sorted(list(set(target_level_category_indices))),
        # Full vocabulary mappings for reference (excluding ceramics since they use attributes)
        "full_vocabulary": {
            "function_id_to_pos": function_id_to_pos,
            "feature_id_to_pos": feature_id_to_pos,
            "category_id_to_pos": category_id_to_pos
        },
        "stats": { 
            "study_name": study_name,
            "target_connection_level": target_connection_level,
            "encoding_type": "mixed_ceramic_attributes_and_onehot",
            "full_vocab_functions": len(all_function_ids),
            "full_vocab_features": len(all_feature_ids),
            "full_vocab_categories": len(all_category_ids),
            "ceramic_attribute_dim": ceramic_attr_dim,
            "ceramics_in_sample": len(ceramic_ids_in_sample),
            "functions_in_sample": len(function_ids_in_sample),
            "features_in_sample": len(feature_ids_in_sample),
            "categories_in_sample": len(category_ids_in_sample),
            "total_embedding_dim": total_embedding_dim,
            "ceramic_offset": ceramic_offset,
            "function_offset": function_offset,
            "feature_offset": feature_offset,
            "category_offset": category_offset,
            "total_nodes_in_graph": len(node_to_idx),
            "total_relations": len(relation_to_idx),
            "ceramic_embedding_info": ceramic_embed_info,
            "ceramic_attribute_maps": ceramic_maps
        }
    }
    
    print(f"    ✅ Formatted data for {study_name} with MIXED EMBEDDINGS.")
    print(f"      🏺 Ceramics: Attribute-based embeddings ({ceramic_attr_dim}D)")
    print(f"      ⚙️🔧📂 Functions/Features/Categories: One-hot encodings")
    print(f"      Train triplets: {len(rgcn_data['training_triplets'])}")
    print(f"      Eval triplets: {len(rgcn_data['evaluation_triplets'])}")
    print(f"      Target categories: {len(rgcn_data['target_category_node_indices'])}")
    print(f"      📊 Final embedding matrix shape: {embedding_matrix.shape}")
    print(f"      🌍 Embedding covers ceramic attributes + full vocabulary of {len(all_function_ids) + len(all_feature_ids) + len(all_category_ids)} F/F/C entities")

    return rgcn_data


def prepare_all_level_based_studies_onehot(dfs, auto_save=True, base_output_dir="output/rgcn_data/ontology_onehot"):
    """
    Main orchestrator for the comparative study with ONE-HOT ENCODINGS instead of BERT.
    
    This version uses one-hot encodings where:
    - Each node type (ceramic, function, feature, category) gets its own section in the embedding
    - Within each section, each specific node gets a unique one-hot position
    - Total embedding length = num_ceramics + num_functions + num_features + num_categories
    
    Args:
        dfs: Dictionary of DataFrames
        auto_save: If True, automatically saves the datasets
        base_output_dir: Directory to save the datasets
    """
    print("======================================================")
    print("=== STARTING DATA PREPARATION (ONE-HOT ENCODINGS) ===")
    print("======================================================")

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
    print("\n--- STEP 3: Generating All 9 Datasets with ONE-HOT Encodings ---")
    all_study_datasets = {}

    for linking_level in [2, 1, 0]:
        repo_name = f"level_{linking_level}_connections_onehot"
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

            rgcn_data = format_rgcn_data_for_study_onehot_fixed(
                dfs=dfs,
                triplets_for_study=triplets_for_study,
                study_name=f"{repo_name}_{etude_name}",
                category_hierarchy=hierarchy,
                target_connection_level=linking_level
            )
            
            all_study_datasets[repo_name][etude_name] = rgcn_data
            if rgcn_data:
                print(f"    ✅ Successfully prepared dataset for {repo_name} / {etude_name}")
                print(f"    📊 One-Hot Embedding Dimensions: {rgcn_data['stats']['total_embedding_dim']}D")
                print(f"      (C:{rgcn_data['stats']['ceramics_in_sample']} + F:{rgcn_data['stats']['functions_in_sample']} + Feat:{rgcn_data['stats']['features_in_sample']} + Cat:{rgcn_data['stats']['categories_in_sample']})")
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


def debug_missing_ceramics(ceramic_summary, ceramic_ids_in_sample):
    """Debug function to identify why ceramics are missing"""
    
    # Check original ceramic_summary before any processing
    print("🔍 DEBUGGING MISSING CERAMICS:")
    
    # Get the problematic ceramic IDs from your log
    problematic_ids = list(range(10594, 11062))  # Based on your error log
    
    print(f"📊 Original ceramic_summary shape: {ceramic_summary.shape}")
    print(f"📊 Ceramic IDs in sample: {len(ceramic_ids_in_sample)}")
    print(f"📊 Problematic IDs to check: {len(problematic_ids)}")
    
    # Check if problematic IDs exist in original data
    original_ids = set(ceramic_summary['ceramic_id'].astype(str))  # Convert to string to avoid type issues
    problematic_ids_str = set(str(id) for id in problematic_ids)
    
    found_in_original = problematic_ids_str.intersection(original_ids)
    missing_from_original = problematic_ids_str - original_ids
    
    print(f"✅ Found in original ceramic_summary: {len(found_in_original)}")
    print(f"❌ Missing from original ceramic_summary: {len(missing_from_original)}")
    
    if missing_from_original:
        print(f"   Missing IDs: {sorted(list(missing_from_original))[:10]}...")  # Show first 10
    
    # Check for data type issues
    try:
        numeric_conversion = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce')
        num_failed_conversions = numeric_conversion.isna().sum()
        print(f"🔢 Failed numeric conversions: {num_failed_conversions}")
        
        if num_failed_conversions > 0:
            failed_values = ceramic_summary.loc[numeric_conversion.isna(), 'ceramic_id'].unique()
            print(f"   Failed values: {failed_values[:10]}")  # Show first 10
            
    except Exception as e:
        print(f"❌ Error in numeric conversion check: {e}")
    
    # Check for duplicates
    duplicates = ceramic_summary['ceramic_id'].duplicated().sum()
    print(f"🔄 Duplicate ceramic_ids: {duplicates}")
    
    # Check the processed ceramic_summary index
    if hasattr(ceramic_summary, 'index'):
        available_ids = set(ceramic_summary.index.astype(str))
        sample_ids_str = set(str(id) for id in ceramic_ids_in_sample)
        
        missing_from_processed = sample_ids_str - available_ids
        print(f"❌ IDs in sample but missing from processed ceramic_summary: {len(missing_from_processed)}")
        
        if missing_from_processed:
            print(f"   Missing processed IDs: {sorted(list(missing_from_processed))[:10]}...")



def save_all_study_datasets(all_study_results, base_output_dir="output/rgcn_data/without_ontology"):
    """
    Saves the RGCN data for each repo and etude into a nested directory structure.
    MODIFIED: Default output directory changed to without_ontology
    e.g., output/rgcn_data/without_ontology/level_2_connections/etude_1/
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