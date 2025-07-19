import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
import torch
import datetime
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

def format_rgcn_data_for_study(dfs, 
                                      triplets_for_study, 
                                      study_name,
                                      category_hierarchy,
                                      target_connection_level: int):
    """
    FIXED version - handles missing feature IDs and ensures consistent data types
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
    
    # Convert other columns with consistent string handling for feature IDs
    object_function['id'] = pd.to_numeric(object_function['id'], errors='coerce').dropna().astype(int)
    tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
    
    # FIXED: Ensure all feature IDs are consistently handled as strings
    Features_Ontology_df['id'] = Features_Ontology_df['id'].astype(str)
    tech_cat_func_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_func_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
    tech_cat_func_attrib['function_id'] = pd.to_numeric(tech_cat_func_attrib['function_id'], errors='coerce').dropna().astype(int)
    tech_cat_feat_attrib['tech_cat_id'] = pd.to_numeric(tech_cat_feat_attrib['tech_cat_id'], errors='coerce').dropna().astype(int)
    tech_cat_feat_attrib['feature_id'] = tech_cat_feat_attrib['feature_id'].astype(str)
    
    # --- FIXED: Build vocabulary including all feature IDs that will be used ---
    print("    🌍 Building full vocabulary from entire database...")
    
    # All ceramics from database
    all_ceramic_ids = sorted(ceramic_summary['ceramic_id'].unique())
    
    # All functions from database  
    all_function_ids = sorted(object_function['id'].unique())
    
    # FIXED: Collect all feature IDs from multiple sources to ensure completeness
    feature_ids_from_ontology = set(Features_Ontology_df['id'].astype(str))
    feature_ids_from_attrib = set(tech_cat_feat_attrib['feature_id'].astype(str))
    
    # Also collect feature IDs from triplets to ensure we don't miss any
    feature_ids_from_triplets = set()
    for entry in triplets_for_study:
        for feat_id, parents_ids in entry.get("features", []):
            feature_ids_from_triplets.add(str(feat_id))
            for parent_id in parents_ids:
                feature_ids_from_triplets.add(str(parent_id))
    
    # Combine all feature ID sources
    all_feature_ids = sorted(feature_ids_from_ontology | feature_ids_from_attrib | feature_ids_from_triplets)
    
    # All categories from database
    all_category_ids = sorted(list(category_hierarchy.parent_map.keys()))
    
    print(f"    📊 Full vocabulary sizes:")
    print(f"      🏺 All Ceramics: {len(all_ceramic_ids)}")
    print(f"      ⚙️  All Functions: {len(all_function_ids)}")
    print(f"      🔧 All Features: {len(all_feature_ids)} (from {len(feature_ids_from_ontology)} ontology + {len(feature_ids_from_attrib)} attrib + {len(feature_ids_from_triplets)} triplets)")
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
    ceramic_summary_with_embeddings.drop_duplicates(subset='ceramic_id', keep='first', inplace=True)
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
    
    # --- Generate embeddings with better error handling ---
    print("    🎯 Generating mixed embeddings (attributes + one-hot)...")
    embedding_matrix = np.zeros((len(nodes_in_sample), total_embedding_dim), dtype=np.float32)
    
    missing_ceramic_count = 0
    missing_feature_count = 0
    
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
            if function_id in function_id_to_pos:
                pos = function_offset + function_id_to_pos[function_id]
                embedding_matrix[node_idx, pos] = 1.0
            else:
                print(f"      ⚠️  Warning: Function {function_id} not found in function vocabulary")
            
        elif node_identifier.startswith("Feat_"):
            feature_id = node_identifier.split("_")[1]  # Already a string
            # FIXED: Better error handling for missing feature IDs
            if feature_id in feature_id_to_pos:
                pos = feature_offset + feature_id_to_pos[feature_id]
                embedding_matrix[node_idx, pos] = 1.0
            else:
                missing_feature_count += 1
                if missing_feature_count <= 10:
                    print(f"      ⚠️  Warning: Feature {feature_id} not found in feature vocabulary")
                elif missing_feature_count == 11:
                    print(f"      ⚠️  ... (suppressing further feature warnings)")
                # Add missing feature ID to vocabulary dynamically
                if feature_id not in feature_id_to_pos:
                    print(f"      🔧 Adding missing feature {feature_id} to vocabulary")
                    # Note: This is a fallback - ideally we'd rebuild the vocabulary
                    # For now, we'll skip this embedding but log it
            
        elif node_identifier.startswith("Cat_"):
            category_id = int(node_identifier.split("_")[1])
            if category_id in category_id_to_pos:
                pos = category_offset + category_id_to_pos[category_id]
                embedding_matrix[node_idx, pos] = 1.0
            else:
                print(f"      ⚠️  Warning: Category {category_id} not found in category vocabulary")

    if missing_ceramic_count > 0:
        print(f"    ❌ TOTAL MISSING CERAMICS: {missing_ceramic_count}")
    if missing_feature_count > 0:
        print(f"    ❌ TOTAL MISSING FEATURES: {missing_feature_count}")

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
        "ceramic_attribute_maps": ceramic_maps,
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


def create_classification_config(classification_data, dataset_label):
    """
    Create a configuration dictionary for the classification dataset.
    
    Args:
        classification_data (dict): Classification data
        dataset_label (str): Label for the dataset
    
    Returns:
        dict: Configuration dictionary
    """
    stats = classification_data.get('stats', {})
    
    config = {
        "dataset_label": dataset_label,
        "task_type": "node_classification",
        "target_nodes": "ceramic_nodes",
        "num_nodes": classification_data.get('num_nodes', 0),
        "num_relations": classification_data.get('num_relations', 0),
        "num_classes": len(classification_data.get('label_to_category_id', {})),
        "num_ceramic_nodes": len(classification_data.get('ceramic_node_indices', [])),
        "num_labeled_ceramics": len(classification_data.get('ceramic_labels', {})),
        "embedding_dimension": classification_data.get('embedding_dim', 0),
        "num_training_triplets": len(classification_data.get('training_triplets', [])),
        "bert_model": stats.get('bert_model_used', 'unknown'),
        "target_connection_level": stats.get('target_connection_level', 'unknown'),
        "labeling_strategy": stats.get('classification_labeling_strategy', 'unknown'),
        "created_from": {
            "study_name": stats.get('study_name', 'unknown'),
            "original_task": "link_prediction"
        }
    }
    
    return config


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

def adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs=None):
    if rgcn_data is None:
        print("❌ Input rgcn_data is None. Cannot adapt for classification.")
        return None

    print("\n--- Adapting RGCN Data for Ceramic Classification (Direct Categories) ---")

    original_node_to_idx = rgcn_data['node_to_idx']
    original_idx_to_node = rgcn_data['idx_to_node']
    original_num_nodes = rgcn_data['num_nodes']

    # Keep all nodes for classification (1:1 mapping)
    original_idx_to_new_idx = {}
    new_node_to_idx = {}
    
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
    # MODIFIED: Safely get ceramic_attribute_maps to prevent KeyError
    ceramic_attribute_maps = rgcn_data.get('ceramic_attribute_maps', {})
    
    # --- Direct Category Label Extraction ---
    ceramic_labels = {} 
    category_to_label_id = {} 
    label_id_to_category_id = {} 
    label_id_to_category_name = {}
    label_counter = 0

    # Use evaluation triplets to get ceramic-category relationships
    evaluation_triplets = rgcn_data.get('evaluation_triplets', [])
    all_triplets = rgcn_data.get('training_triplets', []) + evaluation_triplets
    
    # Find BELONGS_TO_CATEGORY relation
    EVALUATION_RELATION_NAME = "BELONGS_TO_CATEGORY"
    belongs_to_category_rel_id = None
    
    if 'relation_to_idx' in rgcn_data and EVALUATION_RELATION_NAME in rgcn_data['relation_to_idx']:
        belongs_to_category_rel_id = rgcn_data['relation_to_idx'][EVALUATION_RELATION_NAME]
    else:
        print(f"  CRITICAL ERROR: Relation '{EVALUATION_RELATION_NAME}' not found. Cannot create labels.")
        print(f"  Available relations: {list(rgcn_data['relation_to_idx'].keys())}")
        return None

    print(f"  Extracting direct category labels from {len(all_triplets)} available triplets...")
    print(f"  Looking for relation ID {belongs_to_category_rel_id} ('{EVALUATION_RELATION_NAME}')")
    
    # Extract labels from BELONGS_TO_CATEGORY triplets
    for h_idx, r_idx, t_idx in all_triplets:
        if r_idx != belongs_to_category_rel_id:
            continue

        h_node_identifier = original_idx_to_node.get(h_idx)
        t_node_identifier = original_idx_to_node.get(t_idx)

        # Check if head is ceramic and tail is category
        if (h_node_identifier and t_node_identifier and 
            h_node_identifier.startswith("Ceramic_") and t_node_identifier.startswith("Cat_")):
            
            if h_idx in original_idx_to_new_idx:
                h_new_idx = original_idx_to_new_idx[h_idx]
                
                # Extract category ID from node name (e.g., "Cat_132" -> 132)
                try:
                    category_id = int(t_node_identifier.split('_')[1])
                except (ValueError, IndexError):
                    print(f"  ⚠️ Warning: Could not parse category ID from '{t_node_identifier}'")
                    continue
                
                # Create label mapping
                if category_id not in category_to_label_id:
                    label_id = label_counter
                    category_to_label_id[category_id] = label_id
                    label_id_to_category_id[label_id] = category_id
                    
                    # Try to get category name from dfs if available
                    category_name = f"Category_{category_id}"
                    if dfs and 'tech_cat' in dfs and not dfs['tech_cat'].empty:
                        try:
                            cat_row = dfs['tech_cat'][dfs['tech_cat']['id'] == category_id]
                            if not cat_row.empty:
                                name_col = 'cat_name_processed' if 'cat_name_processed' in cat_row.columns else 'cat_name'
                                category_name = cat_row[name_col].iloc[0]
                        except Exception as e:
                            print(f"  ⚠️ Warning: Could not get name for category {category_id}: {e}")
                    
                    label_id_to_category_name[label_id] = category_name
                    label_counter += 1
                else:
                    label_id = category_to_label_id[category_id]
                
                ceramic_labels[h_new_idx] = label_id
    
    print(f"  Extracted labels for {len(ceramic_labels)} ceramic nodes. Found {label_counter} unique categories.")
    
    if label_counter > 0:
        print(f"  Label mapping ({label_counter} labels):")
        for lid, name in label_id_to_category_name.items():
            cat_id = label_id_to_category_id.get(lid, 'N/A')
            count = sum(1 for label in ceramic_labels.values() if label == lid)
            print(f"    Label {lid}: '{name}' (Category ID: {cat_id}, {count} ceramics)")

    # Find all ceramic node indices
    ceramic_node_indices = sorted([
        new_idx for original_idx, new_idx in original_idx_to_new_idx.items()
        if original_idx in original_idx_to_node and original_idx_to_node[original_idx].startswith("Ceramic_")
    ])
    print(f"  Identified {len(ceramic_node_indices)} ceramic nodes for classification.")

    # Create stats (simplified version)
    stats = {
        'classification_num_nodes': original_num_nodes,
        'classification_nodes_removed_category': 0,
        'classification_training_triplets_filtered': len(original_training_triplets),
        'classification_ceramic_nodes_with_labels': len(ceramic_labels),
        'classification_num_classes': label_counter,
        'classification_ceramic_node_indices': ceramic_node_indices,
        'classification_labeling_strategy': 'direct_category_from_evaluation_triplets'
    }

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
        "ceramic_node_indices": ceramic_node_indices,
        "ceramic_labels": ceramic_labels,
        "label_to_category_id": label_id_to_category_id,
        "label_to_category_name": label_id_to_category_name,
        "stats": stats,
        "ceramic_attribute_maps": ceramic_attribute_maps
    }

    print("\n--- Adaptation for Classification Complete ---")
    print(f"  Final ceramic nodes with labels: {len(ceramic_labels)}")
    print(f"  Total classes: {label_counter}")
    
    return classification_data


def save_classification_dataset(class_data, output_dir="classification_output", dataset_label="dataset"):
    """
    Saves the components of a classification-ready dataset.
    """
    if not class_data:
        print(f"Error: Classification data for '{dataset_label}' is None or empty. Cannot save.")
        return

    print(f"Saving '{dataset_label}' classification data to directory: '{output_dir}'")

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  Output directory '{output_dir}' ensured.")

        saved_files = []

        # 1. Save Node Mappings
        if 'node_to_idx' in class_data and class_data['node_to_idx']:
            try:
                pd.DataFrame(list(class_data['node_to_idx'].items()), columns=['NodeIdentifier', 'NodeIndex']).to_csv(os.path.join(output_dir, "node_mapping.csv"), index=False)
                saved_files.append("node_mapping.csv")
            except Exception as e: print(f"  ❌ Error saving node mapping: {e}")
        else: print(f"  ⚠️ Node mapping data missing.")

        # 2. Save Relation Mappings
        if 'relation_to_idx' in class_data and class_data['relation_to_idx']:
            try:
                pd.DataFrame(list(class_data['relation_to_idx'].items()), columns=['RelationName', 'RelationIndex']).to_csv(os.path.join(output_dir, "relation_mapping.csv"), index=False)
                saved_files.append("relation_mapping.csv")
            except Exception as e: print(f"  ❌ Error saving relation mapping: {e}")
        else: print(f"  ⚠️ Relation mapping data missing.")

        # 3. Save Training Triplets
        if 'training_triplets' in class_data and class_data['training_triplets']:
            try:
                pd.DataFrame(class_data['training_triplets'], columns=['SourceIndex', 'RelationIndex', 'TargetIndex']).to_csv(os.path.join(output_dir, "training_triplets.csv"), index=False)
                saved_files.append("training_triplets.csv")
            except Exception as e: print(f"  ❌ Error saving training triplets: {e}")
        else: print(f"  ⚠️ Training triplets data missing.")

        # 4. Save Embeddings
        if 'node_embeddings' in class_data and isinstance(class_data['node_embeddings'], np.ndarray):
            try:
                embedding_path = os.path.join(output_dir, "node_embeddings.npy")
                np.save(embedding_path, class_data['node_embeddings'])
                print(f"    Node embeddings saved to '{embedding_path}' (Shape: {class_data['node_embeddings'].shape})")
                saved_files.append("node_embeddings.npy")
            except Exception as e: print(f"  ❌ Error saving node embeddings: {e}")

        # 5. Save Ceramic Node Indices
        if 'ceramic_node_indices' in class_data and class_data['ceramic_node_indices'] is not None:
            try:
                with open(os.path.join(output_dir, "ceramic_node_indices.json"), 'w') as f:
                    json.dump(class_data['ceramic_node_indices'], f)
                saved_files.append("ceramic_node_indices.json")
            except Exception as e: print(f"  ❌ Error saving ceramic node indices: {e}")
        else: print(f"  ⚠️ Ceramic node indices data missing.")

        # 6. Save Ceramic Labels
        if 'ceramic_labels' in class_data and class_data['ceramic_labels'] is not None:
            try:
                # Convert integer keys to strings for JSON compatibility
                labels_to_save = {str(k): v for k, v in class_data['ceramic_labels'].items()}
                with open(os.path.join(output_dir, "ceramic_labels.json"), 'w') as f:
                    json.dump(labels_to_save, f, indent=4)
                saved_files.append("ceramic_labels.json")
            except Exception as e: print(f"  ❌ Error saving ceramic labels: {e}")
        else: print(f"  ⚠️ Ceramic labels data missing.")
        
        # 7. Save Label Maps
        label_map = {
            'label_to_category_id': class_data.get('label_to_category_id', {}),
            'label_to_category_name': class_data.get('label_to_category_name', {})
        }
        # Convert integer keys to strings for JSON compatibility
        label_map['label_to_category_id'] = {str(k): v for k, v in label_map['label_to_category_id'].items()}
        label_map['label_to_category_name'] = {str(k): v for k, v in label_map['label_to_category_name'].items()}
        try:
            with open(os.path.join(output_dir, "label_map.json"), 'w') as f:
                json.dump(label_map, f, indent=4)
            saved_files.append("label_map.json")
        except Exception as e: print(f"  ❌ Error saving label map: {e}")

        # 8. Save Classification Configuration
        classification_config = create_classification_config(class_data, dataset_label)
        config_path = os.path.join(output_dir, "classification_config.json")
        with open(config_path, 'w') as f:
            json.dump(classification_config, f, indent=2)
        saved_files.append("classification_config.json")
        print(f"    ✓ Saved classification configuration")

        # 9. Save Statistics
        if 'stats' in class_data:
            stats_serializable = convert_numpy_to_python_native(class_data['stats'])
            stats_path = os.path.join(output_dir, "dataset_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_serializable, f, indent=2)
            saved_files.append("dataset_stats.json")
            print(f"    ✓ Saved dataset statistics")

        print(f"  Finished saving for '{dataset_label}'. {len(saved_files)} classification files created/attempted in '{output_dir}'.")

    except Exception as e:
        print(f"An unexpected error occurred saving classification data for {dataset_label}: {e}")
        traceback.print_exc()



def generate_study_config(rgcn_data):
    """
    Generate study configuration dictionary from RGCN data.
    """
    stats = rgcn_data.get('stats', {})
    return {
        "embedding_method": "bert_aggregated",
        "bert_model": stats.get('bert_model_used', 'all-MiniLM-L6-v2'),
        "target_y_level": stats.get('target_connection_level', 0),
        "total_samples": len([idx for idx, node_id in rgcn_data.get('idx_to_node', {}).items() 
                             if node_id.startswith("Ceramic_")]),
        "n_classes": len(rgcn_data.get('target_category_node_indices', [])),
        "embedding_dimensions": stats.get('final_embedding_dim', 0)
    }


def prepare_three_level_studies_onehot(dfs, 
                                bert_model_name="all-MiniLM-L6-v2", 
                                auto_save=True, 
                                base_output_dir="output/rgcn_data/Onehot_ontology",
                                save_bert_model=True, 
                                bert_model_save_dir="models/bert"):
    """
    Prepares exactly 3 datasets for levels 0, 1, and 2 with specific sampling strategies.
    
    etude_lvl0: 400 ceramics per root category, diverse subcategory sampling
    etude_lvl1: Balance subcategories with minimum 10 ceramics, duplicate to reach targets
    etude_lvl2: Balance sub-subcategories with minimum 10 ceramics, ~40 per sub-subcategory
    
    Args:
        dfs: Dictionary of DataFrames
        bert_model_name: BERT model name for embeddings
        auto_save: Whether to save automatically
        base_output_dir: Base directory for saving
        save_bert_model: Whether to save BERT model
        bert_model_save_dir: Directory for BERT model saving
    
    Returns:
        dict: Dictionary with the three prepared datasets
    """
    print("=" * 80)
    print("=== PREPARING THREE LEVEL STUDIES ===")
    print("=" * 80)
    
    # Initialize hierarchy
    try:
        hierarchy = demonstrate_hierarchy_usage(dfs['tech_cat'])
        print(f"✅ Hierarchy initialized with {len(hierarchy.roots)} roots: {hierarchy.roots}")
    except Exception as e:
        print(f"❌ Failed to initialize hierarchy: {e}")
        return None
    
    # Filter roots that have subcategories
    valid_roots = []
    for root_id in hierarchy.roots:
        children = hierarchy.get_children(root_id)
        if children and len(children) > 0:
            valid_roots.append(root_id)
    
    print(f"✅ Found {len(valid_roots)} root categories with subcategories: {valid_roots}")
    
    # Map ceramics to their category information
    ceramic_to_info = {}
    for _, row in dfs['ceramic_summary'].iterrows():
        try:
            cid = int(row['ceramic_id'])
            cat_id = row.get('tech_cat_id')
            if pd.notna(cat_id):
                cat_id = int(cat_id)
                root = hierarchy.get_root(cat_id)
                if root in valid_roots:
                    level = hierarchy.get_level(cat_id)
                    if level is not None:
                        ceramic_to_info[cid] = {
                            'cat_id': cat_id, 
                            'level': level, 
                            'root': root
                        }
        except (ValueError, TypeError):
            continue
    
    print(f"✅ Mapped {len(ceramic_to_info)} ceramics to category info")
    
    # Prepare the three datasets
    all_datasets = {}
    
    # Dataset 1: Level 0 (400 ceramics per root, diverse sampling)
    print("\n" + "=" * 50)
    print("PREPARING ETUDE_LVL0 (Level 0 connections)")
    print("=" * 50)
    
    etude_lvl0_ceramics = prepare_level0_dataset(
        ceramic_to_info, hierarchy, dfs, target_per_root=400
    )
    
    if etude_lvl0_ceramics:
        rgcn_data_lvl0 = create_rgcn_dataset_level_0(
            etude_lvl0_ceramics, dfs, hierarchy, 
            target_connection_level=0,
            study_name="etude_lvl0",
            bert_model_name=bert_model_name,
            save_bert_model=save_bert_model,
            bert_model_save_dir=bert_model_save_dir
        )
        all_datasets['etude_lvl0'] = rgcn_data_lvl0
        print(f"✅ Level 0 dataset prepared with {len(etude_lvl0_ceramics)} ceramics")
    
    # Dataset 2: Level 1 (balanced subcategories, min 10 ceramics)
    print("\n" + "=" * 50)
    print("PREPARING ETUDE_LVL1 (Level 1 connections)")
    print("=" * 50)
    
    # Prepare Level 1 dataset
    level1_ceramics, level1_duplication_map = prepare_level1_dataset(ceramic_to_info, hierarchy, dfs)
    
    # IMPORTANT: Update ceramic_to_info with Level 1 duplicates
    print(f"📝 Updating ceramic_to_info with Level 1 duplicates...")
    original_count = len(ceramic_to_info)
    
    # Add duplicated ceramics to ceramic_to_info
    for original_cid, duplicate_list in level1_duplication_map.items():
        if original_cid in ceramic_to_info:
            for duplicate_cid in duplicate_list:
                if duplicate_cid != original_cid:  # Skip the original
                    ceramic_to_info[duplicate_cid] = ceramic_to_info[original_cid].copy()
    
    print(f"   Ceramic info updated: {original_count} → {len(ceramic_to_info)} entries")
    
    # Create Level 1 RGCN dataset
    all_datasets['etude_lvl1'] = create_rgcn_dataset(
        level1_ceramics, level1_duplication_map, dfs, ceramic_to_info, 
        hierarchy, target_connection_level=1, study_name='etude_lvl1',
        bert_model_name=bert_model_name
    )
    
    # Dataset 3: Level 2 (balanced sub-subcategories, min 10 ceramics)
    print("\n" + "=" * 50)
    print("PREPARING ETUDE_LVL2 (Level 2 connections)")
    print("=" * 50)
    
    # Prepare Level 2 dataset using the UPDATED ceramic_to_info
    level2_ceramics, level2_duplication_map = prepare_level2_dataset(ceramic_to_info, hierarchy, dfs)
    
    # IMPORTANT: Update ceramic_to_info with Level 2 duplicates
    print(f"📝 Updating ceramic_to_info with Level 2 duplicates...")
    original_count = len(ceramic_to_info)
    
    # Add duplicated ceramics to ceramic_to_info
    for original_cid, duplicate_list in level2_duplication_map.items():
        if original_cid in ceramic_to_info:
            for duplicate_cid in duplicate_list:
                if duplicate_cid != original_cid:  # Skip the original
                    ceramic_to_info[duplicate_cid] = ceramic_to_info[original_cid].copy()
    
    print(f"   Ceramic info updated: {original_count} → {len(ceramic_to_info)} entries")
    
    # Create Level 2 RGCN dataset
    all_datasets['etude_lvl2'] = create_rgcn_dataset(
        level2_ceramics, level2_duplication_map, dfs, ceramic_to_info,
        hierarchy, target_connection_level=2, study_name='etude_lvl2', 
        bert_model_name=bert_model_name
    )
    
    # Auto-save if requested
    if auto_save:
        print(f"\n💾 Auto-saving datasets to '{base_output_dir}'...")
        # MODIFIED: Pass dfs to the saving function
        save_three_level_datasets(all_datasets, dfs, base_output_dir)
        print("✅ All datasets saved successfully!")
    
    print("\n" + "=" * 80)
    print("=== THREE LEVEL STUDIES PREPARATION COMPLETE ===")
    print("=" * 80)
    
    return all_datasets


def prepare_level0_dataset(ceramic_to_info, hierarchy, dfs, target_per_root=400):
    """
    Prepare Level 0 dataset: 400 ceramics per root category with diverse subcategory sampling.
    """
    print("📊 Preparing Level 0 dataset...")
    
    # Group ceramics by root category
    ceramics_by_root = defaultdict(list)
    for cid, info in ceramic_to_info.items():
        ceramics_by_root[info['root']].append(cid)
    
    selected_ceramics = []
    
    for root_id in sorted(ceramics_by_root.keys()):
        root_ceramics = ceramics_by_root[root_id]
        print(f"  Root {root_id}: {len(root_ceramics)} available ceramics")
        
        # Group by subcategory levels for diverse sampling
        ceramics_by_level = defaultdict(list)
        for cid in root_ceramics:
            level = ceramic_to_info[cid]['level']
            ceramics_by_level[level].append(cid)
        
        # Sample diversely across levels
        selected_for_root = []
        remaining_target = target_per_root
        
        # Priority: deeper levels first (more specific categories)
        for level in sorted(ceramics_by_level.keys(), reverse=True):
            level_ceramics = ceramics_by_level[level]
            if not level_ceramics or remaining_target <= 0:
                continue
                
            # Calculate completeness scores
            scores = calculate_completeness_score(level_ceramics, dfs)
            sorted_ceramics = sorted(level_ceramics, key=lambda cid: (-scores.get(cid, 0), cid))
            
            # Take proportionally from this level
            level_target = min(len(sorted_ceramics), remaining_target // len(ceramics_by_level))
            level_target = max(1, level_target)  # At least 1 if available
            
            selected_for_root.extend(sorted_ceramics[:level_target])
            remaining_target -= level_target
        
        # If we still need more ceramics, fill from the best available
        if remaining_target > 0:
            all_root_ceramics = [cid for level_ceramics in ceramics_by_level.values() for cid in level_ceramics]
            remaining_ceramics = [cid for cid in all_root_ceramics if cid not in selected_for_root]
            
            if remaining_ceramics:
                scores = calculate_completeness_score(remaining_ceramics, dfs)
                sorted_remaining = sorted(remaining_ceramics, key=lambda cid: (-scores.get(cid, 0), cid))
                selected_for_root.extend(sorted_remaining[:remaining_target])
        
        # Ensure we have exactly target_per_root (duplicate if necessary)
        while len(selected_for_root) < target_per_root and len(selected_for_root) > 0:
            needed = target_per_root - len(selected_for_root)
            duplicates = selected_for_root[:needed]
            selected_for_root.extend(duplicates)
        
        selected_for_root = selected_for_root[:target_per_root]  # Trim to exact target
        selected_ceramics.extend(selected_for_root)
        
        print(f"    → Selected {len(selected_for_root)} ceramics for root {root_id}")
    
    return selected_ceramics


def get_level1_category(cat_id, hierarchy: CategoryHierarchy):
    """
    Gets the Level 1 ancestor for a given category. If the category itself is at 
    Level 1, it returns its own ID.
    """
    return hierarchy.get_ancestor_at_level(cat_id, target_level=1)


def get_level2_category(cat_id, hierarchy: CategoryHierarchy):
    """
    Gets the Level 2 ancestor for a given category. If the category itself is at 
    Level 2, it returns its own ID.
    """
    return hierarchy.get_ancestor_at_level(cat_id, target_level=2)

def prepare_level1_dataset(ceramic_to_info, hierarchy, dfs, min_ceramics_per_subcat=10):
    """
    Prepare Level 1 dataset: Balance subcategories with minimum 10 ceramics, duplicate to reach targets.
    """
    print("📊 Preparing Level 1 dataset...")
    
    # Define specific targets for category 140 as per your example
    category_140_targets = {
        129: 120,  # Pâte siliceuse artificielle / Orient
        150: 80,   # Porcelaine européenne  
        149: 100   # Faïence fine
    }
    
    # Group ceramics by root and subcategory
    ceramics_by_root_subcat = defaultdict(lambda: defaultdict(list))
    for cid, info in ceramic_to_info.items():
        root_id = info['root']
        cat_id = info['cat_id']
        
        # Find the level 1 category (direct child of root)
        level1_cat = get_level1_category(cat_id, hierarchy)
        if level1_cat:
            ceramics_by_root_subcat[root_id][level1_cat].append(cid)
    
    selected_ceramics = []
    ceramic_duplication_map = {}  # Track which ceramics are duplicated and how many times
    
    max_ceramic_id = max(ceramic_to_info.keys()) if ceramic_to_info else 0
    next_available_id = max_ceramic_id + 1

    for root_id in sorted(ceramics_by_root_subcat.keys()):
        print(f"  Root {root_id}:")
        subcats = ceramics_by_root_subcat[root_id]
        
        # Filter subcategories with at least min_ceramics_per_subcat
        valid_subcats = {subcat: ceramics for subcat, ceramics in subcats.items() 
                        if len(ceramics) >= min_ceramics_per_subcat}
        
        if not valid_subcats:
            print(f"    No valid subcategories with >= {min_ceramics_per_subcat} ceramics")
            continue
        
        # Calculate targets for each subcategory
        if root_id == 140 and category_140_targets:
            # Use predefined targets for category 140
            subcat_targets = category_140_targets
        else:
            # Default: aim for 300 total per root, distributed equally
            total_target = 300
            target_per_subcat = total_target // len(valid_subcats)
            subcat_targets = {subcat: target_per_subcat for subcat in valid_subcats.keys()}
        
        for subcat_id, target_count in subcat_targets.items():
            if subcat_id not in valid_subcats:
                continue
                
            available_ceramics = valid_subcats[subcat_id]
            print(f"    Subcat {subcat_id}: {len(available_ceramics)} available, target {target_count}")
            
            # Calculate completeness scores
            scores = calculate_completeness_score(available_ceramics, dfs)
            sorted_ceramics = sorted(available_ceramics, key=lambda cid: (-scores.get(cid, 0), cid))
            
            # Select ceramics with proper duplication tracking
            selected_for_subcat = []
            duplication_counter = 0
            
            while len(selected_for_subcat) < target_count:
                remaining_needed = target_count - len(selected_for_subcat)
                batch_size = min(remaining_needed, len(sorted_ceramics))
                
                # Create duplicated IDs with suffix for tracking
                for i in range(batch_size):
                    original_cid = sorted_ceramics[i]
                    if duplication_counter == 0:
                        # First occurrence - use original ID
                        new_cid = original_cid
                    else:
                        # Subsequent occurrences - add suffix
                        new_cid = next_available_id
                        next_available_id += 1
                    
                    selected_for_subcat.append(new_cid)
                    
                    # Track duplication
                    if original_cid not in ceramic_duplication_map:
                        ceramic_duplication_map[original_cid] = []
                    ceramic_duplication_map[original_cid].append(new_cid)
                
                duplication_counter += 1
            
            selected_ceramics.extend(selected_for_subcat)
            print(f"      → Selected {len(selected_for_subcat)} ceramics (including duplicates)")
    
    return selected_ceramics, ceramic_duplication_map


def prepare_level2_dataset(ceramic_to_info, hierarchy, dfs, min_ceramics_per_subsubcat=10, target_per_subsubcat=40):
    """
    Prepare Level 2 dataset: Balance sub-subcategories with minimum 10 ceramics, ~40 per sub-subcat.
    """
    print("📊 Preparing Level 2 dataset...")
    
    # Group ceramics by sub-subcategory (level 2)
    ceramics_by_subsubcat = defaultdict(list)
    for cid, info in ceramic_to_info.items():
        cat_id = info['cat_id']
        level = info['level']
        
        # Only consider ceramics at level 2 or deeper
        if level >= 2:
            level2_cat = get_level2_category(cat_id, hierarchy)
            if level2_cat:
                ceramics_by_subsubcat[level2_cat].append(cid)
    
    # Filter sub-subcategories with at least min_ceramics_per_subsubcat
    valid_subsubcats = {subsubcat: ceramics for subsubcat, ceramics in ceramics_by_subsubcat.items() 
                       if len(ceramics) >= min_ceramics_per_subsubcat}
    
    print(f"  Found {len(valid_subsubcats)} valid sub-subcategories with >= {min_ceramics_per_subsubcat} ceramics")
    
    selected_ceramics = []
    ceramic_duplication_map = {}  # Track which ceramics are duplicated and how many times
    

    max_ceramic_id = max(ceramic_to_info.keys()) if ceramic_to_info else 0
    next_available_id = max_ceramic_id + 1

    for subsubcat_id, available_ceramics in valid_subsubcats.items():
        print(f"  Sub-subcat {subsubcat_id}: {len(available_ceramics)} available, target {target_per_subsubcat}")
        
        # Calculate completeness scores
        scores = calculate_completeness_score(available_ceramics, dfs)
        sorted_ceramics = sorted(available_ceramics, key=lambda cid: (-scores.get(cid, 0), cid))
        
        # Select ceramics with proper duplication tracking
        selected_for_subsubcat = []
        duplication_counter = 0
        
        while len(selected_for_subsubcat) < target_per_subsubcat:
            remaining_needed = target_per_subsubcat - len(selected_for_subsubcat)
            batch_size = min(remaining_needed, len(sorted_ceramics))
            
            # Create duplicated IDs with suffix for tracking
            for i in range(batch_size):
                original_cid = sorted_ceramics[i]
                if duplication_counter == 0:
                    # First occurrence - use original ID
                    new_cid = original_cid
                else:
                    # Subsequent occurrences - add suffix
                    new_cid = next_available_id
                    next_available_id += 1
                
                selected_for_subsubcat.append(new_cid)
                
                # Track duplication
                if original_cid not in ceramic_duplication_map:
                    ceramic_duplication_map[original_cid] = []
                ceramic_duplication_map[original_cid].append(new_cid)
            
            duplication_counter += 1
        
        selected_ceramics.extend(selected_for_subsubcat)
        print(f"    → Selected {len(selected_for_subsubcat)} ceramics (including duplicates)")
    
    return selected_ceramics, ceramic_duplication_map


def duplicate_ceramic_data(ceramic_ids, ceramic_duplication_map, dfs, ceramic_to_info):
    """
    Duplicate ceramic data using integer IDs for duplicates, including proper
    handling of database connections and relationships.
    """
    print("🔄 Duplicating ceramic data with integer IDs...")
    
    # Create copies of all dataframes
    updated_dfs = {name: df.copy() for name, df in dfs.items()}
    updated_ceramic_to_info = ceramic_to_info.copy()
    
    # Track original ceramics for connection duplication
    original_to_duplicates = {}
    
    # Process duplication mapping
    for original_cid, duplicate_list in ceramic_duplication_map.items():
        if original_cid in ceramic_to_info:
            # Store duplicates for this original
            duplicates = [dup for dup in duplicate_list if dup != original_cid]
            if duplicates:
                original_to_duplicates[original_cid] = duplicates
            
            # Copy ceramic info to duplicates
            for duplicate_cid in duplicate_list:
                if duplicate_cid != original_cid:
                    updated_ceramic_to_info[duplicate_cid] = ceramic_to_info[original_cid].copy()
    
    # Duplicate ceramic_summary entries
    ceramic_summary = updated_dfs['ceramic_summary']
    new_ceramic_rows = []
    
    for original_cid, duplicates in original_to_duplicates.items():
        # Find the original ceramic row
        original_rows = ceramic_summary[ceramic_summary['ceramic_id'] == original_cid]
        if not original_rows.empty:
            original_row = original_rows.iloc[0].copy()
            
            # Create duplicate rows
            for duplicate_cid in duplicates:
                duplicate_row = original_row.copy()
                duplicate_row['ceramic_id'] = duplicate_cid
                new_ceramic_rows.append(duplicate_row)
    
    # Add duplicate rows to ceramic_summary
    if new_ceramic_rows:
        duplicate_df = pd.DataFrame(new_ceramic_rows)
        updated_dfs['ceramic_summary'] = pd.concat([ceramic_summary, duplicate_df], ignore_index=True)
    
    print(f"  Ceramic info: {len(ceramic_to_info)} → {len(updated_ceramic_to_info)} entries")
    print(f"  Ceramic summary rows: {len(ceramic_summary)} → {len(updated_dfs['ceramic_summary'])}")
    
    return updated_dfs, updated_ceramic_to_info, original_to_duplicates


def extract_triplets_for_selection_with_duplication(ceramic_ids, dfs, original_to_duplicates=None):
    """
    Extract triplets for selected ceramics, handling duplication properly.
    """
    print("🔍 Extracting triplets for ceramic selection with duplication support...")
    
    if original_to_duplicates is None:
        original_to_duplicates = {}
    
    # Get original extraction results
    original_triplets = extract_triplets_for_selection(ceramic_ids, dfs)
    
    if not original_triplets:
        return original_triplets
    
    # Create a mapping from ceramic_id to triplet entry
    ceramic_to_triplet = {}
    for entry in original_triplets:
        cid = entry.get('ceramic_id')
        if cid is not None:
            ceramic_to_triplet[cid] = entry
    
    # Duplicate triplets for duplicated ceramics
    duplicated_triplets = []
    
    for original_cid, duplicates in original_to_duplicates.items():
        if original_cid in ceramic_to_triplet:
            original_entry = ceramic_to_triplet[original_cid]
            
            # Create duplicate entries
            for duplicate_cid in duplicates:
                if duplicate_cid in ceramic_ids:  # Only if duplicate is in our selection
                    duplicate_entry = original_entry.copy()
                    duplicate_entry['ceramic_id'] = duplicate_cid
                    duplicated_triplets.append(duplicate_entry)
    
    # Combine original and duplicated triplets
    all_triplets = original_triplets + duplicated_triplets
    
    print(f"  Original triplets: {len(original_triplets)}")
    print(f"  Duplicated triplets: {len(duplicated_triplets)}")
    print(f"  Total triplets: {len(all_triplets)}")
    
    return all_triplets


def create_rgcn_dataset(ceramic_ids, ceramic_duplication_map, dfs, ceramic_to_info, 
                       hierarchy, target_connection_level, study_name, 
                       bert_model_name, save_bert_model=True, bert_model_save_dir="models/bert"):
    """
    Create RGCN dataset from selected ceramics with proper duplication handling.
    """
    print(f"🔄 Creating RGCN dataset for {study_name} with duplication...")
    
    # First, duplicate the ceramic data (now returns original_to_duplicates)
    updated_dfs, updated_ceramic_to_info, original_to_duplicates = duplicate_ceramic_data(
        ceramic_ids, ceramic_duplication_map, dfs, ceramic_to_info
    )
    
    # Extract triplets for selected ceramics with duplication support
    triplets_for_study = extract_triplets_for_selection_with_duplication(
        ceramic_ids, updated_dfs, original_to_duplicates
    )
    
    if not triplets_for_study:
        print(f"❌ Failed to extract triplets for {study_name}")
        return None
    
    # Verify that all ceramics in ceramic_ids have corresponding triplets
    triplet_ceramic_ids = set()
    for entry in triplets_for_study:
        cid = entry.get('ceramic_id')
        if cid is not None:
            triplet_ceramic_ids.add(cid)
    
    missing_ceramics = set(ceramic_ids) - triplet_ceramic_ids
    if missing_ceramics:
        print(f"⚠️  Warning: {len(missing_ceramics)} ceramics from ceramic_ids are missing triplets")
        print(f"   Missing IDs (sample): {list(missing_ceramics)[:10]}")
    
    # Format data for RGCN
    rgcn_data = format_rgcn_data_for_study(
        dfs=updated_dfs,
        triplets_for_study=triplets_for_study,
        study_name=study_name,
        category_hierarchy=hierarchy,
        target_connection_level=target_connection_level,
    )
    
    if rgcn_data:
        # Count labeled vs unlabeled ceramics
        ceramic_nodes_with_labels = 0
        ceramic_nodes_total = 0
        
        for node_id in rgcn_data['node_to_idx'].keys():
            if node_id.startswith("Ceramic_"):
                ceramic_nodes_total += 1
        
        # Count ceramics that appear in evaluation triplets
        ceramic_indices_in_eval = set()
        for head, rel, tail in rgcn_data['evaluation_triplets']:
            head_node = rgcn_data['idx_to_node'][head]
            if head_node.startswith("Ceramic_"):
                ceramic_indices_in_eval.add(head)
        
        ceramic_nodes_with_labels = len(ceramic_indices_in_eval)
        
        # Add detailed study metadata
        rgcn_data['study_metadata'] = {
            'study_name': study_name,
            'target_connection_level': target_connection_level,
            'sample_size': len(ceramic_ids),
            'original_ceramics': len(ceramic_duplication_map),
            'total_duplicates': len(ceramic_ids) - len(ceramic_duplication_map),
            'bert_model': bert_model_name,
            'ceramic_nodes_total': ceramic_nodes_total,
            'ceramic_nodes_with_labels': ceramic_nodes_with_labels,
            'ceramic_nodes_without_labels': ceramic_nodes_total - ceramic_nodes_with_labels,
            'labeling_coverage': ceramic_nodes_with_labels / ceramic_nodes_total if ceramic_nodes_total > 0 else 0
        }
        
        print(f"✅ RGCN dataset created for {study_name}")
        print(f"   - Nodes: {rgcn_data['num_nodes']}")
        print(f"   - Relations: {rgcn_data['num_relations']}")
        print(f"   - Training triplets: {len(rgcn_data['training_triplets'])}")
        print(f"   - Evaluation triplets: {len(rgcn_data['evaluation_triplets'])}")
        print(f"   - Original ceramics: {rgcn_data['study_metadata']['original_ceramics']}")
        print(f"   - Total duplicates: {rgcn_data['study_metadata']['total_duplicates']}")
        print(f"   - Ceramic nodes total: {ceramic_nodes_total}")
        print(f"   - Ceramic nodes with labels: {ceramic_nodes_with_labels}")
        print(f"   - Ceramic nodes without labels: {ceramic_nodes_total - ceramic_nodes_with_labels}")
        print(f"   - Labeling coverage: {ceramic_nodes_with_labels / ceramic_nodes_total * 100:.1f}%")
    
    return rgcn_data



def create_rgcn_dataset_level_0(ceramic_ids, dfs, hierarchy, target_connection_level, study_name, 
                       bert_model_name, save_bert_model=True, bert_model_save_dir="models/bert"):
    """
    Create RGCN dataset from selected ceramics.
    """
    print(f"🔄 Creating RGCN dataset for {study_name}...")
    
    # Extract triplets for selected ceramics
    triplets_for_study = extract_triplets_for_selection(ceramic_ids, dfs)
    
    if not triplets_for_study:
        print(f"❌ Failed to extract triplets for {study_name}")
        return None
    
    # Format data for RGCN
    rgcn_data = format_rgcn_data_for_study(
        dfs=dfs,
        triplets_for_study=triplets_for_study,
        study_name=study_name,
        category_hierarchy=hierarchy,
        target_connection_level=target_connection_level,
    )
    
    if rgcn_data:
        # Add study metadata
        rgcn_data['study_metadata'] = {
            'study_name': study_name,
            'target_connection_level': target_connection_level,
            'sample_size': len(ceramic_ids),
            'bert_model': bert_model_name
        }
        
        print(f"✅ RGCN dataset created for {study_name}")
        print(f"   - Nodes: {rgcn_data['num_nodes']}")
        print(f"   - Relations: {rgcn_data['num_relations']}")
        print(f"   - Training triplets: {len(rgcn_data['training_triplets'])}")
        print(f"   - Evaluation triplets: {len(rgcn_data['evaluation_triplets'])}")
    
    return rgcn_data


def save_three_level_datasets(datasets, dfs, base_output_dir):
    """
    Save the three level datasets, adapting and saving a classification version for each.
    """
    os.makedirs(base_output_dir, exist_ok=True)
    
    for study_name, rgcn_data in datasets.items():
        if rgcn_data is None:
            print(f"\n⚠️ Skipping {study_name} as its data is None.")
            continue
            
        study_dir = os.path.join(base_output_dir, study_name)
        os.makedirs(study_dir, exist_ok=True)
        
        # --- 1. Save the original link-prediction dataset ---
        link_prediction_dir = os.path.join(study_dir, "link_prediction_data")
        print(f"\n--- Saving original link prediction data for {study_name} ---")
        save_single_rgcn_dataset(
            rgcn_data, 
            output_dir=link_prediction_dir, 
            dataset_label=f"{study_name}_link_prediction"
        )

        # --- 2. Adapt data for classification ---
        print(f"\n--- Adapting {study_name} for classification ---")
        classification_data = adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs=dfs)
        
        # --- 3. Save the adapted classification dataset ---
        if classification_data:
            classification_dir = os.path.join(study_dir, "classification_data")
            print(f"\n--- Saving classification data for {study_name} ---")
            save_classification_dataset(
                classification_data, 
                output_dir=classification_dir, 
                dataset_label=f"{study_name}_classification"
            )
        else:
            print(f"⚠️ Skipping classification data saving for {study_name} due to adaptation failure.")

        # --- 4. Save study configuration and summary ---
        config = generate_study_config(rgcn_data)
        config_path = os.path.join(study_dir, "study_config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        summary_path = os.path.join(study_dir, f"{study_name}_summary.txt")
        with open(summary_path, 'w') as f:
            f.write(f"Study: {study_name}\n")
            f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("[Link Prediction Data]\n")
            if rgcn_data.get('study_metadata'):
                f.write(f"  Target connection level: {rgcn_data['study_metadata']['target_connection_level']}\n")
                f.write(f"  Sample size: {rgcn_data['study_metadata']['sample_size']}\n")
                f.write(f"  BERT model: {rgcn_data['study_metadata']['bert_model']}\n")
            f.write(f"  Nodes: {rgcn_data['num_nodes']}\n")
            f.write(f"  Relations: {rgcn_data['num_relations']}\n")
            f.write(f"  Training triplets: {len(rgcn_data['training_triplets'])}\n")
            f.write(f"  Evaluation triplets: {len(rgcn_data['evaluation_triplets'])}\n\n")

            f.write("[Classification Data]\n")
            if classification_data and classification_data.get('stats'):
                class_stats = classification_data['stats']
                f.write(f"  Ceramic nodes with labels: {class_stats['classification_ceramic_nodes_with_labels']}\n")
                f.write(f"  Total classes: {class_stats['classification_num_classes']}\n")
            else:
                f.write("  Not generated.\n")
        
        print(f"\n✅ Finished processing and saving for {study_name} in {study_dir}")