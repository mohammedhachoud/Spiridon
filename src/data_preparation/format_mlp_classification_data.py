# src/data_preparation/format_mlp_classification_data.py
import pandas as pd
import numpy as np
import os
import json
from collections import Counter
from .. import config # Assuming config.py has OUTPUT_BASE_DIR

def create_mlp_input_data(dfs, study_name="etude1", embedding_type=1, max_functions=None, max_features=None):
    """
    Create MLP input data with different embedding types.
    
    Args:
        dfs: Dictionary of DataFrames
        study_name: str, one of ["etude1", "etude1_prime", "etude2"]
            - "etude1": 138 ceramics per class (default)
            - "etude1_prime": 276 ceramics per class (double the minimum class)
            - "etude2": Remove minimum class, select 950 from each remaining class
        embedding_type: int, one of [0, 1, 2]
            - 0: Only ceramic attributes (origin, color, context, source, reuse, production_fail)
            - 1: Only functions + features (default, current behavior)
            - 2: Combined ceramic attributes + functions + features
        max_functions: Deprecated parameter (ignored)
        max_features: Deprecated parameter (ignored)
    """
    print(f"Preparing data for MLP Classifier...")
    print(f"Study configuration: {study_name}")
    print(f"Embedding type: {embedding_type}")
    
    # Map embedding type to description
    embedding_descriptions = {
        0: "Ceramic attributes only (origin, color, context, source, reuse, production_fail)",
        1: "Functions + Features multi-label one-hot encoding",
        2: "Combined: Ceramic attributes + Functions + Features"
    }
    print(f"Embedding strategy: {embedding_descriptions.get(embedding_type, 'Unknown')}")

    # Validate inputs
    valid_studies = ["etude1", "etude1_prime", "etude2"]
    if study_name not in valid_studies:
        print(f"  ERROR: Invalid study_name '{study_name}'. Must be one of {valid_studies}")
        return None, None, None, None, None
    
    valid_embedding_types = [0, 1, 2]
    if embedding_type not in valid_embedding_types:
        print(f"  ERROR: Invalid embedding_type '{embedding_type}'. Must be one of {valid_embedding_types}")
        return None, None, None, None, None

    # Check required DataFrames
    if 'ceramic_summary' not in dfs or dfs['ceramic_summary'].empty:
        print("  ERROR: 'ceramic_summary' DataFrame is missing or empty.")
        return None, None, None, None, None
    if 'tech_cat' not in dfs or dfs['tech_cat'].empty:
        print("  ERROR: 'tech_cat' DataFrame is missing or empty.")
        return None, None, None, None, None
    
    # Check for function/feature data when needed
    if embedding_type in [1, 2]:
        if 'object_function' not in dfs or dfs['object_function'].empty:
            print("  ERROR: 'object_function' DataFrame is missing or empty (required for embedding types 1 and 2).")
            return None, None, None, None, None
        if 'object_feature' not in dfs or dfs['object_feature'].empty:
            print("  ERROR: 'object_feature' DataFrame is missing or empty (required for embedding types 1 and 2).")
            return None, None, None, None, None

    ceramic_summary_df = dfs['ceramic_summary'].copy()
    tech_cat_df = dfs['tech_cat'].copy()
    
    if embedding_type in [1, 2]:
        object_function_df = dfs['object_function'].copy()
        object_feature_df = dfs['object_feature'].copy()

    # --- 1. Generate ceramic attribute embeddings (for types 0 and 2) ---
    def generate_ceramic_attribute_embeddings(df):
        """Generate one-hot embeddings for ceramic attributes."""
        print("  Generating ceramic attribute embeddings...")
        
        if df.empty:
            print("    DataFrame is empty. Cannot generate embeddings.")
            return df, {}, {}

        # Get unique values for each attribute
        origin_values = sorted(df['origin'].dropna().unique())
        
        # Handle color list
        all_colors = set()
        for sublist in df['color_name_list'].dropna():
            if isinstance(sublist, list):
                for color in sublist:
                    if pd.notna(color):
                        all_colors.add(str(color).strip())
        color_values = sorted(list(all_colors))

        context_values = sorted(df['context_type_name'].dropna().unique())
        source_values = sorted(df['identifier_origin_source_name'].dropna().unique())

        # Create mappings
        origin_map = {v: i for i, v in enumerate(origin_values)}
        color_map = {v: i for i, v in enumerate(color_values)}
        context_map = {v: i for i, v in enumerate(context_values)}
        source_map = {v: i for i, v in enumerate(source_values)}

        print(f"    Attribute map sizes: origin={len(origin_map)}, color={len(color_map)}, context={len(context_map)}, source={len(source_map)}")

        def get_ceramic_embedding(row):
            embedding = []
            
            # Origin
            origin_vector = [0] * len(origin_map)
            if pd.notna(row['origin']) and row['origin'] in origin_map:
                origin_vector[origin_map[row['origin']]] = 1
            embedding.extend(origin_vector)
            
            # Colors (multi-label)
            color_vector = [0] * len(color_map)
            if isinstance(row['color_name_list'], list):
                for c_val in row['color_name_list']:
                    c_str = str(c_val).strip()
                    if c_str in color_map:
                        color_vector[color_map[c_str]] = 1
            embedding.extend(color_vector)
            
            # Context
            context_vector = [0] * len(context_map)
            if pd.notna(row['context_type_name']) and row['context_type_name'] in context_map:
                context_vector[context_map[row['context_type_name']]] = 1
            embedding.extend(context_vector)
            
            # Source
            source_vector = [0] * len(source_map)
            if pd.notna(row['identifier_origin_source_name']) and row['identifier_origin_source_name'] in source_map:
                source_vector[source_map[row['identifier_origin_source_name']]] = 1
            embedding.extend(source_vector)
            
            # Reuse (binary with NA handling)
            if pd.isna(row['reuse']): 
                embedding.extend([0, 0])  # NA state
            else: 
                embedding.extend([0, 1] if row['reuse'] else [1, 0])  # True/False
            
            # Production Fail (binary with NA handling)
            if pd.isna(row['production_fail']): 
                embedding.extend([0, 0])  # NA state
            else: 
                embedding.extend([0, 1] if row['production_fail'] else [1, 0])  # True/False
            
            return embedding

        df['ceramic_attribute_embedding'] = df.apply(get_ceramic_embedding, axis=1)
        
        # Calculate embedding dimensions
        ceramic_embedding_info = {
            'origin_dim': len(origin_map),
            'color_dim': len(color_map),
            'context_dim': len(context_map),
            'source_dim': len(source_map),
            'reuse_dim': 2,
            'production_fail_dim': 2,
            'total_dim': len(origin_map) + len(color_map) + len(context_map) + len(source_map) + 4
        }
        
        attribute_maps = {
            'origin_map': origin_map,
            'color_map': color_map,
            'context_map': context_map,
            'source_map': source_map
        }
        
        if not df.empty and 'ceramic_attribute_embedding' in df.columns:
            actual_length = len(df['ceramic_attribute_embedding'].iloc[0])
            print(f"    Ceramic attribute embedding generated. Length: {actual_length}")
            print(f"    Breakdown: origin({len(origin_map)}) + color({len(color_map)}) + context({len(context_map)}) + source({len(source_map)}) + reuse(2) + prod_fail(2)")
        
        return df, ceramic_embedding_info, attribute_maps

    # --- 2. Generate function/feature embeddings (for types 1 and 2) ---
    def generate_function_feature_embeddings(df):
        """Generate multi-label one-hot embeddings for functions and features."""
        print("  Generating function/feature embeddings...")
        
        # Create mapping for functions and features
        function_values = sorted(object_function_df['id'].dropna().unique().astype(int))
        function_map = {v: i for i, v in enumerate(function_values)}
        num_distinct_functions = len(function_map)
        print(f"    Function embedding length: {num_distinct_functions}")

        feature_values = sorted(object_feature_df['id'].dropna().unique().astype(int))
        feature_map = {v: i for i, v in enumerate(feature_values)}
        num_distinct_features = len(feature_map)
        print(f"    Feature embedding length: {num_distinct_features}")

        def get_function_feature_embedding(row):
            # Function multi-label one-hot vector
            function_vector = [0] * num_distinct_functions
            if isinstance(row['function_id'], list):
                func_ids_in_row = [int(f) for f in row['function_id'] if pd.notna(f) and int(f) in function_map]
                for func_id in func_ids_in_row:
                    if func_id in function_map:
                        function_vector[function_map[func_id]] = 1

            # Feature multi-label one-hot vector
            feature_vector = [0] * num_distinct_features
            if isinstance(row['feature_id'], list):
                feat_ids_in_row = [int(f) for f in row['feature_id'] if pd.notna(f) and int(f) in feature_map]
                for feat_id in feat_ids_in_row:
                    if feat_id in feature_map:
                        feature_vector[feature_map[feat_id]] = 1
                
            # Concatenate function and feature vectors
            embedding = function_vector + feature_vector
            return embedding

        df['function_feature_embedding'] = df.apply(get_function_feature_embedding, axis=1)
        
        function_feature_info = {
            'function_dim': num_distinct_functions,
            'feature_dim': num_distinct_features,
            'total_dim': num_distinct_functions + num_distinct_features
        }
        
        maps = {
            'function_map': function_map,
            'feature_map': feature_map
        }
        
        if not df.empty and 'function_feature_embedding' in df.columns:
            actual_length = len(df['function_feature_embedding'].iloc[0])
            print(f"    Function/feature embedding generated. Length: {actual_length}")
            print(f"    Breakdown: functions({num_distinct_functions}) + features({num_distinct_features})")
        
        return df, function_feature_info, maps

    # --- 3. Find root categories (same as before) ---
    def find_root_category(cat_id, tc_df):
        visited = set()
        current_id = cat_id
        while current_id not in visited:
            visited.add(current_id)
            parent_series = tc_df[tc_df['id'] == current_id]['inherit_from']
            if parent_series.empty or pd.isna(parent_series.iloc[0]):
                return current_id
            parent_id = parent_series.iloc[0]
            if pd.isna(parent_id): 
                return current_id
            current_id = int(parent_id)
            if current_id in visited: 
                return current_id
        return current_id

    root_categories_ids = [row_id for row_id in tech_cat_df[pd.isna(tech_cat_df['inherit_from'])]['id'].tolist() if pd.notna(row_id)]
    print(f"  Root categories found (IDs): {root_categories_ids}")

    category_to_root = {}
    for _, row in tech_cat_df.iterrows():
        if pd.notna(row['id']):
            category_to_root[int(row['id'])] = find_root_category(int(row['id']), tech_cat_df)

    ceramic_summary_df['root_tech_cat_id'] = ceramic_summary_df['tech_cat_id'].apply(
        lambda x: category_to_root.get(int(x), -1) if pd.notna(x) and int(x) in category_to_root else -1
    )
    
    # Filter for categorized ceramics
    categorized_ceramics_df = ceramic_summary_df[ceramic_summary_df['root_tech_cat_id'] != -1].copy()
    print(f"  Total categorized ceramics before sampling: {len(categorized_ceramics_df)}")

    if categorized_ceramics_df.empty:
        print("  No categorized ceramics found. MLP data preparation cannot proceed.")
        return None, None, None, None, None

    # --- 4. Apply study-specific sampling strategy (same as before) ---
    def apply_sampling_strategy(df, study_name):
        """Apply sampling strategy based on study configuration"""
        # Get class distribution
        class_counts = df['root_tech_cat_id'].value_counts().sort_index()
        print(f"  Original class distribution:")
        for class_id, count in class_counts.items():
            cat_name_col = 'cat_name_processed' if 'cat_name_processed' in tech_cat_df.columns else 'cat_name'
            name_series = tech_cat_df[tech_cat_df['id'] == class_id][cat_name_col]
            class_name = name_series.iloc[0] if not name_series.empty else f"Unknown_{class_id}"
            print(f"    Class {class_id} ({class_name}): {count} ceramics")
        
        min_class_count = class_counts.min()
        min_class_id = class_counts.idxmin()
        print(f"  Minimum class: {min_class_id} with {min_class_count} ceramics")
        
        if study_name == "etude1":
            target_count = 138
            print(f"  Applying etude1 strategy: {target_count} ceramics per class")
            
            sampled_dfs = []
            for class_id in class_counts.index:
                class_df = df[df['root_tech_cat_id'] == class_id]
                if len(class_df) >= target_count:
                    sampled_df = class_df.sample(n=target_count, random_state=42)
                else:
                    print(f"    WARNING: Class {class_id} has only {len(class_df)} ceramics, less than target {target_count}")
                    sampled_df = class_df
                sampled_dfs.append(sampled_df)
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            
        elif study_name == "etude1_prime":
            target_count = min_class_count * 2
            print(f"  Applying etude1_prime strategy: {target_count} ceramics per class (2x minimum)")
            
            sampled_dfs = []
            for class_id in class_counts.index:
                class_df = df[df['root_tech_cat_id'] == class_id]
                if len(class_df) >= target_count:
                    sampled_df = class_df.sample(n=target_count, random_state=42)
                else:
                    print(f"    WARNING: Class {class_id} has only {len(class_df)} ceramics, less than target {target_count}")
                    sampled_df = class_df
                sampled_dfs.append(sampled_df)
            result_df = pd.concat(sampled_dfs, ignore_index=True)
            
        elif study_name == "etude2":
            target_count = 950
            print(f"  Applying etude2 strategy: Remove minimum class ({min_class_id}), sample {target_count} from remaining classes")
            
            df_filtered = df[df['root_tech_cat_id'] != min_class_id].copy()
            remaining_class_counts = df_filtered['root_tech_cat_id'].value_counts().sort_index()
            
            print(f"  Remaining classes after removing minimum:")
            for class_id, count in remaining_class_counts.items():
                cat_name_col = 'cat_name_processed' if 'cat_name_processed' in tech_cat_df.columns else 'cat_name'
                name_series = tech_cat_df[tech_cat_df['id'] == class_id][cat_name_col]
                class_name = name_series.iloc[0] if not name_series.empty else f"Unknown_{class_id}"
                print(f"    Class {class_id} ({class_name}): {count} ceramics")
            
            sampled_dfs = []
            for class_id in remaining_class_counts.index:
                class_df = df_filtered[df_filtered['root_tech_cat_id'] == class_id]
                if len(class_df) >= target_count:
                    sampled_df = class_df.sample(n=target_count, random_state=42)
                else:
                    print(f"    WARNING: Class {class_id} has only {len(class_df)} ceramics, less than target {target_count}")
                    sampled_df = class_df
                sampled_dfs.append(sampled_df)
            result_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Print final distribution
        final_counts = result_df['root_tech_cat_id'].value_counts().sort_index()
        print(f"  Final class distribution after {study_name} sampling:")
        for class_id, count in final_counts.items():
            cat_name_col = 'cat_name_processed' if 'cat_name_processed' in tech_cat_df.columns else 'cat_name'
            name_series = tech_cat_df[tech_cat_df['id'] == class_id][cat_name_col]
            class_name = name_series.iloc[0] if not name_series.empty else f"Unknown_{class_id}"
            print(f"    Class {class_id} ({class_name}): {count} ceramics")
        
        return result_df
    
    # Apply sampling strategy
    categorized_ceramics_df = apply_sampling_strategy(categorized_ceramics_df, study_name)
    print(f"  Total ceramics after {study_name} sampling: {len(categorized_ceramics_df)}")

    # --- 5. Generate embeddings based on type ---
    embedding_info = {}
    all_maps = {}
    
    if embedding_type == 0:
        # Only ceramic attributes
        categorized_ceramics_df, ceramic_info, ceramic_maps = generate_ceramic_attribute_embeddings(categorized_ceramics_df)
        categorized_ceramics_df['mlp_embedding'] = categorized_ceramics_df['ceramic_attribute_embedding']
        embedding_info.update(ceramic_info)
        all_maps.update(ceramic_maps)
        
    elif embedding_type == 1:
        # Only functions + features
        categorized_ceramics_df, func_feat_info, func_feat_maps = generate_function_feature_embeddings(categorized_ceramics_df)
        categorized_ceramics_df['mlp_embedding'] = categorized_ceramics_df['function_feature_embedding']
        embedding_info.update(func_feat_info)
        all_maps.update(func_feat_maps)
        
    elif embedding_type == 2:
        # Combined: ceramic attributes + functions + features
        categorized_ceramics_df, ceramic_info, ceramic_maps = generate_ceramic_attribute_embeddings(categorized_ceramics_df)
        categorized_ceramics_df, func_feat_info, func_feat_maps = generate_function_feature_embeddings(categorized_ceramics_df)
        
        # Combine embeddings
        def combine_embeddings(row):
            ceramic_emb = row['ceramic_attribute_embedding']
            func_feat_emb = row['function_feature_embedding']
            return ceramic_emb + func_feat_emb
        
        categorized_ceramics_df['mlp_embedding'] = categorized_ceramics_df.apply(combine_embeddings, axis=1)
        
        # Combine embedding info
        embedding_info = {
            'ceramic_attributes': ceramic_info,
            'function_features': func_feat_info,
            'total_dim': ceramic_info['total_dim'] + func_feat_info['total_dim']
        }
        all_maps.update(ceramic_maps)
        all_maps.update(func_feat_maps)

    # --- 6. Verify and prepare final data ---
    if not categorized_ceramics_df.empty and 'mlp_embedding' in categorized_ceramics_df.columns:
        actual_len = len(categorized_ceramics_df['mlp_embedding'].iloc[0])
        print(f"  Final MLP embedding generated. Length: {actual_len}")
    
    X = np.array(categorized_ceramics_df['mlp_embedding'].tolist())
    y = categorized_ceramics_df['root_tech_cat_id'].values.astype(int)

    # Create root category names map
    root_category_names = {}
    cat_name_col = 'cat_name_processed' if 'cat_name_processed' in tech_cat_df.columns else 'cat_name'
    for cat_id_val in np.unique(y):
        name_series = tech_cat_df[tech_cat_df['id'] == cat_id_val][cat_name_col]
        if not name_series.empty:
            root_category_names[cat_id_val] = name_series.iloc[0]
        else:
            root_category_names[cat_id_val] = f"UnknownRoot ({cat_id_val})"
    
    print(f"\n  Final MLP data shapes:")
    print(f"    X shape: {X.shape}")
    print(f"    y shape: {y.shape}")
    print(f"    Unique labels: {np.unique(y)}")

    # --- 7. Save processed data ---
    output_mlp_data_dir = os.path.join(config.OUTPUT_BASE_DIR, "mlp_classification_data", f"{study_name}_type{embedding_type}")
    os.makedirs(output_mlp_data_dir, exist_ok=True)
    
    np.save(os.path.join(output_mlp_data_dir, "X_mlp_embeddings.npy"), X)
    np.save(os.path.join(output_mlp_data_dir, "y_mlp_labels.npy"), y)
    
    with open(os.path.join(output_mlp_data_dir, "root_category_names_mlp.json"), 'w') as f:
        json.dump({int(k):v for k,v in root_category_names.items()}, f, indent=4)
    
    # Save all maps
    with open(os.path.join(output_mlp_data_dir, "embedding_maps.json"), 'w') as f:
        # Convert all map values to int for JSON serialization
        serializable_maps = {}
        for key, mapping in all_maps.items():
            if isinstance(mapping, dict):
                serializable_maps[key] = {str(k): int(v) if isinstance(v, (int, np.integer)) else v for k, v in mapping.items()}
            else:
                serializable_maps[key] = mapping
        json.dump(serializable_maps, f, indent=4)
    
    # Save study configuration info
    study_config = {
        "study_name": study_name,
        "embedding_type": embedding_type,
        "embedding_description": embedding_descriptions[embedding_type],
        "sampling_strategy": {
            "etude1": "138 ceramics per class",
            "etude1_prime": f"276 ceramics per class (2x minimum class size)",
            "etude2": "Remove minimum class, 950 ceramics per remaining class"
        }[study_name],
        "total_samples": len(X),
        "n_classes": len(np.unique(y)),
        "embedding_dimensions": X.shape[1],
        "embedding_info": embedding_info
    }
    
    with open(os.path.join(output_mlp_data_dir, "study_config.json"), 'w') as f:
        json.dump(study_config, f, indent=4)

    print(f"  MLP data saved to: {output_mlp_data_dir}")
    print(f"  Study configuration saved as study_config.json")
    
    return X, y, root_category_names, all_maps, embedding_info