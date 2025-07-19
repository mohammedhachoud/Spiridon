import re
import ast
import json
import os
import pandas as pd
import numpy as np
from collections import defaultdict
from utils import CategoryHierarchy

# Assume CategoryHierarchy class is defined elsewhere

def _build_global_vocab_maps(dfs: dict):
    """
    Scans all dataframes to build comprehensive, global vocabulary maps
    for all categorical features. This ensures embedding consistency across all datasets.
    """
    print("  Building global vocabularies from all available data...")
    
    # --- Ceramic Attributes from ceramic_summary ---
    ceramic_df = dfs['ceramic_summary']
    
    origin_map = {v: i for i, v in enumerate(sorted(ceramic_df['origin'].dropna().unique()))}
    context_map = {v: i for i, v in enumerate(sorted(ceramic_df['context_type_name'].dropna().unique()))}
    source_map = {v: i for i, v in enumerate(sorted(ceramic_df['identifier_origin_source_name'].dropna().unique()))}
    
    all_colors = set()
    for color_list in ceramic_df['color_name_list'].dropna():
        if isinstance(color_list, str):
            # If it's a string representation of a list, parse it
            try:
                color_list = ast.literal_eval(color_list)
            except (ValueError, SyntaxError):
                # If parsing fails, skip this entry
                continue
        
        if isinstance(color_list, list):
            for color in color_list:
                if pd.notna(color):
                    all_colors.add(str(color).strip())
    color_map = {v: i for i, v in enumerate(sorted(list(all_colors)))}
    
    # --- Functions and Features from their respective tables ---
    function_map = {v: i for i, v in enumerate(sorted(dfs['object_function']['id'].dropna().unique().astype(int)))}
    feature_map = {v: i for i, v in enumerate(sorted(dfs['object_feature']['id'].dropna().unique().astype(int)))}
    
    # --- Mappings for debugging/readability ---
    func_id_to_name = dict(zip(dfs['object_function']['id'], dfs['object_function'].get('function_name_fr', 'Unknown')))
    feat_id_to_name = dict(zip(dfs['object_feature']['id'], dfs['object_feature'].get('feature_name_fr', 'Unknown')))

    print(f"  Global vocabularies built.")
    print(f"    - Origins: {len(origin_map)}, Colors: {len(color_map)}, Contexts: {len(context_map)}, Sources: {len(source_map)}")
    print(f"    - Functions: {len(function_map)}, Features: {len(feature_map)}")

    return {
        "origin_map": origin_map, "color_map": color_map, "context_map": context_map,
        "source_map": source_map, "function_map": function_map, "feature_map": feature_map,
        "func_id_to_name": func_id_to_name, "feat_id_to_name": feat_id_to_name
    }


def calculate_completeness_score(ceramic_ids, dfs):
    """
    Calculate completeness scores for ceramics based on available data.
    """
    scores = {}
    ceramic_df = dfs['ceramic_summary']
    
    for cid in ceramic_ids:
        ceramic_row = ceramic_df[ceramic_df['ceramic_id'] == cid]
        if ceramic_row.empty:
            scores[cid] = 0
            continue
            
        row = ceramic_row.iloc[0]
        score = 0
        
        # Helper function to check if a value is not null/empty
        def is_not_empty(value):
            # Handle None/NaN values first
            if value is None:
                return False
            
            # For scalar values, use pd.isna() safely
            try:
                if pd.isna(value):
                    return False
            except (ValueError, TypeError):
                # If pd.isna() fails, try alternative approaches
                if hasattr(value, '__len__') and not isinstance(value, str):
                    # For arrays/lists, check if any element is not NaN
                    try:
                        if isinstance(value, np.ndarray):
                            return not np.isnan(value).all()
                        elif isinstance(value, (list, tuple)):
                            return len(value) > 0 and not all(pd.isna(x) for x in value if x is not None)
                    except:
                        return len(value) > 0
                
            # Check for different data types
            if isinstance(value, (list, tuple, np.ndarray)):
                return len(value) > 0
            if isinstance(value, str):
                return value.strip() != ''
            
            return True
        
        # Score based on available attributes
        if is_not_empty(row.get('origin')):
            score += 1
        if is_not_empty(row.get('context_type_name')):
            score += 1
        if is_not_empty(row.get('identifier_origin_source_name')):
            score += 1
        if is_not_empty(row.get('color_name_list')):
            score += 1
        if is_not_empty(row.get('reuse')):
            score += 1
        if is_not_empty(row.get('production_fail')):
            score += 1
        if is_not_empty(row.get('function_id')):
            score += 1
        if is_not_empty(row.get('feature_id')):
            score += 1
            
        scores[cid] = score
    
    return scores
def get_level1_category(cat_id, hierarchy):
    """
    Gets the Level 1 ancestor for a given category. If the category itself is at 
    Level 1, it returns its own ID.
    """
    return hierarchy.get_ancestor_at_level(cat_id, target_level=1)


def get_level2_category(cat_id, hierarchy):
    """
    Gets the Level 2 ancestor for a given category. If the category itself is at 
    Level 2, it returns its own ID.
    """
    return hierarchy.get_ancestor_at_level(cat_id, target_level=2)


def prepare_level0_dataset(ceramic_to_info, hierarchy, dfs, target_per_root=400):
    """
    Prepare Level 0 dataset: 400 ceramics per root category with diverse subcategory sampling.
    """
    print("📊 Preparing Level 0 dataset...")
    
    # Group ceramics by root category, filtering out None values
    ceramics_by_root = defaultdict(list)
    for cid, info in ceramic_to_info.items():
        root_id = info['root']
        if root_id is not None:  # Only include ceramics with valid root categories
            ceramics_by_root[root_id].append(cid)
    
    if not ceramics_by_root:
        print("  ❌ No ceramics found with valid root categories")
        return []
    
    selected_ceramics = []
    
    for root_id in sorted(ceramics_by_root.keys()):
        root_ceramics = ceramics_by_root[root_id]
        print(f"  Root {root_id}: {len(root_ceramics)} available ceramics")
        
        # Group by subcategory levels for diverse sampling
        ceramics_by_level = defaultdict(list)
        for cid in root_ceramics:
            level = ceramic_to_info[cid]['level']
            if level is not None:  # Only include ceramics with valid levels
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
        level = hierarchy.get_level(cat_id)
        
        # Only consider ceramics at level 2 or deeper
        if level is not None and level >= 2:
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


def _generate_single_mlp_dataset(
    dfs: dict,
    hierarchy,
    ceramic_ids_to_process: list,
    ceramic_duplication_map: dict,
    target_y_level: int,
    output_dir: str,
    global_maps: dict,
    debug_examples: bool = True
):
    """
    Generates and saves a single MLP dataset for a pre-defined set of ceramics.
    Uses pre-built global vocabularies to ensure consistent embeddings.
    Handles ceramic duplication for balanced datasets.
    Only generates type 2 embeddings (combined attributes + functions + features).
    """
    
    # --- 1. Handle ceramic duplication for balanced datasets ---
    print(f"    Processing {len(ceramic_ids_to_process)} ceramics (including duplicates)...")
    
    # Create expanded ceramic summary with duplicates
    ceramic_summary_df = dfs['ceramic_summary'].copy()
    
    # Add duplicate entries to the dataframe
    duplicate_rows = []
    for original_cid, duplicate_list in ceramic_duplication_map.items():
        if original_cid in ceramic_ids_to_process:
            original_row = ceramic_summary_df[ceramic_summary_df['ceramic_id'] == original_cid]
            if not original_row.empty:
                original_row = original_row.iloc[0]
                
                # Create duplicate rows with new ceramic_ids
                for duplicate_cid in duplicate_list:
                    if duplicate_cid != original_cid and duplicate_cid in ceramic_ids_to_process:
                        duplicate_row = original_row.copy()
                        duplicate_row['ceramic_id'] = duplicate_cid
                        duplicate_rows.append(duplicate_row)
    
    # Add duplicate rows to the dataframe
    if duplicate_rows:
        duplicate_df = pd.DataFrame(duplicate_rows)
        ceramic_summary_df = pd.concat([ceramic_summary_df, duplicate_df], ignore_index=True)
    
    # Filter to only ceramics we're processing
    ceramic_summary_df = ceramic_summary_df[ceramic_summary_df['ceramic_id'].isin(ceramic_ids_to_process)].copy()
    
    if ceramic_summary_df.empty:
        print("    ❌ No matching ceramics found in summary. Skipping generation.")
        return False
    
    # --- 2. Generate 'y' labels ---
    print(f"    Generating 'y' labels by finding ancestors at Level {target_y_level}...")
    y_labels = [hierarchy.get_ancestor_at_level(row['tech_cat_id'], target_y_level) or -1 for _, row in ceramic_summary_df.iterrows()]
    ceramic_summary_df['target_y_label'] = y_labels
    final_df = ceramic_summary_df[ceramic_summary_df['target_y_label'] != -1].copy()
    
    if final_df.empty:
        print("    ❌ Could not determine target labels for any of the selected ceramics. Aborting.")
        return False

    def robust_parse_ids(value):
        """Robustly parses a cell value that may contain function/feature IDs."""
        if hasattr(value, '__iter__') and not isinstance(value, str):
            return [int(v) for v in value if pd.notna(v)]
        if pd.isna(value):
            return []
        if isinstance(value, str):
            return [int(num) for num in re.findall(r'\d+', value)]
        if isinstance(value, (int, float, np.number)):
            return [int(value)]
        return []

    def _generate_ceramic_attribute_embeddings(df, maps):
        print("    🧱 Generating ceramic attribute embeddings (using global vocab)...")
        origin_map, color_map, context_map, source_map = maps['origin_map'], maps['color_map'], maps['context_map'], maps['source_map']

        # Build offset tracking
        offset_origin = 0
        offset_color = offset_origin + len(origin_map)
        offset_context = offset_color + len(color_map)
        offset_source = offset_context + len(context_map)
        offset_reuse = offset_source + len(source_map)
        offset_prod_fail = offset_reuse + 2  # Binary reuse indicator uses 2 positions
        total_dim = offset_prod_fail + 2     # Binary production failure uses 2 positions

        # Create a detailed index map
        index_map = {}

        for idx, key in enumerate(origin_map.keys()):
            index_map[offset_origin + idx] = {"type": "origin", "value": key}

        for idx, key in enumerate(color_map.keys()):
            index_map[offset_color + idx] = {"type": "color", "value": key}

        for idx, key in enumerate(context_map.keys()):
            index_map[offset_context + idx] = {"type": "context_type_name", "value": key}

        for idx, key in enumerate(source_map.keys()):
            index_map[offset_source + idx] = {"type": "identifier_origin_source_name", "value": key}

        index_map[offset_reuse] = {"type": "reuse", "value": "False"}
        index_map[offset_reuse + 1] = {"type": "reuse", "value": "True"}

        index_map[offset_prod_fail] = {"type": "production_fail", "value": "False"}
        index_map[offset_prod_fail + 1] = {"type": "production_fail", "value": "True"}

        # Save this into the maps
        maps["ceramic_index_map"] = index_map

        def get_ceramic_embedding(row):
            origin_vector = [0] * len(origin_map)
            color_vector = [0] * len(color_map)
            context_vector = [0] * len(context_map)
            source_vector = [0] * len(source_map)

            if pd.notna(row['origin']) and row['origin'] in origin_map:
                origin_vector[origin_map[row['origin']]] = 1

            color_list_str = row['color_name_list']
            try:
                colors = ast.literal_eval(color_list_str) if isinstance(color_list_str, str) else []
                if isinstance(colors, list):
                    for c_val in colors:
                        c_str = str(c_val).strip()
                        if c_str in color_map:
                            color_vector[color_map[c_str]] = 1
            except:
                pass

            if pd.notna(row['context_type_name']) and row['context_type_name'] in context_map:
                context_vector[context_map[row['context_type_name']]] = 1

            if pd.notna(row['identifier_origin_source_name']) and row['identifier_origin_source_name'] in source_map:
                source_vector[source_map[row['identifier_origin_source_name']]] = 1

            reuse_emb = [0, 0] if pd.isna(row['reuse']) else ([0, 1] if row['reuse'] else [1, 0])
            prod_fail_emb = [0, 0] if pd.isna(row['production_fail']) else ([0, 1] if row['production_fail'] else [1, 0])

            return origin_vector + color_vector + context_vector + source_vector + reuse_emb + prod_fail_emb

        df['ceramic_attribute_embedding'] = df.apply(get_ceramic_embedding, axis=1)

        info = {
            'total_dim': total_dim,
            'offsets': {
                'origin': (offset_origin, offset_color),
                'color': (offset_color, offset_context),
                'context': (offset_context, offset_source),
                'source': (offset_source, offset_reuse),
                'reuse': (offset_reuse, offset_prod_fail),
                'prod_fail': (offset_prod_fail, total_dim)
            }
        }

        

        print(f"      ✅ Ceramic attribute embedding generated. Total dimensions: {info['total_dim']}")
        return df, info
    
    def _generate_function_feature_embeddings(df, maps):
        print("    🔧 Generating function/feature embeddings (using global vocab)...")
        function_map, feature_map = maps['function_map'], maps['feature_map']
        func_id_to_name, feat_id_to_name = maps['func_id_to_name'], maps['feat_id_to_name']

        offset_func = 0
        offset_feat = offset_func + len(function_map)
        total_dim = offset_feat + len(feature_map)

        index_map = {}

        for idx, func_id in enumerate(function_map.keys()):
            index_map[offset_func + idx] = {"type": "function_id", "id": int(func_id), "name": func_id_to_name.get(int(func_id), "Unknown")}

        for idx, feat_id in enumerate(feature_map.keys()):
            index_map[offset_feat + idx] = {"type": "feature_id", "id": int(feat_id), "name": feat_id_to_name.get(int(feat_id), "Unknown")}

        maps["function_feature_index_map"] = index_map

        def get_func_feat_embedding(row):
            function_vector = [0] * len(function_map)
            feature_vector = [0] * len(feature_map)

            func_ids = robust_parse_ids(row.get('function_id'))
            for func_id in func_ids:
                if func_id in function_map:
                    function_vector[function_map[func_id]] = 1

            feat_ids = robust_parse_ids(row.get('feature_id'))
            for feat_id in feat_ids:
                if feat_id in feature_map:
                    feature_vector[feature_map[feat_id]] = 1

            return function_vector + feature_vector

        df['function_feature_embedding'] = df.apply(get_func_feat_embedding, axis=1)

        info = {
            'function_dim': len(function_map),
            'feature_dim': len(feature_map),
            'total_dim': total_dim,
            'offsets': {
                'functions': (offset_func, offset_feat),
                'features': (offset_feat, total_dim)
            }
        }

        print(f"      ✅ Function/feature embedding generated. Total dimensions: {info['total_dim']}")
        return df, info

    # --- Generate combined embeddings (type 2 only) ---
    print("    🔧 Generating combined embeddings (ceramic attributes + functions + features)...")
    final_df, ceramic_info = _generate_ceramic_attribute_embeddings(final_df, global_maps)
    final_df, func_feat_info = _generate_function_feature_embeddings(final_df, global_maps)
    final_df['mlp_embedding'] = final_df.apply(lambda r: r['ceramic_attribute_embedding'] + r['function_feature_embedding'], axis=1)
    embedding_info = {
        'ceramic_attributes': ceramic_info, 
        'function_features': func_feat_info, 
        'total_dim': ceramic_info['total_dim'] + func_feat_info['total_dim']
    }

    # --- 4. Prepare and Save Final Data ---
    if 'mlp_embedding' not in final_df.columns or final_df['mlp_embedding'].isnull().all():
        print("    ❌ MLP embedding generation failed. Aborting save.")
        return False
        
    X = np.array(final_df['mlp_embedding'].tolist())
    y = final_df['target_y_label'].values.astype(int)
    
    

    # Create target category names map
    target_category_names = {cat_id: hierarchy.cat_names.get(cat_id, f"Unknown_{cat_id}") for cat_id in np.unique(y)}
    
    print(f"    Final MLP data shapes: X={X.shape}, y={y.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_mlp.npy"), X)
    np.save(os.path.join(output_dir, "y_mlp.npy"), y)
    with open(os.path.join(output_dir, "target_category_names.json"), 'w') as f:
        json.dump({int(k): v for k, v in target_category_names.items()}, f, indent=4)
    study_config = {
            "embedding_method": "One_hot_encoding",
            "target_y_level": target_y_level,
            "total_samples": len(X), 
            "n_classes": len(np.unique(y)),
            "embedding_dimensions": X.shape[1],
            "embedding_info": embedding_info
        }
    with open(os.path.join(output_dir, "study_config.json"), 'w') as f:
        json.dump(study_config, f, indent=4)
    # Build full index map for combined embeddings
    full_index_map = {}
    
    # Add ceramic attributes index map
    full_index_map.update(global_maps.get("ceramic_index_map", {}))
    
    # Add function/feature index map with offset
    offset_shift = len(global_maps.get("ceramic_index_map", {}))
    ff_map = global_maps.get("function_feature_index_map", {})
    for k, v in ff_map.items():
        full_index_map[k + offset_shift] = v

    # Save index map to JSON
    with open(os.path.join(output_dir, "embedding_index_map.json"), 'w') as f:
        json.dump({str(k): v for k, v in full_index_map.items()}, f, indent=4)
    with open(os.path.join(output_dir, "embedding_maps.json"), 'w') as f:
        json.dump({k: {str(k2): v2 for k2, v2 in v.items()} for k, v in global_maps.items()}, f, indent=4)
        
    print(f"    ✅ Combined embedding (type 2) generated with {embedding_info['total_dim']} dimensions")
    return True


def prepare_all_mlp_studies(dfs, base_output_dir="output/mlp_data/ohe", debug_examples=True):
    """
    Main orchestrator to prepare all MLP datasets.
    1. Builds GLOBAL vocabularies for all features.
    2. Loops through études to generate consistently-structured datasets.
    Only generates type 2 embeddings (combined attributes + functions + features).
    """
    print("======================================================")
    print("=== STARTING PREPARATION FOR ALL MLP STUDIES       ===")
    print("======================================================")
    
    # Initialize hierarchy
    hierarchy = CategoryHierarchy(dfs['tech_cat'])
    
    # Build ceramic_to_info mapping
    ceramic_to_info = {}
    ceramic_df = dfs['ceramic_summary']
    
    for _, row in ceramic_df.iterrows():
        cid = row['ceramic_id']
        cat_id = row['tech_cat_id']
        level = hierarchy.get_level(cat_id)
        root = hierarchy.get_root(cat_id)
        
        ceramic_to_info[cid] = {
            'cat_id': cat_id,
            'level': level,
            'root': root
        }
    
    # Build global vocabularies for consistent embeddings
    print("🌍 Building global vocabularies...")
    global_maps = _build_global_vocab_maps(dfs)
    
    # Prepare datasets for each level
    studies = {
        'level0': {
            'ceramics': prepare_level0_dataset(ceramic_to_info, hierarchy, dfs, target_per_root=400),
            'duplication_map': {},
            'target_y_level': 0,
            'description': "Level 0: 400 ceramics per root category with diverse subcategory sampling"
        },
        'level1': {
            'ceramics': None,
            'duplication_map': {},
            'target_y_level': 1,
            'description': "Level 1: Balanced subcategories with minimum 10 ceramics, duplicate to reach targets"
        },
        'level2': {
            'ceramics': None,
            'duplication_map': {},
            'target_y_level': 2,
            'description': "Level 2: Balanced sub-subcategories with minimum 10 ceramics, ~40 per sub-subcat"
        }
    }
    
    # Prepare level 1 dataset
    level1_ceramics, level1_duplication_map = prepare_level1_dataset(
        ceramic_to_info, hierarchy, dfs, min_ceramics_per_subcat=10
    )
    studies['level1']['ceramics'] = level1_ceramics
    studies['level1']['duplication_map'] = level1_duplication_map
    
    # Prepare level 2 dataset
    level2_ceramics, level2_duplication_map = prepare_level2_dataset(
        ceramic_to_info, hierarchy, dfs, min_ceramics_per_subsubcat=10, target_per_subsubcat=40
    )
    studies['level2']['ceramics'] = level2_ceramics
    studies['level2']['duplication_map'] = level2_duplication_map
    
    # Generate MLP datasets for each study
    print("\n🔬 Generating MLP datasets for all studies...")
    
    for study_name, study_config in studies.items():
        print(f"\n📊 Processing {study_name}...")
        print(f"    {study_config['description']}")
        
        if study_config['ceramics'] is None:
            print(f"    ❌ No ceramics prepared for {study_name}. Skipping.")
            continue
            
        output_dir = os.path.join(base_output_dir, study_name)
        
        success = _generate_single_mlp_dataset(
            dfs=dfs,
            hierarchy=hierarchy,
            ceramic_ids_to_process=study_config['ceramics'],
            ceramic_duplication_map=study_config['duplication_map'],
            target_y_level=study_config['target_y_level'],
            output_dir=output_dir,
            global_maps=global_maps,
            debug_examples=debug_examples
        )
        
        if success:
            print(f"    ✅ {study_name} dataset generated successfully at {output_dir}")
        else:
            print(f"    ❌ Failed to generate {study_name} dataset")
    
    print("\n======================================================")
    print("=== ALL MLP STUDIES PREPARATION COMPLETED          ===")
    print("======================================================")
    
    return studies