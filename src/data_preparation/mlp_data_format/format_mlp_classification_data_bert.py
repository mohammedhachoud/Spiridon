import pandas as pd
import numpy as np
import os
import re
import json
import ast
from collections import Counter, defaultdict
import config 
from utils import CategoryHierarchy
from sentence_transformers import SentenceTransformer
from graph_utils import (
    calculate_completeness_score,
    extract_triplets_for_selection
)



def parse_id_list(value):
    """
    Improved version that handles more data formats and provides better debugging.
    """
    # Debug logging
    debug = False  # Set to True for debugging
    
    if debug:
        print(f"Parsing: {repr(value)} (type: {type(value)})")
    
    # Case 1: Already a list
    if isinstance(value, list):
        result = [int(v) for v in value if isinstance(v, (int, float, np.integer))]
        if debug: print(f"  -> List case: {result}")
        return result

    # Case 2: NaN, None, or not a string
    if pd.isna(value) or not isinstance(value, str):
        if debug: print(f"  -> NaN/None case: []")
        return []

    # Case 3: String processing
    value = str(value).strip()
    if debug: print(f"  -> Cleaned string: {repr(value)}")
    
    # Handle empty or malformed strings
    if not value or value in ['[]', 'nan', 'None']:
        if debug: print(f"  -> Empty/malformed: []")
        return []
    
    try:
        # Method 1: Try ast.literal_eval for standard Python formats
        if value.startswith('[') and value.endswith(']'):
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    result = [int(v) for v in parsed if isinstance(v, (int, float, np.integer))]
                    if debug: print(f"  -> ast.literal_eval: {result}")
                    return result
            except (ValueError, SyntaxError) as e:
                if debug: print(f"  -> ast.literal_eval failed: {e}")
        
        # Method 2: Handle numpy formats like [np.int64(123), np.int64(456)]
        if 'np.' in value:
            # Pattern for np.TYPE(NUMBER)
            import re
            np_pattern = r'np\.\w+\((\d+)\)'
            matches = re.findall(np_pattern, value)
            if matches:
                result = [int(match) for match in matches]
                if debug: print(f"  -> Numpy pattern: {result}")
                return result
        
        # Method 3: Simple comma/space separated numbers
        # Remove brackets and split by common separators
        cleaned = value.strip('[](){}').replace(',', ' ').replace(';', ' ')
        numbers = re.findall(r'\d+', cleaned)
        if numbers:
            result = [int(num) for num in numbers]
            if debug: print(f"  -> Number extraction: {result}")
            return result
            
        # Method 4: Handle single number
        if value.isdigit():
            result = [int(value)]
            if debug: print(f"  -> Single number: {result}")
            return result
            
    except Exception as e:
        if debug: print(f"  -> Exception: {e}")
        pass
    
    if debug: print(f"  -> Failed to parse: []")
    return []


# Alternative: Simple regex-based parser
def parse_id_list_regex(value):
    """Simple regex-based parser that extracts all numbers from any string format."""
    if pd.isna(value) or not isinstance(value, str):
        return []
    
    import re
    # Extract all numbers from the string
    numbers = re.findall(r'\d+', str(value))
    return [int(num) for num in numbers] if numbers else []


def validate_embedding_maps(feature_embedding_map, function_embedding_map, sample_df):
    """Validate that embedding maps contain the IDs we need"""
    all_feature_ids = set()
    all_function_ids = set()
    
    for _, row in sample_df.iterrows():
        feature_ids = parse_id_list(row.get('feature_id'))
        function_ids = parse_id_list(row.get('function_id'))
        all_feature_ids.update(feature_ids)
        all_function_ids.update(function_ids)
    
    missing_features = all_feature_ids - set(feature_embedding_map.keys())
    missing_functions = all_function_ids - set(function_embedding_map.keys())
    
    print(f"Feature IDs needed: {len(all_feature_ids)}, available: {len(feature_embedding_map)}")
    print(f"Missing feature IDs: {len(missing_features)} - {list(missing_features)[:10]}")
    print(f"Function IDs needed: {len(all_function_ids)}, available: {len(function_embedding_map)}")
    print(f"Missing function IDs: {len(missing_functions)} - {list(missing_functions)[:10]}")
def _generate_single_bert_mlp_dataset(
    dfs: dict,
    hierarchy: 'CategoryHierarchy',
    ceramic_ids_to_process: list,
    target_y_level: int,
    bert_model: 'SentenceTransformer',
    feature_embedding_map: dict,
    function_embedding_map: dict,
    output_dir: str,
    ceramic_duplication_map: dict = None
):
    """
    Generates and saves a single MLP dataset using aggregated BERT embeddings.
    - X is a concatenation of (ceramic_desc_emb, mean_feature_emb, mean_function_emb).
    - y is the ancestor category ID at the target level.
    - Handles ceramic duplication mapping for proper data expansion.
    """
    bert_dim = bert_model.get_sentence_embedding_dimension()
    
    # --- 1. Handle ceramic duplication if provided ---
    if ceramic_duplication_map:
        # Create expanded ceramic summary with duplicates
        expanded_ceramic_summary = []
        ceramic_summary_df = dfs['ceramic_summary']
        
        for ceramic_id in ceramic_ids_to_process:
            # Find the original ceramic ID if this is a duplicate
            original_id = None
            for orig_id, duplicate_list in ceramic_duplication_map.items():
                if ceramic_id in duplicate_list:
                    original_id = orig_id
                    break
            
            if original_id is None:
                # This ceramic_id is not in duplication map, skip
                continue
                
            # Get the original ceramic row
            original_row = ceramic_summary_df[ceramic_summary_df['ceramic_id'] == original_id]
            if original_row.empty:
                continue
                
            # Create a new row with the new ceramic_id
            new_row = original_row.copy()
            new_row['ceramic_id'] = ceramic_id
            expanded_ceramic_summary.append(new_row)
        
        if expanded_ceramic_summary:
            ceramic_summary_df = pd.concat(expanded_ceramic_summary, ignore_index=True)
        else:
            ceramic_summary_df = ceramic_summary_df[ceramic_summary_df['ceramic_id'].isin(ceramic_ids_to_process)].copy()
    else:
        ceramic_summary_df = dfs['ceramic_summary'][dfs['ceramic_summary']['ceramic_id'].isin(ceramic_ids_to_process)].copy()
    
    if ceramic_summary_df.empty:
        print("    ❌ No matching ceramics found in summary. Skipping generation.")
        return False
        
    print(f"    Processing {len(ceramic_summary_df)} ceramics for this dataset.")
    
    # --- 2. Generate the target labels 'y' ---
    print(f"    Generating 'y' labels for Level {target_y_level}...")
    ceramic_summary_df['target_y_label'] = ceramic_summary_df['tech_cat_id'].apply(
        lambda cid: hierarchy.get_ancestor_at_level(cid, target_y_level)
    )
    final_df = ceramic_summary_df.dropna(subset=['target_y_label']).copy()
    final_df['target_y_label'] = final_df['target_y_label'].astype(int)

    if final_df.empty:
        print("    ❌ No ceramics remaining after creating labels. Skipping.")
        return False

    # --- 3. Generate the input features 'X' ---
    print("    Generating aggregated BERT embeddings for 'X'...")

    # Batch encode ceramic descriptions for efficiency to prevent memory crashes
    descriptions = final_df['description'].fillna("").tolist()
    print(f"    Encoding {len(descriptions)} descriptions in batches...")
    
    # Ensure descriptions are not empty
    if not descriptions or all(not desc.strip() for desc in descriptions):
        print("    ❌ All descriptions are empty!")
        return False
    
    ceramic_desc_embeddings = bert_model.encode(
        descriptions,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # CRITICAL: Validate embeddings immediately
    print(f"    ✅ Generated embeddings shape: {ceramic_desc_embeddings.shape}")
    print(f"    ✅ Embeddings dtype: {ceramic_desc_embeddings.dtype}")
    print(f"    ✅ Embeddings range: [{ceramic_desc_embeddings.min():.4f}, {ceramic_desc_embeddings.max():.4f}]")
    
    if np.allclose(ceramic_desc_embeddings, 0):
        print("    ❌ WARNING: All ceramic embeddings are zero!")
        print("    First few descriptions:")
        for i, desc in enumerate(descriptions[:3]):
            print(f"      {i}: {desc[:100]}...")
    
    # Assemble X matrix with better validation
    X_rows = []
    zero_feature_count = 0
    zero_function_count = 0
    
    for i, (_, row) in enumerate(final_df.iterrows()):
        # Part 1: Ceramic Description Embedding (ensure it's not zero)
        vec_c = ceramic_desc_embeddings[i].astype(np.float64)  # Use float64 for precision
        
        # Part 2: Feature Embedding
        feature_ids = parse_id_list(row.get('feature_id'))
        feature_vectors = []
        for fid in feature_ids:
            if fid in feature_embedding_map:
                feature_vectors.append(feature_embedding_map[fid])
        
        if feature_vectors:
            vec_f = np.mean(feature_vectors, axis=0).astype(np.float64)
        else:
            vec_f = np.zeros(bert_dim, dtype=np.float64)
            zero_feature_count += 1
            
        # Part 3: Function Embedding
        function_ids = parse_id_list(row.get('function_id'))
        function_vectors = []
        for fid in function_ids:
            if fid in function_embedding_map:
                function_vectors.append(function_embedding_map[fid])
        
        if function_vectors:
            vec_fn = np.mean(function_vectors, axis=0).astype(np.float64)
        else:
            vec_fn = np.zeros(bert_dim, dtype=np.float64)
            zero_function_count += 1
            
        # Concatenate and validate
        final_x_row = np.concatenate([vec_c, vec_f, vec_fn])
        
        # Debug first few rows
        if i < 3:
            print(f"    Row {i} - vec_c range: [{vec_c.min():.4f}, {vec_c.max():.4f}]")
            print(f"    Row {i} - vec_f range: [{vec_f.min():.4f}, {vec_f.max():.4f}]")
            print(f"    Row {i} - vec_fn range: [{vec_fn.min():.4f}, {vec_fn.max():.4f}]")
            print(f"    Row {i} - final_x_row range: [{final_x_row.min():.4f}, {final_x_row.max():.4f}]")
        
        X_rows.append(final_x_row)
    
    print(f"    ⚠️ Rows with zero feature embeddings: {zero_feature_count}/{len(X_rows)}")
    print(f"    ⚠️ Rows with zero function embeddings: {zero_function_count}/{len(X_rows)}")
    
    X = np.array(X_rows, dtype=np.float32)  # Convert to float32 at the end
    y = final_df['target_y_label'].values
    
    # Final validation
    print(f"    Final X shape: {X.shape}, dtype: {X.dtype}")
    print(f"    Final X range: [{X.min():.4f}, {X.max():.4f}]")
    
    if np.allclose(X, 0):
        print("    ❌ CRITICAL: Final X matrix is all zeros!")
        return False

    # --- 4. Prepare and Save Final Data ---
    target_category_names = {cat_id: hierarchy.cat_names.get(cat_id, f"Unknown_{cat_id}") for cat_id in np.unique(y)}
    print(f"    Final MLP data shapes: X={X.shape}, y={y.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "X_mlp_bert.npy"), X)
    np.save(os.path.join(output_dir, "y_mlp_bert.npy"), y)
    with open(os.path.join(output_dir, "target_category_names.json"), 'w') as f:
        json.dump({int(k): v for k, v in target_category_names.items()}, f, indent=4)
        
    study_config = {
        "embedding_method": "bert_aggregated",
        "bert_model": bert_model.get_max_seq_length(),
        "target_y_level": target_y_level,
        "total_samples": len(X), 
        "n_classes": len(np.unique(y)),
        "embedding_dimensions": X.shape[1],
        "embedding_info": {
            "ceramic_desc_dim": bert_dim,
            "aggregated_feature_dim": bert_dim,
            "aggregated_function_dim": bert_dim
        }
    }
    with open(os.path.join(output_dir, "study_config.json"), 'w') as f:
        json.dump(study_config, f, indent=4)

    print(f"    ✅ BERT-MLP data successfully saved to: {output_dir}")
    return True


def prepare_three_bert_mlp_datasets(
    dfs, 
    base_output_dir="output/mlp_data/Bert",
    bert_model_name="all-MiniLM-L6-v2"
):
    """
    Main orchestrator for preparing 3 BERT-based MLP datasets using the same logic as RGCN.
    Creates: etude_lvl0, etude_lvl1, etude_lvl2
    """
    print("=" * 80)
    print("=== STARTING PREPARATION FOR THREE BERT-MLP DATASETS ===")
    print("=" * 80)

    try:
        print(f"Loading Sentence-BERT model: '{bert_model_name}'...")
        bert_model = SentenceTransformer(bert_model_name)
        
        print("Initializing CategoryHierarchy...")
        hierarchy = CategoryHierarchy(dfs['tech_cat'])
    except Exception as e:
        print(f"❌ CRITICAL ERROR during initialization: {e}")
        return

    # --- 1. Pre-compute BERT Embeddings for all Functions and Features ---
    print("\n--- STEP 1: Pre-computing BERT embeddings for all functions and features ---")
    
    # Functions
    func_df = dfs['object_function'].dropna(subset=['id', 'function_name_fr']).copy()
    func_df['id'] = func_df['id'].astype(int)
    func_names = func_df['function_name_fr'].tolist()
    func_embeddings = bert_model.encode(func_names, show_progress_bar=True, convert_to_numpy=True)
    function_embedding_map = {fid: emb for fid, emb in zip(func_df['id'], func_embeddings)}
    print(f"  ✅ Pre-computed {len(function_embedding_map)} function embeddings.")

    # Features - Using a robust method to avoid data loss
    feat_df = dfs['Features_Ontology'].copy()
    feat_df.dropna(subset=['id'], inplace=True) # Must have an ID
    feat_df['feature_name_fr'] = feat_df['feature_name_fr'].fillna("") # Fill missing names with empty string
    feat_df['id'] = feat_df['id'].astype(int)
    feat_names = feat_df['feature_name_fr'].tolist()
    feat_embeddings = bert_model.encode(feat_names, show_progress_bar=True, convert_to_numpy=True)
    feature_embedding_map = {fid: emb for fid, emb in zip(feat_df['id'], feat_embeddings)}
    print(f"  ✅ Pre-computed {len(feature_embedding_map)} feature embeddings.")

    # --- 2. Initialize hierarchy and ceramic mapping ---
    print("\n--- STEP 2: Initializing hierarchy and ceramic mapping ---")
    
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

    # --- 3. Prepare the three datasets ---
    print("\n--- STEP 3: Preparing three datasets ---")
    
    # Dataset 1: etude_lvl0 (Level 0 - 400 ceramics per root, diverse sampling)
    print("\n" + "=" * 50)
    print("PREPARING ETUDE_LVL0 (Level 0 dataset)")
    print("=" * 50)
    
    etude_lvl0_ceramics = prepare_level0_dataset(
        ceramic_to_info, hierarchy, dfs, target_per_root=400
    )
    
    if etude_lvl0_ceramics:
        output_path = os.path.join(base_output_dir, "etude_lvl0")
        success = _generate_single_bert_mlp_dataset(
            dfs=dfs,
            hierarchy=hierarchy,
            ceramic_ids_to_process=etude_lvl0_ceramics,
            target_y_level=0,  # Level 0 connections
            bert_model=bert_model,
            feature_embedding_map=feature_embedding_map,
            function_embedding_map=function_embedding_map,
            output_dir=output_path,
            ceramic_duplication_map=None  # No duplication for level 0
        )
        if success:
            print(f"✅ Level 0 dataset prepared with {len(etude_lvl0_ceramics)} ceramics")
        else:
            print("❌ Failed to prepare Level 0 dataset")
    
    # Dataset 2: etude_lvl1 (Level 1 - balanced subcategories, min 10 ceramics)
    print("\n" + "=" * 50)
    print("PREPARING ETUDE_LVL1 (Level 1 dataset)")
    print("=" * 50)
    
    level1_ceramics, level1_duplication_map = prepare_level1_dataset(
        ceramic_to_info, hierarchy, dfs
    )
    
    # Update ceramic_to_info with Level 1 duplicates
    print(f"📝 Updating ceramic_to_info with Level 1 duplicates...")
    original_count = len(ceramic_to_info)
    
    for original_cid, duplicate_list in level1_duplication_map.items():
        if original_cid in ceramic_to_info:
            for duplicate_cid in duplicate_list:
                if duplicate_cid != original_cid:
                    ceramic_to_info[duplicate_cid] = ceramic_to_info[original_cid].copy()
    
    print(f"   Ceramic info updated: {original_count} → {len(ceramic_to_info)} entries")
    
    if level1_ceramics:
        output_path = os.path.join(base_output_dir, "etude_lvl1")
        success = _generate_single_bert_mlp_dataset(
            dfs=dfs,
            hierarchy=hierarchy,
            ceramic_ids_to_process=level1_ceramics,
            target_y_level=1,  # Level 1 connections
            bert_model=bert_model,
            feature_embedding_map=feature_embedding_map,
            function_embedding_map=function_embedding_map,
            output_dir=output_path,
            ceramic_duplication_map=level1_duplication_map
        )
        if success:
            print(f"✅ Level 1 dataset prepared with {len(level1_ceramics)} ceramics")
        else:
            print("❌ Failed to prepare Level 1 dataset")
    
    # Dataset 3: etude_lvl2 (Level 2 - balanced sub-subcategories, min 10 ceramics)
    print("\n" + "=" * 50)
    print("PREPARING ETUDE_LVL2 (Level 2 dataset)")
    print("=" * 50)
    
    level2_ceramics, level2_duplication_map = prepare_level2_dataset(
        ceramic_to_info, hierarchy, dfs
    )
    
    # Update ceramic_to_info with Level 2 duplicates
    print(f"📝 Updating ceramic_to_info with Level 2 duplicates...")
    original_count = len(ceramic_to_info)
    
    for original_cid, duplicate_list in level2_duplication_map.items():
        if original_cid in ceramic_to_info:
            for duplicate_cid in duplicate_list:
                if duplicate_cid != original_cid:
                    ceramic_to_info[duplicate_cid] = ceramic_to_info[original_cid].copy()
    
    print(f"   Ceramic info updated: {original_count} → {len(ceramic_to_info)} entries")
    
    if level2_ceramics:
        output_path = os.path.join(base_output_dir, "etude_lvl2")
        success = _generate_single_bert_mlp_dataset(
            dfs=dfs,
            hierarchy=hierarchy,
            ceramic_ids_to_process=level2_ceramics,
            target_y_level=2,  # Level 2 connections
            bert_model=bert_model,
            feature_embedding_map=feature_embedding_map,
            function_embedding_map=function_embedding_map,
            output_dir=output_path,
            ceramic_duplication_map=level2_duplication_map
        )
        if success:
            print(f"✅ Level 2 dataset prepared with {len(level2_ceramics)} ceramics")
        else:
            print("❌ Failed to prepare Level 2 dataset")

    print("\n" + "=" * 80)
    print("=== FINISHED PREPARATION FOR THREE BERT-MLP DATASETS ===")
    print("=" * 80)


# Helper functions from the RGCN script (need to be included or imported)
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
    ceramic_duplication_map = {}
    
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
    ceramic_duplication_map = {}
    
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
                    # Subsequent occurrences - create new ID
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