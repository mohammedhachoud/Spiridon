# src/data_preparation/format_mlp_classification_data.py
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


def parse_numpy_array_string(value):
    """Parse string representations of numpy arrays like '[np.int64(3), np.int64(16)]'"""
    if pd.isna(value) or not isinstance(value, str):
        return []
    
    value = value.strip()
    if not (value.startswith('[') and value.endswith(']')):
        return []
    
    try:
        # Method 1: Use regex to extract numbers from np.int64() format
        numbers = re.findall(r'np\.int64\((\d+)\)', value)
        if numbers:
            return [int(num) for num in numbers]
        
        # Method 2: Try to extract just numbers if it's a simple list
        numbers = re.findall(r'\b(\d+)\b', value)
        if numbers:
            return [int(num) for num in numbers]
        
        # Method 3: Try ast.literal_eval after cleaning
        cleaned = value.replace('np.int64(', '').replace(')', '')
        return ast.literal_eval(cleaned)
        
    except Exception as e:
        print(f"    Warning: Could not parse '{value}': {e}")
        return []


# --- 3. Find root categories ---
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





def _generate_single_bert_mlp_dataset(
    dfs: dict,
    hierarchy: 'CategoryHierarchy',
    ceramic_ids_to_process: list,
    target_y_level: int,
    bert_model: 'SentenceTransformer',
    feature_embedding_map: dict,
    function_embedding_map: dict,
    output_dir: str
):
    """
    Generates and saves a single MLP dataset using aggregated BERT embeddings.
    - X is a concatenation of (ceramic_desc_emb, mean_feature_emb, mean_function_emb).
    - y is the ancestor category ID at the target level.
    """
    bert_dim = bert_model.get_sentence_embedding_dimension()
    
    # --- 1. Filter main DataFrame for the pre-selected ceramics ---
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

    # --- 3. Generate the input features 'X' ---
    print("    Generating aggregated BERT embeddings for 'X'...")

    # Batch encode ceramic descriptions for efficiency
    descriptions = final_df['description'].fillna("").tolist()
    ceramic_desc_embeddings = bert_model.encode(descriptions, show_progress_bar=False, convert_to_numpy=True)
    
    # Robust parser for feature/function ID strings
    def parse_ids(value):
            # Convert to scalar if it's a single-element array/Series
            if hasattr(value, '__len__') and hasattr(value, 'iloc') and len(value) == 1:
                value = value.iloc[0]
            elif hasattr(value, '__len__') and not isinstance(value, str):
                # If it's an array-like object with multiple elements, return empty list
                return []
            
            # Now check if the scalar value is null or not a string
            try:
                if pd.isna(value) or not isinstance(value, str): 
                    return []
            except (ValueError, TypeError):
                # Fallback for edge cases where pd.isna fails
                if value is None or not isinstance(value, str):
                    return []
            
            try: 
                return [int(num) for num in re.findall(r'\d+', value)]
            except: 
                return []

    # Assemble the final X matrix row by row
    X_rows = []
    for i, (_, row) in enumerate(final_df.iterrows()):
        # Part 1: Ceramic Description Embedding
        vec_c = ceramic_desc_embeddings[i]
        
        # Part 2: Aggregated Feature Embedding
        feature_ids = parse_ids(row.get('feature_id'))
        feature_vectors = [feature_embedding_map[fid] for fid in feature_ids if fid in feature_embedding_map]
        
        if feature_vectors:
            vec_f = np.mean(feature_vectors, axis=0)
        else:
            vec_f = np.zeros(bert_dim, dtype=np.float32)
            
        # Part 3: Aggregated Function Embedding
        function_ids = parse_ids(row.get('function_id'))
        function_vectors = [function_embedding_map[fid] for fid in function_ids if fid in function_embedding_map]
        
        if function_vectors:
            vec_fn = np.mean(function_vectors, axis=0)
        else:
            vec_fn = np.zeros(bert_dim, dtype=np.float32)
            
        # Concatenate the three parts
        final_x_row = np.concatenate([vec_c, vec_f, vec_fn])
        X_rows.append(final_x_row)
        
    X = np.array(X_rows, dtype=np.float32)
    y = final_df['target_y_label'].values

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
        "bert_model": bert_model.get_max_seq_length(), # A way to reference the model used
        "target_y_level": target_y_level,
        "total_samples": len(X), "n_classes": len(np.unique(y)),
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



def prepare_all_bert_mlp_studies(
    dfs, 
    base_output_dir="output/mlp_data/mlp_bert_level_studies",
    bert_model_name="all-MiniLM-L6-v2"
):
    """
    Main orchestrator for preparing BERT-based MLP datasets for the comparative study.

    1.  Loads the Sentence-BERT model.
    2.  Pre-computes embeddings for ALL features and functions.
    3.  Selects ONE master set of ceramics (those at Level 2 or deeper).
    4.  Samples this master set for each of the 3 "études".
    5.  Loops through all 9 combinations of (target_level, etude) to generate
        and save every required BERT-MLP dataset.
    """
    print("======================================================")
    print("=== STARTING PREPARATION FOR ALL BERT-MLP STUDIES  ===")
    print("======================================================")

    # --- 0. Load Model and Initialize Hierarchy ---
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

    # Features (using Features_Ontology as in RGCN)
    # Ensure 'Features_Ontology' is the correct key in your dfs dictionary
    feat_df = dfs['Features_Ontology'].dropna(subset=['id', 'Nom complet']).copy()
    feat_df['id'] = feat_df['id'].astype(float).astype(int)
    feat_names = feat_df['Nom complet'].tolist()
    feat_embeddings = bert_model.encode(feat_names, show_progress_bar=True, convert_to_numpy=True)
    feature_embedding_map = {fid: emb for fid, emb in zip(feat_df['id'], feat_embeddings)}
    print(f"  ✅ Pre-computed {len(feature_embedding_map)} feature embeddings.")

    # --- 2. Select and Sample Master Ceramic Set ---
    print("\n--- STEP 2: Selecting and Sampling Master Ceramic Set (Level >= 2) ---")
    candidate_ceramics_by_root = defaultdict(list)
    for _, row in dfs['ceramic_summary'].iterrows():
        cat_id = row.get('tech_cat_id')
        if pd.notna(cat_id):
            level = hierarchy.get_level(int(cat_id))
            if level is not None and level >= 2:
                root = hierarchy.get_root(int(cat_id))
                if root is not None:
                    candidate_ceramics_by_root[root].append(int(row['ceramic_id']))
    
    candidate_counts = {r: len(c) for r, c in candidate_ceramics_by_root.items()}
    
    min_class_id_for_etude2 = min(candidate_counts, key=candidate_counts.get) if candidate_counts else None

    etude_definitions = {
        'etude1': {'target_size': 138, 'exclude_root': None},
        'etude1_prime': {'target_size': 276, 'exclude_root': None},
        'etude2': {'target_size': 950, 'exclude_root': min_class_id_for_etude2}
    }
    
    sampled_ceramics_per_etude = {}
    for etude_name, params in etude_definitions.items():
        selected_ids = []
        roots_to_sample = set(candidate_ceramics_by_root.keys())
        if params['exclude_root'] is not None: 
            roots_to_sample.discard(params['exclude_root'])
        
        for root_id in sorted(list(roots_to_sample)):
            candidates = candidate_ceramics_by_root.get(root_id, [])
            num_to_sample = min(len(candidates), params['target_size'])
            if num_to_sample > 0:
                # Using your completeness score logic would be better here if available
                # For now, using random sampling as a robust placeholder
                selected_ids.extend(np.random.choice(candidates, num_to_sample, replace=False))
        
        sampled_ceramics_per_etude[etude_name] = sorted(list(set(selected_ids)))
        print(f"  -> Selected {len(sampled_ceramics_per_etude[etude_name])} unique ceramics for {etude_name}.")

    # --- 3. Generate All 9 Datasets via Nested Loops ---
    print("\n--- STEP 3: Generating All BERT-MLP Datasets ---")
    
    for etude_name, ceramic_ids in sampled_ceramics_per_etude.items():
        if not ceramic_ids:
            print(f"\nSkipping {etude_name} as no ceramics were sampled.")
            continue
            
        for target_level in [2, 1, 0]:
            study_identifier = f"level_{target_level}_target/{etude_name}"
            print(f"\n--- Generating: {study_identifier} ---")
            
            output_path = os.path.join(base_output_dir, f"level_{target_level}", etude_name)

            _generate_single_bert_mlp_dataset(
                dfs=dfs,
                hierarchy=hierarchy,
                ceramic_ids_to_process=ceramic_ids,
                # THIS IS THE CORRECTED LINE:
                target_y_level=target_level,
                bert_model=bert_model,
                feature_embedding_map=feature_embedding_map,
                function_embedding_map=function_embedding_map,
                output_dir=output_path
            )

    print("\n======================================================")
    print("=== FINISHED PREPARATION FOR ALL BERT-MLP STUDIES  ===")
    print("======================================================")