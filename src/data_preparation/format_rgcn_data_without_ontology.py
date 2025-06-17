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



def format_rgcn_data_for_study(dfs, 
                               triplets_for_study, 
                               study_name,
                               category_hierarchy: CategoryHierarchy,
                               target_connection_level: int,
                               bert_model_name="all-MiniLM-L6-v2"):
    """
    Formats data for RGCN with ALL nodes using BERT embeddings, with flexible
    category linking. MODIFIED: Removes ontology reasoning - no IS_A relations
    and only HAS_FEATURE relations for features.

    - All nodes (Ceramic, Function, Feature, Category) use BERT embeddings.
    - BELONGS_TO_CATEGORY links are established from Ceramic_Node -> Category_Node
      at the specified `target_connection_level`.
    - Category-to-feature/function links originate ONLY from Root Category Nodes.
    - NO IS_A relations between parent/child nodes
    - Only HAS_FEATURE relation used for all features
    
    Args:
        ...
        category_hierarchy (CategoryHierarchy): An initialized hierarchy manager object.
        target_connection_level (int): The hierarchy level (0=root, 1, 2, etc.) to
                                       link ceramics to.
    Returns:
        dict: RGCN formatted data, or None on failure.
    """
    print(f"\n  🔄 Formatting {study_name} data for RGCN (WITHOUT ONTOLOGY)...")
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
    
    # Add all functions and features from the sample (NO PARENTS - removing ontology)
    for entry in triplets_for_study:
        # Add Ceramic node
        try:
            nodes_in_sample.add(f"Ceramic_{int(float(entry.get('ceramic_id')))}")
        except (TypeError, ValueError):
            continue

        # Add Function nodes ONLY (no parents - removing IS_A relations)
        for fid, parents_ids in entry.get("functions", []):
            nodes_in_sample.add(f"Func_{fid}")
            # REMOVED: No longer adding parent function nodes

        # Add Feature nodes ONLY (no parents - removing IS_A relations)
        for fid, parents_ids in entry.get("features", []):
            nodes_in_sample.add(f"Feat_{str(fid)}")
            # REMOVED: No longer adding parent feature nodes
        
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
    
    # --- Triplet Generation (Modified - NO ONTOLOGY) ---
    print("    🔗 Processing triplets using pre-assigned graph indices (WITHOUT ONTOLOGY)...")
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
    # REMOVED: IS_A relation - no longer used
    HAS_FEATURE_REL = get_or_assign_relation_idx("HAS_FEATURE")  # Only relation for features
    
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

        # Ceramic -> Function ONLY (NO IS_A hierarchy)
        for func_id, parents_ids in entry.get("functions", []):
            func_idx = node_to_idx.get(f"Func_{func_id}")
            if func_idx is not None:
                training_triplets.append((ceramic_idx, HAS_FUNCTION_REL, func_idx))
                # REMOVED: No IS_A links between functions and their parents
        
        # Ceramic -> Feature ONLY (NO IS_A hierarchy, ONLY HAS_FEATURE relation)
        for feat_id, parents_ids in entry.get("features", []):
            feat_id_str = str(feat_id)
            feat_idx = node_to_idx.get(f"Feat_{feat_id_str}")
            if feat_idx is not None:
                # MODIFIED: Always use HAS_FEATURE relation, ignore ceramic_relation column
                training_triplets.append((ceramic_idx, HAS_FEATURE_REL, feat_idx))
                # REMOVED: No IS_A links between features and their parents

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

        # Root -> Features (MODIFIED: Always use HAS_FEATURE)
        linked_feats = tech_cat_feat_attrib[tech_cat_feat_attrib['tech_cat_id'] == root_id]
        for _, row in linked_feats.iterrows():
            feat_id_str = str(row['feature_id'])
            feat_idx = node_to_idx.get(f"Feat_{feat_id_str}")
            if feat_idx is not None:
                # MODIFIED: Always use HAS_FEATURE relation
                training_triplets.append((root_idx, HAS_FEATURE_REL, feat_idx))

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
            "total_relations": len(relation_to_idx),
            "ontology_removed": True  # Flag to indicate ontology was removed
        }
    }
    
    print(f"    ✅ Formatted data for {study_name} (WITHOUT ONTOLOGY). "
          f"Train triplets: {len(rgcn_data['training_triplets'])}, "
          f"Eval triplets: {len(rgcn_data['evaluation_triplets'])}. "
          f"Target categories: {len(rgcn_data['target_category_node_indices'])}")

    return rgcn_data


def prepare_all_level_based_studies(dfs, bert_model_name="all-MiniLM-L6-v2", auto_save=True, base_output_dir="output/rgcn_data/without_ontology"):
    """
    Main orchestrator for the comparative study with DYNAMIC ROOT DISCOVERY.
    MODIFIED: Removes ontology reasoning and changes output directory.

    This version implements the corrected logic:
    1.  **Dynamically discovers root categories** (those with no parent).
    2.  **Selects ONE master set of ceramics**: All those belonging to categories
        at Level 2 or deeper.
    3.  **Samples this master set THREE times**: Once for each "étude".
    4.  **Generates NINE datasets**: By taking each of the three sampled sets and
        linking them to their ancestors at Level 2, Level 1, and Level 0.
    5.  **MODIFIED**: Removes ontology reasoning (no IS_A relations, only HAS_FEATURE)
    
    Args:
        dfs: Dictionary of DataFrames
        bert_model_name: Name of the BERT model to use
        auto_save: If True, automatically saves the datasets
        base_output_dir: Directory to save the datasets (MODIFIED: without_ontology)
    """
    print("======================================================")
    print("=== STARTING DATA PREPARATION (WITHOUT ONTOLOGY) ===")
    print("======================================================")
    print(f"🤖 BERT Model: {bert_model_name}")
    print(f"📁 Output Directory: {base_output_dir}")

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
    print("\n--- STEP 3: Generating All 9 Datasets with Variable Linking (WITHOUT ONTOLOGY) ---")
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
                print(f"    ✅ Successfully prepared dataset for {repo_name} / {etude_name} (WITHOUT ONTOLOGY)")
                print(f"    📊 Embedding Info: {bert_model_name} -> {bert_embedding_dim}D")
            else:
                print(f"    ❌ FAILED to prepare dataset for {repo_name} / {etude_name}")

    print("\n======================================================")
    print("=== FINISHED ALL COMPARATIVE STUDY PREPARATION     ===")
    print("=== (WITHOUT ONTOLOGY REASONING)                   ===")
    print("======================================================")
    
    # --- AUTO-SAVE if requested ---
    if auto_save:
        print(f"\n🔄 AUTO-SAVING datasets to '{base_output_dir}'...")
        save_all_study_datasets(all_study_datasets, base_output_dir)
        print(f"✅ All datasets saved successfully!")
    else:
        print(f"\n⚠️  Auto-save disabled. Call save_all_study_datasets() manually to save.")
    
    return all_study_datasets


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