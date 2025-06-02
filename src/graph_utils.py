import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
# import torch # Not used in the provided snippet relevant to the fix
# from sklearn.decomposition import PCA # Not used
# from . import config # Assuming this is external and not needing change for the bug
# from .utils import get_feature_parent_relation_label # Not used
from IPython.display import display

import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
import ast
from collections import Counter, defaultdict
from collections.abc import Iterable
# import torch # Not used in the provided snippet relevant to the fix
# from sklearn.decomposition import PCA # Not used
# from . import config # Assuming this is external and not needing change for the bug
# from .utils import get_feature_parent_relation_label # Not used
from IPython.display import display

memo_parents = {}

# --- Helper Functions (assuming get_parents is the one from the previous good state)
def get_parents(item_id, p_map, is_string_id=False):
    lookup_key = (item_id, id(p_map))
    if lookup_key in memo_parents: return memo_parents[lookup_key]

    parents = []
    curr = item_id
    path = {curr} 

    while True:
        curr_for_lookup_and_cmp = curr
        if not is_string_id:
            try:
                curr_for_lookup_and_cmp = int(float(curr)) # Allow curr to be float initially
            except (ValueError, TypeError):
                break
        
        parent = p_map.get(curr_for_lookup_and_cmp)

        if parent is None or pd.isna(parent):
            break

        stop_due_to_parent_value = False
        if is_string_id:
            parent_as_str = str(parent)
            # curr_for_lookup_and_cmp is original curr if is_string_id=True, else it's int(curr)
            curr_as_str = str(curr_for_lookup_and_cmp) 
            if parent_as_str in ['-1', 'nan', ''] or parent_as_str == curr_as_str:
                stop_due_to_parent_value = True
        else: 
            try:
                parent_as_int = int(float(parent)) # Allow parent to be float from map
                if parent_as_int == 0 or parent_as_int == -1 or parent_as_int == curr_for_lookup_and_cmp:
                    stop_due_to_parent_value = True
            except (ValueError, TypeError):
                pass

        if stop_due_to_parent_value:
            break
            
        if parent in path:
            break 
        
        parents.append(parent)
        path.add(parent) 
        curr = parent    

        if len(parents) > 50: 
            break
            
    memo_parents[lookup_key] = parents
    return parents


# --- Hierarchy and Sampling Logic ---
def build_category_hierarchy_and_map_ceramics(dfs):
    """ Builds the category hierarchy, identifies root categories, and maps each
    ceramic to its root category. """
    
    print("Building category hierarchy and mapping ceramics...")
    

    tech_cat = dfs['tech_cat'].copy()
    ceramic_summary = dfs['ceramic_summary'].copy()

    
    try:
        tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
        tech_cat['parent_id'] = pd.to_numeric(tech_cat['inherit_from'], errors='coerce').fillna(-1).astype(int)
        tech_cat.set_index('id', inplace=True) 

        ceramic_summary['ceramic_id'] = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce').dropna().astype(int)
        ceramic_summary['tech_cat_id'] = pd.to_numeric(ceramic_summary['tech_cat_id'], errors='coerce')
        ceramic_summary.dropna(subset=['tech_cat_id'], inplace=True)
        ceramic_summary['tech_cat_id'] = ceramic_summary['tech_cat_id'].astype(int)

    except Exception as e:
        print(f"Error standardizing IDs in tech_cat or ceramic_summary: {e}")
        traceback.print_exc()
        return None

    cat_to_root_map = {}
    memo = {} # Memoization for finding root

    def find_root(cat_id):
        if cat_id in memo:
            return memo[cat_id]
        if cat_id not in tech_cat.index:
            print(f"Warning: Category ID {cat_id} not found in tech_cat index.")
            return None 

        parent_id = tech_cat.loc[cat_id, 'parent_id']

        # Check if it's a root (parent is -1 or doesn't exist in index)
        if parent_id == -1 or parent_id not in tech_cat.index:
            memo[cat_id] = cat_id
            return cat_id
        else:
            root = find_root(parent_id)
            memo[cat_id] = root # Store result for current cat_id
            return root

    # Calculate root for every category in the tech_cat table
    print("Finding root for each category...")
    all_cat_ids = tech_cat.index.tolist()
    for cat_id in all_cat_ids:
        root = find_root(cat_id)
        if root is not None:
            cat_to_root_map[cat_id] = root
    print(f"Mapped {len(cat_to_root_map)} categories to their roots.")

    # Map each ceramic to its root category
    print("Mapping ceramics to root categories...")
    ceramic_to_root_map = {}
    valid_ceramics = 0
    for _, row in ceramic_summary.iterrows():
        ceramic_id = row['ceramic_id']
        direct_cat_id = row['tech_cat_id']
        root_cat_id = cat_to_root_map.get(direct_cat_id) # Use the precomputed map

        if root_cat_id is not None:
            ceramic_to_root_map[ceramic_id] = root_cat_id
            valid_ceramics += 1
        #else: print(f"Warning: Could not find root for direct category {direct_cat_id} of ceramic {ceramic_id}")

    print(f"Mapped {valid_ceramics} ceramics to a root category.")

    # Count ceramics per root category
    root_category_counts = Counter(ceramic_to_root_map.values())

    # Add root category names for clarity
    root_category_info = {}
    for root_id, count in root_category_counts.items():
         if root_id in tech_cat.index:
              name = tech_cat.loc[root_id, 'cat_name']
              root_category_info[root_id] = {'name': name, 'count': count}
         else:
              root_category_info[root_id] = {'name': f"[Unknown Root ID: {root_id}]", 'count': count}


    print("\n📊 Ceramic Counts per Root Category:")
    for root_id, info in sorted(root_category_info.items(), key=lambda item: item[1]['count']):
        print(f"  - Root ID {root_id} ('{info['name']}'): {info['count']} ceramics")

    return cat_to_root_map, ceramic_to_root_map, root_category_counts

def calculate_completeness_score(ceramic_ids, dfs):
    """
    Calculates a completeness score for each ceramic based on non-null values
    in relevant columns of ceramic_summary.
    """
    
    if 'ceramic_summary' not in dfs or dfs['ceramic_summary'] is None:
        print("Error: 'ceramic_summary' needed for completeness scoring.")
        return {}

    ceramic_summary = dfs['ceramic_summary']
    try:
        # Ensure ceramic_id is int for consistent indexing and lookup
        ceramic_summary['ceramic_id'] = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce').dropna().astype(int)
    except Exception as e:
        print(f"Warning: Could not process ceramic_id in calculate_completeness_score: {e}")


    # Select relevant columns for scoring
    score_cols = ['origin', 'tech_cat_id', 'reuse', 'production_fail',
                  'period_name_fr', 'context_type_name', 'identifier_origin',
                  'function_id', 'feature_id', 'color_name_fr']
    
    # Filter summary for relevant ceramics first for efficiency
    # Make sure ceramic_ids are of a comparable type (e.g., int) if ceramic_summary['ceramic_id'] is int
    try:
        ceramic_ids_int = [int(cid) for cid in ceramic_ids]
    except (ValueError, TypeError):
        print("Warning: Not all ceramic_ids for completeness score could be converted to int. Results may be incomplete.")
        ceramic_ids_int = [cid for cid in ceramic_ids if isinstance(cid, (int, float)) and not pd.isna(cid)] # Best effort


    summary_subset = ceramic_summary[ceramic_summary['ceramic_id'].isin(ceramic_ids_int)].copy()

    # Keep only columns that actually exist in the dataframe
    valid_score_cols = [col for col in score_cols if col in summary_subset.columns]
    if not valid_score_cols:
        print("Warning: No valid scoring columns found in ceramic_summary. Returning zero scores.")
        return {cid: 0 for cid in ceramic_ids} # Return score for original list of cids

    # Calculate score: count non-null values in the selected columns
    summary_subset['completeness_score'] = summary_subset[valid_score_cols].notna().sum(axis=1)

    scores = summary_subset.set_index('ceramic_id')['completeness_score'].to_dict()

    # Ensure all requested ceramic_ids (original format) have a score
    # Map back to original cid format if conversion happened
    final_scores = {}
    for original_cid in ceramic_ids:
        try:
            processed_cid = int(original_cid)
            final_scores[original_cid] = scores.get(processed_cid, 0)
        except (ValueError, TypeError):
            final_scores[original_cid] = 0 # Assign 0 if ID was not processable

    return final_scores


def extract_triplets_for_selection(selected_ceramic_ids, dfs):
    print(f"Extracting triplets for {len(selected_ceramic_ids)} selected ceramics (handling lists)...")
    required = ['ceramic_summary', 'tech_cat', 'object_function', 'Features_Ontology']
    if not all(req in dfs and not dfs[req].empty for req in required):
        missing = [req for req in required if req not in dfs or dfs[req].empty]
        print(f"❌ Error: Missing or empty required DataFrames: {missing}")
        return None
    
    required_summary_cols = ['ceramic_id', 'tech_cat_id']
    optional_list_cols = ['function_id', 'feature_id']
    missing_summary_cols = [col for col in required_summary_cols if col not in dfs['ceramic_summary'].columns]
    if missing_summary_cols: 
        print(f"❌ Error: Missing required columns in ceramic_summary: {missing_summary_cols}")
        return None
    for col in optional_list_cols:
        if col not in dfs['ceramic_summary'].columns: 
            print(f"    ℹ️ Info: Optional list column '{col}' not found in ceramic_summary.")

    try:
        ceramic_summary = dfs['ceramic_summary'].copy()
        ceramic_summary['ceramic_id'] = pd.to_numeric(ceramic_summary['ceramic_id'], errors='coerce').dropna().astype(int)
        original_summary_count = len(ceramic_summary)
        ceramic_summary.drop_duplicates(subset=['ceramic_id'], keep='first', inplace=True)
        unique_summary_count = len(ceramic_summary)
        if original_summary_count != unique_summary_count: 
            print(f"    ⚠️ Warning: Dropped {original_summary_count - unique_summary_count} duplicate ceramic_id rows from ceramic_summary.")
        
        ceramic_summary['tech_cat_id'] = pd.to_numeric(ceramic_summary['tech_cat_id'], errors='coerce')

        for col in optional_list_cols:
            if col in ceramic_summary.columns:
                new_col_values = []
                for item_idx, item in enumerate(ceramic_summary[col]): # Use enumerate for better error reporting if needed
                    if isinstance(item, str):
                        if item.startswith('[') and item.endswith(']'):
                            try:
                                parsed_list = ast.literal_eval(item)
                                if isinstance(parsed_list, list):
                                     new_col_values.append([pd.to_numeric(i, errors='ignore') for i in parsed_list])
                                else: 
                                     new_col_values.append(pd.to_numeric(parsed_list, errors='ignore'))
                            except (ValueError, SyntaxError):
                                new_col_values.append(item) 
                        else:
                            # Attempt to convert single string numbers, otherwise keep as string
                            val = pd.to_numeric(item, errors='ignore')
                            new_col_values.append(val)
                    elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)): # Handles lists, tuples, arrays
                        # Ensure elements within existing iterables are numeric where possible
                        new_col_values.append([pd.to_numeric(i, errors='ignore') for i in item])
                    else: # Handles NaN, numbers
                        new_col_values.append(item)
                ceramic_summary[col] = pd.Series(new_col_values, index=ceramic_summary.index)
        
        ceramic_summary_indexed = ceramic_summary.set_index('ceramic_id', drop=False)
        
        tech_cat = dfs['tech_cat'].copy()
        tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
        tech_cat['inherit_from'] = pd.to_numeric(tech_cat['inherit_from'], errors='coerce')
        cat_parent_map = tech_cat.set_index('id')['inherit_from'].to_dict()
        
        object_function = dfs['object_function'].copy()
        object_function['id'] = pd.to_numeric(object_function['id'], errors='coerce').dropna().astype(int)
        object_function['function_parent'] = pd.to_numeric(object_function['function_parent'], errors='coerce')
        func_parent_map = object_function.set_index('id')['function_parent'].to_dict()
        
        Features_Ontology_df = dfs['Features_Ontology'].copy()
        Features_Ontology_df['id'] = pd.to_numeric(Features_Ontology_df['id'], errors='coerce').dropna().astype(float).astype(int).astype(str)
        Features_Ontology_df['feature_parent'] = pd.to_numeric(Features_Ontology_df['feature_parent'], errors='coerce').fillna(-1).astype(float).astype(int).astype(str)
        feat_parent_map = Features_Ontology_df.set_index('id')['feature_parent'].to_dict()

    except Exception as e:
        print(f"❌ Error preparing DataFrames/Mappings: {e}"); traceback.print_exc(); return None

    results = []
    memo_parents.clear()

    for i, cid_original in enumerate(selected_ceramic_ids):
        try:
            cid = int(cid_original) 
        except (ValueError, TypeError):
            print(f"Warning: Skipping ceramic_id '{cid_original}' as it's not a valid integer.")
            results.append({"ceramic_id": cid_original, "categories": [], "functions": [], "features": [], "error": "Invalid ID format"})
            continue
        
        entry = {"ceramic_id": cid, "categories": [], "functions": [], "features": []}
        
        if cid not in ceramic_summary_indexed.index:
            results.append(entry)
            continue
        
        ceramic_data_series = ceramic_summary_indexed.loc[cid]
        if isinstance(ceramic_data_series, pd.DataFrame): # Should not happen with drop_duplicates
            ceramic_data_series = ceramic_data_series.iloc[0]

        # --- Category Hierarchy ---
        tech_cat_id_val = ceramic_data_series.get('tech_cat_id')
        if pd.notna(tech_cat_id_val): 
            try:
                cat_id_int = int(float(tech_cat_id_val))
                cat_parents = get_parents(cat_id_int, cat_parent_map, is_string_id=False)
                hierarchy_ids = [cat_id_int]
                for p_val in cat_parents:
                    if pd.notna(p_val): 
                        try:
                            parent_int = int(float(p_val))
                            hierarchy_ids.append(parent_int)
                        except (ValueError, TypeError, OverflowError):
                            continue 
                entry["categories"] = [{"category_id": hid} for hid in hierarchy_ids]
            except (ValueError, TypeError, OverflowError) as e:
                print(f"Warning: Could not fully process tech_cat_id {tech_cat_id_val} for ceramic {cid}: {e}")
                try: entry["categories"] = [{"category_id": int(float(tech_cat_id_val))}]
                except: pass

        # --- Direct Functions & Hierarchy ---
        func_id_input_value = ceramic_data_series.get('function_id')
        func_ids_to_process = []
        if func_id_input_value is not None: # Check for Python None
            if isinstance(func_id_input_value, (list, tuple, pd.Series, np.ndarray)):
                temp_list = list(func_id_input_value) if not isinstance(func_id_input_value, list) else func_id_input_value
                func_ids_to_process = [item for item in temp_list if pd.notna(item) and item is not None]
            elif pd.notna(func_id_input_value): # Scalar and not NaN
                func_ids_to_process = [func_id_input_value]
        
        if func_ids_to_process:
            # print(f"Debug: Processing function_ids {func_ids_to_process} for ceramic {cid}")
            processed_functions = []
            for func_id_item in func_ids_to_process:
                try:
                    func_id_int = int(float(func_id_item)) 
                    func_parents_raw = get_parents(func_id_int, func_parent_map, is_string_id=False)
                    func_parents_int = []
                    for p_val in func_parents_raw:
                        if pd.notna(p_val):
                            try:
                                parent_int = int(float(p_val))
                                func_parents_int.append(parent_int)
                            except (ValueError, TypeError, OverflowError):
                                continue 
                    processed_functions.append((func_id_int, func_parents_int))
                except (ValueError, TypeError, OverflowError) as e:
                    print(f"Warning: Could not fully process function_id {func_id_item} for ceramic {cid}: {e}")
            entry["functions"] = processed_functions

        # --- Direct Features & Hierarchy ---
        feat_id_input_value = ceramic_data_series.get('feature_id')
        feat_ids_to_process = []
        if feat_id_input_value is not None: # Check for Python None
            if isinstance(feat_id_input_value, (list, tuple, pd.Series, np.ndarray)):
                temp_list = list(feat_id_input_value) if not isinstance(feat_id_input_value, list) else feat_id_input_value
                feat_ids_to_process = [item for item in temp_list if pd.notna(item) and item is not None]
            elif pd.notna(feat_id_input_value): # Scalar and not NaN
                feat_ids_to_process = [feat_id_input_value]
        
        if feat_ids_to_process:
            processed_features = []
            for feat_id_item in feat_ids_to_process:
                try:
                    feat_id_str = str(int(float(feat_id_item))) 
                    feat_parents_str = get_parents(feat_id_str, feat_parent_map, is_string_id=True)
                    processed_features.append((feat_id_str, feat_parents_str)) 
                except (ValueError, TypeError, OverflowError) as e:
                     print(f"Warning: Could not process feature_id {feat_id_item} for ceramic {cid}: {e}")
            entry["features"] = processed_features
            
        results.append(entry)

    print(f"Finished extraction for selection. Got results structure for {len(results)} ceramics.")
    return results