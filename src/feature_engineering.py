import pandas as pd
import numpy as np
import config

def create_ceramic_summary(dfs):
    """
    Creates the ceramic_summary DataFrame by merging and processing various
    input DataFrames. This function encapsulates the logic from notebook cells 7-9.
    It uses the English translated columns where available.
    """
    print("Creating ceramic_summary DataFrame...")
    required_dfs = [
        'ceramic', 'archaeological_sites', 'object_data_source', 'tech_cat',
        'object_function_attrib', 'object_function', 'object_feature_attrib',
        'object_feature', 'object_colors_attrib', 'object_colors',
        'historical_period', 'context_type_list'
    ]
    for req_df_key in required_dfs:
        if req_df_key not in dfs or dfs[req_df_key].empty:
            print(f"  ERROR: Required DataFrame '{req_df_key}' is missing or empty.")
            return pd.DataFrame()

    ceramic_df = dfs['ceramic'].copy()
    archaeological_sites_df = dfs['archaeological_sites'].copy()
    object_data_source_df = dfs['object_data_source'].copy()
    tech_cat_df = dfs['tech_cat'].copy()
    object_function_attrib_df = dfs['object_function_attrib'].copy()
    object_function_df = dfs['object_function'].copy()
    object_feature_attrib_df = dfs['object_feature_attrib'].copy()
    feature_name_df = dfs['Features_Ontology'] if 'Features_Ontology' in dfs else dfs['object_feature']

    object_colors_attrib_df = dfs['object_colors_attrib'].copy()
    object_colors_df = dfs['object_colors'].copy()
    historical_period_df = dfs['historical_period'].copy()
    context_type_list_df = dfs['context_type_list'].copy()
    # Features_Ontology_df = dfs['Features_Ontology'].copy() # Not directly used in ceramic_summary creation

    # --- Define column names to use  ---
    source_name_col = 'source_name_en' if 'source_name_en' in object_data_source_df.columns else 'source_name'
    tech_cat_name_col = 'cat_name_processed' if 'cat_name_processed' in tech_cat_df.columns else 'cat_name'
    function_name_col = 'function_name_fr' 
    feature_name_col = 'feature_name_fr'  
    color_name_col = 'color_name_en' if 'color_name_en' in object_colors_df.columns else 'color_name_fr'
    context_name_col = 'context_name_en' if 'context_name_en' in context_type_list_df.columns else 'context_name_fr'
    period_name_col = 'period_name_fr'    
    # --- Merging archaeological_sites ---
    ceramic_df = ceramic_df.merge(
        archaeological_sites_df[['id', 'site_name', 'site_country_code']],
        left_on='origin',
        right_on='id',
        how='left',
        suffixes=('', '_arch_site') # Use a unique suffix
    )
    # Ensure 'site_country_code' is string before map
    ceramic_df['site_country_code'] = ceramic_df['site_country_code'].astype(str).fillna('??')
    ceramic_df['origin_combined'] = ceramic_df['site_name'].fillna('Unknown') + ', ' + ceramic_df['site_country_code'].map(config.CODE_TO_COUNTRY)
    
    # Drop original 'origin' and merged 'id' and other site cols to avoid clashes and keep clean
    cols_to_drop_arch = ['origin', 'id_arch_site']
    if 'site_name' in ceramic_df.columns: cols_to_drop_arch.append('site_name')
    if 'site_country_code' in ceramic_df.columns: cols_to_drop_arch.append('site_country_code')
    ceramic_df.drop(columns=[col for col in cols_to_drop_arch if col in ceramic_df.columns], inplace=True)
    ceramic_df.rename(columns={'origin_combined': 'origin'}, inplace=True)


    # --- Merging object_data_source ---
    ceramic_df = ceramic_df.merge(
        object_data_source_df[['id', source_name_col]],
        left_on='identifier_origin',
        right_on='id',
        how='left',
        suffixes=('', '_obj_source')
    )
    if 'id_obj_source' in ceramic_df.columns: ceramic_df.drop(columns=['id_obj_source'], inplace=True)
    ceramic_df.rename(columns={source_name_col: 'identifier_origin_source_name'}, inplace=True)


    # --- Tech Cat ID handling ---
    ceramic_df['tech_cat_id'] = ceramic_df['tech_cat_id'].fillna(-1).astype(int)
    tech_cat_df_copy = tech_cat_df.copy()
    if -1 not in tech_cat_df_copy['id'].unique(): # Ensure 'id' column exists and is unique
        tech_cat_template = {"id": -1, tech_cat_name_col: "Unknown Category"}
        for col in tech_cat_df_copy.columns:
            if col not in tech_cat_template:
                tech_cat_template[col] = pd.NA if tech_cat_df_copy[col].dtype == 'object' else np.nan
        tech_cat_df_copy = pd.concat([tech_cat_df_copy, pd.DataFrame([tech_cat_template])], ignore_index=True)

    # --- Main Merge for ceramic_summary ---
    # Start with the ceramic_df, which will be 'merged_data'
    # Rename its 'id' column to avoid conflicts during merges
    merged_data = ceramic_df.rename(columns={'id': 'ceramic_id_main'})

    # Merge tech_cat
    merged_data = merged_data.merge(
        tech_cat_df_copy[['id', tech_cat_name_col]].rename(columns={'id': 'tech_cat_table_id'}), # Rename to avoid clash with other ids
        left_on="tech_cat_id", 
        right_on="tech_cat_table_id", 
        how="left"
    )
    if 'tech_cat_table_id' in merged_data.columns: merged_data.drop(columns=['tech_cat_table_id'], inplace=True)
    # tech_cat_name_col is now in merged_data

    # Merge function attributes and function names
    merged_data = merged_data.merge(
        object_function_attrib_df[['object_id', 'function_id']].rename(columns={'object_id':'object_id_func_attr'}),
        left_on="ceramic_id_main", 
        right_on="object_id_func_attr", 
        how="left"
    )
    merged_data = merged_data.merge(
        object_function_df[['id', function_name_col]].rename(columns={'id': 'func_table_id', function_name_col:'function_name_final'}),
        left_on="function_id", 
        right_on="func_table_id", 
        how="left"
    )
    if 'object_id_func_attr' in merged_data.columns: merged_data.drop(columns=['object_id_func_attr'], inplace=True)
    if 'func_table_id' in merged_data.columns: merged_data.drop(columns=['func_table_id'], inplace=True)


    # Merge feature attributes and feature names
    merged_data = merged_data.merge(
        object_feature_attrib_df[['object_id', 'feature_id']].rename(columns={'object_id':'object_id_feat_attr'}),
        left_on="ceramic_id_main", 
        right_on="object_id_feat_attr", 
        how="left"
    )
    merged_data = merged_data.merge(
        feature_name_df[['id', feature_name_col]].rename(columns={'id': 'feat_table_id', feature_name_col:'feature_name_final'}),
        left_on="feature_id", 
        right_on="feat_table_id", 
        how="left"
    )
    if 'object_id_feat_attr' in merged_data.columns: merged_data.drop(columns=['object_id_feat_attr'], inplace=True)
    if 'feat_table_id' in merged_data.columns: merged_data.drop(columns=['feat_table_id'], inplace=True)


    # Merge color attributes and color names
    merged_data = merged_data.merge(
        object_colors_attrib_df[['object_id', 'color_id']].rename(columns={'object_id':'object_id_col_attr'}),
        left_on="ceramic_id_main", 
        right_on="object_id_col_attr", 
        how="left"
    )
    merged_data = merged_data.merge(
        object_colors_df[['id', color_name_col]].rename(columns={'id': 'col_table_id', color_name_col:'color_name_final'}),
        left_on="color_id", 
        right_on="col_table_id", 
        how="left"
    )
    if 'object_id_col_attr' in merged_data.columns: merged_data.drop(columns=['object_id_col_attr'], inplace=True)
    if 'col_table_id' in merged_data.columns: merged_data.drop(columns=['col_table_id'], inplace=True)

    # Merge historical period
    merged_data = merged_data.merge(
        historical_period_df[['id', period_name_col]].rename(columns={'id': 'hist_table_id', period_name_col: 'period_name_final'}),
        left_on="historical_period", 
        right_on="hist_table_id", 
        how="left"
    )
    if 'hist_table_id' in merged_data.columns: merged_data.drop(columns=['hist_table_id'], inplace=True)

    # Merge context type
    merged_data = merged_data.merge(
        context_type_list_df[['id', context_name_col]].rename(columns={'id': 'context_table_id', context_name_col:'context_name_final'}),
        left_on="context_type", 
        right_on="context_table_id", 
        how="left"
    )
    if 'context_table_id' in merged_data.columns: merged_data.drop(columns=['context_table_id'], inplace=True)


    # --- Aggregation ---
    # Now, the column names in merged_data should be clean or predictably suffixed.
    agg_dict = {
        'ceramic_identifier': ('identifier', 'first'),
        'origin': ('origin', 'first'), 
        'tech_cat_id': ('tech_cat_id', 'first'),
        'tech_cat_name': (tech_cat_name_col, 'first'),
        'reuse': ('reuse', 'first'),
        'production_fail': ('production_fail', 'first'),
        'period_name_fr': ('period_name_final', 'first'), 
        'context_type_name': ('context_name_final', 'first'),
        'identifier_origin_source_name': ('identifier_origin_source_name', 'first'), 
        'function_id': ("function_id", lambda x: sorted(list(x.dropna().unique().astype(int)))),
        'function_name_fr': ("function_name_final", lambda x: sorted(list(x.dropna().unique()))),
        'feature_id': ("feature_id", lambda x: sorted(list(x.dropna().unique().astype(int)))),
        'feature_name_fr': ("feature_name_final", lambda x: sorted(list(x.dropna().unique()))),
        'color_name_list': ('color_name_final', lambda x: sorted(list(x.dropna().unique()))) 
    }
    
    # Ensure all keys in agg_dict exist in merged_data.columns before aggregation
    missing_agg_keys = [val[0] for val in agg_dict.values() if val[0] not in merged_data.columns]
    if missing_agg_keys:
        print(f"  ERROR: Missing columns in merged_data for aggregation: {missing_agg_keys}")
        print(f"  Available columns in merged_data: {merged_data.columns.tolist()}")
        return pd.DataFrame()

    ceramic_summary = (
        merged_data
        .groupby("ceramic_id_main")
        .agg(**agg_dict)
        .reset_index()
        .rename(columns={"ceramic_id_main": "ceramic_id"}) # color_name already renamed to color_name_list
    )

    # --- Description Generation ---
    def generate_ceramic_description(row, code_to_country_map):
        parts = []
        colors = row['color_name_list'] # Use the aggregated list column
        valid_colors = []
        if isinstance(colors, list) and colors:
            valid_colors = [str(c).strip().lower() for c in colors if pd.notna(c) and str(c).strip()]
        color_text = " ".join(sorted(set(valid_colors)))

        if color_text:
            color_text = color_text.capitalize()
        else:
            color_text = "Ceramic"

        base_phrase = f"A {color_text} ceramic"

        if row.get('production_fail') == True:
            base_phrase += " with a production fault"

        origin_text = row['origin']
        site_name, country_name_full = "Unknown", ""
        if pd.notna(origin_text) and isinstance(origin_text, str):
            if ', ' in origin_text:
                origin_parts_desc = origin_text.split(', ', 1)
                site_name = origin_parts_desc[0]
                if len(origin_parts_desc) > 1:
                    country_name_full = origin_parts_desc[1] 
            else:
                site_name = origin_text

        site = site_name if site_name != "Unknown" else None
        country = country_name_full if country_name_full else None

        context = row.get('context_type_name')
        origin_parts_desc = []
        if context and pd.notna(context) and str(context).lower() != 'unknown':
            origin_parts_desc.append(f"from the {context}")
        if site:
            origin_parts_desc.append(f"in {site}")
        # elif site and country: # site implies country is already part of origin
        #     origin_parts_desc.append(f"in {site}") # country_name_full from origin
        elif country and not site : # Only add country if site is not already mentioned (which would include country)
            origin_parts_desc.append(f"in {country}")

        if origin_parts_desc:
            base_phrase += " " + " ".join(origin_parts_desc)

        return base_phrase.strip() + "."


    ceramic_summary['description'] = ceramic_summary.apply(lambda row: generate_ceramic_description(row, config.CODE_TO_COUNTRY), axis=1)

    print(f"  ceramic_summary created with {len(ceramic_summary)} rows.")
    return ceramic_summary


import ast
import pandas as pd

def generate_one_hot_embeddings(ceramic_summary_df):
    """Generates one-hot embeddings and adds them as a column."""
    print("Generating one-hot embeddings...")
    if ceramic_summary_df.empty:
        print("  ceramic_summary_df is empty. Cannot generate embeddings.")
        return ceramic_summary_df

    origin_values = sorted(ceramic_summary_df['origin'].dropna().unique())
    
    # Fixed color processing
    all_colors = set()
    for color_list in ceramic_summary_df['color_name_list'].dropna():
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
    
    color_values = sorted(list(all_colors))

    context_values = sorted(ceramic_summary_df['context_type_name'].dropna().unique())
    source_values = sorted(ceramic_summary_df['identifier_origin_source_name'].dropna().unique())

    origin_map = {v: i for i, v in enumerate(origin_values)}
    color_map = {v: i for i, v in enumerate(color_values)}
    context_map = {v: i for i, v in enumerate(context_values)}
    source_map = {v: i for i, v in enumerate(source_values)}

    print(f"  OHE map sizes: origin={len(origin_map)}, color={len(color_map)}, context={len(context_map)}, source={len(source_map)}")

    def get_embedding(row):
        embedding = []
        
        # Origin
        origin_vector = [0] * len(origin_map)
        if pd.notna(row['origin']) and row['origin'] in origin_map:
            origin_vector[origin_map[row['origin']]] = 1
        embedding.extend(origin_vector)
        
        # Colors - Fixed processing with proper null checking
        color_vector = [0] * len(color_map)
        color_list = row['color_name_list']
        
        # Fix: Check if color_list is not None and not NaN before processing
        if color_list is not None and not (isinstance(color_list, float) and pd.isna(color_list)):
            if isinstance(color_list, str):
                # If it's a string representation of a list, parse it
                try:
                    color_list = ast.literal_eval(color_list)
                except (ValueError, SyntaxError):
                    color_list = []
            
            if isinstance(color_list, list):
                for c_val in color_list:
                    if pd.notna(c_val):
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
        
        # Reuse
        if pd.isna(row['reuse']): 
            embedding.extend([0, 0])
        else: 
            embedding.extend([0, 1] if row['reuse'] else [1, 0])
        
        # Production Fail
        if pd.isna(row['production_fail']): 
            embedding.extend([0, 0])
        else: 
            embedding.extend([0, 1] if row['production_fail'] else [1, 0])
        
        return embedding

    ceramic_summary_df['Embedding'] = ceramic_summary_df.apply(get_embedding, axis=1)
    
    if not ceramic_summary_df.empty and 'Embedding' in ceramic_summary_df.columns:
        print(f"  'Embedding' column added. Example length: {len(ceramic_summary_df['Embedding'].iloc[0])}")
    
    return ceramic_summary_df