import pandas as pd
import re
import config

def translate_dataframes(dfs):
    if 'object_colors' in dfs and not dfs['object_colors'].empty:
        dfs['object_colors']['color_name_en'] = dfs['object_colors']['color_name_fr'].map(config.COLOR_FR_TO_EN)
    if 'context_type_list' in dfs and not dfs['context_type_list'].empty:
        dfs['context_type_list']['context_name_en'] = dfs['context_type_list']['context_name_fr'].map(config.CONTEXT_TYPE_FR_TO_EN)
    if 'object_data_source' in dfs and not dfs['object_data_source'].empty:
        dfs['object_data_source']['source_name_en'] = dfs['object_data_source']['source_name'].map(config.IDENTIFIER_SOURCE_FR_TO_EN)
    return dfs

def _clean_category_name_single(name):
    if pd.isna(name) or not isinstance(name, str) or name.strip() == "":
        return 'Category'
    
    processed_name = name.strip()
    processed_name = re.sub(r'^\s*(Categories|categorie|categories|Catégories|Category|category)\s*:?\s*', '', processed_name, flags=re.IGNORECASE)
    processed_name = re.sub(r'^\s*(Categories|categorie|categories|Catégories|Category|category)\s*:?\s*', '', processed_name, flags=re.IGNORECASE)
    processed_name = re.sub(r'\s+', ' ', processed_name).strip()
    
    if not processed_name:
        final_name = 'Category'
    else:
        if not re.search(r'^\s*(Categories|categorie|categories|Catégories|Category|category)\s*:?\s*', processed_name, flags=re.IGNORECASE):
            final_name = ' Category ' + processed_name 
        else:
            final_name = processed_name
    
    return final_name

def process_tech_cat_names(dfs):
    if 'tech_cat' in dfs and not dfs['tech_cat'].empty:
        dfs['tech_cat']['cat_name_processed'] = dfs['tech_cat']['cat_name'].apply(_clean_category_name_single)
    return dfs
