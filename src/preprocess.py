import pandas as pd
import re
from . import config

def translate_dataframes(dfs):
    """Applies French to English translations to relevant DataFrames."""
    print("Applying translations...")
    if 'object_colors' in dfs and not dfs['object_colors'].empty:
        dfs['object_colors']['color_name_en'] = dfs['object_colors']['color_name_fr'].map(config.COLOR_FR_TO_EN)
        # print("  Translated 'object_colors'")
    if 'context_type_list' in dfs and not dfs['context_type_list'].empty:
        dfs['context_type_list']['context_name_en'] = dfs['context_type_list']['context_name_fr'].map(config.CONTEXT_TYPE_FR_TO_EN)
        # print("  Translated 'context_type_list'")
    if 'object_data_source' in dfs and not dfs['object_data_source'].empty:
        dfs['object_data_source']['source_name_en'] = dfs['object_data_source']['source_name'].map(config.IDENTIFIER_SOURCE_FR_TO_EN)
        # print("  Translated 'object_data_source'")
    return dfs

def _clean_category_name_single(name):
    """Cleans a single category name."""
    if pd.isna(name) or not isinstance(name, str) or name.strip() == "":
        return 'Category'
    
    processed_name = name.strip()
    
    # Remove category-related words from the beginning or end, but preserve the core content
    # First, remove from the beginning
    processed_name = re.sub(r'^\s*(Categories|categorie|categories|Catégories|Category|category)\s*:?\s*', '', processed_name, flags=re.IGNORECASE)
    
    # Then remove from the end
    processed_name = re.sub(r'^\s*(Categories|categorie|categories|Catégories|Category|category)\s*:?\s*', '', processed_name, flags=re.IGNORECASE)
    
    # Clean up multiple spaces
    processed_name = re.sub(r'\\s+', ' ', processed_name).strip()
    
    # If nothing is left after cleaning, use default
    if not processed_name:
        final_name = 'Category'
    else:
        # Only add "Category" if it doesn't already end with a category-related word
        if not re.search(r'^\s*(Categories|categorie|categories|Catégories|Category|category)\s*:?\s*', processed_name, flags=re.IGNORECASE):
            final_name = ' Category' + processed_name 
        else:
            final_name = processed_name
    
    return final_name

def process_tech_cat_names(dfs):
    """Processes 'cat_name' in the tech_cat DataFrame."""
    print("Processing tech_cat names...")
    if 'tech_cat' in dfs and not dfs['tech_cat'].empty:
        dfs['tech_cat']['cat_name_processed'] = dfs['tech_cat']['cat_name'].apply(_clean_category_name_single)
        print("  Processed 'tech_cat' names into 'cat_name_processed'")
    return dfs