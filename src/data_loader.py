import pandas as pd
import os
from . import config

def load_all_dataframes(data_base_path):
    """Loads all specified CSV files into a dictionary of DataFrames."""
    dfs = {}
    print(f"Loading data from: {data_base_path}")
    for filename, key in config.FILENAME_TO_KEY_MAP.items():
        file_path = os.path.join(data_base_path, filename)
        try:
            dfs[key] = pd.read_csv(file_path)
            print(f"  Loaded {filename} as dfs['{key}']")
        except FileNotFoundError:
            print(f"  WARNING: File not found {file_path} for key '{key}'")
            dfs[key] = pd.DataFrame() 
        except Exception as e:
            print(f"  ERROR loading {filename}: {e}")
            dfs[key] = pd.DataFrame()
    return dfs