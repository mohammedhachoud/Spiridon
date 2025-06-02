import os
import torch

# --- Paths ---
# Adjust these paths if your project root is different
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Assumes src is one level down
KAGGLE_INPUT_PATH = "/kaggle/input/db-data/" # Original Kaggle path
LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT, "data")
OUTPUT_BASE_DIR = os.path.join(PROJECT_ROOT, "output")

RGCN_STUDY_DATASETS_DIR = os.path.join(OUTPUT_BASE_DIR, "rgcn_study_datasets")
CLASSIFICATION_DATA_DIR = os.path.join(OUTPUT_BASE_DIR, "classification_data")
LIGHTNING_MODELS_DIR = os.path.join(OUTPUT_BASE_DIR, "lightning_models")
LIGHTNING_LOGS_DIR = os.path.join(OUTPUT_BASE_DIR, "lightning_logs")
PREDICTIONS_DIR = os.path.join(OUTPUT_BASE_DIR, "predictions")


# --- Model & Training Hyperparameters (Defaults) ---
BERT_MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
TARGET_EMBEDDING_DIM = 85 # For hybrid embeddings (OHE dimension)

# Link Prediction (RGCNLinkPredictor)
LP_BATCH_SIZE = 16
LP_LEARNING_RATE = 5e-4
LP_NUM_EPOCHS = 200
LP_RGCN_HIDDEN_DIM = 32
LP_NUM_RGCN_LAYERS = 2
LP_NUM_BASES = 4
LP_DROPOUT = 0.2
LP_L2_REG = 0.2
LP_PATIENCE_EARLY_STOPPING = 10
LP_NEG_SAMPLES_BTC = 4
LP_USE_SIMILARITY_REL_INIT = False # Default, can be overridden
LP_TRAIN_RATIO_BTC = 0.7
LP_VAL_RATIO_BTC = 0.15

EVALUATION_RELATION_NAME = "BELONGS_TO_CATEGORY" # For link prediction

# Classification (RGCNClassifier)
CLF_BATCH_SIZE = 64 # Or adjust as needed, often larger for classification
CLF_NUM_EPOCHS = 300
CLF_RGCN_HIDDEN_DIM = 64
CLF_NUM_RGCN_LAYERS = 2
CLF_NUM_BASES = None  # Or a default value if RGCNConv needs it
CLF_LEARNING_RATE = 1e-3     # Much more conservative
CLF_L2_REG = 0             # Your current L2 might be too strong
CLF_DROPOUT = 0.2 # Typical L2 for classifiers
CLF_PATIENCE_EARLY_STOPPING = 5 # From notebook
CLF_NEG_SAMPLES_TRAINING = 4 # For augmented training in classifier

# --- Translations & Mappings ---
COLOR_FR_TO_EN = {
    'Vert': 'Green', 'Brun': 'Brown', 'Violacé': 'Violet',
    'Bleu': 'Blue', 'Jaune moutarde': 'Mustard yellow', 'Brun-noir': 'Brown-black',
    'Jaune': 'Yellow', 'Turquoise': 'Turquoise', 'Orange': 'Orange',
    'Rouge-brun': 'Red-brown', 'Blanc': 'White', 'Lustre': 'Luster'
}

CONTEXT_TYPE_FR_TO_EN = {
    'production': 'production', 'consommation': 'consumption',
    'inconnu': 'unknown', 'dépôt funéraire': 'funerary deposit'
}

IDENTIFIER_SOURCE_FR_TO_EN = {
    'Inventaire LA3M/LAMM': 'LA3M/LAMM Inventory',
    'Diapositives LA3M/LAMM': 'LA3M/LAMM Slides',
    'Archives photos LA3M/LAMM': 'LA3M/LAMM Photo Archives',
    'Archives chantier': 'Site Archives'
}

CODE_TO_COUNTRY = {
    'FRA': 'France', 'SYR': 'Syria', 'CYP': 'Cyprus', 'ESP': 'Spain',
    'BGR': 'Bulgaria', 'TUR': 'Turkey', 'UZB': 'Uzbekistan', 'DZA': 'Algeria',
    'EGY': 'Egypt', 'UKR': 'Ukraine', 'MAR': 'Morocco', '??': 'Unknown country'
}

# --- Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CSV Filenames (as keys for dfs dictionary) ---
CSV_FILES = [
    "ceramic", "object_colors", "object_colors_attrib", "object_feature",
    "object_feature_combined_names", "object_feature_attrib",
    "object_function_translated", "object_function_attrib",
    "tech_cat_translated", "archaeological_sites", "traditional_designation",
    "historical_period", "tech_cat_color_attrib", "tech_cat_feature_attrib",
    "tech_cat_function_attrib", "relations_type", "relations",
    "context_type_list", "object_data_source", "category_hierarchy_combined_names",
    "Features_Ontology_PF_translated"
]

# Mapping notebook filenames to simpler keys for dfs dictionary
# Adjust keys if you prefer different ones
FILENAME_TO_KEY_MAP = {
    "ceramic.csv": "ceramic",
    "object_colors.csv": "object_colors",
    "object_colors_attrib.csv": "object_colors_attrib",
    "object_feature.csv": "object_feature",
    "object_feature_combined_names.csv": "object_feature_combined_names",
    "object_feature_attrib.csv": "object_feature_attrib",
    "object_function_translated.csv": "object_function",
    "object_function_attrib.csv": "object_function_attrib",
    "tech_cat_translated.csv": "tech_cat",
    "archaeological_sites.csv": "archaeological_sites",
    "traditional_designation.csv": "traditional_designation",
    "historical_period.csv": "historical_period",
    "tech_cat_color_attrib.csv": "tech_cat_color_attrib",
    "tech_cat_feature_attrib.csv": "tech_cat_feature_attrib",
    "tech_cat_function_attrib.csv": "tech_cat_function_attrib",
    "relations_type.csv": "relations_type",
    "relations.csv": "relations",
    "context_type_list.csv": "context_type_list",
    "object_data_source.csv": "object_data_source",
    "category_hierarchy_combined_names.csv": "category_hierarchy_combined_names",
    "Features_Ontology_PF_translated.csv": "Features_Ontology"
}

RELATION_HAS_FUNCTION_NAME = "HAS_FUNCTION"
DEFAULT_RELATION_HAS_FEATURE_NAME = "HAS_FEATURE"