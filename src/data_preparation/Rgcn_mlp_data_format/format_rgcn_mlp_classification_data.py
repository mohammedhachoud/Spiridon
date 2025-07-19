# src/data_preparation/format_rgcn_classification_data.py

import pandas as pd
import numpy as np
import os
import json
import traceback
import warnings
from datetime import datetime

# --- Helper: get_level1_root_group ---
def get_level1_root_group(node_str):
    if not isinstance(node_str, str) or not node_str.startswith('Cat_'):
        return None 
    
    if "_Root_" not in node_str:
        parts = node_str.split('_')
        if len(parts) == 2 and parts[0] == 'Cat':
             try:
                  _ = int(float(parts[1])) 
             except (ValueError, TypeError):
                  warnings.warn(f"Could not parse category ID from node string '{node_str}' lacking _Root_.")
                  return None
        else:
             warnings.warn(f"Unexpected category node format lacking _Root_: '{node_str}'")
             return None

    parts = node_str.split("_Root_")
    if len(parts) != 2:
        warnings.warn(f"Unexpected category node format with _Root_: '{node_str}'")
        return None

    try:
        specific_id_part = parts[0] # e.g., "Cat_135"
        root_id_str_from_node = parts[1] # e.g., "135"      
        if specific_id_part.split('_')[1] == root_id_str_from_node:
            return node_str 
        else:
            warnings.warn(f"Node string '{node_str}' does not seem to be a canonical root identifier. Returning as is for label key.")
            return node_str
            
    except Exception as e:
        warnings.warn(f"Error parsing node string '{node_str}' for root group: {e}")
        return None

def adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs=None):
    if rgcn_data is None:
        print("❌ Input rgcn_data is None. Cannot adapt for classification.")
        return None

    print("\n--- Adapting RGCN Data for Ceramic Classification (Direct Categories) ---")

    original_node_to_idx = rgcn_data['node_to_idx']
    original_idx_to_node = rgcn_data['idx_to_node']
    original_num_nodes = rgcn_data['num_nodes']

    # Keep all nodes for classification (1:1 mapping)
    original_idx_to_new_idx = {}
    new_node_to_idx = {}
    
    for original_idx in range(original_num_nodes):
        if original_idx in original_idx_to_node:
            node_identifier = original_idx_to_node[original_idx]
            original_idx_to_new_idx[original_idx] = original_idx
            new_node_to_idx[node_identifier] = original_idx

    print(f"  Original nodes: {original_num_nodes}")
    print(f"  Nodes kept for classification: {len(new_node_to_idx)}")

    original_embeddings = rgcn_data['node_embeddings']
    embedding_dim = rgcn_data['embedding_dim']
    original_training_triplets = rgcn_data['training_triplets']
    relation_to_idx = rgcn_data['relation_to_idx']
    idx_to_relation = rgcn_data['idx_to_relation']
    ceramic_attribute_maps = rgcn_data['ceramic_attribute_maps']
    
    # --- Direct Category Label Extraction ---
    ceramic_labels = {} 
    category_to_label_id = {} 
    label_id_to_category_id = {} 
    label_id_to_category_name = {}
    label_counter = 0

    # Use evaluation triplets to get ceramic-category relationships
    evaluation_triplets = rgcn_data.get('evaluation_triplets', [])
    all_triplets = rgcn_data.get('training_triplets', []) + evaluation_triplets
    
    # Find BELONGS_TO_CATEGORY relation
    EVALUATION_RELATION_NAME = "BELONGS_TO_CATEGORY"
    belongs_to_category_rel_id = None
    
    if 'relation_to_idx' in rgcn_data and EVALUATION_RELATION_NAME in rgcn_data['relation_to_idx']:
        belongs_to_category_rel_id = rgcn_data['relation_to_idx'][EVALUATION_RELATION_NAME]
    else:
        print(f"  CRITICAL ERROR: Relation '{EVALUATION_RELATION_NAME}' not found. Cannot create labels.")
        print(f"  Available relations: {list(rgcn_data['relation_to_idx'].keys())}")
        return None

    print(f"  Extracting direct category labels from {len(all_triplets)} available triplets...")
    print(f"  Looking for relation ID {belongs_to_category_rel_id} ('{EVALUATION_RELATION_NAME}')")
    
    # Extract labels from BELONGS_TO_CATEGORY triplets
    for h_idx, r_idx, t_idx in all_triplets:
        if r_idx != belongs_to_category_rel_id:
            continue

        h_node_identifier = original_idx_to_node.get(h_idx)
        t_node_identifier = original_idx_to_node.get(t_idx)

        # Check if head is ceramic and tail is category
        if (h_node_identifier and t_node_identifier and 
            h_node_identifier.startswith("Ceramic_") and t_node_identifier.startswith("Cat_")):
            
            if h_idx in original_idx_to_new_idx:
                h_new_idx = original_idx_to_new_idx[h_idx]
                
                # Extract category ID from node name (e.g., "Cat_132" -> 132)
                try:
                    category_id = int(t_node_identifier.split('_')[1])
                except (ValueError, IndexError):
                    print(f"  ⚠️ Warning: Could not parse category ID from '{t_node_identifier}'")
                    continue
                
                # Create label mapping
                if category_id not in category_to_label_id:
                    label_id = label_counter
                    category_to_label_id[category_id] = label_id
                    label_id_to_category_id[label_id] = category_id
                    
                    # Try to get category name from dfs if available
                    category_name = f"Category_{category_id}"
                    if dfs and 'tech_cat' in dfs and not dfs['tech_cat'].empty:
                        try:
                            cat_row = dfs['tech_cat'][dfs['tech_cat']['id'] == category_id]
                            if not cat_row.empty:
                                name_col = 'cat_name_processed' if 'cat_name_processed' in cat_row.columns else 'cat_name'
                                category_name = cat_row[name_col].iloc[0]
                        except Exception as e:
                            print(f"  ⚠️ Warning: Could not get name for category {category_id}: {e}")
                    
                    label_id_to_category_name[label_id] = category_name
                    label_counter += 1
                else:
                    label_id = category_to_label_id[category_id]
                
                ceramic_labels[h_new_idx] = label_id
    
    print(f"  Extracted labels for {len(ceramic_labels)} ceramic nodes. Found {label_counter} unique categories.")
    
    if label_counter > 0:
        print(f"  Label mapping ({label_counter} labels):")
        for lid, name in label_id_to_category_name.items():
            cat_id = label_id_to_category_id.get(lid, 'N/A')
            count = sum(1 for label in ceramic_labels.values() if label == lid)
            print(f"    Label {lid}: '{name}' (Category ID: {cat_id}, {count} ceramics)")

    # Find all ceramic node indices
    ceramic_node_indices = sorted([
        new_idx for original_idx, new_idx in original_idx_to_new_idx.items()
        if original_idx in original_idx_to_node and original_idx_to_node[original_idx].startswith("Ceramic_")
    ])
    print(f"  Identified {len(ceramic_node_indices)} ceramic nodes for classification.")

    # Create stats (simplified version)
    stats = {
        'classification_num_nodes': original_num_nodes,
        'classification_nodes_removed_category': 0,
        'classification_training_triplets_filtered': len(original_training_triplets),
        'classification_ceramic_nodes_with_labels': len(ceramic_labels),
        'classification_num_classes': label_counter,
        'classification_ceramic_node_indices': ceramic_node_indices,
        'classification_labeling_strategy': 'direct_category_from_evaluation_triplets'
    }

    classification_data = {
        "num_nodes": original_num_nodes,
        "num_relations": len(relation_to_idx), 
        "node_to_idx": new_node_to_idx,
        "idx_to_node": {v: k for k, v in new_node_to_idx.items()}, 
        "relation_to_idx": relation_to_idx,
        "idx_to_relation": idx_to_relation,
        "training_triplets": original_training_triplets,
        "node_embeddings": original_embeddings,
        "embedding_dim": embedding_dim,
        "ceramic_node_indices": ceramic_node_indices,
        "ceramic_labels": ceramic_labels,
        "label_to_category_id": label_id_to_category_id,
        "label_to_category_name": label_id_to_category_name,
        "stats": stats,
        "ceramic_attribute_maps": ceramic_attribute_maps
    }

    print("\n--- Adaptation for Classification Complete ---")
    print(f"  Final ceramic nodes with labels: {len(ceramic_labels)}")
    print(f"  Total classes: {label_counter}")
    
    return classification_data


# --- Helper: convert_numpy_to_python_native---
def convert_numpy_to_python_native(data):
    if isinstance(data, dict):
        return {k: convert_numpy_to_python_native(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_to_python_native(i) for i in data]
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.floating):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    elif isinstance(data, np.ndarray): 
        return convert_numpy_to_python_native(data.tolist())
    return data


import os
import json
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict
import traceback

def process_and_save_classification_data(study_name, study_datasets_dict, dfs_dict, base_output_dir="output/rgcn_data/ontology"):
    """
    Process and save classification data for all études in a study.
    
    Args:
        study_name (str): Name of the study (e.g., "level_2_connections")
        study_datasets_dict (dict): Dictionary containing étude datasets
        dfs_dict (dict): Dictionary of source dataframes
        base_output_dir (str): Base directory for saving classification data
    """
    print(f"\n🎯 Processing Classification Data for Study: {study_name}")
    print("=" * 60)
    
    if not study_datasets_dict:
        print(f"❌ No datasets found for study '{study_name}'. Skipping.")
        return
    
    study_output_dir = os.path.join(base_output_dir, study_name)
    os.makedirs(study_output_dir, exist_ok=True)
    
    classification_results = {}
    
    for etude_name, rgcn_data in study_datasets_dict.items():
        print(f"\n📊 Processing Étude: {etude_name}")
        print("-" * 40)
        
        if rgcn_data is None:
            print(f"⚠️ RGCN data is None for {etude_name}. Skipping.")
            classification_results[etude_name] = None
            continue
        
        try:
            # Adapt RGCN data for classification
            classification_data = adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs_dict)
            
            if classification_data is None:
                print(f"❌ Failed to adapt data for classification: {etude_name}")
                classification_results[etude_name] = None
                continue
            
            # Save classification data
            etude_output_dir = os.path.join(study_output_dir, etude_name)
            success = save_classification_dataset(
                classification_data, 
                etude_output_dir, 
                f"{study_name}_{etude_name}"
            )
            
            if success:
                classification_results[etude_name] = classification_data
                print(f"✅ Successfully processed and saved: {etude_name}")
            else:
                classification_results[etude_name] = None
                print(f"❌ Failed to save classification data: {etude_name}")
                
        except Exception as e:
            print(f"❌ Error processing {etude_name}: {e}")
            traceback.print_exc()
            classification_results[etude_name] = None
    
    # Save study summary
    save_study_classification_summary(classification_results, study_output_dir, study_name)
    
    print(f"\n✅ Completed processing classification data for study: {study_name}")
    print(f"📁 Results saved to: {study_output_dir}")
    
    return classification_results


def save_classification_dataset(classification_data, output_dir, dataset_label):
    """
    Save a single classification dataset with all necessary components.
    
    Args:
        classification_data (dict): Processed classification data
        output_dir (str): Output directory path
        dataset_label (str): Label for the dataset
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"  💾 Saving classification data to: {output_dir}")
        
        saved_files = []
        
        # 1. Save Node Mappings
        if 'node_to_idx' in classification_data and classification_data['node_to_idx']:
            node_mapping_df = pd.DataFrame(
                list(classification_data['node_to_idx'].items()), 
                columns=['NodeIdentifier', 'NodeIndex']
            )
            node_mapping_path = os.path.join(output_dir, "node_mapping.csv")
            node_mapping_df.to_csv(node_mapping_path, index=False)
            saved_files.append("node_mapping.csv")
            print(f"    ✓ Saved node mapping ({len(node_mapping_df)} nodes)")
        
        # 2. Save Relation Mappings
        if 'relation_to_idx' in classification_data and classification_data['relation_to_idx']:
            relation_mapping_df = pd.DataFrame(
                list(classification_data['relation_to_idx'].items()), 
                columns=['RelationName', 'RelationIndex']
            )
            relation_mapping_path = os.path.join(output_dir, "relation_mapping.csv")
            relation_mapping_df.to_csv(relation_mapping_path, index=False)
            saved_files.append("relation_mapping.csv")
            print(f"    ✓ Saved relation mapping ({len(relation_mapping_df)} relations)")
        
        # 3. Save Training Triplets
        if 'training_triplets' in classification_data and classification_data['training_triplets']:
            triplets_df = pd.DataFrame(
                classification_data['training_triplets'], 
                columns=['SourceIndex', 'RelationIndex', 'TargetIndex']
            )
            triplets_path = os.path.join(output_dir, "training_triplets.csv")
            triplets_df.to_csv(triplets_path, index=False)
            saved_files.append("training_triplets.csv")
            print(f"    ✓ Saved training triplets ({len(triplets_df)} triplets)")
        
        # 4. Save Node Embeddings
        if 'node_embeddings' in classification_data and isinstance(classification_data['node_embeddings'], np.ndarray):
            embeddings_path = os.path.join(output_dir, "node_embeddings.npy")
            np.save(embeddings_path, classification_data['node_embeddings'])
            saved_files.append("node_embeddings.npy")
            print(f"    ✓ Saved node embeddings {classification_data['node_embeddings'].shape}")
        
        # 5. Save Ceramic Node Indices
        if 'ceramic_node_indices' in classification_data:
            ceramic_indices_path = os.path.join(output_dir, "ceramic_node_indices.json")
            with open(ceramic_indices_path, 'w') as f:
                json.dump(classification_data['ceramic_node_indices'], f)
            saved_files.append("ceramic_node_indices.json")
            print(f"    ✓ Saved ceramic node indices ({len(classification_data['ceramic_node_indices'])} ceramics)")
        
        # 6. Save Ceramic Labels
        if 'ceramic_labels' in classification_data:
            # Convert to serializable format
            ceramic_labels_serializable = convert_numpy_to_python_native(classification_data['ceramic_labels'])
            ceramic_labels_path = os.path.join(output_dir, "ceramic_labels.json")
            with open(ceramic_labels_path, 'w') as f:
                json.dump(ceramic_labels_serializable, f)
            saved_files.append("ceramic_labels.json")
            print(f"    ✓ Saved ceramic labels ({len(ceramic_labels_serializable)} labeled ceramics)")
        
        # 7. Save Label Mappings
        if 'label_to_category_id' in classification_data:
            label_mapping_data = {
                'label_to_category_id': convert_numpy_to_python_native(classification_data['label_to_category_id']),
                'label_to_category_name': convert_numpy_to_python_native(classification_data['label_to_category_name'])
            }
            label_mapping_path = os.path.join(output_dir, "label_mappings.json")
            with open(label_mapping_path, 'w') as f:
                json.dump(label_mapping_data, f, indent=2)
            saved_files.append("label_mappings.json")
            print(f"    ✓ Saved label mappings ({len(label_mapping_data['label_to_category_id'])} classes)")
        
        # 8. Save Classification Configuration
        classification_config = create_classification_config(classification_data, dataset_label)
        config_path = os.path.join(output_dir, "classification_config.json")
        with open(config_path, 'w') as f:
            json.dump(classification_config, f, indent=2)
        saved_files.append("classification_config.json")
        print(f"    ✓ Saved classification configuration")
        
        # 9. Save Statistics
        if 'stats' in classification_data:
            stats_serializable = convert_numpy_to_python_native(classification_data['stats'])
            stats_path = os.path.join(output_dir, "dataset_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(stats_serializable, f, indent=2)
            saved_files.append("dataset_stats.json")
            print(f"    ✓ Saved dataset statistics")
        
        print(f"  ✅ Successfully saved {len(saved_files)} files for {dataset_label}")

        # Add this debugging code right before the ceramic attribute maps section 
        # in your save_classification_dataset function

        print("🔍 DEBUGGING CERAMIC ATTRIBUTE MAPS LOCATION:")
        print(f"  'ceramic_attribute_maps' in classification_data: {'ceramic_attribute_maps' in classification_data}")

        if 'stats' in classification_data:
            print(f"  'stats' exists in classification_data")
            print(f"  Keys in stats: {list(classification_data['stats'].keys())}")
            print(f"  'ceramic_attribute_maps' in stats: {'ceramic_attribute_maps' in classification_data['stats']}")
            
            if 'ceramic_attribute_maps' in classification_data['stats']:
                ceramic_maps_value = classification_data['stats']['ceramic_attribute_maps']
                print(f"  ceramic_attribute_maps value type: {type(ceramic_maps_value)}")
                print(f"  ceramic_attribute_maps value: {ceramic_maps_value}")
                
                # Check if it's a valid value (not None, not 'unknown', not empty)
                if ceramic_maps_value and ceramic_maps_value != 'unknown' and ceramic_maps_value is not None:
                    print(f"  ✅ Valid ceramic_attribute_maps found!")
                else:
                    print(f"  ❌ ceramic_attribute_maps is invalid: {ceramic_maps_value}")
        else:
            print(f"  'stats' does not exist in classification_data")

        if 'study_metadata' in classification_data:
            print(f"  'study_metadata' exists in classification_data")
            print(f"  'ceramic_attribute_maps' in study_metadata: {'ceramic_attribute_maps' in classification_data['study_metadata']}")
        else:
            print(f"  'study_metadata' does not exist in classification_data")

        print(f"  All top-level keys in classification_data: {list(classification_data.keys())}")
        
        ceramic_maps = None
        
        # Try to find ceramic attribute maps in different locations
        if 'ceramic_attribute_maps' in classification_data:
            ceramic_maps = classification_data['ceramic_attribute_maps']
        elif 'stats' in classification_data and 'ceramic_attribute_maps' in classification_data['stats']:
            ceramic_maps = classification_data['stats']['ceramic_attribute_maps']
        elif 'study_metadata' in classification_data and 'ceramic_attribute_maps' in classification_data['study_metadata']:
            ceramic_maps = classification_data['study_metadata']['ceramic_attribute_maps']
        
        if (ceramic_maps is not None and 
            ceramic_maps != 'unknown' and 
            ceramic_maps != {} and 
            ceramic_maps != []):
            try:
                ceramic_maps_serializable = convert_numpy_to_python_native(ceramic_maps)
                ceramic_maps_path = os.path.join(output_dir, "ceramic_attribute_maps.json")
                with open(ceramic_maps_path, 'w') as f:
                    json.dump(ceramic_maps_serializable, f, indent=2)
                saved_files.append("ceramic_attribute_maps.json")
                print(f"    ✓ Saved ceramic attribute maps")
            except Exception as e:
                print(f"    ⚠️ Warning: Could not save ceramic attribute maps: {e}")
        else:
            print(f"    ⚠️ No ceramic attribute maps found to save")

        return True
        
    except Exception as e:
        print(f"  ❌ Error saving classification dataset {dataset_label}: {e}")
        traceback.print_exc()
        return False


def create_classification_config(classification_data, dataset_label):
    """
    Create a configuration dictionary for the classification dataset.
    
    Args:
        classification_data (dict): Classification data
        dataset_label (str): Label for the dataset
    
    Returns:
        dict: Configuration dictionary
    """
    stats = classification_data.get('stats', {})
    
    config = {
        "dataset_label": dataset_label,
        "task_type": "node_classification",
        "target_nodes": "ceramic_nodes",
        "num_nodes": classification_data.get('num_nodes', 0),
        "num_relations": classification_data.get('num_relations', 0),
        "num_classes": len(classification_data.get('label_to_category_id', {})),
        "num_ceramic_nodes": len(classification_data.get('ceramic_node_indices', [])),
        "num_labeled_ceramics": len(classification_data.get('ceramic_labels', {})),
        "embedding_dimension": classification_data.get('embedding_dim', 0),
        "num_training_triplets": len(classification_data.get('training_triplets', [])),
        "bert_model": stats.get('bert_model_used', 'unknown'),
        "target_connection_level": stats.get('target_connection_level', 'unknown'),
        "labeling_strategy": stats.get('classification_labeling_strategy', 'unknown'),
        "created_from": {
            "study_name": stats.get('study_name', 'unknown'),
            "original_task": "link_prediction"
        }
    }
    
    return config


def save_study_classification_summary(classification_results, study_output_dir, study_name):
    """
    Save a summary of all classification datasets in the study.
    
    Args:
        classification_results (dict): Results from all études
        study_output_dir (str): Output directory for the study
        study_name (str): Name of the study
    """
    try:
        summary = {
            "study_name": study_name,
            "total_etudes": len(classification_results),
            "successful_etudes": sum(1 for result in classification_results.values() if result is not None),
            "failed_etudes": sum(1 for result in classification_results.values() if result is None),
            "etude_details": {}
        }
        
        for etude_name, result in classification_results.items():
            if result is not None:
                summary["etude_details"][etude_name] = {
                    "status": "success",
                    "num_nodes": result.get('num_nodes', 0),
                    "num_classes": len(result.get('label_to_category_id', {})),
                    "num_ceramic_nodes": len(result.get('ceramic_node_indices', [])),
                    "num_labeled_ceramics": len(result.get('ceramic_labels', {})),
                    "embedding_dim": result.get('embedding_dim', 0),
                    "bert_model": result.get('stats', {}).get('bert_model_used', 'unknown'),
                    "target_level": result.get('stats', {}).get('target_connection_level', 'unknown')
                }
            else:
                summary["etude_details"][etude_name] = {
                    "status": "failed",
                    "reason": "Processing failed or data was None"
                }
        
        summary_path = os.path.join(study_output_dir, "study_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  📋 Study summary saved: {summary_path}")
        print(f"     • Total études: {summary['total_etudes']}")
        print(f"     • Successful: {summary['successful_etudes']}")
        print(f"     • Failed: {summary['failed_etudes']}")
        
    except Exception as e:
        print(f"  ❌ Error saving study summary: {e}")


def process_all_studies_for_classification(all_study_datasets, dfs_dict, base_output_dir="./classification_data"):
    """
    Process all studies for classification tasks.
    
    Args:
        all_study_datasets (dict): Dictionary containing all study datasets
        dfs_dict (dict): Dictionary of source dataframes
        base_output_dir (str): Base output directory
    
    Returns:
        dict: All classification results
    """
    print("\n🎯 PROCESSING ALL STUDIES FOR CLASSIFICATION")
    print("=" * 60)
    
    os.makedirs(base_output_dir, exist_ok=True)
    all_classification_results = {}
    
    for study_name, study_datasets in all_study_datasets.items():
        print(f"\n📚 Processing Study: {study_name}")
        
        classification_results = process_and_save_classification_data(
            study_name, 
            study_datasets, 
            dfs_dict, 
            base_output_dir
        )
        
        all_classification_results[study_name] = classification_results
    
    # Save overall summary
    save_overall_classification_summary(all_classification_results, base_output_dir)
    
    print(f"\n✅ ALL STUDIES PROCESSED FOR CLASSIFICATION")
    print(f"📁 Results saved to: {base_output_dir}")
    
    return all_classification_results


def save_overall_classification_summary(all_classification_results, base_output_dir):
    """
    Save an overall summary of all classification processing.
    
    Args:
        all_classification_results (dict): All classification results
        base_output_dir (str): Base output directory
    """
    try:
        overall_summary = {
            "total_studies": len(all_classification_results),
            "processing_timestamp": pd.Timestamp.now().isoformat(),
            "studies": {}
        }
        
        total_successful_etudes = 0
        total_failed_etudes = 0
        
        for study_name, study_results in all_classification_results.items():
            if study_results:
                successful = sum(1 for result in study_results.values() if result is not None)
                failed = sum(1 for result in study_results.values() if result is None)
                
                overall_summary["studies"][study_name] = {
                    "total_etudes": len(study_results),
                    "successful_etudes": successful,
                    "failed_etudes": failed,
                    "success_rate": successful / len(study_results) if study_results else 0
                }
                
                total_successful_etudes += successful
                total_failed_etudes += failed
            else:
                overall_summary["studies"][study_name] = {
                    "total_etudes": 0,
                    "successful_etudes": 0,
                    "failed_etudes": 0,
                    "success_rate": 0
                }
        
        overall_summary["total_successful_etudes"] = total_successful_etudes
        overall_summary["total_failed_etudes"] = total_failed_etudes
        overall_summary["overall_success_rate"] = (
            total_successful_etudes / (total_successful_etudes + total_failed_etudes) 
            if (total_successful_etudes + total_failed_etudes) > 0 else 0
        )
        
        summary_path = os.path.join(base_output_dir, "overall_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(overall_summary, f, indent=2)
        
        print(f"\n📋 Overall Summary:")
        print(f"   • Total Studies: {overall_summary['total_studies']}")
        print(f"   • Total Successful Études: {total_successful_etudes}")
        print(f"   • Total Failed Études: {total_failed_etudes}")
        print(f"   • Overall Success Rate: {overall_summary['overall_success_rate']:.2%}")
        print(f"   • Summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"❌ Error saving overall summary: {e}")


def process_single_study_classification(rgcn_data, 
                                        dfs_dict, 
                                        study_name="single_study",
                                        output_dir="output/classification_data",
                                        auto_save=True):
    """
    Process and prepare classification data for a single RGCN study.
    
    This function takes RGCN data and converts it into a format suitable for 
    ceramic classification tasks, with options to save the results.
    
    Args:
        rgcn_data (dict): RGCN formatted data from prepare_single_study()
        dfs_dict (dict): Dictionary of source dataframes
        study_name (str): Name for this classification study
        output_dir (str): Directory to save classification data
        auto_save (bool): Whether to automatically save the results
        
    Returns:
        dict: Classification data dictionary, or None on failure
    """
    print("=" * 60)
    print(f"=== PROCESSING CLASSIFICATION DATA: {study_name.upper()} ===")
    print("=" * 60)
    
    if rgcn_data is None:
        print("❌ RGCN data is None. Cannot process classification data.")
        return None
    
    try:
        print("🔄 Adapting RGCN data for ceramic classification...")
        
        # Adapt RGCN data for classification
        classification_data = adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs_dict)
        
        if classification_data is None:
            print("❌ Failed to adapt RGCN data for classification.")
            return None
        
        # Add study metadata
        classification_data['study_metadata'] = {
            'study_name': study_name,
            'source_rgcn_study': rgcn_data.get('stats', {}).get('study_name', 'unknown'),
            'target_connection_level': rgcn_data.get('stats', {}).get('target_connection_level', 'unknown'),
            'processing_timestamp': datetime.now().isoformat(),
            'total_ceramics': len([k for k in rgcn_data.get('node_to_idx', {}).keys() if k.startswith('Ceramic_')]),
            'total_categories': len(rgcn_data.get('target_category_node_indices', [])),
            'embedding_dim': rgcn_data.get('embedding_dim', 'unknown'),
            "ceramic_attribute_maps":  rgcn_data.get('ceramic_attribute_maps', 'unknown'),
        }
        
        print("✅ Successfully adapted data for classification!")
        print(f"📊 Classification Data Stats:")
        print(f"   - Total samples: {classification_data.get('num_samples', 'unknown')}")
        print(f"   - Feature dimension: {classification_data.get('feature_dim', 'unknown')}")
        print(f"   - Number of classes: {classification_data.get('num_classes', 'unknown')}")
        
        # Auto-save if requested
        if auto_save:
            print(f"\n💾 Saving classification data to {output_dir}...")
            success = save_single_classification_dataset(
                classification_data, 
                output_dir, 
                study_name
            )
            
            if success:
                print(f"✅ Successfully saved classification data!")
            else:
                print(f"❌ Failed to save classification data.")
                return classification_data  # Return data even if save failed
        
        return classification_data
        
    except Exception as e:
        print(f"❌ Error processing classification data: {e}")
        traceback.print_exc()
        return None


def save_single_classification_dataset(classification_data, output_dir, study_name):
    """
    Save classification data for a single study with comprehensive outputs.
    
    Args:
        classification_data (dict): The classification data to save
        output_dir (str): Directory to save the data
        study_name (str): Name of the study for file naming
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import os
        import pickle
        import json
        from datetime import datetime
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main classification data as pickle
        main_file = os.path.join(output_dir, f"{study_name}_classification_data.pkl")
        with open(main_file, 'wb') as f:
            pickle.dump(classification_data, f)
        print(f"📦 Saved main data: {main_file}")
        
        # Save metadata as JSON
        metadata = classification_data.get('study_metadata', {})
        metadata_file = os.path.join(output_dir, f"{study_name}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"📋 Saved metadata: {metadata_file}")
        
        # Save summary report
        summary_file = os.path.join(output_dir, f"{study_name}_classification_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Classification Data Summary\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Study Name: {study_name}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source RGCN Study: {metadata.get('source_rgcn_study', 'unknown')}\n")
            f.write(f"Target Connection Level: {metadata.get('target_connection_level', 'unknown')}\n")
            f.write(f"\nData Statistics:\n")
            f.write(f"- Total Samples: {classification_data.get('num_samples', 'unknown')}\n")
            f.write(f"- Feature Dimension: {classification_data.get('feature_dim', 'unknown')}\n")
            f.write(f"- Number of Classes: {classification_data.get('num_classes', 'unknown')}\n")
            f.write(f"- Total Ceramics: {metadata.get('total_ceramics', 'unknown')}\n")
            f.write(f"- Total Categories: {metadata.get('total_categories', 'unknown')}\n")
            f.write(f"- Embedding Dimension: {metadata.get('embedding_dim', 'unknown')}\n")
            
            # Add class distribution if available
            if 'class_distribution' in classification_data:
                f.write(f"\nClass Distribution:\n")
                for class_idx, count in classification_data['class_distribution'].items():
                    f.write(f"- Class {class_idx}: {count} samples\n")
        
        print(f"📊 Saved summary: {summary_file}")
        
        # Save features and labels separately for easy loading
        if 'features' in classification_data and 'labels' in classification_data:
            features_file = os.path.join(output_dir, f"{study_name}_features.npy")
            labels_file = os.path.join(output_dir, f"{study_name}_labels.npy")
            
            np.save(features_file, classification_data['features'])
            np.save(labels_file, classification_data['labels'])
            
            print(f"🔢 Saved features: {features_file}")
            print(f"🏷️ Saved labels: {labels_file}")

        ceramic_maps = None
        if 'study_metadata' in classification_data and 'ceramic_attribute_maps' in classification_data['study_metadata']:
            ceramic_maps = classification_data['study_metadata']['ceramic_attribute_maps']
        elif 'stats' in classification_data and 'ceramic_attribute_maps' in classification_data['stats']:
            ceramic_maps = classification_data['stats']['ceramic_attribute_maps']

        if ceramic_maps and ceramic_maps != 'unknown':
            try:
                ceramic_maps_file = os.path.join(output_dir, f"{study_name}_ceramic_attribute_maps.json")
                with open(ceramic_maps_file, 'w') as f:
                    json.dump(ceramic_maps, f, indent=4)
                print(f"🗺️ Saved ceramic attribute maps: {ceramic_maps_file}")
            except Exception as e:
                print(f"❌ Error saving ceramic attribute maps: {e}")
        else:
            print("⚠️ No ceramic attribute maps found to save")


        return True
    
        
    except Exception as e:
        print(f"❌ Error saving classification data: {e}")
        traceback.print_exc()
        return False
