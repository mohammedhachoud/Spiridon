import os
import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
import traceback

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
        "relation_mapping": relation_to_idx,
        "idx_to_relation": idx_to_relation,
        "training_triplets": original_training_triplets,
        "node_embeddings": original_embeddings,
        "embedding_dim": embedding_dim,
        "ceramic_node_indices": ceramic_node_indices,
        "ceramic_labels": ceramic_labels,
        "label_to_category_id": label_id_to_category_id,
        "label_to_category_name": label_id_to_category_name,
        "stats": stats
    }

    print("\n--- Adaptation for Classification Complete ---")
    print(f"  Final ceramic nodes with labels: {len(ceramic_labels)}")
    print(f"  Total classes: {label_counter}")
    
    return classification_data

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



def generate_relation_embeddings(classification_data, bert_model_name="all-MiniLM-L6-v2", output_dir=None):
    """
    Generate relation embeddings using sentence BERT for TransGNN.
    
    Args:
        classification_data (dict): Classification data containing relation_mapping
        bert_model_name (str): Name of the BERT model to use
        output_dir (str): Directory to save the embeddings (optional)
    
    Returns:
        dict: Updated classification data with relation embeddings
    """
    print(f"\n🔗 Generating Relation Embeddings with {bert_model_name}")
    print("=" * 60)
    
    try:
        # Load the BERT model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"📱 Using device: {device}")
        
        bert_model = SentenceTransformer(bert_model_name, device=device)
        embedding_dim = bert_model.get_sentence_embedding_dimension()
        print(f"📊 Embedding dimension: {embedding_dim}")
        
        # Get relation mapping
        relation_mapping = classification_data.get('relation_mapping', {})
        if not relation_mapping:
            print("❌ No relation mapping found in classification data")
            return classification_data
        
        print(f"🔍 Found {len(relation_mapping)} relations to embed")
        
        # Prepare relation texts for embedding
        relation_texts = []
        relation_ids = []
        
        print(f"📝 Preparing relation names for embedding:")
        for relation_name, relation_id in relation_mapping.items():
            # Use the cleaned relation name directly for embedding
            relation_text = prepare_relation_text(relation_name)
            relation_texts.append(relation_text)
            relation_ids.append(relation_id)
            print(f"  Relation {relation_id}: '{relation_name}' -> '{relation_text}'")
        
        # Show some examples of what will be embedded
        print(f"\n🎯 Examples of relation texts that will be embedded:")
        for i, text in enumerate(relation_texts[:5]):  # Show first 5 examples
            print(f"  {i+1}. '{text}'")
        if len(relation_texts) > 5:
            print(f"  ... and {len(relation_texts) - 5} more relations")
        
        # Generate embeddings
        print(f"\n🚀 Generating embeddings for {len(relation_texts)} relations...")
        
        # Encode all relation texts at once (more efficient)
        relation_embeddings = bert_model.encode(
            relation_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        print(f"✅ Generated embeddings with shape: {relation_embeddings.shape}")
        
        # Create relation embeddings array ordered by relation ID
        max_relation_id = max(relation_ids) if relation_ids else 0
        edge_embeddings = np.zeros((max_relation_id + 1, embedding_dim))
        
        for i, relation_id in enumerate(relation_ids):
            edge_embeddings[relation_id] = relation_embeddings[i]
        
        # Add to classification data
        classification_data['edge_embeddings'] = edge_embeddings
        classification_data['edge_embedding_dim'] = embedding_dim
        classification_data['relation_texts'] = dict(zip(relation_ids, relation_texts))
        
        # Save embeddings if output directory is provided
        if output_dir:
            embeddings_path = os.path.join(output_dir, 'edge_embeddings.npy')
            np.save(embeddings_path, edge_embeddings)
            print(f"💾 Saved edge embeddings to: {embeddings_path}")
            
            # Save relation texts for reference
            relation_texts_path = os.path.join(output_dir, 'relation_texts.json')
            with open(relation_texts_path, 'w') as f:
                json.dump(classification_data['relation_texts'], f, indent=2)
            print(f"💾 Saved relation texts to: {relation_texts_path}")
            
            # Update classification config
            config_path = os.path.join(output_dir, 'classification_config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                config['edge_embedding_dim'] = embedding_dim
                config['bert_model_name'] = bert_model_name
                config['has_edge_embeddings'] = True
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                print(f"💾 Updated classification config with edge embedding info")
        
        # Clean up
        del bert_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✅ Successfully generated relation embeddings!")
        return classification_data
        
    except Exception as e:
        print(f"❌ Error generating relation embeddings: {e}")
        traceback.print_exc()
        return classification_data

def prepare_relation_text(relation_name):
    """
    Prepare relation name for embedding by cleaning it but keeping the semantic meaning.
    
    Args:
        relation_name (str): The relation name
    
    Returns:
        str: Cleaned relation name for embedding
    """
    # Simply clean the relation name by replacing underscores with spaces
    # This preserves the semantic meaning of the relation names
    cleaned_name = relation_name.replace('_', ' ').replace('-', ' ')
    return cleaned_name


def update_classification_pipeline_with_relations(classification_data, bert_model_name="all-MiniLM-L6-v2"):
    """
    Update existing classification data with relation embeddings.
    
    Args:
        classification_data (dict): Existing classification data
        bert_model_name (str): BERT model name for consistency
    
    Returns:
        dict: Updated classification data with relation embeddings
    """
    print(f"\n🔄 Updating classification pipeline with relation embeddings...")
    
    if 'edge_embeddings' in classification_data:
        print("✅ Relation embeddings already exist in classification data")
        return classification_data
    
    return generate_relation_embeddings(classification_data, bert_model_name)

def save_classification_dataset_with_relations(classification_data, output_dir, dataset_name, bert_model_name="all-MiniLM-L6-v2"):
    """
    Enhanced version of save_classification_dataset that includes relation embeddings.
    
    Args:
        classification_data (dict): Classification data
        output_dir (str): Output directory
        dataset_name (str): Name of the dataset
        bert_model_name (str): BERT model name
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"\n💾 Saving classification dataset with relation embeddings: {dataset_name}")
    
    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate relation embeddings if not present
        if 'edge_embeddings' not in classification_data:
            classification_data = generate_relation_embeddings(
                classification_data, 
                bert_model_name, 
                output_dir
            )

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
        classification_config = create_classification_config(classification_data, dataset_name)
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
        
        print(f"  ✅ Successfully saved {len(saved_files)} files for {dataset_name}")

        
        # Save all the standard classification files
        # (assuming the original save_classification_dataset function handles these)
        
        # Save edge embeddings
        if 'edge_embeddings' in classification_data:
            edge_embeddings_path = os.path.join(output_dir, 'edge_embeddings.npy')
            np.save(edge_embeddings_path, classification_data['edge_embeddings'])
            print(f"💾 Saved edge embeddings to: {edge_embeddings_path}")
        
        # Save relation texts for reference
        if 'relation_texts' in classification_data:
            relation_texts_path = os.path.join(output_dir, 'relation_texts.json')
            with open(relation_texts_path, 'w') as f:
                json.dump(classification_data['relation_texts'], f, indent=2)
            print(f"💾 Saved relation texts to: {relation_texts_path}")
        
        # Update/create classification config
        config_path = os.path.join(output_dir, 'classification_config.json')
        config = {
            'dataset_name': dataset_name,
            'bert_model_name': bert_model_name,
            'node_embedding_dim': classification_data.get('node_embedding_dim', 'unknown'),
            'edge_embedding_dim': classification_data.get('edge_embedding_dim', 'unknown'),
            'has_node_embeddings': 'node_embeddings' in classification_data,
            'has_edge_embeddings': 'edge_embeddings' in classification_data,
            'num_relations': len(classification_data.get('relation_mapping', {})),
            'num_nodes': len(classification_data.get('node_mapping', {})),
            'created_for_transgnn': True
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"💾 Updated classification config: {config_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving classification dataset with relations: {e}")
        traceback.print_exc()
        return False

def process_all_studies_for_classification_with_relations(all_study_datasets, dfs_dict, base_output_dir="./classification_data", bert_model_name="all-MiniLM-L6-v2"):
    """
    Enhanced version that processes all studies with both node and edge embeddings for TransGNN.
    
    Args:
        all_study_datasets (dict): Dictionary containing all study datasets
        dfs_dict (dict): Dictionary of source dataframes
        base_output_dir (str): Base output directory
        bert_model_name (str): BERT model name for embeddings
    
    Returns:
        dict: All classification results with relation embeddings
    """
    print("\n🎯 PROCESSING ALL STUDIES FOR CLASSIFICATION WITH RELATION EMBEDDINGS")
    print("=" * 70)
    
    os.makedirs(base_output_dir, exist_ok=True)
    all_classification_results = {}
    
    for study_name, study_datasets in all_study_datasets.items():
        print(f"\n📚 Processing Study: {study_name}")
        
        classification_results = process_and_save_classification_data_with_relations(
            study_name, 
            study_datasets, 
            dfs_dict, 
            base_output_dir,
            bert_model_name
        )
        
        all_classification_results[study_name] = classification_results
    
    # Save overall summary
    save_overall_classification_summary_with_relations(all_classification_results, base_output_dir)
    
    print(f"\n✅ ALL STUDIES PROCESSED FOR CLASSIFICATION WITH RELATION EMBEDDINGS")
    print(f"📁 Results saved to: {base_output_dir}")
    
    return all_classification_results

def process_and_save_classification_data_with_relations(study_name, study_datasets_dict, dfs_dict, base_output_dir, bert_model_name="all-MiniLM-L6-v2"):
    """
    Enhanced version that processes and saves classification data with relation embeddings.
    
    Args:
        study_name (str): Name of the study
        study_datasets_dict (dict): Dictionary containing étude datasets
        dfs_dict (dict): Dictionary of source dataframes
        base_output_dir (str): Base directory for saving classification data
        bert_model_name (str): BERT model name
    
    Returns:
        dict: Classification results with relation embeddings
    """
    print(f"\n🎯 Processing Classification Data with Relations for Study: {study_name}")
    print("=" * 70)
    
    if not study_datasets_dict:
        print(f"❌ No datasets found for study '{study_name}'. Skipping.")
        return {}
    
    study_output_dir = os.path.join(base_output_dir, study_name)
    os.makedirs(study_output_dir, exist_ok=True)
    
    classification_results = {}
    
    for etude_name, rgcn_data in study_datasets_dict.items():
        print(f"\n📊 Processing Étude: {etude_name}")
        print("-" * 50)
        
        if rgcn_data is None:
            print(f"⚠️ RGCN data is None for {etude_name}. Skipping.")
            classification_results[etude_name] = None
            continue
        
        try:
            # Adapt RGCN data for classification (using your existing function)
            classification_data = adapt_rgcn_data_for_ceramic_classification(rgcn_data, dfs_dict)
            
            if classification_data is None:
                print(f"❌ Failed to adapt data for classification: {etude_name}")
                classification_results[etude_name] = None
                continue
            
            # Generate relation embeddings
            classification_data = generate_relation_embeddings(
                classification_data, 
                bert_model_name
            )
            
            # Save classification data with relations
            etude_output_dir = os.path.join(study_output_dir, etude_name)
            success = save_classification_dataset_with_relations(
                classification_data,
                etude_output_dir,
                f"{study_name}_{etude_name}",
                bert_model_name
            )
            
            if success:
                classification_results[etude_name] = classification_data
                print(f"✅ Successfully processed and saved with relations: {etude_name}")
            else:
                classification_results[etude_name] = None
                print(f"❌ Failed to save classification data with relations: {etude_name}")
                
        except Exception as e:
            print(f"❌ Error processing {etude_name}: {e}")
            traceback.print_exc()
            classification_results[etude_name] = None
    
    # Save study summary
    save_study_classification_summary_with_relations(classification_results, study_output_dir, study_name)
    
    print(f"\n✅ Completed processing classification data with relations for study: {study_name}")
    print(f"📁 Results saved to: {study_output_dir}")
    
    return classification_results

def save_overall_classification_summary_with_relations(all_classification_results, base_output_dir):
    """
    Save overall summary including relation embedding information.
    """
    print(f"\n📋 Saving overall classification summary with relations...")
    
    summary = {
        'total_studies': len(all_classification_results),
        'studies': {}
    }
    
    for study_name, study_results in all_classification_results.items():
        study_summary = {
            'total_etudes': len(study_results),
            'successful_etudes': sum(1 for r in study_results.values() if r is not None),
            'failed_etudes': sum(1 for r in study_results.values() if r is None),
            'etudes': {}
        }
        
        for etude_name, classification_data in study_results.items():
            if classification_data is not None:
                study_summary['etudes'][etude_name] = {
                    'has_node_embeddings': 'node_embeddings' in classification_data,
                    'has_edge_embeddings': 'edge_embeddings' in classification_data,
                    'node_embedding_dim': classification_data.get('node_embedding_dim', 'unknown'),
                    'edge_embedding_dim': classification_data.get('edge_embedding_dim', 'unknown'),
                    'num_relations': len(classification_data.get('relation_mapping', {})),
                    'num_nodes': len(classification_data.get('node_mapping', {})),
                    'ready_for_transgnn': (
                        'node_embeddings' in classification_data and 
                        'edge_embeddings' in classification_data
                    )
                }
            else:
                study_summary['etudes'][etude_name] = None
        
        summary['studies'][study_name] = study_summary
    
    summary_path = os.path.join(base_output_dir, 'classification_summary_with_relations.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Saved overall summary to: {summary_path}")

def save_study_classification_summary_with_relations(classification_results, study_output_dir, study_name):
    """
    Save study-specific summary including relation embedding information.
    """
    print(f"📋 Saving study summary with relations for: {study_name}")
    
    summary = {
        'study_name': study_name,
        'total_etudes': len(classification_results),
        'successful_etudes': sum(1 for r in classification_results.values() if r is not None),
        'failed_etudes': sum(1 for r in classification_results.values() if r is None),
        'etudes': {}
    }
    
    for etude_name, classification_data in classification_results.items():
        if classification_data is not None:
            summary['etudes'][etude_name] = {
                'has_node_embeddings': 'node_embeddings' in classification_data,
                'has_edge_embeddings': 'edge_embeddings' in classification_data,
                'node_embedding_dim': classification_data.get('node_embedding_dim', 'unknown'),
                'edge_embedding_dim': classification_data.get('edge_embedding_dim', 'unknown'),
                'num_relations': len(classification_data.get('relation_mapping', {})),
                'num_nodes': len(classification_data.get('node_mapping', {})),
                'ready_for_transgnn': (
                    'node_embeddings' in classification_data and 
                    'edge_embeddings' in classification_data
                )
            }
        else:
            summary['etudes'][etude_name] = None
    
    summary_path = os.path.join(study_output_dir, f'{study_name}_summary_with_relations.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Saved study summary to: {summary_path}")