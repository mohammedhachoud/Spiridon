import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import traceback
from collections import Counter, defaultdict


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Seeded everything with seed {seed}")

# --- Helper Function to plot training history ---
def plot_training_history(log_dir, study_name):
    """
    Plots training history from the CSVLogger output.
    """
    metrics_file = None
    version_dirs = [d for d in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, d)) and d.startswith('version_')]
    if version_dirs:
        version_dirs.sort()
        latest_version_dir = os.path.join(log_dir, version_dirs[-1])
        metrics_file = os.path.join(latest_version_dir, 'metrics.csv')
        print(f"  Looking for metrics file in: {metrics_file}")
    else:
        metrics_file_direct = os.path.join(log_dir, 'metrics.csv')
        if os.path.exists(metrics_file_direct):
            metrics_file = metrics_file_direct
            print(f"  Looking for metrics file directly in: {metrics_file}")


    if not metrics_file or not os.path.exists(metrics_file):
        print(f"  Metrics file not found after checking standard paths under {log_dir}. Skipping plot.")
        return

    try:
        metrics_df = pd.read_csv(metrics_file)
        if 'epoch' not in metrics_df.columns:
             print("  Warning: 'epoch' column not found in metrics.csv. Plotting against 'step' instead.")
             metrics_df['epoch'] = metrics_df['step']
             x_label = 'Step'
        else:
             x_label = 'Epoch'

        train_loss = metrics_df.dropna(subset=['train_loss']).groupby('epoch')['train_loss'].last()
        val_loss = metrics_df.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].last()
        train_acc = metrics_df.dropna(subset=['train_acc']).groupby('epoch')['train_acc'].last()
        val_acc = metrics_df.dropna(subset=['val_acc']).groupby('epoch')['val_acc'].last()
        test_acc_row = metrics_df.dropna(subset=['test_acc'])


        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        if not train_loss.empty:
             plt.plot(train_loss.index, train_loss.values, label='Train Loss')
        if not val_loss.empty:
            plt.plot(val_loss.index, val_loss.values, label='Validation Loss')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.title(f'{study_name} Training & Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        if not train_acc.empty:
            plt.plot(train_acc.index, train_acc.values, label='Train Accuracy')
        if not val_acc.empty:
            plt.plot(val_acc.index, val_acc.values, label='Validation Accuracy')
        if not test_acc_row.empty:
             # Get the last test accuracy logged
             test_acc_value = test_acc_row['test_acc'].iloc[-1]
             plt.axhline(y=test_acc_value, color='r', linestyle='--', label=f'Test Accuracy ({test_acc_value:.4f})')
             if 'epoch' in test_acc_row.columns:
                  test_epoch = test_acc_row['epoch'].iloc[-1]
                  plt.plot(test_epoch, test_acc_value, 'ro')

        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.title(f'{study_name} Training & Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        # Save the plot
        plot_path = os.path.join(log_dir, f'{study_name}_training_history.png')
        plt.savefig(plot_path)
        print(f"  Saved training history plot to {plot_path}")
        plt.show() # Display the plot

    except Exception as e:
        print(f"❌ Error plotting training history for '{study_name}' from {metrics_file}: {e}")
        traceback.print_exc()


    if not metrics_file or not os.path.exists(metrics_file):
        print(f"  Metrics file not found after checking standard paths under {log_dir}. Skipping plot.")
        return

    try:
        metrics_df = pd.read_csv(metrics_file)
        if 'epoch' not in metrics_df.columns:
             print("  Warning: 'epoch' column not found in metrics.csv. Plotting against 'step' instead.")
             if 'step' in metrics_df.columns:
                 metrics_df['epoch'] = metrics_df.groupby((metrics_df['step'] != metrics_df['step'].shift()).cumsum() -1).ngroup()

             if 'epoch' not in metrics_df.columns : #if still not there
                  metrics_df['epoch'] = metrics_df.index # Fallback to index
             x_label = 'Step/Index'
        else:
             x_label = 'Epoch'


        train_loss = metrics_df.dropna(subset=['train_loss']).groupby('epoch')['train_loss'].last()
        val_loss = metrics_df.dropna(subset=['val_loss']).groupby('epoch')['val_loss'].last()
        train_acc = metrics_df.dropna(subset=['train_acc']).groupby('epoch')['train_acc'].last()
        val_acc = metrics_df.dropna(subset=['val_acc']).groupby('epoch')['val_acc'].last()
        test_acc_row = metrics_df.dropna(subset=['test_acc'])


        plt.figure(figsize=(12, 5))

        # Plot Loss
        plt.subplot(1, 2, 1)
        if not train_loss.empty:
             plt.plot(train_loss.index, train_loss.values, label='Train Loss')
        if not val_loss.empty:
            plt.plot(val_loss.index, val_loss.values, label='Validation Loss')
        plt.xlabel(x_label)
        plt.ylabel('Loss')
        plt.title(f'{study_name} Training & Validation Loss')
        plt.legend()
        plt.grid(True)

        # Plot Accuracy
        plt.subplot(1, 2, 2)
        if not train_acc.empty:
            plt.plot(train_acc.index, train_acc.values, label='Train Accuracy')
        if not val_acc.empty:
            plt.plot(val_acc.index, val_acc.values, label='Validation Accuracy')
        if not test_acc_row.empty:
             test_acc_value = test_acc_row['test_acc'].iloc[-1]
             plt.axhline(y=test_acc_value, color='r', linestyle='--', label=f'Test Accuracy ({test_acc_value:.4f})')
             if 'epoch' in test_acc_row.columns: # Check if 'epoch' column exists
                  test_epoch = test_acc_row['epoch'].iloc[-1]
                  plt.plot(test_epoch, test_acc_value, 'ro')


        plt.xlabel(x_label)
        plt.ylabel('Accuracy')
        plt.title(f'{study_name} Training & Validation Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        # Save the plot to the specific study's log directory
        plot_save_dir = os.path.join(base_output_dir, "lightning_logs", study_name, "version_0" if not version_dirs else version_dirs[-1] )
        os.makedirs(plot_save_dir, exist_ok=True)
        plot_path = os.path.join(plot_save_dir, f'{study_name}_training_history.png')

        plt.savefig(plot_path)
        print(f"  Saved training history plot to {plot_path}")
        # plt.show() # In scripts, plt.show() might block or not work in non-interactive envs.
        plt.close()


    except Exception as e:
        print(f"❌ Error plotting training history for '{study_name}' from {metrics_file}: {e}")
        traceback.print_exc()

def plot_mlp_training_history(loss_curve, validation_scores, test_accuracy, scenario_name, output_dir):
    """
    Plot MLP training history for scikit-learn MLPClassifier.
    
    Args:
        loss_curve: List of loss values during training
        validation_scores: List of validation scores during training
        test_accuracy: Final test accuracy
        scenario_name: Name of the scenario for plot title
        output_dir: Directory to save the plot
    """
    try:
        plt.figure(figsize=(12, 5))
        
        # Plot Loss
        plt.subplot(1, 2, 1)
        if loss_curve is not None and len(loss_curve) > 0:
            plt.plot(range(1, len(loss_curve) + 1), loss_curve, 'b-', label='Training Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'{scenario_name} - Training Loss')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'No loss curve available', ha='center', va='center', transform=plt.gca().transAxes)
            plt.title(f'{scenario_name} - Training Loss (Not Available)')
        
        # Plot Validation Accuracy
        plt.subplot(1, 2, 2)
        if validation_scores is not None and len(validation_scores) > 0:
            plt.plot(range(1, len(validation_scores) + 1), validation_scores, 'g-', label='Validation Accuracy')
            
        # Add test accuracy as horizontal line
        if test_accuracy is not None:
            plt.axhline(y=test_accuracy, color='r', linestyle='--', label=f'Test Accuracy ({test_accuracy:.4f})')
            
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title(f'{scenario_name} - Validation & Test Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(output_dir, f'{scenario_name}_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  Saved MLP training history plot to {plot_path}")
        plt.close()
        
    except Exception as e:
        print(f"❌ Error plotting MLP training history for '{scenario_name}': {e}")
        traceback.print_exc()

# Place this class at the top of your script or in a utils file.


class CategoryHierarchy:
    """
    Manages the category hierarchy, providing levels, parent-child maps,
    and methods to find ancestors.
    """
    def __init__(self, tech_cat_df):
        print("Initializing CategoryHierarchy...")
        if not isinstance(tech_cat_df, pd.DataFrame) or tech_cat_df.empty:
            raise ValueError("tech_cat_df must be a non-empty Pandas DataFrame.")
        
        df = tech_cat_df.copy()
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        df['inherit_from'] = pd.to_numeric(df['inherit_from'], errors='coerce')
        df.dropna(subset=['id'], inplace=True)
        df['id'] = df['id'].astype(int)
        
        self.parent_map = df.set_index('id')['inherit_from'].to_dict()
        self.cat_names = df.set_index('id')['cat_name'].to_dict()
        
        self.roots = set()
        self.levels = {}
        self.ancestor_map = {} # {cat_id: [root, L1_ancestor, L2_ancestor, ...]}

        self._build_hierarchy()
        print(f"Hierarchy built. Found {len(self.roots)} roots. Processed {len(self.levels)} categories.")

    def _build_hierarchy(self):
        memo = {}

        def find_path_and_level(cat_id):
            if cat_id in memo:
                return memo[cat_id]
            
            if cat_id not in self.parent_map:
                memo[cat_id] = ([cat_id], 0)
                return [cat_id], 0

            path_to_root = []  # Will build path from current node to root
            current_id = cat_id
            visited = set()  # Track visited nodes to detect cycles
            
            # Build path from current node up to root
            while current_id is not None:
                if current_id in visited:
                    print(f"  ⚠️ Cycle detected in hierarchy at ID {current_id}. Breaking path.")
                    break
                    
                visited.add(current_id)
                path_to_root.append(current_id)
                
                # Check if we've already computed this node's path
                if current_id in memo:
                    # Remove the current node from path_to_root since it's already in memo
                    path_to_root.pop()
                    # Get the pre-computed path and extend our path
                    parent_path, _ = memo[current_id]
                    full_path = parent_path + path_to_root[::-1]  # Reverse path_to_root
                    break
                
                # Get parent
                parent_id = self.parent_map.get(current_id)
                if parent_id is None or pd.isna(parent_id) or parent_id == 0 or int(parent_id) == -1:
                    # Reached root
                    full_path = path_to_root[::-1]  # Reverse to get root -> ... -> current
                    break
                    
                current_id = int(parent_id)
            else:
                # If we exit the while loop without breaking, we have the full path
                full_path = path_to_root[::-1]  # Reverse to get root -> ... -> current
            
            # Store results for all nodes in this newly computed path
            for i, node_id in enumerate(full_path):
                if node_id not in memo:
                    # The path from root to this node
                    path_to_node = full_path[:i+1]
                    memo[node_id] = (path_to_node, i)

            return memo[cat_id]

        for cat_id in self.parent_map.keys():
            path, level = find_path_and_level(cat_id)
            self.levels[cat_id] = level
            self.ancestor_map[cat_id] = path
            if level == 0:
                self.roots.add(cat_id)

    def get_level(self, cat_id):
        return self.levels.get(int(cat_id))

    def get_root(self, cat_id):
        ancestors = self.ancestor_map.get(int(cat_id))
        return ancestors[0] if ancestors else None
        
    def get_ancestor_at_level(self, cat_id, target_level):
        """
        Finds the ancestor of a category at a specific target level.
        Returns the category ID of the ancestor.
        """
        cat_id = int(cat_id)
        current_level = self.get_level(cat_id)
        if current_level is None or target_level < 0:
            return None
        
        # If the category itself is at or above the target level, it's its own ancestor at that level.
        if current_level < target_level:
            return None # Cannot find an ancestor at a deeper level
        
        ancestors = self.ancestor_map.get(cat_id)
        if ancestors and len(ancestors) > target_level:
            return ancestors[target_level]
            
        return None

    def get_full_path(self, cat_id):
        """
        Returns the full path from root to the given category as a list of category IDs.
        """
        return self.ancestor_map.get(int(cat_id), [])
    
    def get_path_names(self, cat_id):
        """
        Returns the full path from root to the given category as a list of category names.
        """
        path_ids = self.get_full_path(cat_id)
        return [self.cat_names.get(cat_id, f"Unknown({cat_id})") for cat_id in path_ids]
    
    def get_path_string(self, cat_id, separator=" -> "):
        """
        Returns the full path as a formatted string with category names.
        """
        path_names = self.get_path_names(cat_id)
        return separator.join(path_names)


def export_hierarchy_to_csv(hierarchy, output_filepath):
    """
    Exports the computed category hierarchy to a CSV file for easy inspection.
    """
    print(f"\n--- Exporting Category Hierarchy to CSV ---")
    if not hasattr(hierarchy, 'levels') or not hierarchy.levels:
        print("  ❌ Error: Hierarchy object is not initialized or is empty. Cannot export.")
        return

    # Prepare data for the DataFrame
    hierarchy_data = []
    
    # Iterate through all categories known to the hierarchy
    for cat_id in sorted(hierarchy.levels.keys()):
        name = hierarchy.cat_names.get(cat_id, "Unknown Name")
        level = hierarchy.levels.get(cat_id)
        path_ids = hierarchy.ancestor_map.get(cat_id, [])
        
        # The root is the first element in the path
        root_id = path_ids[0] if path_ids else None
        root_name = hierarchy.cat_names.get(root_id, "Unknown Root") if root_id else "Unknown"
        
        # Format the path for readability (IDs)
        path_ids_str = " -> ".join(map(str, path_ids))
        
        # Format the path with names
        path_names = [hierarchy.cat_names.get(pid, f"Unknown({pid})") for pid in path_ids]
        path_names_str = " -> ".join(path_names)
        
        hierarchy_data.append({
            'id': cat_id,
            'name': name,
            'level': level,
            'root_id': root_id,
            'root_name': root_name,
            'path_ids': path_ids_str,
            'path_names': path_names_str
        })

    if not hierarchy_data:
        print("  ⚠️ Warning: No data to export.")
        return

    try:
        # Create a DataFrame and save to CSV
        df = pd.DataFrame(hierarchy_data)
        df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
        print(f"  ✅ Successfully exported hierarchy details for {len(df)} categories to:")
        print(f"     '{output_filepath}'")
        return df

    except Exception as e:
        print(f"  ❌ Error exporting hierarchy to CSV: {e}")
        traceback.print_exc()
        return None

def demonstrate_hierarchy_usage(tech_cat_df):
    """
    Demonstrates how to use the CategoryHierarchy class to get paths for categories.
    """
    print("=== Category Hierarchy Path Extraction Demo ===\n")
    
    # Initialize the hierarchy
    hierarchy = CategoryHierarchy(tech_cat_df)
    
    print(f"\n=== Hierarchy Summary ===")
    print(f"Total categories: {len(hierarchy.levels)}")
    print(f"Root categories: {len(hierarchy.roots)}")
    print(f"Root IDs: {sorted(hierarchy.roots)}")
    
    # Show some example paths
    print(f"\n=== Example Category Paths ===")
    sample_categories = sorted(hierarchy.levels.keys())[:10]  # First 10 categories
    
    for cat_id in sample_categories:
        level = hierarchy.get_level(cat_id)
        path_ids = hierarchy.get_full_path(cat_id)
        path_names = hierarchy.get_path_names(cat_id)
        path_string = hierarchy.get_path_string(cat_id)
        
        print(f"\nCategory ID: {cat_id}")
        print(f"  Name: {hierarchy.cat_names.get(cat_id, 'Unknown')}")
        print(f"  Level: {level}")
        print(f"  Path (IDs): {' -> '.join(map(str, path_ids))}")
        print(f"  Path (Names): {path_string}")
    
    return hierarchy


def get_feature_display_label(feature_id, ontology_df):
    try:
        row = ontology_df[ontology_df['id'] == str(feature_id)]
        if not row.empty:
            return str(row.iloc[0].get('Nom complet', f"Feature ID: {feature_id}"))
    except Exception: pass
    return f"Feature ID: {feature_id}"



def get_root_and_level1_ancestor(feature_id, feature_df):
     try:
         row = feature_df[feature_df['id'] == str(feature_id)]
         if not row.empty:
             parent_id = row.iloc[0].get('feature_parent')
             if parent_id and parent_id != "-1" and pd.notna(parent_id):
                 parent_row = feature_df[feature_df['id'] == str(parent_id)]
                 if not parent_row.empty:
                     parent_name = parent_row.iloc[0].get('Nom complet', f"ID: {parent_id}")
                     if "surface" in parent_name.lower(): return parent_name, parent_name
                     return parent_name, parent_name # Placeholder
             else:
                 return row.iloc[0].get('Nom complet', f"ID: {feature_id}"), None
     except Exception: pass
     return f"Root_Unknown_{feature_id}", f"Ancestor_Unknown_{feature_id}"



def get_feature_parent_relation_label(child_feature_data):
     relation_val = child_feature_data.get('Relation', '')
     nom_abrege_val = child_feature_data.get('Nom abrégé avec la relation', '')
     relation_val = "" if pd.isna(relation_val) else str(relation_val).strip()
     nom_abrege_val = "" if pd.isna(nom_abrege_val) else str(nom_abrege_val).strip()
     return "IS_A"


def analyze_ceramic_distribution_by_hierarchy_level(dfs):
    """
    Analyzes the distribution of ceramics across different levels of the category hierarchy.
    
    Returns:
    - hierarchy_info: Dict with complete hierarchy structure
    - ceramic_level_distribution: Dict showing ceramic counts at each level for each root
    - summary_stats: Overall statistics about the distribution
    """
    
    print("Analyzing ceramic distribution across hierarchy levels...")
    
    tech_cat = dfs['tech_cat'].copy()
    ceramic_summary = dfs['ceramic'].copy()
    
    # Standardize IDs
    try:
        tech_cat['id'] = pd.to_numeric(tech_cat['id'], errors='coerce').dropna().astype(int)
        tech_cat['parent_id'] = pd.to_numeric(tech_cat['inherit_from'], errors='coerce').fillna(-1).astype(int)
        tech_cat.set_index('id', inplace=True)
        
        ceramic_summary['id'] = pd.to_numeric(ceramic_summary['id'], errors='coerce').dropna().astype(int)
        ceramic_summary['tech_cat_id'] = pd.to_numeric(ceramic_summary['tech_cat_id'], errors='coerce')
        ceramic_summary.dropna(subset=['tech_cat_id'], inplace=True)
        ceramic_summary['tech_cat_id'] = ceramic_summary['tech_cat_id'].astype(int)
        
    except Exception as e:
        print(f"Error standardizing IDs: {e}")
        traceback.print_exc()
        return None, None, None
    
    # Build hierarchy with levels
    category_info = {}
    
    def calculate_level_and_root(cat_id, visited=None):
        """Calculate the level and root category for a given category ID"""
        if visited is None:
            visited = set()
        
        if cat_id in visited:
            print(f"Warning: Circular reference detected for category {cat_id}")
            return None, None
        
        if cat_id in category_info:
            return category_info[cat_id]['level'], category_info[cat_id]['root']
        
        if cat_id not in tech_cat.index:
            print(f"Warning: Category ID {cat_id} not found in tech_cat")
            return None, None
        
        visited.add(cat_id)
        parent_id = tech_cat.loc[cat_id, 'parent_id']
        
        # If it's a root category (parent is -1 or doesn't exist)
        if parent_id == -1 or parent_id not in tech_cat.index:
            level = 0
            root = cat_id
        else:
            parent_level, parent_root = calculate_level_and_root(parent_id, visited.copy())
            if parent_level is None:
                return None, None
            level = parent_level + 1
            root = parent_root
        
        # Store the information
        category_info[cat_id] = {
            'level': level,
            'root': root,
            'parent': parent_id if parent_id != -1 else None,
            'name': tech_cat.loc[cat_id, 'cat_name']
        }
        
        return level, root
    
    # Calculate levels and roots for all categories
    print("Calculating hierarchy levels...")
    for cat_id in tech_cat.index:
        calculate_level_and_root(cat_id)
    
    # Group categories by root and level
    hierarchy_structure = defaultdict(lambda: defaultdict(list))
    root_names = {}
    
    for cat_id, info in category_info.items():
        root_id = info['root']
        level = info['level']
        hierarchy_structure[root_id][level].append({
            'id': cat_id,
            'name': info['name']
        })
        if level == 0:  # Store root names
            root_names[root_id] = info['name']
    
    # Count ceramics at each level for each root category
    print("Counting ceramics at each hierarchy level...")
    ceramic_level_distribution = defaultdict(lambda: defaultdict(int))
    ceramic_assignments = defaultdict(list)  # Track which ceramics are assigned where
    
    for _, row in ceramic_summary.iterrows():
        id = row['id']
        direct_cat_id = row['tech_cat_id']
        
        if direct_cat_id in category_info:
            cat_info = category_info[direct_cat_id]
            root_id = cat_info['root']
            level = cat_info['level']
            
            ceramic_level_distribution[root_id][level] += 1
            ceramic_assignments[root_id].append({
                'id': id,
                'assigned_category': direct_cat_id,
                'assigned_category_name': cat_info['name'],
                'level': level
            })
    
    # Create summary statistics
    summary_stats = {
        'total_categories': len(category_info),
        'total_root_categories': len([c for c in category_info.values() if c['level'] == 0]),
        'max_depth': max([c['level'] for c in category_info.values()]) if category_info else 0,
        'total_ceramics_analyzed': len(ceramic_summary),
        'ceramics_with_valid_categories': sum([sum(levels.values()) for levels in ceramic_level_distribution.values()])
    }
    
    # Print detailed results
    print("\n" + "="*80)
    print("📊 CERAMIC DISTRIBUTION ACROSS HIERARCHY LEVELS")
    print("="*80)
    
    print(f"\n📈 Overall Statistics:")
    print(f"  • Total categories: {summary_stats['total_categories']}")
    print(f"  • Root categories: {summary_stats['total_root_categories']}")
    print(f"  • Maximum hierarchy depth: {summary_stats['max_depth']}")
    print(f"  • Total ceramics analyzed: {summary_stats['total_ceramics_analyzed']}")
    print(f"  • Ceramics with valid categories: {summary_stats['ceramics_with_valid_categories']}")
    
    # Sort root categories by total ceramic count
    root_totals = {root_id: sum(levels.values()) for root_id, levels in ceramic_level_distribution.items()}
    sorted_roots = sorted(root_totals.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🏛️ Distribution by Root Category (sorted by total ceramics):")
    print("-" * 80)
    
    for root_id, total_ceramics in sorted_roots:
        root_name = root_names.get(root_id, f"Unknown Root {root_id}")
        print(f"\n🔹 Root Category: {root_name} (ID: {root_id}) - Total: {total_ceramics} ceramics")
        
        levels = ceramic_level_distribution[root_id]
        max_level_in_root = max(levels.keys()) if levels else 0
        
        for level in range(max_level_in_root + 1):
            count = levels.get(level, 0)
            percentage = (count / total_ceramics * 100) if total_ceramics > 0 else 0
            
            # Show categories at this level
            categories_at_level = hierarchy_structure[root_id][level]
            category_names = [cat['name'] for cat in categories_at_level]
            
            level_name = "Root" if level == 0 else f"Level {level}"
            print(f"    └─ {level_name}: {count} ceramics ({percentage:.1f}%)")
            if category_names:
                print(f"       Categories: {', '.join(category_names[:3])}" + 
                      (f" (+{len(category_names)-3} more)" if len(category_names) > 3 else ""))
    
    # Level distribution summary across all roots
    print(f"\n📊 Overall Level Distribution Summary:")
    print("-" * 50)
    
    level_totals = defaultdict(int)
    for root_levels in ceramic_level_distribution.values():
        for level, count in root_levels.items():
            level_totals[level] += count
    
    total_ceramics = sum(level_totals.values())
    for level in sorted(level_totals.keys()):
        count = level_totals[level]
        percentage = (count / total_ceramics * 100) if total_ceramics > 0 else 0
        level_name = "Root level" if level == 0 else f"Level {level}"
        print(f"  • {level_name}: {count} ceramics ({percentage:.1f}%)")
    
    return {
        'hierarchy_structure': dict(hierarchy_structure),
        'category_info': category_info,
        'root_names': root_names
    }, dict(ceramic_level_distribution), summary_stats