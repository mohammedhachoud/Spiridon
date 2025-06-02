import random
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import pandas as pd
import traceback

def seed_everything(seed=42):
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