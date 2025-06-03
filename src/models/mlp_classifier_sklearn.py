import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from collections import Counter

def train_and_evaluate_mlp_model(X_train, X_test, y_train, y_test, scenario_name, root_category_names_map, mlp_params=None):
    """Train and evaluate MLP classifier without normalization"""
    print(f"\n{'='*50}")
    print(f"SCENARIO: {scenario_name} - MLP Classifier")
    print(f"{'='*50}")
    
    default_mlp_params = {
        'hidden_layer_sizes': (128, 64),
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.0001,
        'batch_size': 'auto',
        'learning_rate': 'adaptive',
        'max_iter': 1000,
        'random_state': 42,
        'verbose': False,
        'early_stopping': True,
        'validation_fraction': 0.15,
        'n_iter_no_change': 10
    }
    if mlp_params:
        default_mlp_params.update(mlp_params)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input features: {X_train.shape[1]}")
    
    unique_train_labels = np.unique(y_train)
    unique_test_labels = np.unique(y_test)
    print(f"Number of classes (train): {len(unique_train_labels)}")
    print(f"Number of classes (test): {len(unique_test_labels)}")
    
    print(f"Training distribution: {Counter(y_train)}")
    print(f"Test distribution: {Counter(y_test)}")
    print(f"\nNeural Network Architecture:")
    print(f"Input layer: {X_train.shape[1]} features")
    print(f"Hidden layers: {default_mlp_params['hidden_layer_sizes']}")
    print(f"Output layer: {len(unique_train_labels)} neurons (softmax)") 

    mlp = MLPClassifier(**default_mlp_params)
    
    print(f"\nTraining MLP model...")
    mlp.fit(X_train, y_train)
    
    train_accuracy = mlp.score(X_train, y_train)
    test_accuracy = mlp.score(X_test, y_test)
    
    print(f"\n{'='*30}\nTRAINING METRICS:\n{'='*30}")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    y_pred = mlp.predict(X_test)
    print(f"\n{'='*30}\nDETAILED CLASSIFICATION REPORT:\n{'='*30}")
    target_names_report = [root_category_names_map.get(int(cls), f"Class_{cls}") for cls in mlp.classes_]
    print(classification_report(y_test, y_pred, target_names=target_names_report, zero_division=0))
    
    print(f"\n{'='*30}\nPERFORMANCE BY ROOT CATEGORY:\n{'='*30}")
    for cat_id in sorted(np.unique(y_test)): 
        cat_name_display = root_category_names_map.get(int(cat_id), f"UnknownCat ({cat_id})")
        cat_indices = (y_test == cat_id)
        if np.sum(cat_indices) > 0:
            cat_accuracy = accuracy_score(y_test[cat_indices], y_pred[cat_indices])
            count = np.sum(cat_indices)
            print(f"{cat_name_display} (ID: {cat_id}): {cat_accuracy:.4f} accuracy ({count} samples)")
    
    # Confusion Matrix
    print(f"\n{'='*30}")
    print("CONFUSION MATRIX:")
    print(f"{'='*30}")
    cm = confusion_matrix(y_test, y_pred)
    print("Rows: True labels, Columns: Predicted labels")
    class_names = [root_category_names_map.get(cat_id, f"Cat_{cat_id}") for cat_id in sorted(set(y_test))]
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
    
    return mlp, None, test_accuracy