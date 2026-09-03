import os
import json
import pandas as pd
import numpy as np

def get_default_canonical_split_path():
    """Return absolute path to results/canonical_wafer_split.json."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, 'results', 'canonical_wafer_split.json')

def load_canonical_wafer_split(split_json_path=None):
    """
    Load canonical 128/32 wafer split from canonical_wafer_split.json artifact.
    Fails loudly if the split artifact is missing.
    """
    if split_json_path is None:
        split_json_path = get_default_canonical_split_path()
        
    if not os.path.exists(split_json_path):
        raise FileNotFoundError(
            f"CRITICAL ERROR: Canonical wafer split artifact not found at '{split_json_path}'. "
            f"Independent random splitting is strictly prohibited to prevent data leakage."
        )
        
    with open(split_json_path, 'r') as f:
        split_data = json.load(f)
        
    train_wafers = set(split_data['train_wafers'])
    val_wafers = set(split_data['validation_wafers'])
    
    # HARD SAFETY INTEGRITY CHECKS
    if len(train_wafers) != 128:
        raise ValueError(f"Integrity check failed: Expected 128 train wafers, got {len(train_wafers)}")
    if len(val_wafers) != 32:
        raise ValueError(f"Integrity check failed: Expected 32 val wafers, got {len(val_wafers)}")
        
    intersection = train_wafers.intersection(val_wafers)
    if len(intersection) > 0:
        raise ValueError(
            f"CRITICAL LEAKAGE DETECTED: Canonical split contains {len(intersection)} overlapping wafers! "
            f"Overlapping: {sorted(list(intersection))}"
        )
        
    return train_wafers, val_wafers, split_data.get('canonical_split_metadata', {})

def load_eligible_data(csv_path):
    """Load dataset and filter strictly to eligible dies (old_label == 0)."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Filter strictly to eligible dies
    if 'old_label' in df.columns:
        eligible_df = df[df['old_label'] == 0].copy().reset_index(drop=True)
    else:
        eligible_df = df.copy().reset_index(drop=True)
        
    return eligible_df

def get_wafer_split(eligible_df, split_json_path=None, seed=None, train_pct=None):
    """
    Get canonical wafer split masks for train and validation sets.
    Consumes results/canonical_wafer_split.json directly.
    """
    train_wafers, val_wafers, metadata = load_canonical_wafer_split(split_json_path)
    
    if 'wafer_id' not in eligible_df.columns:
        raise KeyError("Eligible dataframe must contain 'wafer_id' column.")
        
    train_mask = eligible_df['wafer_id'].isin(train_wafers)
    val_mask = eligible_df['wafer_id'].isin(val_wafers)
    
    # Verify dataset wafer coverage and zero contamination
    val_df_check = eligible_df[val_mask]
    train_df_check = eligible_df[train_mask]
    
    val_wafers_in_train = set(train_df_check['wafer_id'].unique()).intersection(val_wafers)
    if len(val_wafers_in_train) > 0:
        raise ValueError(f"CRITICAL CONTAMINATION: Validation wafers found in training slice: {val_wafers_in_train}")
        
    if 'old_label' in eligible_df.columns:
        if not (eligible_df['old_label'] == 0).all():
            raise ValueError("Evaluation Leakage: Non-eligible dies (old_label != 0) found in dataset!")
            
    return train_mask, val_mask, train_wafers, val_wafers
