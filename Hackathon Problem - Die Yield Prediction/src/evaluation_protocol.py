import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from src.data_loader import load_canonical_wafer_split

TARGET_COL = 'label'
OLD_LABEL_COL = 'old_label'
WAFER_ID_COL = 'wafer_id'
FORBIDDEN_FEATURE_COLS = {'wafer_id', 'old_label', 'label', 'block_readings'}
MODEL_A_LOCKED_THRESHOLD = 0.5100

def validate_eligible_dataset(df):
    """Ensure dataset strictly contains only eligible pre-test passing dies."""
    if WAFER_ID_COL not in df.columns:
        raise KeyError(f"Dataset missing required '{WAFER_ID_COL}' column.")
        
    if OLD_LABEL_COL in df.columns:
        if not (df[OLD_LABEL_COL] == 0).all():
            bad_count = (df[OLD_LABEL_COL] != 0).sum()
            raise ValueError(
                f"DATA LEAKAGE ERROR: {bad_count} dies with old_label != 0 found in eligible dataset! "
                f"All post-test yield evaluations must strictly filter to old_label == 0."
            )
            
    if TARGET_COL in df.columns:
        unique_targets = set(df[TARGET_COL].unique())
        if not unique_targets.issubset({0, 1}):
            raise ValueError(f"Target column '{TARGET_COL}' contains invalid values: {unique_targets}")
            
    return True

def validate_features_not_leaked(feature_names):
    """Verify that forbidden identifier/target columns are not passed as features."""
    feat_set = set(feature_names)
    leaked = feat_set.intersection(FORBIDDEN_FEATURE_COLS)
    if len(leaked) > 0:
        raise ValueError(
            f"CRITICAL FEATURE LEAKAGE: Forbidden columns {sorted(list(leaked))} present in feature matrix!"
        )
    return True

def validate_wafer_partition_integrity(train_df, val_df, train_wafers=None, val_wafers=None):
    """Enforce strict 0-intersection holdout partition checks."""
    if train_wafers is None or val_wafers is None:
        train_wafers, val_wafers, _ = load_canonical_wafer_split()
        
    actual_train_wafers = set(train_df[WAFER_ID_COL].unique())
    actual_val_wafers = set(val_df[WAFER_ID_COL].unique())
    
    # Check 1: Zero intersection between active train and val slices
    slice_intersection = actual_train_wafers.intersection(actual_val_wafers)
    if len(slice_intersection) > 0:
        raise ValueError(
            f"CRITICAL SPLIT CONTAMINATION: {len(slice_intersection)} wafers overlap between train and val slices: "
            f"{sorted(list(slice_intersection))}"
        )
        
    # Check 2: All validation wafers must belong to canonical val set
    illegal_val = actual_val_wafers - val_wafers
    if len(illegal_val) > 0:
        raise ValueError(
            f"CRITICAL SPLIT LEAKAGE: Validation slice contains {len(illegal_val)} non-canonical validation wafers: "
            f"{sorted(list(illegal_val))}"
        )
        
    # Check 3: No training wafer in canonical val set
    train_in_val = actual_train_wafers.intersection(val_wafers)
    if len(train_in_val) > 0:
        raise ValueError(
            f"CRITICAL SPLIT LEAKAGE: Training slice contains {len(train_in_val)} canonical validation wafers: "
            f"{sorted(list(train_in_val))}"
        )
        
    return True

def calculate_evaluation_metrics(y_true, y_prob, threshold=MODEL_A_LOCKED_THRESHOLD):
    """Calculate standardized evaluation metrics with precision-recall curve integration."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    
    p_curve, r_curve, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = float(auc(r_curve, p_curve))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'threshold': float(threshold),
        'f1': f1,
        'auc_pr': auc_pr,
        'roc_auc': roc_auc,
        'precision': prec,
        'recall': rec,
        'confusion_matrix': {
            'tp': int(tp),
            'fn': int(fn),
            'fp': int(fp),
            'tn': int(tn)
        },
        'total_samples': int(len(y_true)),
        'positive_samples': int(y_true.sum()),
        'negative_samples': int((1 - y_true).sum())
    }
