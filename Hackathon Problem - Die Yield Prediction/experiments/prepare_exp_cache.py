#!/usr/bin/env python3
"""
Pre-experiment data preparation and caching script for EXP-05 through EXP-09.
Caches tabular feature matrices and block sequences for canonical 128 train / 32 validation partitions.
CRITICAL: input/test.csv is NOT touched here.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import load_eligible_data, get_wafer_split
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix

CACHE_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

print("[1/5] Loading train.csv and computing spatial features...")
t0 = time.time()
csv_path = os.path.join(PROJECT_ROOT, 'input', 'train.csv')
raw_df = pd.read_csv(csv_path)
eligible_df = raw_df[raw_df['old_label'] == 0].copy().reset_index(drop=True)

train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible_df)
train_df = eligible_df[train_mask].reset_index(drop=True)
val_df = eligible_df[val_mask].reset_index(drop=True)

# Spatial features using full wafer map
sp_all = compute_spatial_features(eligible_df, full_df=raw_df)
sp_tr = sp_all[train_mask].reset_index(drop=True)
sp_val = sp_all[val_mask].reset_index(drop=True)

feat_cols = [f'feature_{i}' for i in range(1, 501)]
spatial_cols = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']

X_tr_A = pd.concat([train_df[feat_cols], sp_tr[spatial_cols]], axis=1).values.astype(np.float32)
X_val_A = pd.concat([val_df[feat_cols], sp_val[spatial_cols]], axis=1).values.astype(np.float32)
feature_names_A = feat_cols + spatial_cols

y_tr = train_df['label'].values.astype(np.int32)
y_val = val_df['label'].values.astype(np.int32)
w_tr = train_df['wafer_id'].values
w_val = val_df['wafer_id'].values

print(f"  • Model A matrices: Train {X_tr_A.shape}, Val {X_val_A.shape} (Time: {time.time() - t0:.1f}s)")

print("[2/5] Loading cached B0/B1 and B2 candidate block features from EXP-03...")
exp03_cache = np.load(os.path.join(PROJECT_ROOT, 'experiments', 'EXP-03', 'cached_block_features.npz'), allow_pickle=True)
tr_b_eng = exp03_cache['tr_b_eng']
val_b_eng = exp03_cache['val_b_eng']
b_eng_cols = list(exp03_cache['b_eng_cols'])

tr_b_cand = exp03_cache['tr_b_cand']
val_b_cand = exp03_cache['val_b_cand']
b_cand_cols = list(exp03_cache['b_cand_cols'])

X_tr_B2 = np.hstack([X_tr_A, tr_b_eng, tr_b_cand]).astype(np.float32)
X_val_B2 = np.hstack([X_val_A, val_b_eng, val_b_cand]).astype(np.float32)
feature_names_B2 = feature_names_A + b_eng_cols + b_cand_cols
print(f"  • Model B2 matrices: Train {X_tr_B2.shape}, Val {X_val_B2.shape}")

print("[3/5] Parsing raw 2,000-block sequences for validation set...")
t_val_b = time.time()
val_blocks_2000 = parse_block_matrix(val_df['block_readings'])
print(f"  • Val blocks parsed: {val_blocks_2000.shape} in {time.time() - t_val_b:.1f}s")

print("[4/5] Parsing raw 2,000-block sequences for training set...")
t_tr_b = time.time()
tr_blocks_2000 = parse_block_matrix(train_df['block_readings'])
print(f"  • Train blocks parsed: {tr_blocks_2000.shape} in {time.time() - t_tr_b:.1f}s")

print("[5/5] Saving cached arrays...")
np.save(os.path.join(CACHE_DIR, 'X_tr_A.npy'), X_tr_A)
np.save(os.path.join(CACHE_DIR, 'X_val_A.npy'), X_val_A)
np.save(os.path.join(CACHE_DIR, 'X_tr_B2.npy'), X_tr_B2)
np.save(os.path.join(CACHE_DIR, 'X_val_B2.npy'), X_val_B2)
np.save(os.path.join(CACHE_DIR, 'y_tr.npy'), y_tr)
np.save(os.path.join(CACHE_DIR, 'y_val.npy'), y_val)
np.save(os.path.join(CACHE_DIR, 'w_tr.npy'), w_tr)
np.save(os.path.join(CACHE_DIR, 'w_val.npy'), w_val)
np.save(os.path.join(CACHE_DIR, 'val_blocks_2000.npy'), val_blocks_2000)
np.save(os.path.join(CACHE_DIR, 'tr_blocks_2000.npy'), tr_blocks_2000)

with open(os.path.join(CACHE_DIR, 'feature_names_A.json'), 'w') as f:
    json.dump(feature_names_A, f)
with open(os.path.join(CACHE_DIR, 'feature_names_B2.json'), 'w') as f:
    json.dump(feature_names_B2, f)

print(f"✅ Pre-experiment cache built successfully in {time.time() - t0:.1f}s!")
