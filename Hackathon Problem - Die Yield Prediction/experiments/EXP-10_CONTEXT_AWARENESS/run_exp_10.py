#!/usr/bin/env python3
"""
EXP-10: Context-Aware Die Intelligence Runner.
Executes controlled experimentation across Families A through E against frozen B2.
Zero-leakage wafer-grouped training and evaluation.
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
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.ndimage import uniform_filter1d

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'experiments', 'EXP-10_CONTEXT_AWARENESS'))

from experiments.common_eval import (
    evaluate_predictions, evaluate_cascade_system, compute_wafer_robustness_summary
)
from extract_context_features import (
    extract_family_A, extract_family_B, extract_family_C, extract_family_D, extract_family_E,
    FAM_A_COLS, FAM_B_COLS, FAM_C_COLS, FAM_D_COLS, FAM_E_COLS
)
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix, extract_B1_engineered_features
from src.live_inference import extract_advanced_candidate_block_features

EXP_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'EXP-10_CONTEXT_AWARENESS')
CACHE_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache')

SEED = 42
LOCKED_TAU_A = 0.5100
LOCKED_TAU_B = 0.5376
GATE_LOWER = 0.295
GATE_UPPER = 0.550

print("=" * 80)
print("EXP-10: CONTEXT-AWARE DIE INTELLIGENCE EXPERIMENT")
print("=" * 80)

# [1/7] Load Pre-Cached Baseline Matrices
print("\n[1/7] Loading pre-cached baseline matrices from exp_cache...")
X_tr_B2 = np.load(os.path.join(CACHE_DIR, 'X_tr_B2.npy'))
X_val_B2 = np.load(os.path.join(CACHE_DIR, 'X_val_B2.npy'))
y_tr = np.load(os.path.join(CACHE_DIR, 'y_tr.npy'))
y_val = np.load(os.path.join(CACHE_DIR, 'y_val.npy'))
w_tr = np.load(os.path.join(CACHE_DIR, 'w_tr.npy'), allow_pickle=True)
w_val = np.load(os.path.join(CACHE_DIR, 'w_val.npy'), allow_pickle=True)
tr_blocks = np.load(os.path.join(CACHE_DIR, 'tr_blocks_2000.npy'))
val_blocks = np.load(os.path.join(CACHE_DIR, 'val_blocks_2000.npy'))

with open(os.path.join(CACHE_DIR, 'feature_names_B2.json'), 'r') as f:
    feature_names_B2 = json.load(f)

print(f"  • Train: {X_tr_B2.shape}, Defect Rate: {y_tr.mean()*100:.2f}% across {len(np.unique(w_tr))} wafers")
print(f"  • Val:   {X_val_B2.shape}, Defect Rate: {y_val.mean()*100:.2f}% across {len(np.unique(w_val))} wafers")

# Load frozen Model A to establish baseline cascade gating
with open(os.path.join(PROJECT_ROOT, 'experiments', 'EXP-01', 'model_A_spatial.pkl'), 'rb') as f:
    clf_A_frozen = pickle.load(f)

X_val_A = np.load(os.path.join(CACHE_DIR, 'X_val_A.npy'))
p_val_A = clf_A_frozen.predict_proba(X_val_A)[:, 1]

# Load frozen Model B2 for baseline comparison
with open(os.path.join(PROJECT_ROOT, 'experiments', 'EXP-03B', 'models', 'model_B_B2_confirmed.pkl'), 'rb') as f:
    clf_B2_frozen = pickle.load(f)
p_val_B2_frozen = clf_B2_frozen.predict_proba(X_val_B2)[:, 1]

m_base_casc, r_base_casc, _, preds_base_casc = evaluate_cascade_system(
    p_val_A, p_val_B2_frozen, y_val, tau_A=LOCKED_TAU_A, tau_B=LOCKED_TAU_B, name="Frozen_B2_Baseline"
)
print(f"  • Frozen B2 Control Cascade: F1={m_base_casc['f1']:.4f}, PR-AUC={m_base_casc['pr_auc']:.4f}, Prec={m_base_casc['precision']*100:.2f}%, Rec={m_base_casc['recall']*100:.2f}%, Escalation={m_base_casc['escalation_rate_pct']:.2f}%")

# [2/7] Feature Extraction for Families A through E
print("\n[2/7] Extracting Context Families A through E...")
t_feat0 = time.time()

# Helper coordinate and electrical slices
rows_tr, cols_tr = X_tr_B2[:, 504].astype(int), X_tr_B2[:, 505].astype(int)
rows_val, cols_val = X_val_B2[:, 504].astype(int), X_val_B2[:, 505].astype(int)

f171_tr, f460_tr, f336_tr = X_tr_B2[:, 170], X_tr_B2[:, 459], X_tr_B2[:, 335]
f171_val, f460_val, f336_val = X_val_B2[:, 170], X_val_B2[:, 459], X_val_B2[:, 335]

f33_tr, f109_tr = X_tr_B2[:, 32], X_tr_B2[:, 108]
f33_val, f109_val = X_val_B2[:, 32], X_val_B2[:, 108]

d3_tr, d5_tr = X_tr_B2[:, 501], X_tr_B2[:, 502]
d3_val, d5_val = X_val_B2[:, 501], X_val_B2[:, 502]

cluster_area_110_tr = X_tr_B2[:, 529]
cluster_area_110_val = X_val_B2[:, 529]

longest_run_104_tr = X_tr_B2[:, 522]
longest_run_104_val = X_val_B2[:, 522]

# Family A
print("  • Computing Family A (Local Physical Neighbourhood, 7 features)...")
famA_tr = extract_family_A(w_tr, rows_tr, cols_tr, f171_tr, f460_tr, f336_tr)
famA_val = extract_family_A(w_val, rows_val, cols_val, f171_val, f460_val, f336_val)

# Family B
print("  • Computing Family B (Spatial Topology & Morphology, 6 features)...")
raw_train_spatial = pd.read_csv(os.path.join(PROJECT_ROOT, 'input', 'train.csv'), usecols=['wafer_id', 'die_row', 'die_col', 'old_label'])
famB_tr = extract_family_B(w_tr, rows_tr, cols_tr, raw_train_spatial, d3_tr, d5_tr)
famB_val = extract_family_B(w_val, rows_val, cols_val, raw_train_spatial, d3_val, d5_val)

# Family C
print("  • Computing Family C (Wafer-Relative Context, 8 features)...")
famC_tr = extract_family_C(w_tr, f171_tr, f460_tr, f336_tr, f33_tr, f109_tr)
famC_val = extract_family_C(w_val, f171_val, f460_val, f336_val, f33_val, f109_val)

# Family D
print("  • Computing Family D (Signal Geometry & Location, 8 features)...")
famD_tr = extract_family_D(tr_blocks, longest_run_104_tr)
famD_val = extract_family_D(val_blocks, longest_run_104_val)

# Family E
print("  • Computing Family E (Evidence Agreement & Concordance, 5 features)...")
famE_tr = extract_family_E(famC_tr[:, 0], d5_tr, cluster_area_110_tr)
famE_val = extract_family_E(famC_val[:, 0], d5_val, cluster_area_110_val)

print(f"  ✅ Feature extraction finished in {time.time() - t_feat0:.1f}s.")

# [3/7] Assemble Controlled Candidate Datasets
print("\n[3/7] Assembling controlled candidate datasets...")
candidate_datasets = {
    "B0_Control": {
        "desc": "Frozen B2 Baseline (536 features)",
        "X_tr": X_tr_B2,
        "X_val": X_val_B2,
        "n_feat": 536,
        "new_cols": []
    },
    "C1_Neighbourhood": {
        "desc": "B2 + Family A: Local Physical Neighbourhood (543 features)",
        "X_tr": np.hstack([X_tr_B2, famA_tr]).astype(np.float32),
        "X_val": np.hstack([X_val_B2, famA_val]).astype(np.float32),
        "n_feat": 536 + 7,
        "new_cols": FAM_A_COLS
    },
    "C2_SpatialTopology": {
        "desc": "B2 + Family B: Spatial Topology & Morphology (542 features)",
        "X_tr": np.hstack([X_tr_B2, famB_tr]).astype(np.float32),
        "X_val": np.hstack([X_val_B2, famB_val]).astype(np.float32),
        "n_feat": 536 + 6,
        "new_cols": FAM_B_COLS
    },
    "C3_WaferRelative": {
        "desc": "B2 + Family C: Wafer-Relative Context (544 features)",
        "X_tr": np.hstack([X_tr_B2, famC_tr]).astype(np.float32),
        "X_val": np.hstack([X_val_B2, famC_val]).astype(np.float32),
        "n_feat": 536 + 8,
        "new_cols": FAM_C_COLS
    },
    "C4_SignalGeometry": {
        "desc": "B2 + Family D: Signal Geometry & Location (544 features)",
        "X_tr": np.hstack([X_tr_B2, famD_tr]).astype(np.float32),
        "X_val": np.hstack([X_val_B2, famD_val]).astype(np.float32),
        "n_feat": 536 + 8,
        "new_cols": FAM_D_COLS
    },
    "C5_EvidenceAgreement": {
        "desc": "B2 + Family E: Evidence Agreement (541 features)",
        "X_tr": np.hstack([X_tr_B2, famE_tr]).astype(np.float32),
        "X_val": np.hstack([X_val_B2, famE_val]).astype(np.float32),
        "n_feat": 536 + 5,
        "new_cols": FAM_E_COLS
    },
    "C6_AllContextFamilies": {
        "desc": "B2 + All Families A-E (570 features)",
        "X_tr": np.hstack([X_tr_B2, famA_tr, famB_tr, famC_tr, famD_tr, famE_tr]).astype(np.float32),
        "X_val": np.hstack([X_val_B2, famA_val, famB_val, famC_val, famD_val, famE_val]).astype(np.float32),
        "n_feat": 536 + 34,
        "new_cols": FAM_A_COLS + FAM_B_COLS + FAM_C_COLS + FAM_D_COLS + FAM_E_COLS
    }
}

for name, d in candidate_datasets.items():
    print(f"  • {name:22s}: {d['n_feat']} features (Train {d['X_tr'].shape}, Val {d['X_val'].shape})")

# [4/7] Train and Evaluate Candidates C1 through C6
print("\n[4/7] Training and evaluating candidate models in selective cascade...")
lgb_params = {
    'n_estimators': 150,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'scale_pos_weight': 10.0,
    'min_child_samples': 20,
    'random_state': SEED,
    'verbose': -1,
    'n_jobs': 1
}

trained_models = {}
candidate_metrics = {}
cascade_metrics = {}
rescue_metrics = {}
inference_latencies = {}
feature_importances = {}
wafer_robustness_records = {}

for name, d in candidate_datasets.items():
    print(f"\n  ▶ Evaluating {name} ({d['desc']})...")
    
    if name == "B0_Control":
        clf = clf_B2_frozen
        t_tr = 25.2
        p_val_cand = p_val_B2_frozen
        lat_die = 0.0085
    else:
        t0 = time.time()
        clf = lgb.LGBMClassifier(**lgb_params).fit(d['X_tr'], y_tr)
        t_tr = time.time() - t0
        
        t_inf0 = time.perf_counter()
        p_val_cand = clf.predict_proba(d['X_val'])[:, 1]
        lat_die = (time.perf_counter() - t_inf0) / len(y_val) * 1000.0

    trained_models[name] = clf
    inference_latencies[name] = {
        'train_time_s': round(t_tr, 2),
        'latency_ms_per_die': round(lat_die, 5),
        'throughput_dies_per_sec': int(1000.0 / lat_die) if lat_die > 0 else 0
    }
    
    # Standalone Model B evaluation
    m_stand = evaluate_predictions(y_val, p_val_cand, LOCKED_TAU_B, name=f"Standalone_{name}")
    candidate_metrics[name] = m_stand
    
    # Complete Selective Cascade evaluation
    c_m, r_m, c_probs, c_preds = evaluate_cascade_system(
        p_val_A, p_val_cand, y_val, tau_A=LOCKED_TAU_A, tau_B=LOCKED_TAU_B, name=f"Cascade_{name}"
    )
    cascade_metrics[name] = c_m
    rescue_metrics[name] = r_m
    
    # Per-wafer robustness summary
    df_w, summary_w = compute_wafer_robustness_summary(w_val, y_val, preds_base_casc, c_preds)
    wafer_robustness_records[name] = {
        'df': df_w,
        'summary': summary_w
    }
    
    print(f"    - Standalone Model B: F1={m_stand['f1']:.4f}, PR-AUC={m_stand['pr_auc']:.4f}, Prec={m_stand['precision']*100:.2f}%, Rec={m_stand['recall']*100:.2f}%")
    print(f"    - Complete Cascade:   F1={c_m['f1']:.4f}, PR-AUC={c_m['pr_auc']:.4f}, Prec={c_m['precision']*100:.2f}%, Rec={c_m['recall']*100:.2f}%, Escalation={c_m['escalation_rate_pct']:.2f}%")
    print(f"    - Rescue Mechanics:   FN Rescued={r_m['fn_rescued']}/{r_m['fn_escalated']} ({r_m['rescue_rate_pct']}%), Net TP Gain={r_m['net_tp_gain']}, New FP={r_m['new_fp_introduced']}")
    print(f"    - Wafer Robustness:   Mean F1={summary_w['mean_candidate_f1']:.4f}, Improved={summary_w['improved_wafers']}/32, Degraded={summary_w['degraded_wafers']}/32")

# [5/7] Build and Evaluate C7 (Strongest Selected Context Features)
print("\n[5/7] Analyzing feature importances from C6 to construct C7...")
gain_imp_C6 = trained_models['C6_AllContextFamilies'].booster_.feature_importance(importance_type='gain')
c6_cols = feature_names_B2 + candidate_datasets['C6_AllContextFamilies']['new_cols']

# Inspect top contextual features among the 34 added
new_feat_start_idx = 536
new_feat_gains = []
for i, col in enumerate(candidate_datasets['C6_AllContextFamilies']['new_cols']):
    idx = new_feat_start_idx + i
    new_feat_gains.append((col, float(gain_imp_C6[idx]), idx))

new_feat_gains.sort(key=lambda x: x[1], reverse=True)
print("  • Top 10 Contextual Features by Gain in C6:")
for col, g, idx in new_feat_gains[:10]:
    print(f"    - {col:28s}: {g:10.1f}")

# Select strongest context features (top 8 features with gain > 500)
top_selected_cols = [x[0] for x in new_feat_gains if x[1] > 500.0][:8]
if len(top_selected_cols) < 5:
    top_selected_cols = [x[0] for x in new_feat_gains[:5]]
print(f"  • Selected for C7: {top_selected_cols}")

selected_indices_in_new = [candidate_datasets['C6_AllContextFamilies']['new_cols'].index(col) for col in top_selected_cols]
all_new_tr = np.hstack([famA_tr, famB_tr, famC_tr, famD_tr, famE_tr])
all_new_val = np.hstack([famA_val, famB_val, famC_val, famD_val, famE_val])

X_tr_C7 = np.hstack([X_tr_B2, all_new_tr[:, selected_indices_in_new]]).astype(np.float32)
X_val_C7 = np.hstack([X_val_B2, all_new_val[:, selected_indices_in_new]]).astype(np.float32)

print(f"  ▶ Training C7_StrongestSelected ({536 + len(top_selected_cols)} features)...")
t0 = time.time()
clf_C7 = lgb.LGBMClassifier(**lgb_params).fit(X_tr_C7, y_tr)
t_tr_c7 = time.time() - t0

t_inf0 = time.perf_counter()
p_val_C7 = clf_C7.predict_proba(X_val_C7)[:, 1]
lat_die_c7 = (time.perf_counter() - t_inf0) / len(y_val) * 1000.0

name_c7 = "C7_StrongestSelected"
candidate_datasets[name_c7] = {
    "desc": f"B2 + Top {len(top_selected_cols)} context features",
    "X_tr": X_tr_C7,
    "X_val": X_val_C7,
    "n_feat": 536 + len(top_selected_cols),
    "new_cols": top_selected_cols
}
trained_models[name_c7] = clf_C7
inference_latencies[name_c7] = {
    'train_time_s': round(t_tr_c7, 2),
    'latency_ms_per_die': round(lat_die_c7, 5),
    'throughput_dies_per_sec': int(1000.0 / lat_die_c7) if lat_die_c7 > 0 else 0
}

m_stand_c7 = evaluate_predictions(y_val, p_val_C7, LOCKED_TAU_B, name=f"Standalone_{name_c7}")
candidate_metrics[name_c7] = m_stand_c7

c_m_c7, r_m_c7, _, c_preds_c7 = evaluate_cascade_system(
    p_val_A, p_val_C7, y_val, tau_A=LOCKED_TAU_A, tau_B=LOCKED_TAU_B, name=f"Cascade_{name_c7}"
)
cascade_metrics[name_c7] = c_m_c7
rescue_metrics[name_c7] = r_m_c7

df_w_c7, summary_w_c7 = compute_wafer_robustness_summary(w_val, y_val, preds_base_casc, c_preds_c7)
wafer_robustness_records[name_c7] = {
    'df': df_w_c7,
    'summary': summary_w_c7
}

print(f"    - Standalone Model B: F1={m_stand_c7['f1']:.4f}, PR-AUC={m_stand_c7['pr_auc']:.4f}, Prec={m_stand_c7['precision']*100:.2f}%, Rec={m_stand_c7['recall']*100:.2f}%")
print(f"    - Complete Cascade:   F1={c_m_c7['f1']:.4f}, PR-AUC={c_m_c7['pr_auc']:.4f}, Prec={c_m_c7['precision']*100:.2f}%, Rec={c_m_c7['recall']*100:.2f}%, Escalation={c_m_c7['escalation_rate_pct']:.2f}%")
print(f"    - Rescue Mechanics:   FN Rescued={r_m_c7['fn_rescued']}/{r_m_c7['fn_escalated']} ({r_m_c7['rescue_rate_pct']}%), Net TP Gain={r_m_c7['net_tp_gain']}, New FP={r_m_c7['new_fp_introduced']}")
print(f"    - Wafer Robustness:   Mean F1={summary_w_c7['mean_candidate_f1']:.4f}, Improved={summary_w_c7['improved_wafers']}/32, Degraded={summary_w_c7['degraded_wafers']}/32")

# [6/7] Compile Ablation & Latency Tables
print("\n[6/7] Compiling FEATURE_FAMILY_ABLATION.csv and LATENCY_COMPARISON.csv...")
ablation_rows = []
base_c = cascade_metrics["B0_Control"]
base_lat = inference_latencies["B0_Control"]

for name in candidate_datasets.keys():
    c = cascade_metrics[name]
    lat = inference_latencies[name]
    n_f = candidate_datasets[name]['n_feat']
    
    ablation_rows.append({
        'Feature_Family': name,
        'Num_Features': n_f,
        'Cascade_F1': c['f1'],
        'Cascade_PR_AUC': c['pr_auc'],
        'Cascade_ROC_AUC': c['roc_auc'],
        'Cascade_Precision': c['precision'],
        'Cascade_Recall': c['recall'],
        'Cascade_F2': c['f2'],
        'Cascade_MCC': c['mcc'],
        'Cascade_Brier': c['brier'],
        'Cascade_ECE': c['ece'],
        'Escalation_Pct': c['escalation_rate_pct'],
        'Compute_Saved_Pct': c['compute_saved_pct'],
        'Delta_F1': round(c['f1'] - base_c['f1'], 5),
        'Delta_PR_AUC': round(c['pr_auc'] - base_c['pr_auc'], 5),
        'Delta_Precision': round(c['precision'] - base_c['precision'], 5),
        'Delta_Recall': round(c['recall'] - base_c['recall'], 5),
        'Delta_Escalation': round(c['escalation_rate_pct'] - base_c['escalation_rate_pct'], 4),
        'Inference_Latency_ms_per_die': lat['latency_ms_per_die'],
        'Delta_Latency_ms': round(lat['latency_ms_per_die'] - base_lat['latency_ms_per_die'], 5),
        'Net_TP_Gain': rescue_metrics[name]['net_tp_gain']
    })

df_ablation = pd.DataFrame(ablation_rows)
df_ablation.to_csv(os.path.join(EXP_DIR, 'FEATURE_FAMILY_ABLATION.csv'), index=False)
print("  ✅ Saved FEATURE_FAMILY_ABLATION.csv")

# Latency Comparison
lat_rows = []
for name in candidate_datasets.keys():
    lat = inference_latencies[name]
    c = cascade_metrics[name]
    lat_rows.append({
        'Candidate': name,
        'Num_Features': candidate_datasets[name]['n_feat'],
        'Train_Time_s': lat['train_time_s'],
        'Latency_ms_per_die': lat['latency_ms_per_die'],
        'Throughput_dies_sec': lat['throughput_dies_per_sec'],
        'Escalation_Rate_Pct': c['escalation_rate_pct'],
        'Effective_Inference_Cost_ms': round(0.0066 + (c['escalation_rate_pct']/100.0) * lat['latency_ms_per_die'], 5)
    })
df_lat = pd.DataFrame(lat_rows)
df_lat.to_csv(os.path.join(EXP_DIR, 'LATENCY_COMPARISON.csv'), index=False)
print("  ✅ Saved LATENCY_COMPARISON.csv")

# Save Wafer Robustness for the best candidate vs B0
# Determine best candidate by F1 / PR-AUC
ranked_candidates = sorted(
    [k for k in candidate_datasets.keys() if k != "B0_Control"],
    key=lambda k: (cascade_metrics[k]['f1'], cascade_metrics[k]['pr_auc']),
    reverse=True
)
best_cand_name = ranked_candidates[0]
print(f"\n  • Top-ranked Candidate on Validation Set: {best_cand_name}")
df_best_w = wafer_robustness_records[best_cand_name]['df']
df_best_w.to_csv(os.path.join(EXP_DIR, 'WAFER_ROBUSTNESS.csv'), index=False)
print(f"  ✅ Saved WAFER_ROBUSTNESS.csv (for {best_cand_name})")

# [7/7] Phase 14 & 15 Discipline: Frozen Independent Evaluation on input/test.csv
print("\n[7/7] Executing ONE-TIME Independent Evaluation on input/test.csv (40 unseen wafers)...")
test_raw = pd.read_csv(os.path.join(PROJECT_ROOT, 'input', 'test.csv'))
test_eligible = test_raw[test_raw['old_label'] == 0].copy().reset_index(drop=True)
y_test = test_eligible['label'].values.astype(int)
w_test = test_eligible['wafer_id'].values

# Feature extraction for test population
sp_test = compute_spatial_features(test_eligible, full_df=test_raw)
spatial_cols = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
feat_cols = [f'feature_{i}' for i in range(1, 501)]
X_test_A = pd.concat([test_eligible[feat_cols], sp_test[spatial_cols]], axis=1).values.astype(np.float32)

test_blocks = parse_block_matrix(test_eligible['block_readings'])
b1_feats_test = extract_B1_engineered_features(test_blocks)
b_cand_feats_test = extract_advanced_candidate_block_features(test_blocks)
X_test_B2 = np.hstack([X_test_A, b1_feats_test.values, b_cand_feats_test.values]).astype(np.float32)

p_test_A = clf_A_frozen.predict_proba(X_test_A)[:, 1]
p_test_B2_base = clf_B2_frozen.predict_proba(X_test_B2)[:, 1]

# Independent evaluation of Frozen B2
m_test_base, r_test_base, _, preds_test_base = evaluate_cascade_system(
    p_test_A, p_test_B2_base, y_test, tau_A=LOCKED_TAU_A, tau_B=LOCKED_TAU_B, name="Independent_B2_Baseline"
)
print(f"  • Independent B2 Baseline: F1={m_test_base['f1']:.4f}, PR-AUC={m_test_base['pr_auc']:.4f}, Prec={m_test_base['precision']*100:.2f}%, Rec={m_test_base['recall']*100:.2f}%, Escalation={m_test_base['escalation_rate_pct']:.2f}%")
print(f"  • Rescue mechanics: FNs rescued={r_test_base['fn_rescued']}/{r_test_base['fn_escalated']} ({r_test_base['rescue_rate_pct']}%), Net TP Gain={r_test_base['net_tp_gain']}, New FP={r_test_base['new_fp_introduced']}")

# Extract context features for test set for best candidate
print(f"  • Extracting context features for test set ({best_cand_name})...")
rows_t = X_test_B2[:, 504].astype(int)
cols_t = X_test_B2[:, 505].astype(int)
f171_t, f460_t, f336_t = X_test_B2[:, 170], X_test_B2[:, 459], X_test_B2[:, 335]
f33_t, f109_t = X_test_B2[:, 32], X_test_B2[:, 108]
d3_t, d5_t = X_test_B2[:, 501], X_test_B2[:, 502]
cluster_area_110_t = X_test_B2[:, 529]
longest_run_104_t = X_test_B2[:, 522]

famA_t = extract_family_A(w_test, rows_t, cols_t, f171_t, f460_t, f336_t)
famB_t = extract_family_B(w_test, rows_t, cols_t, test_raw, d3_t, d5_t)
famC_t = extract_family_C(w_test, f171_t, f460_t, f336_t, f33_t, f109_t)
famD_t = extract_family_D(test_blocks, longest_run_104_t)
famE_t = extract_family_E(famC_t[:, 0], d5_t, cluster_area_110_t)

all_new_t = np.hstack([famA_t, famB_t, famC_t, famD_t, famE_t])

if best_cand_name == "C1_Neighbourhood":
    X_test_best = np.hstack([X_test_B2, famA_t]).astype(np.float32)
elif best_cand_name == "C2_SpatialTopology":
    X_test_best = np.hstack([X_test_B2, famB_t]).astype(np.float32)
elif best_cand_name == "C3_WaferRelative":
    X_test_best = np.hstack([X_test_B2, famC_t]).astype(np.float32)
elif best_cand_name == "C4_SignalGeometry":
    X_test_best = np.hstack([X_test_B2, famD_t]).astype(np.float32)
elif best_cand_name == "C5_EvidenceAgreement":
    X_test_best = np.hstack([X_test_B2, famE_t]).astype(np.float32)
elif best_cand_name == "C6_AllContextFamilies":
    X_test_best = np.hstack([X_test_B2, all_new_t]).astype(np.float32)
elif best_cand_name == "C7_StrongestSelected":
    X_test_best = np.hstack([X_test_B2, all_new_t[:, selected_indices_in_new]]).astype(np.float32)

p_test_best = trained_models[best_cand_name].predict_proba(X_test_best)[:, 1]
m_test_best, r_test_best, _, preds_test_best = evaluate_cascade_system(
    p_test_A, p_test_best, y_test, tau_A=LOCKED_TAU_A, tau_B=LOCKED_TAU_B, name=f"Independent_{best_cand_name}"
)

print(f"  • Independent {best_cand_name}: F1={m_test_best['f1']:.4f}, PR-AUC={m_test_best['pr_auc']:.4f}, Prec={m_test_best['precision']*100:.2f}%, Rec={m_test_best['recall']*100:.2f}%, Escalation={m_test_best['escalation_rate_pct']:.2f}%")
print(f"  • Rescue mechanics: FNs rescued={r_test_best['fn_rescued']}/{r_test_best['fn_escalated']} ({r_test_best['rescue_rate_pct']}%), Net TP Gain={r_test_best['net_tp_gain']}, New FP={r_test_best['new_fp_introduced']}")

# Compute delta on independent holdout
delta_test_f1 = m_test_best['f1'] - m_test_base['f1']
delta_test_pr = m_test_best['pr_auc'] - m_test_base['pr_auc']
delta_test_prec = m_test_best['precision'] - m_test_base['precision']
delta_test_rec = m_test_best['recall'] - m_test_base['recall']
delta_test_net_tp = r_test_best['net_tp_gain'] - r_test_base['net_tp_gain']

print(f"\n  • Holdout Deltas ({best_cand_name} vs Frozen B2):")
print(f"    - ΔF1:       {delta_test_f1:+.4f}")
print(f"    - ΔPR-AUC:   {delta_test_pr:+.4f}")
print(f"    - ΔPrecision:{delta_test_prec*100:+.2f}%")
print(f"    - ΔRecall:   {delta_test_rec*100:+.2f}%")
print(f"    - ΔNet TP:   {delta_test_net_tp:+d}")

# Save METRICS.json
all_metrics_dict = {
    "experiment": "EXP-10_CONTEXT_AWARENESS",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "locked_policy": {
        "tau_A": LOCKED_TAU_A,
        "tau_B": LOCKED_TAU_B,
        "gate_lower": GATE_LOWER,
        "gate_upper": GATE_UPPER
    },
    "validation_results": {
        "cascade_metrics": cascade_metrics,
        "rescue_metrics": rescue_metrics,
        "inference_latencies": inference_latencies,
        "wafer_robustness_summaries": {k: v['summary'] for k, v in wafer_robustness_records.items()}
    },
    "independent_holdout_results": {
        "frozen_b2_baseline": {
            "cascade_metrics": m_test_base,
            "rescue_metrics": r_test_base
        },
        f"best_candidate_{best_cand_name}": {
            "cascade_metrics": m_test_best,
            "rescue_metrics": r_test_best
        },
        "deltas": {
            "delta_f1": round(delta_test_f1, 5),
            "delta_pr_auc": round(delta_test_pr, 5),
            "delta_precision": round(delta_test_prec, 5),
            "delta_recall": round(delta_test_rec, 5),
            "delta_net_tp_gain": int(delta_test_net_tp)
        }
    },
    "best_candidate": best_cand_name
}

with open(os.path.join(EXP_DIR, 'METRICS.json'), 'w') as f:
    json.dump(all_metrics_dict, f, indent=2)
print("  ✅ Saved METRICS.json")

print("\n" + "=" * 80)
print("EXP-10 RUNNER COMPLETED SUCCESSFULLY.")
print("=" * 80)
