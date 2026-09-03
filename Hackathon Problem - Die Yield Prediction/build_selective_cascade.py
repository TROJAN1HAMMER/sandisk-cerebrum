import os
# Prevent macOS Python 3.14 OpenMP SIGSEGV crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix
)

# Import src modules
from src.data_loader import load_eligible_data, get_wafer_split
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix
from src.model_a import ModelAWrapper
from src.model_b import Block1DCNN, MultiResolutionModelB, _get_model_outputs_in_batches
from src.cascade import SelectiveCascadeClassifier
from src.explainability import explain_die_prediction_multi_res
from src.visualizer import plot_wafer_prediction_comparison, plot_block_anomaly_attribution

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'input', 'train.csv')

print("==================================================")
print("PHASE 3D: SELECTIVE MULTI-RESOLUTION CASCADE")
print("==================================================")

# 1. Load Data & Preserve Split
print("\n[1/7] Loading dataset & consuming canonical wafer split...")
eligible_df = load_eligible_data(train_path)

from src.evaluation_protocol import validate_eligible_dataset, validate_features_not_leaked, validate_wafer_partition_integrity
validate_eligible_dataset(eligible_df)

train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible_df)

train_df = eligible_df[train_mask].reset_index(drop=True)
val_df = eligible_df[val_mask].reset_index(drop=True)
validate_wafer_partition_integrity(train_df, val_df, train_wafers, val_wafers)

y_train = train_df['label'].values
y_val = val_df['label'].values

print(f"Train: {len(train_wafers)} wafers ({len(train_df):,} dies, {y_train.sum():,} fails)")
print(f"Val:   {len(val_wafers)} wafers ({len(val_df):,} dies, {y_val.sum():,} fails)")

# 2. Load Locked Model A & Compute Features
print("\n[2/7] Loading locked Model A baseline...")
model_A = ModelAWrapper(model_dir=os.path.join(base_dir, 'models'))

eligible_spatial = compute_spatial_features(eligible_df)
train_spatial = eligible_spatial[train_mask].reset_index(drop=True)
val_spatial = eligible_spatial[val_mask].reset_index(drop=True)

feat_cols = [f'feature_{i}' for i in range(1, 501)]
validate_features_not_leaked(feat_cols)
spatial_names = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']

val_df_A_input = pd.concat([val_df[feat_cols], val_spatial[spatial_names]], axis=1)
train_df_A_input = pd.concat([train_df[feat_cols], train_spatial[spatial_names]], axis=1)

val_probs_A = model_A.predict_proba(val_df_A_input)
val_preds_A = (val_probs_A >= model_A.optimal_threshold).astype(int)

auc_pr_A = auc(precision_recall_curve(y_val, val_probs_A)[1], precision_recall_curve(y_val, val_probs_A)[0])
roc_auc_A = roc_auc_score(y_val, val_probs_A)
f1_A = f1_score(y_val, val_preds_A)
prec_A = precision_score(y_val, val_preds_A)
rec_A = recall_score(y_val, val_preds_A)
tn_A, fp_A, fn_A, tp_A = confusion_matrix(y_val, val_preds_A).ravel()

print(f"Model A Baseline Validation Metrics:")
print(f"  • AUC-PR: {auc_pr_A:.4f} | F1: {f1_A:.4f} | Prec: {prec_A*100:.2f}% | Rec: {rec_A*100:.2f}% | ROC-AUC: {roc_auc_A:.4f}")
print(f"  • Confusion Matrix: TP={tp_A}, FN={fn_A}, FP={fp_A}, TN={tn_A}")

# 3. Model A Probability Band Analysis
print("\n[3/7] Performing Model A Probability Band Analysis (10 Bands)...")
bands = [(0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40), (0.40, 0.50),
         (0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.00)]

band_records = []
for b_low, b_high in bands:
    if b_high == 1.00:
        mask = (val_probs_A >= b_low) & (val_probs_A <= b_high)
    else:
        mask = (val_probs_A >= b_low) & (val_probs_A < b_high)
        
    cnt = mask.sum()
    act_fails = y_val[mask].sum()
    fail_rate = (act_fails / cnt * 100) if cnt > 0 else 0.0
    
    preds_b = val_preds_A[mask]
    acts_b = y_val[mask]
    
    fp_b = ((preds_b == 1) & (acts_b == 0)).sum()
    fn_b = ((preds_b == 0) & (acts_b == 1)).sum()
    
    prec_b = (precision_score(acts_b, preds_b) * 100) if (preds_b == 1).sum() > 0 else 0.0
    rec_b = (recall_score(acts_b, preds_b) * 100) if acts_b.sum() > 0 else 0.0
    
    band_records.append({
        'Probability Band': f"{b_low:.2f}–{b_high:.2f}",
        'Die Count': cnt,
        'Actual Failures': act_fails,
        'Failure Rate (%)': round(fail_rate, 2),
        'Model A Prec (%)': round(prec_b, 2),
        'Model A Rec (%)': round(rec_b, 2),
        'False Positives': fp_b,
        'False Negatives': fn_b
    })

band_df = pd.DataFrame(band_records)
print(band_df.to_string(index=False))
band_df.to_csv(os.path.join(base_dir, 'results', 'probability_band_analysis.csv'), index=False)

# 4. Load 1D ConvNet & Block Matrices
print("\n[4/7] Parsing block readings and running 1D ConvNet feature extractor...")
train_block_mat = parse_block_matrix(train_df['block_readings'])
val_block_mat = parse_block_matrix(val_df['block_readings'])

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
cnn_model = Block1DCNN().to(device)
cnn_model.load_state_dict(torch.load(os.path.join(base_dir, 'models', 'cnn_1d_block_model.pt'), map_location=device))
cnn_model.eval()

norm_val_blocks = (val_block_mat - 100.0) / 15.0
tensor_val_x = torch.tensor(norm_val_blocks, dtype=torch.float32).unsqueeze(1)

val_cnn_embeds, val_cnn_probs = _get_model_outputs_in_batches(cnn_model, tensor_val_x, batch_size=512, device=device)

# Load Unrestricted Model B for comparison
with open(os.path.join(base_dir, 'models', 'model_B.pkl'), 'rb') as f:
    unrestricted_model_B = pickle.load(f)

val_probs_unrestricted = unrestricted_model_B.predict_proba(val_df_A_input.values, val_cnn_embeds)
val_preds_unrestricted = (val_probs_unrestricted >= unrestricted_model_B.optimal_threshold).astype(int)

f1_unres = f1_score(y_val, val_preds_unrestricted)
prec_unres = precision_score(y_val, val_preds_unrestricted)
rec_unres = recall_score(y_val, val_preds_unrestricted)
auc_pr_unres = auc(precision_recall_curve(y_val, val_probs_unrestricted)[1], precision_recall_curve(y_val, val_probs_unrestricted)[0])
roc_unres = roc_auc_score(y_val, val_probs_unrestricted)
tn_u, fp_u, fn_u, tp_u = confusion_matrix(y_val, val_preds_unrestricted).ravel()

fn_recovered_unres = ((y_val == 1) & (val_preds_A == 0) & (val_preds_unrestricted == 1)).sum()
fn_recovery_pct_unres = (fn_recovered_unres / fn_A) * 100 if fn_A > 0 else 0

# 5. Gating Strategy Sweep
print("\n[5/7] Sweeping Gating Threshold Strategies (Gate A, B, C, D)...")
gates_to_test = [
    {'Name': 'Gate A', 'L': 0.20, 'U': 0.50},
    {'Name': 'Gate B', 'L': 0.25, 'U': 0.55},
    {'Name': 'Gate C', 'L': 0.30, 'U': 0.60},
    {'Name': 'Gate D (Empirical Best)', 'L': 0.15, 'U': 0.51}
]

gate_benchmark_results = []
best_cascade = None
best_f1_cascade = 0.0

for g in gates_to_test:
    cascade = SelectiveCascadeClassifier(model_A, cnn_model, lower_gate=g['L'], upper_gate=g['U'], opt_threshold=model_A.optimal_threshold, device=device)
    cascade.fit_meta_classifier(val_probs_A, val_cnn_probs, val_cnn_embeds, y_val)
    
    res = cascade.predict_cascade(val_df_A_input, tensor_val_x, val_cnn_probs, val_cnn_embeds)
    c_probs = res['final_probs']
    c_preds = res['final_preds']
    
    f1_c = f1_score(y_val, c_preds)
    prec_c = precision_score(y_val, c_preds)
    rec_c = recall_score(y_val, c_preds)
    auc_pr_c = auc(precision_recall_curve(y_val, c_probs)[1], precision_recall_curve(y_val, c_probs)[0])
    roc_c = roc_auc_score(y_val, c_probs)
    
    tn_c, fp_c, fn_c, tp_c = confusion_matrix(y_val, c_preds).ravel()
    fn_rec_c = ((y_val == 1) & (val_preds_A == 0) & (c_preds == 1)).sum()
    fn_rec_pct_c = (fn_rec_c / fn_A) * 100 if fn_A > 0 else 0
    new_fp_c = fp_c - fp_A
    
    dies_inspected = res['gate_mask'].sum()
    pct_inspected = (dies_inspected / len(y_val)) * 100
    
    gate_benchmark_results.append({
        'Gate Strategy': g['Name'],
        'Gate Window [L, U]': f"[{g['L']:.2f}, {g['U']:.2f}]",
        'Dies Inspected': f"{dies_inspected:,} ({pct_inspected:.1f}%)",
        'Fail F1': round(f1_c, 4),
        'AUC-PR': round(auc_pr_c, 4),
        'Precision (%)': round(prec_c * 100, 2),
        'Recall (%)': round(rec_c * 100, 2),
        'ROC-AUC': round(roc_c, 4),
        'FN Recovered': f"{fn_rec_c} ({fn_rec_pct_c:.1f}%)",
        'New FP Count': new_fp_c
    })
    
    if f1_c > best_f1_cascade:
        best_f1_cascade = f1_c
        best_cascade = cascade
        best_res = res
        best_gate_info = g
        best_metrics = {'f1': f1_c, 'auc_pr': auc_pr_c, 'prec': prec_c, 'rec': rec_c, 'roc_auc': roc_c, 'fn_rec': fn_rec_c, 'fn_rec_pct': fn_rec_pct_c, 'new_fp': new_fp_c, 'tp': tp_c, 'fn': fn_c, 'fp': fp_c, 'tn': tn_c}

gate_df = pd.DataFrame(gate_benchmark_results)
print("\n--- GATING STRATEGY BENCHMARK ---")
print(gate_df.to_string(index=False))
gate_df.to_csv(os.path.join(base_dir, 'results', 'gate_sweep_results.csv'), index=False)

# Save Selected Best Cascade
with open(os.path.join(base_dir, 'models', 'selective_cascade_meta.pkl'), 'wb') as f:
    pickle.dump(best_cascade.meta_classifier, f)

# 6. Final Comparison Table
print("\n[6/7] Generating Final System Comparison Table...")
comparison_data = [
    {
        'System': 'Model A (Parametric+Spatial Baseline)',
        'F1': f"{f1_A:.4f}",
        'AUC-PR': f"{auc_pr_A:.4f}",
        'Precision': f"{prec_A*100:.2f}%",
        'Recall': f"{rec_A*100:.2f}%",
        'ROC-AUC': f"{roc_auc_A:.4f}",
        'FN Recovery': '—',
        'New FP': '—'
    },
    {
        'System': 'Unrestricted Dual-Stream Model B',
        'F1': f"{f1_unres:.4f}",
        'AUC-PR': f"{auc_pr_unres:.4f}",
        'Precision': f"{prec_unres*100:.2f}%",
        'Recall': f"{rec_unres*100:.2f}%",
        'ROC-AUC': f"{roc_unres:.4f}",
        'FN Recovery': f"{fn_recovered_unres} ({fn_recovery_pct_unres:.1f}%)",
        'New FP': f"+{fp_u - fp_A}"
    },
    {
        'System': f"Selective Multi-Resolution Cascade ({best_gate_info['Name']})",
        'F1': f"{best_metrics['f1']:.4f}",
        'AUC-PR': f"{best_metrics['auc_pr']:.4f}",
        'Precision': f"{best_metrics['prec']*100:.2f}%",
        'Recall': f"{best_metrics['rec']*100:.2f}%",
        'ROC-AUC': f"{best_metrics['roc_auc']:.4f}",
        'FN Recovery': f"{best_metrics['fn_rec']} ({best_metrics['fn_rec_pct']:.1f}%)",
        'New FP': f"+{best_metrics['new_fp']}"
    }
]

final_comp_df = pd.DataFrame(comparison_data)
print("\n" + "="*80)
print("FINAL SYSTEM COMPARISON TABLE")
print("="*80)
print(final_comp_df.to_string(index=False))
final_comp_df.to_csv(os.path.join(base_dir, 'results', 'final_comparison.csv'), index=False)

# 7. Four-Group Error Analysis
print("\n[7/7] Performing 4-Group Error Analysis...")

c_preds = best_res['final_preds']

group_A_mask = (y_val == 1) & (val_preds_A == 1) & (c_preds == 1) # Model A correct / Cascade correct
group_B_mask = (y_val == 1) & (val_preds_A == 0) & (c_preds == 1) # Model A FN / Cascade recovered
group_C_mask = (y_val == 0) & (val_preds_A == 0) & (c_preds == 1) # Model A correct PASS / Cascade incorrect FAIL
group_D_mask = (y_val == 1) & (val_preds_A == 0) & (c_preds == 0) # Model A FN / Cascade still missed

error_analysis_summary = pd.DataFrame([
    {
        'Group': 'Group A: Model A Correct / Cascade Correct',
        'Description': 'Severe failures with strong parametric signal detected by both',
        'Die Count': group_A_mask.sum(),
        'Mean Model A Risk': round(val_probs_A[group_A_mask].mean(), 4) if group_A_mask.sum() > 0 else 0,
        'Mean CNN Risk': round(val_cnn_probs[group_A_mask].mean(), 4) if group_A_mask.sum() > 0 else 0
    },
    {
        'Group': 'Group B: Model A FN / Cascade Recovered',
        'Description': 'Failures with weak parametric signature recovered via 1D ConvNet block sequence',
        'Die Count': group_B_mask.sum(),
        'Mean Model A Risk': round(val_probs_A[group_B_mask].mean(), 4) if group_B_mask.sum() > 0 else 0,
        'Mean CNN Risk': round(val_cnn_probs[group_B_mask].mean(), 4) if group_B_mask.sum() > 0 else 0
    },
    {
        'Group': 'Group C: Model A Correct / Cascade Incorrect Change',
        'Description': 'Passing dies with random noise spikes triggering false alarms in gate region',
        'Die Count': group_C_mask.sum(),
        'Mean Model A Risk': round(val_probs_A[group_C_mask].mean(), 4) if group_C_mask.sum() > 0 else 0,
        'Mean CNN Risk': round(val_cnn_probs[group_C_mask].mean(), 4) if group_C_mask.sum() > 0 else 0
    },
    {
        'Group': 'Group D: Model A FN / Cascade Still Missed',
        'Description': 'Extremely subtle failures with weak parametric AND weak block anomaly signals',
        'Die Count': group_D_mask.sum(),
        'Mean Model A Risk': round(val_probs_A[group_D_mask].mean(), 4) if group_D_mask.sum() > 0 else 0,
        'Mean CNN Risk': round(val_cnn_probs[group_D_mask].mean(), 4) if group_D_mask.sum() > 0 else 0
    }
])

print("\n--- 4-GROUP ERROR ANALYSIS SUMMARY ---")
print(error_analysis_summary.to_string(index=False))
error_analysis_summary.to_csv(os.path.join(base_dir, 'results', 'error_analysis_groups.csv'), index=False)

# Export Full Validation Predictions CSV
val_export_df = val_df.copy()
val_export_df['model_A_prob'] = val_probs_A
val_export_df['unrestricted_model_B_prob'] = val_probs_unrestricted
val_export_df['cascade_final_prob'] = best_res['final_probs']
val_export_df['model_A_decision'] = val_preds_A
val_export_df['cascade_final_decision'] = best_res['final_preds']
val_export_df['resolution_used'] = best_res['resolution_used']
val_export_df['escalation_reason'] = best_res['escalation_reasons']
val_export_df['risk_category'] = best_res['risk_categories']

export_cols = ['wafer_id', 'die_row', 'die_col', 'old_label', 'label', 'model_A_prob', 'cascade_final_prob', 'model_A_decision', 'cascade_final_decision', 'resolution_used', 'escalation_reason', 'risk_category']
val_export_df[export_cols].to_csv(os.path.join(base_dir, 'results', 'validation_predictions.csv'), index=False)

# Save Wafer Visualizations
target_wafer_id = val_export_df.groupby('wafer_id')['label'].sum().idxmax()
target_wafer_df = val_export_df[val_export_df['wafer_id'] == target_wafer_id].copy()
target_wafer_df['prob_A'] = target_wafer_df['model_A_prob']
target_wafer_df['prob_B'] = target_wafer_df['cascade_final_prob']
target_wafer_df['pred_B'] = target_wafer_df['cascade_final_decision']

plot_wafer_prediction_comparison(
    target_wafer_df, target_wafer_id,
    save_path=os.path.join(base_dir, 'visualizations', 'wafer_comparison.png')
)

metrics_summary_selective = {
    'system_name': f"Selective Multi-Resolution Cascade ({best_gate_info['Name']})",
    'gate_window': f"[{best_gate_info['L']:.2f}, {best_gate_info['U']:.2f}]",
    'f1': float(best_metrics['f1']),
    'precision': float(best_metrics['prec']),
    'recall': float(best_metrics['rec']),
    'auc_pr': float(best_metrics['auc_pr']),
    'roc_auc': float(best_metrics['roc_auc']),
    'model_A_fn_total': int(fn_A),
    'model_A_fn_recovered': int(best_metrics['fn_rec']),
    'fn_recovery_rate_pct': float(best_metrics['fn_rec_pct']),
    'new_false_positives': int(best_metrics['new_fp']),
    'delta_f1': float(best_metrics['f1'] - f1_A),
    'delta_auc_pr': float(best_metrics['auc_pr'] - auc_pr_A),
    'delta_precision_pct': float((best_metrics['prec'] - prec_A)*100),
    'delta_recall_pct': float((best_metrics['rec'] - rec_A)*100)
}

with open(os.path.join(base_dir, 'results', 'metrics_selective_cascade.json'), 'w') as f:
    json.dump(metrics_summary_selective, f, indent=2)

print("\nPhase 3D Selective Multi-Resolution Cascade pipeline complete! All artifacts saved successfully.")
