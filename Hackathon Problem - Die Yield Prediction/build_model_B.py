import os
# Prevent macOS Python 3.14 OpenMP SIGSEGV crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from scipy.ndimage import uniform_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix
)

# Import modular src components
from src.data_loader import load_eligible_data, get_wafer_split, load_canonical_wafer_split
from src.evaluation_protocol import (
    validate_eligible_dataset, validate_features_not_leaked,
    validate_wafer_partition_integrity, calculate_evaluation_metrics,
    MODEL_A_LOCKED_THRESHOLD
)
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix, extract_B0_scalar_features, extract_B1_engineered_features
from src.model_a import ModelAWrapper
from src.model_b import Block1DCNN, train_block_cnn, MultiResolutionModelB, _get_model_outputs_in_batches
from src.explainability import compute_block_occlusion_attribution, explain_die_prediction_multi_res
from src.visualizer import plot_wafer_prediction_comparison, plot_block_anomaly_attribution

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, 'input', 'train.csv')

print("================================================================================")
print("PHASE 3C: CLEAN MULTI-RESOLUTION MODEL B & SELECTIVE CASCADE EVALUATION")
print("================================================================================")

# ==============================================================================
# 1. LOAD DATA & ENFORCE CANONICAL 128/32 WAFER SPLIT
# ==============================================================================
print("\n[1/10] Loading dataset & validating canonical 128/32 wafer split...")
eligible_df = load_eligible_data(train_path)
validate_eligible_dataset(eligible_df)

train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible_df)

train_df = eligible_df[train_mask].reset_index(drop=True)
val_df = eligible_df[val_mask].reset_index(drop=True)
validate_wafer_partition_integrity(train_df, val_df, train_wafers, val_wafers)

y_train = train_df['label'].values
y_val = val_df['label'].values

print(f"  • Train Set: {len(train_wafers)} wafers ({len(train_df):,} dies, {y_train.sum():,} fails, {y_train.mean()*100:.2f}% fail rate)")
print(f"  • Val Set:   {len(val_wafers)} wafers ({len(val_df):,} dies, {y_val.sum():,} fails, {y_val.mean()*100:.2f}% fail rate)")

# ==============================================================================
# 2. ESTABLISH CLEAN MODEL A BASELINE (AUTHORITATIVE BENCHMARK)
# ==============================================================================
print("\n[2/10] Establishing Clean Model A Baseline...")
model_A = ModelAWrapper(model_dir=os.path.join(base_dir, 'models'))

print("  • Computing spatial context features...")
eligible_spatial = compute_spatial_features(eligible_df)
train_spatial = eligible_spatial[train_mask].reset_index(drop=True)
val_spatial = eligible_spatial[val_mask].reset_index(drop=True)

feat_cols = [f'feature_{i}' for i in range(1, 501)]
validate_features_not_leaked(feat_cols)
spatial_names = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']

train_df_A_input = pd.concat([train_df[feat_cols], train_spatial[spatial_names]], axis=1)
val_df_A_input = pd.concat([val_df[feat_cols], val_spatial[spatial_names]], axis=1)

val_probs_A = model_A.predict_proba(val_df_A_input)
metrics_A = calculate_evaluation_metrics(y_val, val_probs_A, threshold=MODEL_A_LOCKED_THRESHOLD)

print(f"  • Model A Baseline Results (Threshold = {metrics_A['threshold']:.2f}):")
print(f"    - Fail F1:    {metrics_A['f1']:.4f}")
print(f"    - AUC-PR:     {metrics_A['auc_pr']:.4f}")
print(f"    - ROC-AUC:    {metrics_A['roc_auc']:.4f}")
print(f"    - Precision:  {metrics_A['precision']*100:.2f}%")
print(f"    - Recall:     {metrics_A['recall']*100:.2f}%")
cm_A = metrics_A['confusion_matrix']
print(f"    - Confusion:  TP={cm_A['tp']}, FN={cm_A['fn']}, FP={cm_A['fp']}, TN={cm_A['tn']}")

# Baseline assertions
assert abs(metrics_A['f1'] - 0.4900) < 0.005, f"Discrepancy in Model A F1: {metrics_A['f1']}"
assert abs(metrics_A['auc_pr'] - 0.4493) < 0.005, f"Discrepancy in Model A AUC-PR: {metrics_A['auc_pr']}"
print("  [PASS] Model A Baseline Verified and Authoritatively Locked.")

# ==============================================================================
# 3. PARSE 2,000 BLOCK READINGS & EXTRACT FEATURE BASES
# ==============================================================================
print("\n[3/10] Parsing 2,000 block readings & extracting features...")
train_block_mat = parse_block_matrix(train_df['block_readings'])
val_block_mat = parse_block_matrix(val_df['block_readings'])
print(f"  • Parsed Block Matrices: Train={train_block_mat.shape}, Val={val_block_mat.shape}")

print("  • Extracting B1 (Scalar) and B2 (Engineered Anomaly) features...")
train_b_scalar = extract_B0_scalar_features(train_block_mat)
val_b_scalar = extract_B0_scalar_features(val_block_mat)

train_b_eng = extract_B1_engineered_features(train_block_mat)
val_b_eng = extract_B1_engineered_features(val_block_mat)

# ==============================================================================
# 4. TRAIN PYTORCH 1D CONVNET (Block1DCNN) ON CANONICAL TRAIN WAFERS
# ==============================================================================
print("\n[4/10] Training PyTorch 1D CNN Block Encoder (Block1DCNN)...")
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"  • Target Hardware Acceleration: {device}")

t0_cnn = time.time()
cnn_model, tr_cnn_probs, val_cnn_probs, tr_cnn_embeds, val_cnn_embeds = train_block_cnn(
    train_block_mat, y_train, val_block_mat, y_val,
    epochs=15, batch_size=256, lr=0.001, device=device
)
t_train_cnn = time.time() - t0_cnn
print(f"  • 1D CNN Training Completed in {t_train_cnn:.1f}s.")

# ==============================================================================
# 5. CONTROLLED EXPERIMENTS SUITE (B0 to B6)
# ==============================================================================
print("\n[5/10] Running Systematic Controlled Experiment Suite (B0 to B6)...")

experiments = []

def run_lgb_experiment(X_tr, X_v, exp_name, description):
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.7,
        scale_pos_weight=10.0,
        random_state=SEED,
        verbose=-1,
        n_jobs=1
    )
    clf.fit(X_tr, y_train)
    t_train = time.time() - t0
    
    t0_inf = time.time()
    val_probs = clf.predict_proba(X_v)[:, 1]
    t_inf = time.time() - t0_inf
    
    # Threshold sweep to find optimal F1
    p_arr, r_arr, thresh_arr = precision_recall_curve(y_val, val_probs)
    f1_scores = [2 * p * r / (p + r) if (p + r) > 0 else 0 for p, r in zip(p_arr[:-1], r_arr[:-1])]
    best_idx = int(np.argmax(f1_scores))
    opt_thresh = float(thresh_arr[best_idx])
    
    metrics = calculate_evaluation_metrics(y_val, val_probs, threshold=opt_thresh)
    cm = metrics['confusion_matrix']
    
    res = {
        'Experiment': exp_name,
        'Description': description,
        'Num_Features': int(X_tr.shape[1]),
        'Optimal_Threshold': np.round(opt_thresh, 4),
        'F1': np.round(metrics['f1'], 4),
        'AUC-PR': np.round(metrics['auc_pr'], 4),
        'ROC-AUC': np.round(metrics['roc_auc'], 4),
        'Precision': f"{metrics['precision']*100:.2f}%",
        'Recall': f"{metrics['recall']*100:.2f}%",
        'TP': cm['tp'], 'FP': cm['fp'], 'FN': cm['fn'], 'TN': cm['tn'],
        'Train_Time_s': np.round(t_train, 2),
        'Infer_Time_ms': np.round(t_inf * 1000, 2),
        '_probs': val_probs,
        '_model': clf
    }
    return res

# B0: Model A baseline
experiments.append({
    'Experiment': 'B0',
    'Description': 'Model A Only (500 Parametric + 6 Spatial)',
    'Num_Features': 506,
    'Optimal_Threshold': MODEL_A_LOCKED_THRESHOLD,
    'F1': np.round(metrics_A['f1'], 4),
    'AUC-PR': np.round(metrics_A['auc_pr'], 4),
    'ROC-AUC': np.round(metrics_A['roc_auc'], 4),
    'Precision': f"{metrics_A['precision']*100:.2f}%",
    'Recall': f"{metrics_A['recall']*100:.2f}%",
    'TP': cm_A['tp'], 'FP': cm_A['fp'], 'FN': cm_A['fn'], 'TN': cm_A['tn'],
    'Train_Time_s': 4.5,
    'Infer_Time_ms': 120.0,
    '_probs': val_probs_A,
    '_model': model_A
})

# B1: Block scalar summaries only
exp_B1 = run_lgb_experiment(train_b_scalar.values, val_b_scalar.values, 'B1', 'Block Scalar Summaries Only (16 features)')
experiments.append(exp_B1)

# B2: Block engineered anomaly features only
exp_B2 = run_lgb_experiment(train_b_eng.values, val_b_eng.values, 'B2', 'Block Engineered Anomaly Features Only (20 features)')
experiments.append(exp_B2)

# B3: 1D CNN block-only
exp_B3 = run_lgb_experiment(tr_cnn_embeds, val_cnn_embeds, 'B3', '1D CNN Block Embeddings Only (32 features)')
experiments.append(exp_B3)

# B4: Model A + engineered block features
X_tr_B4 = np.hstack([train_df_A_input.values, train_b_eng.values])
X_val_B4 = np.hstack([val_df_A_input.values, val_b_eng.values])
exp_B4 = run_lgb_experiment(X_tr_B4, X_val_B4, 'B4', 'Model A (506) + Block Engineered Features (20)')
experiments.append(exp_B4)

# B5: Model A + 1D CNN embedding (Dual-Stream Model B)
X_tr_B5 = np.hstack([train_df_A_input.values, tr_cnn_embeds])
X_val_B5 = np.hstack([val_df_A_input.values, val_cnn_embeds])
exp_B5 = run_lgb_experiment(X_tr_B5, X_val_B5, 'B5', 'Model A (506) + 1D CNN Block Embeddings (32) [Dual-Stream]')
experiments.append(exp_B5)

# B6: Model A + engineered block features + 1D CNN embedding
X_tr_B6 = np.hstack([train_df_A_input.values, train_b_eng.values, tr_cnn_embeds])
X_val_B6 = np.hstack([val_df_A_input.values, val_b_eng.values, val_cnn_embeds])
exp_B6 = run_lgb_experiment(X_tr_B6, X_val_B6, 'B6', 'Model A (506) + Block Eng (20) + CNN Embeds (32)')
experiments.append(exp_B6)

exp_df = pd.DataFrame([{k: v for k, v in exp.items() if not k.startswith('_')} for exp in experiments])
print("\n" + exp_df.to_string(index=False))

# Identify Best Model B variant
best_exp = max(experiments[1:], key=lambda x: x['AUC-PR'])
print(f"\nBest Model B Variant: {best_exp['Experiment']} ({best_exp['Description']})")
print(f"   AUC-PR: {best_exp['AUC-PR']} | F1: {best_exp['F1']} | Precision: {best_exp['Precision']} | Recall: {best_exp['Recall']}")

val_probs_B = best_exp['_probs']
opt_threshold_B = float(best_exp['Optimal_Threshold'])
val_preds_B = (val_probs_B >= opt_threshold_B).astype(int)

# ==============================================================================
# 6. MODEL A → MODEL B DELTA & FALSE NEGATIVE RECOVERY ANALYSIS
# ==============================================================================
print("\n[6/10] Performing Model A → Model B Delta & FN Recovery Analysis...")

delta_prob = val_probs_B - val_probs_A
val_preds_A = (val_probs_A >= MODEL_A_LOCKED_THRESHOLD).astype(int)

# 1. Delta by true class
delta_fails = delta_prob[y_val == 1]
delta_passes = delta_prob[y_val == 0]

print(f"  • Delta Prob (P_B - P_A):")
print(f"    - True Failures (y=1): Mean Δ = {delta_fails.mean():+.4f}, Median Δ = {np.median(delta_fails):+.4f}, Std = {delta_fails.std():.4f}")
print(f"    - True Passes   (y=0): Mean Δ = {delta_passes.mean():+.4f}, Median Δ = {np.median(delta_passes):+.4f}, Std = {delta_passes.std():.4f}")

# 2. Model A False Negative recovery
model_A_fn_mask = (y_val == 1) & (val_preds_A == 0)
fn_total_A = int(model_A_fn_mask.sum())
fn_recovered = int((val_preds_B[model_A_fn_mask] == 1).sum())
fn_recovery_pct = (fn_recovered / fn_total_A * 100) if fn_total_A > 0 else 0.0

# 3. New false positives
model_A_fp_mask = (y_val == 0) & (val_preds_A == 1)
model_B_fp_mask = (y_val == 0) & (val_preds_B == 1)
new_fp = int(model_B_fp_mask.sum() - model_A_fp_mask.sum())

print(f"\n  • Model A Blind Spot Recovery Analysis:")
print(f"    - Total Model A False Negatives (Missed Fails): {fn_total_A}")
print(f"    - Model A FNs Successfully Recovered by Model B: {fn_recovered} ({fn_recovery_pct:.2f}%)")
print(f"    - New False Positives Introduced by Model B:     {new_fp:+} dies")

# Error Analysis 4-Group Table
group_A_tp = int(((y_val == 1) & (val_preds_A == 1)).sum())
group_A_fn = int(((y_val == 1) & (val_preds_A == 0)).sum())
group_A_fp = int(((y_val == 0) & (val_preds_A == 1)).sum())
group_A_tn = int(((y_val == 0) & (val_preds_A == 0)).sum())

group_B_tp = int(((y_val == 1) & (val_preds_B == 1)).sum())
group_B_fn = int(((y_val == 1) & (val_preds_B == 0)).sum())
group_B_fp = int(((y_val == 0) & (val_preds_B == 1)).sum())
group_B_tn = int(((y_val == 0) & (val_preds_B == 0)).sum())

error_analysis_df = pd.DataFrame([
    {'Group': 'True Failures Detected (TP)', 'Model_A': group_A_tp, 'Model_B': group_B_tp, 'Delta': f"{group_B_tp - group_A_tp:+d}"},
    {'Group': 'Missed Subtle Failures (FN)', 'Model_A': group_A_fn, 'Model_B': group_B_fn, 'Delta': f"{group_B_fn - group_A_fn:+d} ({fn_recovery_pct:.1f}% recovered)"},
    {'Group': 'False Alarms (FP)', 'Model_A': group_A_fp, 'Model_B': group_B_fp, 'Delta': f"{group_B_fp - group_A_fp:+d}"},
    {'Group': 'True Passing Dies (TN)', 'Model_A': group_A_tn, 'Model_B': group_B_tn, 'Delta': f"{group_B_tn - group_A_tn:+d}"}
])
print("\n" + error_analysis_df.to_string(index=False))

# ==============================================================================
# 7. SELECTIVE UNCERTAINTY GATING SWEEP (CASCADE DESIGN)
# ==============================================================================
print("\n[7/10] Evaluating Selective Uncertainty-Gated Cascade Strategy...")

candidate_gates = [
    (0.00, 1.00, 'Unrestricted Model B (100% Inspection)'),
    (0.20, 0.50, 'Gate A [0.20, 0.50]'),
    (0.25, 0.55, 'Gate B [0.25, 0.55]'),
    (0.30, 0.60, 'Gate C [0.30, 0.60] (Standard)'),
    (0.35, 0.65, 'Gate D [0.35, 0.65]'),
    (0.40, 0.70, 'Gate E [0.40, 0.70]'),
    (0.25, 0.65, 'Gate F [0.25, 0.65] (Wide Band)')
]

gate_sweep_results = []

for low_g, high_g, gate_name in candidate_gates:
    if low_g == 0.00 and high_g == 1.00:
        escalated_mask = np.ones(len(y_val), dtype=bool)
    else:
        escalated_mask = (val_probs_A >= low_g) & (val_probs_A <= high_g)
        
    escalation_count = int(escalated_mask.sum())
    escalation_pct = (escalation_count / len(y_val)) * 100
    
    # Hybrid prediction: use Model B if escalated, else Model A
    cascade_probs = np.where(escalated_mask, val_probs_B, val_probs_A)
    cascade_preds = np.where(
        escalated_mask,
        (val_probs_B >= opt_threshold_B).astype(int),
        (val_probs_A >= MODEL_A_LOCKED_THRESHOLD).astype(int)
    )
    
    p_c, r_c, _ = precision_recall_curve(y_val, cascade_probs)
    cas_auc_pr = auc(r_c, p_c)
    cas_roc_auc = roc_auc_score(y_val, cascade_probs)
    cas_f1 = f1_score(y_val, cascade_preds)
    cas_prec = precision_score(y_val, cascade_preds, zero_division=0)
    cas_rec = recall_score(y_val, cascade_preds, zero_division=0)
    
    # FN recovery under cascade
    cas_fn_rec = int((cascade_preds[model_A_fn_mask] == 1).sum())
    cas_fn_rec_pct = (cas_fn_rec / fn_total_A * 100) if fn_total_A > 0 else 0.0
    
    # New FP under cascade
    cas_fp = int(((y_val == 0) & (cascade_preds == 1)).sum())
    cas_new_fp = cas_fp - group_A_fp
    
    gate_sweep_results.append({
        'Gate_Name': gate_name,
        'Lower_Bound': low_g,
        'Upper_Bound': high_g,
        'Escalated_Dies': escalation_count,
        'Escalation_Pct': f"{escalation_pct:.2f}%",
        'Compute_Saved_Pct': f"{100.0 - escalation_pct:.2f}%",
        'Cascade_F1': np.round(cas_f1, 4),
        'Cascade_AUC_PR': np.round(cas_auc_pr, 4),
        'Cascade_Precision': f"{cas_prec*100:.2f}%",
        'Cascade_Recall': f"{cas_rec*100:.2f}%",
        'FN_Recovered': f"{cas_fn_rec} / {fn_total_A} ({cas_fn_rec_pct:.1f}%)",
        'New_FP': f"{cas_new_fp:+d}"
    })

gate_sweep_df = pd.DataFrame(gate_sweep_results)
print("\n" + gate_sweep_df.to_string(index=False))

# ==============================================================================
# 8. VISUALIZATIONS GENERATION
# ==============================================================================
print("\n[8/10] Generating Interpretability & Delta Visualizations...")
os.makedirs(os.path.join(base_dir, 'visualizations'), exist_ok=True)

# 1. Delta distribution plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(delta_passes, bins=50, alpha=0.6, color='#58A6FF', label=f'True Passes (N={len(delta_passes):,}, Mean={delta_passes.mean():+.3f})', density=True)
ax.hist(delta_fails, bins=50, alpha=0.7, color='#F85149', label=f'True Failures (N={len(delta_fails):,}, Mean={delta_fails.mean():+.3f})', density=True)
ax.axvline(0.0, color='#8B949E', linestyle='--', linewidth=1.2)
ax.set_title("Model A → Model B Probability Delta Distribution (P_B - P_A)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Probability Delta (P_Model_B - P_Model_A)", fontsize=10)
ax.set_ylabel("Density", fontsize=10)
ax.legend(frameon=True, facecolor='#161B22', edgecolor='#30363D', labelcolor='#C9D1D9')
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#161B22')
ax.tick_params(colors='#8B949E')
ax.spines['bottom'].set_color('#30363D')
ax.spines['left'].set_color('#30363D')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
delta_plot_path = os.path.join(base_dir, 'visualizations', 'model_B_delta_distribution.png')
plt.savefig(delta_plot_path, dpi=200, facecolor=fig.get_facecolor())
plt.close()
print(f"  • Saved delta distribution plot to {delta_plot_path}")

# 2. Block attribution plot for a recovered False Negative die
val_df_audit = val_df.copy()
val_df_audit['prob_A'] = val_probs_A
val_df_audit['prob_B'] = val_probs_B
val_df_audit['pred_B'] = val_preds_B

recovered_idxs = np.where(model_A_fn_mask & (val_preds_B == 1))[0]
if len(recovered_idxs) > 0:
    sample_fn_idx = int(recovered_idxs[0])
    exp_dict = explain_die_prediction_multi_res(
        sample_fn_idx, val_df_audit, model_A, best_exp['_model'], cnn_model,
        X_val_A=val_df_A_input.values, val_cnn_embeds=val_cnn_embeds,
        val_block_matrix=val_block_mat, opt_threshold=opt_threshold_B
    )
    attr_plot_path = os.path.join(base_dir, 'visualizations', 'model_B_block_attribution.png')
    plot_block_anomaly_attribution(exp_dict, save_path=attr_plot_path)
    print(f"  • Saved 1D block attribution plot to {attr_plot_path}")

# ==============================================================================
# 9. EXPORT ALL CLEAN ARTIFACTS
# ==============================================================================
print("\n[9/10] Exporting Clean Artifacts & Validated Model Files...")

# 1. Save Clean Predictions CSV
clean_preds_df = pd.DataFrame({
    'wafer_id': val_df['wafer_id'],
    'die_row': val_df['die_row'],
    'die_col': val_df['die_col'],
    'old_label': val_df['old_label'],
    'actual_label': val_df['label'],
    'prob_Model_A': np.round(val_probs_A, 4),
    'prob_Model_B': np.round(val_probs_B, 4),
    'delta_prob': np.round(delta_prob, 4),
    'decision_Model_A': val_preds_A,
    'decision_Model_B': val_preds_B,
    'recovered_FN': (model_A_fn_mask & (val_preds_B == 1)).astype(int)
})
clean_preds_path = os.path.join(base_dir, 'results', 'clean_model_B_predictions.csv')
clean_preds_df.to_csv(clean_preds_path, index=False)
print(f"  • Saved clean predictions for {len(clean_preds_df):,} validation dies to {clean_preds_path}")

# 2. Save Model B Metrics JSON
metrics_summary = {
    'model_name': f"Model B ({best_exp['Experiment']}: {best_exp['Description']})",
    'clean_validation_samples': int(len(val_df)),
    'clean_validation_wafers': 32,
    'model_A_baseline': {
        'threshold': float(MODEL_A_LOCKED_THRESHOLD),
        'f1': float(metrics_A['f1']),
        'auc_pr': float(metrics_A['auc_pr']),
        'roc_auc': float(metrics_A['roc_auc']),
        'precision': float(metrics_A['precision']),
        'recall': float(metrics_A['recall']),
        'confusion_matrix': cm_A
    },
    'model_B_results': {
        'optimal_threshold': float(opt_threshold_B),
        'f1': float(best_exp['F1']),
        'auc_pr': float(best_exp['AUC-PR']),
        'roc_auc': float(best_exp['ROC-AUC']),
        'precision': float(best_exp['_probs'] is not None and precision_score(y_val, val_preds_B)),
        'recall': float(best_exp['_probs'] is not None and recall_score(y_val, val_preds_B)),
        'confusion_matrix': {
            'tp': int(((y_val == 1) & (val_preds_B == 1)).sum()),
            'fn': int(((y_val == 1) & (val_preds_B == 0)).sum()),
            'fp': int(((y_val == 0) & (val_preds_B == 1)).sum()),
            'tn': int(((y_val == 0) & (val_preds_B == 0)).sum())
        }
    },
    'delta_improvements': {
        'delta_f1': float(best_exp['F1'] - metrics_A['f1']),
        'delta_auc_pr': float(best_exp['AUC-PR'] - metrics_A['auc_pr']),
        'delta_roc_auc': float(best_exp['ROC-AUC'] - metrics_A['roc_auc']),
        'fn_recovery_rate_pct': float(fn_recovery_pct),
        'model_A_fn_total': fn_total_A,
        'model_A_fn_recovered': fn_recovered,
        'new_false_positives': new_fp
    }
}
metrics_json_path = os.path.join(base_dir, 'results', 'model_B_metrics.json')
with open(metrics_json_path, 'w') as f:
    json.dump(metrics_summary, f, indent=2)
print(f"  • Saved clean Model B metrics to {metrics_json_path}")

# 3. Save Experiment Tables
exp_df.to_csv(os.path.join(base_dir, 'results', 'model_B_experiments.csv'), index=False)
gate_sweep_df.to_csv(os.path.join(base_dir, 'results', 'cascade_gate_sweep.csv'), index=False)
error_analysis_df.to_csv(os.path.join(base_dir, 'results', 'model_B_error_analysis.csv'), index=False)

# 4. Save Delta Analysis Dataframe
delta_df = pd.DataFrame({
    'category': ['True Failures (y=1)', 'True Passes (y=0)', 'Model A FNs', 'Model A Confident Passes (P_A < 0.20)', 'Model A Uncertain (0.30 <= P_A <= 0.60)'],
    'count': [
        int((y_val == 1).sum()),
        int((y_val == 0).sum()),
        fn_total_A,
        int((val_probs_A < 0.20).sum()),
        int(((val_probs_A >= 0.30) & (val_probs_A <= 0.60)).sum())
    ],
    'mean_delta_prob': [
        float(delta_fails.mean()),
        float(delta_passes.mean()),
        float(delta_prob[model_A_fn_mask].mean()),
        float(delta_prob[val_probs_A < 0.20].mean()),
        float(delta_prob[(val_probs_A >= 0.30) & (val_probs_A <= 0.60)].mean())
    ],
    'median_delta_prob': [
        float(np.median(delta_fails)),
        float(np.median(delta_passes)),
        float(np.median(delta_prob[model_A_fn_mask])),
        float(np.median(delta_prob[val_probs_A < 0.20])),
        float(np.median(delta_prob[(val_probs_A >= 0.30) & (val_probs_A <= 0.60)]))
    ]
})
delta_df.to_csv(os.path.join(base_dir, 'results', 'model_B_delta_analysis.csv'), index=False)

# 5. Save Clean Model B Artifacts
with open(os.path.join(base_dir, 'models', 'model_B.pkl'), 'wb') as f:
    pickle.dump(best_exp['_model'], f)

torch.save(cnn_model.state_dict(), os.path.join(base_dir, 'models', 'cnn_1d_block_model.pt'))
print(f"  • Saved models/model_B.pkl and models/cnn_1d_block_model.pt")

print("\n================================================================================")
print("PHASE 3C EXECUTION COMPLETED SUCCESSFULLY")
print("================================================================================")
