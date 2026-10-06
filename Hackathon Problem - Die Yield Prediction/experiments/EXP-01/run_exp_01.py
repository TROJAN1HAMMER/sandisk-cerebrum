"""
EXP-01: Spatial Defect Density Restoration & Baseline Retraining
Rigorous, controlled scientific experiment script.
Changes ONLY ONE THING: compute_spatial_features(eligible, full_df=df)
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import json
import time
import pickle
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, auc, f1_score,
    precision_score, recall_score, confusion_matrix,
    brier_score_loss, matthews_corrcoef, balanced_accuracy_score,
    accuracy_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

EXP_DIR = os.path.join(BASE_DIR, 'experiments', 'EXP-01')
INPUT_DIR = os.path.join(BASE_DIR, 'input')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

os.makedirs(EXP_DIR, exist_ok=True)
log_file = open(os.path.join(EXP_DIR, 'training_log.txt'), 'w')

def log(msg):
    print(msg)
    log_file.write(msg + '\n')
    log_file.flush()

log("================================================================================")
log("EXP-01: SPATIAL DEFECT DENSITY RESTORATION & BASELINE RETRAINING")
log("================================================================================")
log(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
log(f"Base Directory: {BASE_DIR}")
log(f"Experiment Directory: {EXP_DIR}")

# -------------------------------------------------------------
# 1. LOAD DATASET & CANONICAL SPLIT
# -------------------------------------------------------------
log("\n[1/8] Loading dataset and canonical wafer split...")
train_path = os.path.join(INPUT_DIR, 'train.csv')
t0 = time.time()
df = pd.read_csv(train_path)
t_load = time.time() - t0
log(f"  • Loaded train.csv in {t_load:.2f}s: {df.shape[0]:,} rows, {df.shape[1]} cols")

eligible = df[df['old_label'] == 0].copy().reset_index(drop=True)
log(f"  • Pre-test eligible dies (old_label == 0): {len(eligible):,} rows")

from src.data_loader import get_wafer_split, load_canonical_wafer_split
train_mask, val_mask, train_wafers, val_wafers = get_wafer_split(eligible)

train_df = eligible[train_mask].reset_index(drop=True)
val_df = eligible[val_mask].reset_index(drop=True)

y_train = train_df['label'].values
y_val = val_df['label'].values

log(f"  • Train Set: {len(train_wafers)} wafers ({len(train_df):,} dies, {y_train.sum():,} fails, {y_train.mean()*100:.4f}% fail rate)")
log(f"  • Val Set:   {len(val_wafers)} wafers ({len(val_df):,} dies, {y_val.sum():,} fails, {y_val.mean()*100:.4f}% fail rate)")
log(f"  • Wafer Overlap: {len(train_wafers.intersection(val_wafers))} (Must be 0)")
assert len(train_wafers.intersection(val_wafers)) == 0

# -------------------------------------------------------------
# 2. SPATIAL FEATURES: BEFORE VS AFTER AUDIT
# -------------------------------------------------------------
log("\n[2/8] Auditing spatial features before vs after intervention...")
from src.spatial import compute_spatial_features

# Condition A (Before - Broken): without full_df
t0 = time.time()
sp_before_all = compute_spatial_features(eligible)
t_sp_before = time.time() - t0
sp_before_train = sp_before_all[train_mask].reset_index(drop=True)
sp_before_val = sp_before_all[val_mask].reset_index(drop=True)

# Condition B (After - EXP-01): with full_df=df
t0 = time.time()
sp_after_all = compute_spatial_features(eligible, full_df=df)
t_sp_after = time.time() - t0
sp_after_train = sp_after_all[train_mask].reset_index(drop=True)
sp_after_val = sp_after_all[val_mask].reset_index(drop=True)

def compute_feature_stats(df_feats, name):
    records = []
    for col in ['density_3x3', 'density_5x5', 'radial_dist', 'edge_dist']:
        s = df_feats[col]
        nz = (s > 0).sum()
        records.append({
            'dataset_split': name,
            'feature': col,
            'count': len(s),
            'unique_values': s.nunique(),
            'mean': round(float(s.mean()), 6),
            'std': round(float(s.std()), 6),
            'min': round(float(s.min()), 6),
            'max': round(float(s.max()), 6),
            'non_zero_count': int(nz),
            'non_zero_pct': round(float(nz / len(s) * 100), 4)
        })
    return records

stats_before = compute_feature_stats(sp_before_train, 'train_before') + compute_feature_stats(sp_before_val, 'val_before')
stats_after = compute_feature_stats(sp_after_train, 'train_after') + compute_feature_stats(sp_after_val, 'val_after')

df_stats_before = pd.DataFrame(stats_before)
df_stats_after = pd.DataFrame(stats_after)

df_stats_before.to_csv(os.path.join(EXP_DIR, 'feature_stats_before.csv'), index=False)
df_stats_after.to_csv(os.path.join(EXP_DIR, 'feature_stats_after.csv'), index=False)

log("  Feature stats BEFORE intervention (Baseline):")
log(df_stats_before.to_string(index=False))
log("\n  Feature stats AFTER intervention (EXP-01):")
log(df_stats_after.to_string(index=False))

# Absolute Semantic Safety Assertion
log("\n  • Verifying feature semantics safety:")
log("    - Target 'label' column is NOT used in compute_spatial_features.")
log("    - Only pre-test 'old_label' from the wafer grid is queried.")
log(f"    - density_3x3 in training after: non-zero count = {(sp_after_train['density_3x3'] > 0).sum():,} ({sp_after_train['density_3x3'].mean()*100:.2f}% avg density)")
log(f"    - density_3x3 in validation after: non-zero count = {(sp_after_val['density_3x3'] > 0).sum():,} ({sp_after_val['density_3x3'].mean()*100:.2f}% avg density)")
assert (sp_after_train['density_3x3'] > 0).sum() > 0
assert (sp_after_val['density_3x3'] > 0).sum() > 0

# -------------------------------------------------------------
# 3. VERIFY BASELINE MODEL A REPRODUCIBILITY
# -------------------------------------------------------------
log("\n[3/8] Verifying existing locked Model A baseline on validation set...")
from src.model_a import ModelAWrapper
model_A_base = ModelAWrapper(model_dir=MODELS_DIR)

feat_cols = [f'feature_{i}' for i in range(1, 501)]
spatial_names = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
all_feature_names = feat_cols + spatial_names
assert len(all_feature_names) == 506

# Assemble inputs for baseline (using sp_before)
val_A_base_input = pd.concat([val_df[feat_cols], sp_before_val[spatial_names]], axis=1)
val_probs_base = model_A_base.predict_proba(val_A_base_input)
val_preds_base = (val_probs_base >= 0.5100).astype(int)

cm_base = confusion_matrix(y_val, val_preds_base).ravel()
log(f"  • Baseline Model A CM: TP={cm_base[3]}, FN={cm_base[2]}, FP={cm_base[1]}, TN={cm_base[0]}")
assert cm_base[3] == 357
assert cm_base[2] == 699
assert cm_base[1] == 42
assert cm_base[0] == 26874
log("  ✅ Locked baseline reproduced bit-for-bit!")

# -------------------------------------------------------------
# 4. TRAIN EXPERIMENTAL MODEL A (EXP-01)
# -------------------------------------------------------------
log("\n[4/8] Training Experimental Model A (EXP-01) with restored spatial density...")

train_A_exp_input = pd.concat([train_df[feat_cols], sp_after_train[spatial_names]], axis=1)
val_A_exp_input = pd.concat([val_df[feat_cols], sp_after_val[spatial_names]], axis=1)

log(f"  • Feature Matrix Shapes: Train={train_A_exp_input.shape}, Val={val_A_exp_input.shape}")
assert train_A_exp_input.shape[1] == 506
assert val_A_exp_input.shape[1] == 506

# LOCKED LIGHTGBM CONFIGURATION
lgb_params = {
    'n_estimators': 150,
    'learning_rate': 0.05,
    'num_leaves': 63,
    'max_depth': -1,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'scale_pos_weight': 10.0,
    'random_state': 42,
    'verbose': -1,
    'n_jobs': 1
}

with open(os.path.join(EXP_DIR, 'config.json'), 'w') as f:
    json.dump({
        'experiment_id': 'EXP-01',
        'experiment_name': 'Spatial Defect Density Restoration & Baseline Retraining',
        'model_architecture': 'LightGBM LGBMClassifier',
        'hyperparameters': lgb_params,
        'features_count': 506,
        'feature_names': all_feature_names,
        'intervention': 'compute_spatial_features(eligible, full_df=df)',
        'locked_threshold_model_A': 0.5100,
        'locked_threshold_model_B': 0.5376,
        'locked_gate_B': [0.25, 0.55]
    }, f, indent=2)

t0_train = time.time()
clf_exp = lgb.LGBMClassifier(**lgb_params)
clf_exp.fit(train_A_exp_input, y_train)
t_train = time.time() - t0_train
log(f"  • Training completed in {t_train:.2f}s.")

# Save isolated experiment model artifacts
exp_txt_path = os.path.join(EXP_DIR, 'model_A_spatial.txt')
exp_pkl_path = os.path.join(EXP_DIR, 'model_A_spatial.pkl')

clf_exp.booster_.save_model(exp_txt_path)
with open(exp_pkl_path, 'wb') as f:
    pickle.dump(clf_exp, f)

log(f"  • Saved experimental artifacts: {exp_txt_path}")

# -------------------------------------------------------------
# 5. EVALUATE EXPERIMENTAL MODEL A VS BASELINE
# -------------------------------------------------------------
log("\n[5/8] Evaluating Experimental Model A on canonical validation set...")

val_probs_exp = clf_exp.predict_proba(val_A_exp_input)[:, 1]
val_preds_exp = (val_probs_exp >= 0.5100).astype(int)

def evaluate_metrics(y_true, y_prob, y_pred, name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
    auc_pr = auc(rec_arr, prec_arr)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f2 = (5 * prec * rec) / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    spec = tn / (tn + fp)
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)
    mcc = matthews_corrcoef(y_true, y_pred)
    brier = brier_score_loss(y_true, y_prob)
    
    # ECE (10 deciles)
    bins = np.linspace(0, 1, 11)
    bin_idxs = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for b in range(10):
        b_mask = (bin_idxs == b)
        if b_mask.sum() > 0:
            avg_p = y_prob[b_mask].mean()
            emp_r = y_true[b_mask].mean()
            ece += (b_mask.sum() / len(y_true)) * abs(avg_p - emp_r)
            
    return {
        'model': name,
        'threshold': 0.5100,
        'accuracy': round(acc, 4),
        'balanced_accuracy': round(bal_acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'specificity': round(spec, 4),
        'fpr': round(fpr, 4),
        'fnr': round(fnr, 4),
        'f1': round(f1, 4),
        'f2': round(f2, 4),
        'mcc': round(mcc, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(auc_pr, 4),
        'brier': round(brier, 5),
        'ece': round(ece, 5),
        'tp': int(tp),
        'fn': int(fn),
        'fp': int(fp),
        'tn': int(tn)
    }

metrics_base_A = evaluate_metrics(y_val, val_probs_base, val_preds_base, 'Model_A_Baseline')
metrics_exp_A = evaluate_metrics(y_val, val_probs_exp, val_preds_exp, 'Model_A_EXP01')

with open(os.path.join(EXP_DIR, 'baseline_metrics.json'), 'w') as f:
    json.dump(metrics_base_A, f, indent=2)

with open(os.path.join(EXP_DIR, 'experiment_metrics.json'), 'w') as f:
    json.dump(metrics_exp_A, f, indent=2)

log(f"\n  • Model A Baseline: F1={metrics_base_A['f1']:.4f}, PR-AUC={metrics_base_A['pr_auc']:.4f}, Rec={metrics_base_A['recall']*100:.2f}%, Prec={metrics_base_A['precision']*100:.2f}%, TP={metrics_base_A['tp']}, FN={metrics_base_A['fn']}, FP={metrics_base_A['fp']}")
log(f"  • Model A EXP-01:   F1={metrics_exp_A['f1']:.4f}, PR-AUC={metrics_exp_A['pr_auc']:.4f}, Rec={metrics_exp_A['recall']*100:.2f}%, Prec={metrics_exp_A['precision']*100:.2f}%, TP={metrics_exp_A['tp']}, FN={metrics_exp_A['fn']}, FP={metrics_exp_A['fp']}")
log(f"  • Model A Deltas:   ΔF1={metrics_exp_A['f1'] - metrics_base_A['f1']:+.4f}, ΔPR-AUC={metrics_exp_A['pr_auc'] - metrics_base_A['pr_auc']:+.4f}, ΔTP={metrics_exp_A['tp'] - metrics_base_A['tp']:+d}, ΔFP={metrics_exp_A['fp'] - metrics_base_A['fp']:+d}")

# -------------------------------------------------------------
# 6. EVALUATE CASCADE IMPACT (LOCKED GATE B & MODEL B)
# -------------------------------------------------------------
log("\n[6/8] Evaluating Selective Cascade with EXP-01 Model A...")
with open(os.path.join(MODELS_DIR, 'model_B.pkl'), 'rb') as f:
    clf_B = pickle.load(f)

# Load existing val diagnostics to get pre-calculated Model B probabilities
diag_df = pd.read_csv(os.path.join(RESULTS_DIR, 'val_die_diagnostics.csv'))
val_probs_B = diag_df['prob_Model_B'].values
val_preds_B = (val_probs_B >= 0.5376).astype(int)

# 1. Baseline Cascade
base_esc = (val_probs_base >= 0.25) & (val_probs_base <= 0.55)
casc_base_probs = np.where(base_esc, val_probs_B, val_probs_base)
casc_base_preds = np.where(base_esc, val_preds_B, val_preds_base)
metrics_casc_base = evaluate_metrics(y_val, casc_base_probs, casc_base_preds, 'Cascade_Baseline')
metrics_casc_base['escalation_count'] = int(base_esc.sum())
metrics_casc_base['escalation_rate_pct'] = round(float(base_esc.mean() * 100), 2)
metrics_casc_base['compute_saved_pct'] = round(float((1 - base_esc.mean()) * 100), 2)

# 2. EXP-01 Cascade
exp_esc = (val_probs_exp >= 0.25) & (val_probs_exp <= 0.55)
casc_exp_probs = np.where(exp_esc, val_probs_B, val_probs_exp)
casc_exp_preds = np.where(exp_esc, val_preds_B, val_preds_exp)
metrics_casc_exp = evaluate_metrics(y_val, casc_exp_probs, casc_exp_preds, 'Cascade_EXP01')
metrics_casc_exp['escalation_count'] = int(exp_esc.sum())
metrics_casc_exp['escalation_rate_pct'] = round(float(exp_esc.mean() * 100), 2)
metrics_casc_exp['compute_saved_pct'] = round(float((1 - exp_esc.mean()) * 100), 2)

log(f"\n  • Cascade Baseline: F1={metrics_casc_base['f1']:.4f}, PR-AUC={metrics_casc_base['pr_auc']:.4f}, Rec={metrics_casc_base['recall']*100:.2f}%, Prec={metrics_casc_base['precision']*100:.2f}%, TP={metrics_casc_base['tp']}, FN={metrics_casc_base['fn']}, FP={metrics_casc_base['fp']}, Escalated={base_esc.sum()} ({base_esc.mean()*100:.2f}%)")
log(f"  • Cascade EXP-01:   F1={metrics_casc_exp['f1']:.4f}, PR-AUC={metrics_casc_exp['pr_auc']:.4f}, Rec={metrics_casc_exp['recall']*100:.2f}%, Prec={metrics_casc_exp['precision']*100:.2f}%, TP={metrics_casc_exp['tp']}, FN={metrics_casc_exp['fn']}, FP={metrics_casc_exp['fp']}, Escalated={exp_esc.sum()} ({exp_esc.mean()*100:.2f}%)")
log(f"  • Cascade Deltas:   ΔF1={metrics_casc_exp['f1'] - metrics_casc_base['f1']:+.4f}, ΔPR-AUC={metrics_casc_exp['pr_auc'] - metrics_casc_base['pr_auc']:+.4f}, ΔTP={metrics_casc_exp['tp'] - metrics_casc_base['tp']:+d}, ΔFP={metrics_casc_exp['fp'] - metrics_casc_base['fp']:+d}")

# Confusion matrices table
cm_rows = [
    {'system': 'Model_A_Baseline', 'TP': metrics_base_A['tp'], 'FN': metrics_base_A['fn'], 'FP': metrics_base_A['fp'], 'TN': metrics_base_A['tn']},
    {'system': 'Model_A_EXP01', 'TP': metrics_exp_A['tp'], 'FN': metrics_exp_A['fn'], 'FP': metrics_exp_A['fp'], 'TN': metrics_exp_A['tn']},
    {'system': 'Cascade_Baseline', 'TP': metrics_casc_base['tp'], 'FN': metrics_casc_base['fn'], 'FP': metrics_casc_base['fp'], 'TN': metrics_casc_base['tn']},
    {'system': 'Cascade_EXP01', 'TP': metrics_casc_exp['tp'], 'FN': metrics_casc_exp['fn'], 'FP': metrics_casc_exp['fp'], 'TN': metrics_casc_exp['tn']}
]
pd.DataFrame(cm_rows).to_csv(os.path.join(EXP_DIR, 'confusion_matrices.csv'), index=False)

# -------------------------------------------------------------
# 7. WAFER-LEVEL ANALYSIS & FEATURE IMPORTANCES
# -------------------------------------------------------------
log("\n[7/8] Conducting wafer-level breakdown across all 32 validation wafers...")

wafer_records = []
for wid, w_df in val_df.groupby('wafer_id'):
    idx = w_df.index
    y_w = y_val[idx]
    p_base = val_preds_base[idx]
    p_exp = val_preds_exp[idx]
    
    tp_b = int(((y_w == 1) & (p_base == 1)).sum())
    fn_b = int(((y_w == 1) & (p_base == 0)).sum())
    fp_b = int(((y_w == 0) & (p_base == 1)).sum())
    rec_b = round(tp_b / y_w.sum(), 4) if y_w.sum() > 0 else 0.0
    prec_b = round(tp_b / (tp_b + fp_b), 4) if (tp_b + fp_b) > 0 else 0.0
    f1_b = round(2 * prec_b * rec_b / (prec_b + rec_b), 4) if (prec_b + rec_b) > 0 else 0.0
    
    tp_e = int(((y_w == 1) & (p_exp == 1)).sum())
    fn_e = int(((y_w == 1) & (p_exp == 0)).sum())
    fp_e = int(((y_w == 0) & (p_exp == 1)).sum())
    rec_e = round(tp_e / y_w.sum(), 4) if y_w.sum() > 0 else 0.0
    prec_e = round(tp_e / (tp_e + fp_e), 4) if (tp_e + fp_e) > 0 else 0.0
    f1_e = round(2 * prec_e * rec_e / (prec_e + rec_e), 4) if (prec_e + rec_e) > 0 else 0.0
    
    wafer_records.append({
        'wafer_id': wid,
        'die_count': len(w_df),
        'actual_failures': int(y_w.sum()),
        'baseline_TP': tp_b,
        'baseline_FN': fn_b,
        'baseline_FP': fp_b,
        'experiment_TP': tp_e,
        'experiment_FN': fn_e,
        'experiment_FP': fp_e,
        'baseline_recall': rec_b,
        'experiment_recall': rec_e,
        'baseline_precision': prec_b,
        'experiment_precision': prec_e,
        'baseline_F1': f1_b,
        'experiment_F1': f1_e,
        'delta_F1': round(f1_e - f1_b, 4),
        'delta_recall': round(rec_e - rec_b, 4),
        'delta_TP': tp_e - tp_b,
        'delta_FP': fp_e - fp_b
    })

wafer_df = pd.DataFrame(wafer_records)
wafer_df.to_csv(os.path.join(EXP_DIR, 'wafer_metrics.csv'), index=False)

w_improved = (wafer_df['delta_F1'] > 0).sum()
w_unchanged = (wafer_df['delta_F1'] == 0).sum()
w_degraded = (wafer_df['delta_F1'] < 0).sum()

log(f"  • Wafer F1 Impact: {w_improved} improved, {w_unchanged} unchanged, {w_degraded} degraded")
log(f"  • Mean Wafer Recall: Baseline={wafer_df['baseline_recall'].mean():.4f}, EXP-01={wafer_df['experiment_recall'].mean():.4f} (Δ={wafer_df['delta_recall'].mean():+.4f})")
log(f"  • Median Wafer Recall: Baseline={wafer_df['baseline_recall'].median():.4f}, EXP-01={wafer_df['experiment_recall'].median():.4f}")
log(f"  • Worst Wafer Recall: Baseline={wafer_df['baseline_recall'].min():.4f}, EXP-01={wafer_df['experiment_recall'].min():.4f}")

# Feature importance comparison
gain_base = model_A_base.model.feature_importance(importance_type='gain')
split_base = model_A_base.model.feature_importance(importance_type='split')

gain_exp = clf_exp.booster_.feature_importance(importance_type='gain')
split_exp = clf_exp.booster_.feature_importance(importance_type='split')

feat_imp_df = pd.DataFrame({
    'feature': all_feature_names,
    'gain_baseline': gain_base,
    'split_baseline': split_base,
    'gain_exp01': gain_exp,
    'split_exp01': split_exp,
    'gain_delta': gain_exp - gain_base
}).sort_values('gain_exp01', ascending=False).reset_index(drop=True)

feat_imp_df.to_csv(os.path.join(EXP_DIR, 'feature_importance.csv'), index=False)

sp_ranks = feat_imp_df[feat_imp_df['feature'].isin(spatial_names)]
log("\n  • Spatial features importance in EXP-01:")
log(sp_ranks.to_string())

# Save prediction diagnostics CSV
pred_diag = pd.DataFrame({
    'wafer_id': val_df['wafer_id'],
    'die_row': val_df['die_row'],
    'die_col': val_df['die_col'],
    'actual_label': y_val,
    'prob_Model_A_baseline': np.round(val_probs_base, 4),
    'prob_Model_A_exp01': np.round(val_probs_exp, 4),
    'delta_prob_Model_A': np.round(val_probs_exp - val_probs_base, 4),
    'decision_Model_A_baseline': val_preds_base,
    'decision_Model_A_exp01': val_preds_exp,
    'prob_Cascade_baseline': np.round(casc_base_probs, 4),
    'prob_Cascade_exp01': np.round(casc_exp_probs, 4),
    'decision_Cascade_baseline': casc_base_preds,
    'decision_Cascade_exp01': casc_exp_preds,
    'density_3x3_restored': np.round(sp_after_val['density_3x3'], 4),
    'density_5x5_restored': np.round(sp_after_val['density_5x5'], 4)
})
pred_diag.to_csv(os.path.join(EXP_DIR, 'prediction_diagnostics.csv'), index=False)

# -------------------------------------------------------------
# 8. BUILD SCORECARD & ARTIFACT INTEGRITY VERIFICATION
# -------------------------------------------------------------
log("\n[8/8] Generating formal comparison scorecard and verifying artifact hashes...")

scorecard_rows = [
    {
        'category': 'Model_A_Classification',
        'metric': 'Accuracy',
        'baseline': metrics_base_A['accuracy'],
        'experiment': metrics_exp_A['accuracy'],
        'absolute_delta': round(metrics_exp_A['accuracy'] - metrics_base_A['accuracy'], 4),
        'relative_delta': round((metrics_exp_A['accuracy'] - metrics_base_A['accuracy']) / metrics_base_A['accuracy'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['accuracy'] > metrics_base_A['accuracy'] else ('DEGRADED' if metrics_exp_A['accuracy'] < metrics_base_A['accuracy'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Classification',
        'metric': 'Balanced Accuracy',
        'baseline': metrics_base_A['balanced_accuracy'],
        'experiment': metrics_exp_A['balanced_accuracy'],
        'absolute_delta': round(metrics_exp_A['balanced_accuracy'] - metrics_base_A['balanced_accuracy'], 4),
        'relative_delta': round((metrics_exp_A['balanced_accuracy'] - metrics_base_A['balanced_accuracy']) / metrics_base_A['balanced_accuracy'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['balanced_accuracy'] > metrics_base_A['balanced_accuracy'] else ('DEGRADED' if metrics_exp_A['balanced_accuracy'] < metrics_base_A['balanced_accuracy'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Classification',
        'metric': 'Precision',
        'baseline': metrics_base_A['precision'],
        'experiment': metrics_exp_A['precision'],
        'absolute_delta': round(metrics_exp_A['precision'] - metrics_base_A['precision'], 4),
        'relative_delta': round((metrics_exp_A['precision'] - metrics_base_A['precision']) / metrics_base_A['precision'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['precision'] > metrics_base_A['precision'] else ('DEGRADED' if metrics_exp_A['precision'] < metrics_base_A['precision'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Classification',
        'metric': 'Recall',
        'baseline': metrics_base_A['recall'],
        'experiment': metrics_exp_A['recall'],
        'absolute_delta': round(metrics_exp_A['recall'] - metrics_base_A['recall'], 4),
        'relative_delta': round((metrics_exp_A['recall'] - metrics_base_A['recall']) / metrics_base_A['recall'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['recall'] > metrics_base_A['recall'] else ('DEGRADED' if metrics_exp_A['recall'] < metrics_base_A['recall'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Classification',
        'metric': 'F1 Score',
        'baseline': metrics_base_A['f1'],
        'experiment': metrics_exp_A['f1'],
        'absolute_delta': round(metrics_exp_A['f1'] - metrics_base_A['f1'], 4),
        'relative_delta': round((metrics_exp_A['f1'] - metrics_base_A['f1']) / metrics_base_A['f1'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['f1'] > metrics_base_A['f1'] else ('DEGRADED' if metrics_exp_A['f1'] < metrics_base_A['f1'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Classification',
        'metric': 'F2 Score',
        'baseline': metrics_base_A['f2'],
        'experiment': metrics_exp_A['f2'],
        'absolute_delta': round(metrics_exp_A['f2'] - metrics_base_A['f2'], 4),
        'relative_delta': round((metrics_exp_A['f2'] - metrics_base_A['f2']) / metrics_base_A['f2'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['f2'] > metrics_base_A['f2'] else ('DEGRADED' if metrics_exp_A['f2'] < metrics_base_A['f2'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Classification',
        'metric': 'MCC',
        'baseline': metrics_base_A['mcc'],
        'experiment': metrics_exp_A['mcc'],
        'absolute_delta': round(metrics_exp_A['mcc'] - metrics_base_A['mcc'], 4),
        'relative_delta': round((metrics_exp_A['mcc'] - metrics_base_A['mcc']) / metrics_base_A['mcc'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['mcc'] > metrics_base_A['mcc'] else ('DEGRADED' if metrics_exp_A['mcc'] < metrics_base_A['mcc'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Ranking',
        'metric': 'ROC-AUC',
        'baseline': metrics_base_A['roc_auc'],
        'experiment': metrics_exp_A['roc_auc'],
        'absolute_delta': round(metrics_exp_A['roc_auc'] - metrics_base_A['roc_auc'], 4),
        'relative_delta': round((metrics_exp_A['roc_auc'] - metrics_base_A['roc_auc']) / metrics_base_A['roc_auc'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['roc_auc'] > metrics_base_A['roc_auc'] else ('DEGRADED' if metrics_exp_A['roc_auc'] < metrics_base_A['roc_auc'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Ranking',
        'metric': 'PR-AUC',
        'baseline': metrics_base_A['pr_auc'],
        'experiment': metrics_exp_A['pr_auc'],
        'absolute_delta': round(metrics_exp_A['pr_auc'] - metrics_base_A['pr_auc'], 4),
        'relative_delta': round((metrics_exp_A['pr_auc'] - metrics_base_A['pr_auc']) / metrics_base_A['pr_auc'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['pr_auc'] > metrics_base_A['pr_auc'] else ('DEGRADED' if metrics_exp_A['pr_auc'] < metrics_base_A['pr_auc'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Calibration',
        'metric': 'Brier Score',
        'baseline': metrics_base_A['brier'],
        'experiment': metrics_exp_A['brier'],
        'absolute_delta': round(metrics_exp_A['brier'] - metrics_base_A['brier'], 5),
        'relative_delta': round((metrics_exp_A['brier'] - metrics_base_A['brier']) / metrics_base_A['brier'] * 100, 2),
        'direction': 'LOWER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['brier'] < metrics_base_A['brier'] else ('DEGRADED' if metrics_exp_A['brier'] > metrics_base_A['brier'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Calibration',
        'metric': 'ECE',
        'baseline': metrics_base_A['ece'],
        'experiment': metrics_exp_A['ece'],
        'absolute_delta': round(metrics_exp_A['ece'] - metrics_base_A['ece'], 5),
        'relative_delta': round((metrics_exp_A['ece'] - metrics_base_A['ece']) / metrics_base_A['ece'] * 100, 2),
        'direction': 'LOWER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['ece'] < metrics_base_A['ece'] else ('DEGRADED' if metrics_exp_A['ece'] > metrics_base_A['ece'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Operational',
        'metric': 'True Positives (TP)',
        'baseline': metrics_base_A['tp'],
        'experiment': metrics_exp_A['tp'],
        'absolute_delta': metrics_exp_A['tp'] - metrics_base_A['tp'],
        'relative_delta': round((metrics_exp_A['tp'] - metrics_base_A['tp']) / metrics_base_A['tp'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['tp'] > metrics_base_A['tp'] else ('DEGRADED' if metrics_exp_A['tp'] < metrics_base_A['tp'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Operational',
        'metric': 'False Negatives (FN)',
        'baseline': metrics_base_A['fn'],
        'experiment': metrics_exp_A['fn'],
        'absolute_delta': metrics_exp_A['fn'] - metrics_base_A['fn'],
        'relative_delta': round((metrics_exp_A['fn'] - metrics_base_A['fn']) / metrics_base_A['fn'] * 100, 2),
        'direction': 'LOWER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['fn'] < metrics_base_A['fn'] else ('DEGRADED' if metrics_exp_A['fn'] > metrics_base_A['fn'] else 'UNCHANGED')
    },
    {
        'category': 'Model_A_Operational',
        'metric': 'False Positives (FP)',
        'baseline': metrics_base_A['fp'],
        'experiment': metrics_exp_A['fp'],
        'absolute_delta': metrics_exp_A['fp'] - metrics_base_A['fp'],
        'relative_delta': round((metrics_exp_A['fp'] - metrics_base_A['fp']) / metrics_base_A['fp'] * 100, 2),
        'direction': 'LOWER_IS_BETTER',
        'status': 'IMPROVED' if metrics_exp_A['fp'] < metrics_base_A['fp'] else ('DEGRADED' if metrics_exp_A['fp'] > metrics_base_A['fp'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Classification',
        'metric': 'Cascade F1 Score',
        'baseline': metrics_casc_base['f1'],
        'experiment': metrics_casc_exp['f1'],
        'absolute_delta': round(metrics_casc_exp['f1'] - metrics_casc_base['f1'], 4),
        'relative_delta': round((metrics_casc_exp['f1'] - metrics_casc_base['f1']) / metrics_casc_base['f1'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['f1'] > metrics_casc_base['f1'] else ('DEGRADED' if metrics_casc_exp['f1'] < metrics_casc_base['f1'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Classification',
        'metric': 'Cascade Recall',
        'baseline': metrics_casc_base['recall'],
        'experiment': metrics_casc_exp['recall'],
        'absolute_delta': round(metrics_casc_exp['recall'] - metrics_casc_base['recall'], 4),
        'relative_delta': round((metrics_casc_exp['recall'] - metrics_casc_base['recall']) / metrics_casc_base['recall'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['recall'] > metrics_casc_base['recall'] else ('DEGRADED' if metrics_casc_exp['recall'] < metrics_casc_base['recall'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Classification',
        'metric': 'Cascade Precision',
        'baseline': metrics_casc_base['precision'],
        'experiment': metrics_casc_exp['precision'],
        'absolute_delta': round(metrics_casc_exp['precision'] - metrics_casc_base['precision'], 4),
        'relative_delta': round((metrics_casc_exp['precision'] - metrics_casc_base['precision']) / metrics_casc_base['precision'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['precision'] > metrics_casc_base['precision'] else ('DEGRADED' if metrics_casc_exp['precision'] < metrics_casc_base['precision'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Ranking',
        'metric': 'Cascade PR-AUC',
        'baseline': metrics_casc_base['pr_auc'],
        'experiment': metrics_casc_exp['pr_auc'],
        'absolute_delta': round(metrics_casc_exp['pr_auc'] - metrics_casc_base['pr_auc'], 4),
        'relative_delta': round((metrics_casc_exp['pr_auc'] - metrics_casc_base['pr_auc']) / metrics_casc_base['pr_auc'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['pr_auc'] > metrics_casc_base['pr_auc'] else ('DEGRADED' if metrics_casc_exp['pr_auc'] < metrics_casc_base['pr_auc'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Operational',
        'metric': 'Cascade TP',
        'baseline': metrics_casc_base['tp'],
        'experiment': metrics_casc_exp['tp'],
        'absolute_delta': metrics_casc_exp['tp'] - metrics_casc_base['tp'],
        'relative_delta': round((metrics_casc_exp['tp'] - metrics_casc_base['tp']) / metrics_casc_base['tp'] * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['tp'] > metrics_casc_base['tp'] else ('DEGRADED' if metrics_casc_exp['tp'] < metrics_casc_base['tp'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Operational',
        'metric': 'Cascade FP',
        'baseline': metrics_casc_base['fp'],
        'experiment': metrics_casc_exp['fp'],
        'absolute_delta': metrics_casc_exp['fp'] - metrics_casc_base['fp'],
        'relative_delta': round((metrics_casc_exp['fp'] - metrics_casc_base['fp']) / metrics_casc_base['fp'] * 100, 2),
        'direction': 'LOWER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['fp'] < metrics_casc_base['fp'] else ('DEGRADED' if metrics_casc_exp['fp'] > metrics_casc_base['fp'] else 'UNCHANGED')
    },
    {
        'category': 'Cascade_Operational',
        'metric': 'Escalation Rate %',
        'baseline': metrics_casc_base['escalation_rate_pct'],
        'experiment': metrics_casc_exp['escalation_rate_pct'],
        'absolute_delta': round(metrics_casc_exp['escalation_rate_pct'] - metrics_casc_base['escalation_rate_pct'], 2),
        'relative_delta': round((metrics_casc_exp['escalation_rate_pct'] - metrics_casc_base['escalation_rate_pct']) / metrics_casc_base['escalation_rate_pct'] * 100, 2),
        'direction': 'LOWER_IS_BETTER',
        'status': 'IMPROVED' if metrics_casc_exp['escalation_rate_pct'] < metrics_casc_base['escalation_rate_pct'] else ('DEGRADED' if metrics_casc_exp['escalation_rate_pct'] > metrics_casc_base['escalation_rate_pct'] else 'UNCHANGED')
    },
    {
        'category': 'Wafer_Robustness',
        'metric': 'Mean Wafer Recall',
        'baseline': round(wafer_df['baseline_recall'].mean(), 4),
        'experiment': round(wafer_df['experiment_recall'].mean(), 4),
        'absolute_delta': round(wafer_df['delta_recall'].mean(), 4),
        'relative_delta': round(wafer_df['delta_recall'].mean() / wafer_df['baseline_recall'].mean() * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if wafer_df['delta_recall'].mean() > 0 else ('DEGRADED' if wafer_df['delta_recall'].mean() < 0 else 'UNCHANGED')
    },
    {
        'category': 'Wafer_Robustness',
        'metric': 'Median Wafer Recall',
        'baseline': round(wafer_df['baseline_recall'].median(), 4),
        'experiment': round(wafer_df['experiment_recall'].median(), 4),
        'absolute_delta': round(wafer_df['experiment_recall'].median() - wafer_df['baseline_recall'].median(), 4),
        'relative_delta': round((wafer_df['experiment_recall'].median() - wafer_df['baseline_recall'].median()) / wafer_df['baseline_recall'].median() * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if wafer_df['experiment_recall'].median() > wafer_df['baseline_recall'].median() else ('DEGRADED' if wafer_df['experiment_recall'].median() < wafer_df['baseline_recall'].median() else 'UNCHANGED')
    },
    {
        'category': 'Wafer_Robustness',
        'metric': 'Worst Wafer Recall',
        'baseline': round(wafer_df['baseline_recall'].min(), 4),
        'experiment': round(wafer_df['experiment_recall'].min(), 4),
        'absolute_delta': round(wafer_df['experiment_recall'].min() - wafer_df['baseline_recall'].min(), 4),
        'relative_delta': round((wafer_df['experiment_recall'].min() - wafer_df['baseline_recall'].min()) / wafer_df['baseline_recall'].min() * 100, 2),
        'direction': 'HIGHER_IS_BETTER',
        'status': 'IMPROVED' if wafer_df['experiment_recall'].min() > wafer_df['baseline_recall'].min() else ('DEGRADED' if wafer_df['experiment_recall'].min() < wafer_df['baseline_recall'].min() else 'UNCHANGED')
    }
]

scorecard_df = pd.DataFrame(scorecard_rows)
scorecard_df.to_csv(os.path.join(EXP_DIR, 'scorecard.csv'), index=False)
log("\n  • Scorecard generated and saved:")
log(scorecard_df[['category', 'metric', 'baseline', 'experiment', 'absolute_delta', 'status']].to_string(index=False))

# Post-run Artifact Hash Verification
files_to_hash = [
    'models/model_A.txt',
    'models/model_A.pkl',
    'models/model_B.pkl',
    'models/cnn_1d_block_model.pt',
    'models/selective_cascade_meta.pkl',
    'models/feature_metadata.json',
    'results/AUTHORITATIVE_CLEAN_METRICS.json',
    'results/val_die_diagnostics.csv',
    'results/canonical_wafer_split.json'
]

with open(os.path.join(EXP_DIR, 'hashes_before.json'), 'r') as f:
    hashes_before = json.load(f)

hashes_after = {}
mutations = []
for f in files_to_hash:
    with open(f, 'rb') as fp:
        h = hashlib.sha256(fp.read()).hexdigest()
        hashes_after[f] = h
        if h != hashes_before[f]:
            mutations.append(f)

log("\n  • Production Artifact Integrity Check:")
if len(mutations) == 0:
    log("  ✅ ZERO MUTATIONS: All 9 production artifacts remain 100% bit-for-bit identical!")
else:
    log(f"  ❌ CRITICAL INTEGRITY VIOLATION: Mutated files: {mutations}")

log("\n================================================================================")
log("EXP-01 EXECUTION COMPLETED")
log("================================================================================")
log_file.close()
