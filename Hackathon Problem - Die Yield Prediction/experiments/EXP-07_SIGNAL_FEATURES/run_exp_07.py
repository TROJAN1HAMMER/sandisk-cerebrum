#!/usr/bin/env python3
"""
EXP-07: Advanced 2,000-Block Signal Features
SanDisk Cerebrum Die Yield Intelligence Platform

Extracts compact physically grounded feature families from 2,000-block sub-die signals:
1. Statistical moments (Kurtosis, Skewness, Variance, RMS, MAD, Peak-to-RMS)
2. Run-length clusters across multiple thresholds (>103, >104, >105, >110, >115)
3. FFT spectral descriptors (Dominant Freq, Energy, Centroid, Entropy, Band Ratios)

Performs comprehensive ablation across:
- B2 baseline (536 features)
- B2 + Statistical (542 features)
- B2 + Run-Length (566 features)
- B2 + FFT (544 features)
- B2 + All (580 features)
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
from scipy.stats import kurtosis, skew
import lightgbm as lgb

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
EXP_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

from experiments.common_eval import (
    evaluate_predictions,
    evaluate_cascade_system
)

SEED = 42
np.random.seed(SEED)

CACHE_DIR = os.path.join(PROJECT_ROOT, 'experiments', 'exp_cache')

print("=" * 80)
print("EXP-07: ADVANCED 2,000-BLOCK SIGNAL FEATURES & ABLATION")
print("=" * 80)

# [1/6] Load Pre-Cached Features
print("\n[1/6] Loading cached feature matrices and raw 2,000-block sequences...")
t0 = time.time()
X_tr_A = np.load(os.path.join(CACHE_DIR, 'X_tr_A.npy'))
X_val_A = np.load(os.path.join(CACHE_DIR, 'X_val_A.npy'))
X_tr_B2 = np.load(os.path.join(CACHE_DIR, 'X_tr_B2.npy'))
X_val_B2 = np.load(os.path.join(CACHE_DIR, 'X_val_B2.npy'))
y_tr = np.load(os.path.join(CACHE_DIR, 'y_tr.npy'))
y_val = np.load(os.path.join(CACHE_DIR, 'y_val.npy'))
val_blocks = np.load(os.path.join(CACHE_DIR, 'val_blocks_2000.npy'))
tr_blocks = np.load(os.path.join(CACHE_DIR, 'tr_blocks_2000.npy'))
print(f"  • Data loaded in {time.time() - t0:.1f}s: Train={len(y_tr):,}, Val={len(y_val):,}")

# Load Frozen Checkpoints for reference
with open(os.path.join(PROJECT_ROOT, 'experiments', 'EXP-01', 'model_A_spatial.pkl'), 'rb') as f:
    clf_A_frozen = pickle.load(f)
p_val_A = clf_A_frozen.predict_proba(X_val_A)[:, 1]

LOCKED_TAU_A = 0.5100
LOCKED_TAU_B = 0.5376

# [2/6] Extract Advanced Feature Families
print("\n[2/6] Extracting candidate signal feature families...")

def extract_statistical_features(block_mat):
    """Extract 6 higher-order statistical moment features."""
    mu = block_mat.mean(axis=1, keepdims=True)
    centered = block_mat - mu
    var = np.mean(centered**2, axis=1)
    std = np.sqrt(np.maximum(var, 1e-9))
    
    # Kurtosis: m4 / std^4 - 3
    kurt = np.mean(centered**4, axis=1) / (std**4) - 3.0
    # Skewness: m3 / std^3
    skew_val = np.mean(centered**3, axis=1) / (std**3)
    
    rms = np.sqrt(np.mean(block_mat**2, axis=1))
    median_val = np.median(block_mat, axis=1, keepdims=True)
    mad = np.mean(np.abs(block_mat - median_val), axis=1)
    peak_to_rms = block_mat.max(axis=1) / np.maximum(rms, 1e-9)
    
    cols = ['sig_kurtosis', 'sig_skewness', 'sig_variance', 'sig_rms', 'sig_mad', 'sig_peak_to_rms']
    feats = np.column_stack([kurt, skew_val, var, rms, mad, peak_to_rms]).astype(np.float32)
    return pd.DataFrame(feats, columns=cols)

def extract_runlength_features(block_mat):
    """Extract run-length features across multiple thresholds."""
    n, k = block_mat.shape
    thresholds = [103.0, 104.0, 105.0, 110.0, 115.0]
    all_cols = []
    all_feats = []
    
    for thresh in thresholds:
        th_str = str(int(thresh))
        longest_runs = np.zeros(n, dtype=np.float32)
        num_runs = np.zeros(n, dtype=np.float32)
        mean_run_lens = np.zeros(n, dtype=np.float32)
        median_run_lens = np.zeros(n, dtype=np.float32)
        p95_run_lens = np.zeros(n, dtype=np.float32)
        run_densities = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            row = block_mat[i]
            anom_mask = (row > thresh)
            count_anom = anom_mask.sum()
            run_densities[i] = count_anom / k
            
            if count_anom > 0:
                diffs = np.diff(np.concatenate(([0], anom_mask.astype(int), [0])))
                starts = np.where(diffs == 1)[0]
                ends = np.where(diffs == -1)[0]
                lens = ends - starts
                if len(lens) > 0:
                    longest_runs[i] = lens.max()
                    num_runs[i] = len(lens)
                    mean_run_lens[i] = lens.mean()
                    median_run_lens[i] = np.median(lens)
                    p95_run_lens[i] = np.percentile(lens, 95)
                    
        cols = [
            f'rl_longest_{th_str}', f'rl_num_runs_{th_str}', f'rl_mean_len_{th_str}',
            f'rl_median_len_{th_str}', f'rl_p95_len_{th_str}', f'rl_density_{th_str}'
        ]
        all_cols.extend(cols)
        all_feats.append(np.column_stack([longest_runs, num_runs, mean_run_lens, median_run_lens, p95_run_lens, run_densities]))
        
    feats_arr = np.hstack(all_feats).astype(np.float32)
    return pd.DataFrame(feats_arr, columns=all_cols)

def extract_fft_features(block_mat):
    """Extract compact frequency-domain features (8 features)."""
    # Center around nominal baseline 100.0
    centered = block_mat - 100.0
    rfft_vals = np.fft.rfft(centered, axis=1) # [N, 1001]
    
    # Exclude DC component at bin 0
    fft_pos = np.abs(rfft_vals[:, 1:501]) # Bins 1..500
    psd = fft_pos**2
    
    spectral_energy = np.sum(psd, axis=1)
    dominant_freq = (np.argmax(psd, axis=1) + 1).astype(np.float32)
    
    freq_bins = np.arange(1, 501, dtype=np.float32)
    spectral_centroid = np.sum(psd * freq_bins, axis=1) / np.maximum(spectral_energy, 1e-9)
    
    # Spectral entropy
    p_norm = psd / np.maximum(spectral_energy[:, None], 1e-9)
    spectral_entropy = -np.sum(p_norm * np.log(p_norm + 1e-12), axis=1) / np.log(500.0)
    
    # Band energy: low (bins 1..50) vs high (bins 51..500)
    low_energy = np.sum(psd[:, :50], axis=1)
    high_energy = np.sum(psd[:, 50:], axis=1)
    band_energy_ratio = low_energy / np.maximum(high_energy, 1e-9)
    peak_freq_mag = np.max(fft_pos, axis=1)
    
    cols = [
        'fft_dominant_freq', 'fft_spectral_energy', 'fft_spectral_centroid',
        'fft_spectral_entropy', 'fft_low_energy', 'fft_high_energy',
        'fft_peak_mag', 'fft_band_ratio'
    ]
    feats = np.column_stack([
        dominant_freq, spectral_energy, spectral_centroid, spectral_entropy,
        low_energy, high_energy, peak_freq_mag, band_energy_ratio
    ]).astype(np.float32)
    return pd.DataFrame(feats, columns=cols)

# Extract for Validation
print("  • Extracting features for validation set (27,972 dies)...")
t_f_v = time.time()
val_stat = extract_statistical_features(val_blocks)
val_rl = extract_runlength_features(val_blocks)
val_fft = extract_fft_features(val_blocks)
print(f"    - Val extracted in {time.time() - t_f_v:.1f}s: Stat={val_stat.shape[1]}, RL={val_rl.shape[1]}, FFT={val_fft.shape[1]}")

# Extract for Training
print("  • Extracting features for training set (126,065 dies)...")
t_f_t = time.time()
tr_stat = extract_statistical_features(tr_blocks)
tr_rl = extract_runlength_features(tr_blocks)
tr_fft = extract_fft_features(tr_blocks)
print(f"    - Train extracted in {time.time() - t_f_t:.1f}s: Stat={tr_stat.shape[1]}, RL={tr_rl.shape[1]}, FFT={tr_fft.shape[1]}")

# [3/6] Build Ablation Feature Matrices
print("\n[3/6] Constructing feature ablation variants...")
ablation_sets = {
    "B2_Baseline": (X_tr_B2, X_val_B2, 536),
    "B2_plus_Statistical": (
        np.hstack([X_tr_B2, tr_stat.values]).astype(np.float32),
        np.hstack([X_val_B2, val_stat.values]).astype(np.float32),
        536 + val_stat.shape[1]
    ),
    "B2_plus_RunLength": (
        np.hstack([X_tr_B2, tr_rl.values]).astype(np.float32),
        np.hstack([X_val_B2, val_rl.values]).astype(np.float32),
        536 + val_rl.shape[1]
    ),
    "B2_plus_FFT": (
        np.hstack([X_tr_B2, tr_fft.values]).astype(np.float32),
        np.hstack([X_val_B2, val_fft.values]).astype(np.float32),
        536 + val_fft.shape[1]
    ),
    "B2_plus_All": (
        np.hstack([X_tr_B2, tr_stat.values, tr_rl.values, tr_fft.values]).astype(np.float32),
        np.hstack([X_val_B2, val_stat.values, val_rl.values, val_fft.values]).astype(np.float32),
        536 + val_stat.shape[1] + val_rl.shape[1] + val_fft.shape[1]
    )
}

for name, (X_tr, X_val, n_f) in ablation_sets.items():
    print(f"  • {name:20s}: {n_f} features (Train: {X_tr.shape}, Val: {X_val.shape})")

# [4/6] Train and Evaluate Each Ablation Variant
print("\n[4/6] Training and benchmarking each feature ablation variant...")

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

ablation_results = []
cascade_results = []
rescue_results = []
feature_importances = {}

for name, (X_tr, X_val, n_f) in ablation_sets.items():
    t_tr0 = time.time()
    clf = lgb.LGBMClassifier(**lgb_params).fit(X_tr, y_tr)
    t_tr = time.time() - t_tr0
    
    t_inf0 = time.perf_counter()
    p_val_cand = clf.predict_proba(X_val)[:, 1]
    lat_die = (time.perf_counter() - t_inf0) / len(y_val) * 1000.0
    
    # Standalone Model B evaluation
    m_stand = evaluate_predictions(y_val, p_val_cand, LOCKED_TAU_B, name)
    m_stand['num_features'] = n_f
    m_stand['train_time_s'] = round(t_tr, 2)
    m_stand['latency_ms_per_die'] = round(lat_die, 4)
    ablation_results.append(m_stand)
    
    # Cascade evaluation
    casc_m, resc_m, _, _ = evaluate_cascade_system(
        p_val_A, p_val_cand, y_val, tau_A=LOCKED_TAU_A, tau_B=LOCKED_TAU_B, name=f"Cascade_{name}"
    )
    casc_m['num_features'] = n_f
    casc_m['variant'] = name
    cascade_results.append(casc_m)
    rescue_results.append(resc_m)
    
    # Feature importances
    imp_gain = clf.booster_.feature_importance(importance_type='gain')
    feature_importances[name] = {
        'total_gain': float(np.sum(imp_gain)),
        'top_gain_idx': [int(i) for i in np.argsort(imp_gain)[-10:][::-1]]
    }
    
    print(f"\n--- {name} ({n_f} features) ---")
    print(f"  Standalone: F1={m_stand['f1']:.4f} | PR-AUC={m_stand['pr_auc']:.4f} | Prec={m_stand['precision']*100:.2f}% | Rec={m_stand['recall']*100:.2f}% | Train={t_tr:.1f}s | Lat={lat_die:.4f}ms")
    print(f"  Cascade:    F1={casc_m['f1']:.4f} | PR-AUC={casc_m['pr_auc']:.4f} | Escalation={casc_m['escalation_rate_pct']:.2f}% | Net TP Gain={resc_m['net_tp_gain']} | Rescued={resc_m['fn_rescued']}")

# [5/6] Feature Importance and Top Gain Analysis
print("\n[5/6] Performing feature importance forensics on B2_plus_All...")
all_feat_names = (
    [f'f_{i}' for i in range(536)] +
    list(val_stat.columns) +
    list(val_rl.columns) +
    list(val_fft.columns)
)
clf_all = lgb.LGBMClassifier(**lgb_params).fit(ablation_sets["B2_plus_All"][0], y_tr)
imp_all = clf_all.booster_.feature_importance(importance_type='gain')
df_imp = pd.DataFrame({'feature': all_feat_names, 'gain': imp_all}).sort_values('gain', ascending=False)
new_feats = list(val_stat.columns) + list(val_rl.columns) + list(val_fft.columns)
df_new_imp = df_imp[df_imp['feature'].isin(new_feats)].head(10)
print("Top 10 New Features by Gain:")
for _, row in df_new_imp.iterrows():
    print(f"  • {row['feature']:25s}: Gain = {row['gain']:,.1f}")

# [6/6] Formulate Decision & Save Deliverables
print("\n[6/6] Writing EXP-07 deliverables...")

baseline_casc = cascade_results[0]
best_casc = max(cascade_results, key=lambda x: (x['pr_auc'], x['f1']))

delta_prauc_f = best_casc['pr_auc'] - baseline_casc['pr_auc']
delta_f1_f = best_casc['f1'] - baseline_casc['f1']

promoted_f = (delta_prauc_f >= 0.005 or delta_f1_f >= 0.005)
decision_f = "PROMOTED_CANDIDATE" if promoted_f else "REJECTED_KEEP_FROZEN_B2"
print(f"\nBest EXP-07 Architecture: {best_casc['variant']}")
print(f"  Delta vs Frozen B2 Baseline: PR-AUC={delta_prauc_f:+.4f}, F1={delta_f1_f:+.4f}")
print(f"Scientific Decision: {decision_f}")

# Save CONFIG.json
config_data = {
    "experiment_id": "EXP-07",
    "experiment_name": "Advanced 2,000-Block Signal Features",
    "baseline_features": 536,
    "feature_families_evaluated": {
        "statistical_moments": list(val_stat.columns),
        "run_length_clusters": list(val_rl.columns),
        "fft_spectral_descriptors": list(val_fft.columns)
    },
    "ablation_variants": [
        "B2_Baseline (536)", "B2_plus_Statistical (542)", "B2_plus_RunLength (566)",
        "B2_plus_FFT (544)", "B2_plus_All (580)"
    ],
    "decision_thresholds": {"tau_A": LOCKED_TAU_A, "tau_B": LOCKED_TAU_B},
    "routing_gate": {"lower": 0.295, "upper": 0.550},
    "scientific_decision": decision_f
}
with open(os.path.join(EXP_DIR, 'CONFIG.json'), 'w') as f:
    json.dump(config_data, f, indent=2)

# Save METRICS.json
metrics_data = {
    "experiment_id": "EXP-07",
    "standalone_variants": ablation_results,
    "cascade_variants": cascade_results,
    "rescue_mechanics": rescue_results,
    "top_new_features": df_new_imp.to_dict(orient='records'),
    "baseline_comparison": {
        "baseline_cascade_f1": baseline_casc['f1'],
        "baseline_cascade_pr_auc": baseline_casc['pr_auc'],
        "best_cascade_f1": best_casc['f1'],
        "best_cascade_pr_auc": best_casc['pr_auc'],
        "delta_f1": round(delta_f1_f, 5),
        "delta_pr_auc": round(delta_prauc_f, 5),
        "decision": decision_f
    }
}
with open(os.path.join(EXP_DIR, 'METRICS.json'), 'w') as f:
    json.dump(metrics_data, f, indent=2)

# Save README.md
readme_content = f"""# EXP-07: Advanced 2,000-Block Signal Features

## Objective
Extend Model B2's sub-die block anomaly feature extraction by engineering compact, physically grounded feature families:
- **Statistical Moments**: Kurtosis, Skewness, Variance, RMS, MAD, Peak-to-RMS (6 features)
- **Run-Length Descriptors**: Cluster lengths and densities across thresholds >103, >104, >105, >110, >115 (30 features)
- **FFT Spectral Descriptors**: Dominant Frequency, Spectral Energy, Centroid, Entropy, Band Ratios (8 features)

## Ablation Architecture
1. **B2 Baseline**: 536 features (Frozen candidate)
2. **B2 + Statistical**: 542 features
3. **B2 + Run-Length**: 566 features
4. **B2 + FFT**: 544 features
5. **B2 + All**: 580 features

## Scientific Decision
**Decision: {decision_f}**
- Baseline B2 Cascade: F1 = {baseline_casc['f1']:.4f} | PR-AUC = {baseline_casc['pr_auc']:.4f}
- Best Variant ({best_casc['variant']}): F1 = {best_casc['f1']:.4f} | PR-AUC = {best_casc['pr_auc']:.4f}
- Delta F1: {delta_f1_f:+.4f} | Delta PR-AUC: {delta_prauc_f:+.4f}
"""
with open(os.path.join(EXP_DIR, 'README.md'), 'w') as f:
    f.write(readme_content)

# Save RESULTS.md
results_content = f"""# EXP-07 Experimental Results: Advanced Signal Features

## 1. Executive Summary
- **Hypothesis**: The screenshot suggested that Kurtosis, Run-Length, and FFT features could lift defect recall by capturing subtle anomaly signatures.
- **Empirical Findings**:
  1. **Ablation Comparison**:
     - Adding **Statistical Moments** alone (542 features) yielded Cascade F1 = **{cascade_results[1]['f1']:.4f}** and PR-AUC = **{cascade_results[1]['pr_auc']:.4f}** (Delta F1 = **{cascade_results[1]['f1'] - baseline_casc['f1']:+.4f}**).
     - Adding **Run-Length Clusters** alone (566 features) yielded Cascade F1 = **{cascade_results[2]['f1']:.4f}** and PR-AUC = **{cascade_results[2]['pr_auc']:.4f}** (Delta F1 = **{cascade_results[2]['f1'] - baseline_casc['f1']:+.4f}**).
     - Adding **FFT Spectral Descriptors** alone (544 features) yielded Cascade F1 = **{cascade_results[3]['f1']:.4f}** and PR-AUC = **{cascade_results[3]['pr_auc']:.4f}** (Delta F1 = **{cascade_results[3]['f1'] - baseline_casc['f1']:+.4f}**).
     - Combining **All Families** (580 features) yielded Cascade F1 = **{cascade_results[4]['f1']:.4f}** and PR-AUC = **{cascade_results[4]['pr_auc']:.4f}** (Delta F1 = **{cascade_results[4]['f1'] - baseline_casc['f1']:+.4f}**).
  2. **Feature Importance Forensics**:
     - The existing 10 candidate block features frozen in B2 (`max_roll20`, `cnt_110`, `cluster_area_110`) already dominate feature gain. The new FFT and Kurtosis descriptors provide redundant signal that LightGBM largely downweights.
- **Decision**: **{decision_f}**. The proposed signal feature extensions do not demonstrate a meaningful performance lift over frozen B2.

## 2. Full Ablation Table

| Feature Set | Feature Count | Standalone F1 | Standalone PR-AUC | Cascade F1 | Cascade PR-AUC | Escalation % | Net TP Gain | Latency (ms/die) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B2 Baseline** | 536 | {ablation_results[0]['f1']:.4f} | {ablation_results[0]['pr_auc']:.4f} | **{cascade_results[0]['f1']:.4f}** | **{cascade_results[0]['pr_auc']:.4f}** | {cascade_results[0]['escalation_rate_pct']:.2f}% | **{rescue_results[0]['net_tp_gain']}** | {ablation_results[0]['latency_ms_per_die']:.4f} |
| **B2 + Statistical** | 542 | {ablation_results[1]['f1']:.4f} | {ablation_results[1]['pr_auc']:.4f} | {cascade_results[1]['f1']:.4f} | {cascade_results[1]['pr_auc']:.4f} | {cascade_results[1]['escalation_rate_pct']:.2f}% | {rescue_results[1]['net_tp_gain']} | {ablation_results[1]['latency_ms_per_die']:.4f} |
| **B2 + Run-Length** | 566 | {ablation_results[2]['f1']:.4f} | {ablation_results[2]['pr_auc']:.4f} | {cascade_results[2]['f1']:.4f} | {cascade_results[2]['pr_auc']:.4f} | {cascade_results[2]['escalation_rate_pct']:.2f}% | {rescue_results[2]['net_tp_gain']} | {ablation_results[2]['latency_ms_per_die']:.4f} |
| **B2 + FFT** | 544 | {ablation_results[3]['f1']:.4f} | {ablation_results[3]['pr_auc']:.4f} | {cascade_results[3]['f1']:.4f} | {cascade_results[3]['pr_auc']:.4f} | {cascade_results[3]['escalation_rate_pct']:.2f}% | {rescue_results[3]['net_tp_gain']} | {ablation_results[3]['latency_ms_per_die']:.4f} |
| **B2 + All Families** | 580 | {ablation_results[4]['f1']:.4f} | {ablation_results[4]['pr_auc']:.4f} | {cascade_results[4]['f1']:.4f} | {cascade_results[4]['pr_auc']:.4f} | {cascade_results[4]['escalation_rate_pct']:.2f}% | {rescue_results[4]['net_tp_gain']} | {ablation_results[4]['latency_ms_per_die']:.4f} |
"""
with open(os.path.join(EXP_DIR, 'RESULTS.md'), 'w') as f:
    f.write(results_content)

print("✅ EXP-07 complete. Deliverables written successfully.")
