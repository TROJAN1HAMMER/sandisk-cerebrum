import numpy as np
import pandas as pd
import torch

def compute_block_occlusion_attribution(cnn_model, block_array, window_size=50, device=None):
    """
    Compute Occlusion Sensitivity Attribution across 2,000 block readings.
    Slides a 50-block zero/baseline window across the 2,000 array and measures drop in CNN fail probability.
    """
    if device is None:
        device = next(cnn_model.parameters()).device

    cnn_model.eval()
    norm_block = (block_array - 100.0) / 15.0
    
    tensor_orig = torch.tensor(norm_block, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        base_logit = cnn_model(tensor_orig)
        base_prob = torch.sigmoid(base_logit).item()
        
    k = len(block_array)
    attributions = np.zeros(k, dtype=np.float32)
    
    # Occlude in sliding windows of size 50
    for start in range(0, k - window_size + 1, 10):
        end = start + window_size
        occluded = norm_block.copy()
        occluded[start:end] = 0.0 # set to baseline (0 normalized)
        
        tensor_occ = torch.tensor(occluded, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            occ_prob = torch.sigmoid(cnn_model(tensor_occ)).item()
            
        drop = max(0.0, base_prob - occ_prob)
        attributions[start:end] += drop
        
    # Normalize attributions
    if attributions.max() > 0:
        attributions = attributions / attributions.max()
        
    return base_prob, attributions


def compute_tcn_saliency_attribution(tcn_model, block_array, device=None):
    """
    Compute input gradient saliency attribution using the Multi-Scale Dilated TCN.
    Provides fast, exact sensitivity over the 2,000-block sequence for offline forensics.
    """
    if device is None:
        device = next(tcn_model.parameters()).device

    tcn_model.eval()
    norm_block = (block_array - 100.0) / 15.0
    tensor = torch.tensor(norm_block, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    tensor.requires_grad = True

    logit = tcn_model(tensor)
    prob = torch.sigmoid(logit).item()
    logit.backward()

    saliency = torch.abs(tensor.grad).squeeze().cpu().numpy()
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return prob, saliency


def explain_die_prediction_multi_res(
    die_index, df_subset, model_A, model_B_fusion, cnn_model,
    X_val_A, val_cnn_embeds, val_block_matrix, opt_threshold=0.65
):
    """
    Produce a complete, human-readable multi-resolution explanation for a selected die.
    Displays Model A Risk, Model B Risk, Risk Category, Decision Change, and Block Anomaly Attribution.
    """
    die_info = df_subset.iloc[die_index]
    wid = die_info['wafer_id']
    r, c = die_info['die_row'], die_info['die_col']
    actual_label = die_info['label']
    
    # Model A Risk
    if 'prob_A' in die_info:
        prob_A = float(die_info['prob_A'])
    else:
        prob_A = float(model_A.predict_proba(df_subset.iloc[die_index:die_index+1])[0])
    
    # Model B Risk
    if 'prob_B' in die_info:
        prob_B = float(die_info['prob_B'])
    elif hasattr(model_B_fusion, 'predict_proba'):
        try:
            prob_B = float(model_B_fusion.predict_proba(
                X_val_A[die_index:die_index+1],
                val_cnn_embeds[die_index:die_index+1]
            )[0])
        except (TypeError, ValueError):
            prob_B = float(model_B_fusion.predict_proba(X_val_A[die_index:die_index+1])[:, 1][0])
    else:
        prob_B = 0.5
    
    # Risk Category Assignment
    if prob_B < 0.20:
        risk_cat = "Low Risk (0–20%)"
    elif prob_B < 0.50:
        risk_cat = "Moderate Risk (20–50%)"
    elif prob_B < 0.75:
        risk_cat = "High Risk (50–75%)"
    else:
        risk_cat = "Critical Risk (75–100%)"
        
    pred_A = 1 if prob_A >= model_A.optimal_threshold else 0
    pred_B = 1 if prob_B >= opt_threshold else 0
    
    # Decision Change Analysis
    if pred_A == 0 and pred_B == 1:
        decision_msg = "MODEL A MISSED → MODEL B RECOVERED FAILURE! (Sub-die block anomaly detected)"
    elif pred_A == 1 and pred_B == 1:
        decision_msg = "Both Model A & Model B CONFIRMED FAILURE."
    elif pred_A == 0 and pred_B == 0:
        decision_msg = "Both Model A & Model B CONFIRMED PASS."
    else:
        decision_msg = "Model A predicted fail, Model B downgraded to pass."
        
    # Block Attribution
    block_arr = val_block_matrix[die_index]
    _, attributions = compute_block_occlusion_attribution(cnn_model, block_arr)
    top_anom_start = np.argmax(attributions)
    
    return {
        'wafer_id': wid,
        'die_row': int(r),
        'die_col': int(c),
        'actual_label': int(actual_label),
        'prob_A': float(prob_A),
        'prob_B': float(prob_B),
        'pred_A': pred_A,
        'pred_B': pred_B,
        'risk_category': risk_cat,
        'decision_msg': decision_msg,
        'top_anomaly_block_index': int(top_anom_start),
        'block_readings': block_arr,
        'attributions': attributions
    }
