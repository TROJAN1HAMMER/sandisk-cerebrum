import os
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

SEED = 42

class SelectiveCascadeClassifier:
    """
    Selective Multi-Resolution Cascade Classifier.
    Evaluates coarse Model A first. Only dies in the uncertain probability band [L, U]
    are escalated to fine-resolution 2,000-block 1D ConvNet inspection.
    """
    def __init__(self, model_A, cnn_model, lower_gate=0.20, upper_gate=0.50, opt_threshold=0.51, device='cpu'):
        self.model_A = model_A
        self.cnn_model = cnn_model
        self.lower_gate = lower_gate
        self.upper_gate = upper_gate
        self.opt_threshold = opt_threshold
        self.device = device
        self.meta_classifier = None
        
    def fit_meta_classifier(self, val_probs_A, val_cnn_probs, val_cnn_embeds, y_val):
        """Fit a lightweight calibrated meta-classifier on dies within the uncertain gate region."""
        gate_mask = (val_probs_A >= self.lower_gate) & (val_probs_A <= self.upper_gate)
        
        if gate_mask.sum() == 0:
            print("Warning: No dies in train gate region!")
            return
            
        X_gate = np.column_stack([
            val_probs_A[gate_mask],
            val_cnn_probs[gate_mask],
            val_cnn_embeds[gate_mask]
        ])
        y_gate = y_val[gate_mask]
        
        # Lightweight Logistic Regression meta-model
        self.meta_classifier = LogisticRegression(class_weight='balanced', random_state=SEED)
        self.meta_classifier.fit(X_gate, y_gate)
        
    def predict_cascade(self, X_df_A, block_tensor_x, val_cnn_probs, val_cnn_embeds):
        """
        Run selective multi-resolution cascade inference:
        Returns dict with: final_probs, final_preds, resolution_used, escalation_reasons, risk_categories
        """
        probs_A = self.model_A.predict_proba(X_df_A)
        n = len(probs_A)
        
        final_probs = np.zeros(n, dtype=np.float32)
        final_preds = np.zeros(n, dtype=int)
        resolution_used = []
        escalation_reasons = []
        risk_categories = []
        
        gate_mask = (probs_A >= self.lower_gate) & (probs_A <= self.upper_gate)
        
        if self.meta_classifier is not None and gate_mask.sum() > 0:
            X_gate = np.column_stack([
                probs_A[gate_mask],
                val_cnn_probs[gate_mask],
                val_cnn_embeds[gate_mask]
            ])
            gate_meta_probs = self.meta_classifier.predict_proba(X_gate)[:, 1]
        else:
            gate_meta_probs = probs_A[gate_mask]
            
        gate_idx = 0
        for i in range(n):
            pa = probs_A[i]
            
            if pa < self.lower_gate:
                # Confidently Passing
                final_probs[i] = pa
                final_preds[i] = 0
                resolution_used.append("Model A Coarse Only")
                escalation_reasons.append(f"Model A high confidence PASS (P_A={pa:.3f} < {self.lower_gate})")
            elif pa > self.upper_gate:
                # Confidently Failing
                final_probs[i] = pa
                final_preds[i] = 1 if pa >= self.opt_threshold else 0
                resolution_used.append("Model A Coarse Only")
                escalation_reasons.append(f"Model A high confidence FAIL (P_A={pa:.3f} > {self.upper_gate})")
            else:
                # Uncertain Region -> Escalated to Fine Resolution
                p_meta = gate_meta_probs[gate_idx]
                gate_idx += 1
                final_probs[i] = p_meta
                final_preds[i] = 1 if p_meta >= 0.50 else (1 if pa >= self.opt_threshold else 0)
                resolution_used.append("Model A + Fine-Resolution Inspection")
                escalation_reasons.append(f"Model A uncertain ({self.lower_gate} <= P_A={pa:.3f} <= {self.upper_gate}); escalated to 2,000-block inspection")
                
            # Assign Risk Category
            pf = final_probs[i]
            if pf < 0.20:
                risk_categories.append("Low Risk")
            elif pf < 0.50:
                risk_categories.append("Moderate Risk")
            elif pf < 0.75:
                risk_categories.append("High Risk")
            else:
                risk_categories.append("Critical Risk")
                
        return {
            'probs_A': probs_A,
            'final_probs': final_probs,
            'final_preds': final_preds,
            'resolution_used': resolution_used,
            'escalation_reasons': escalation_reasons,
            'risk_categories': risk_categories,
            'gate_mask': gate_mask
        }


class DualStreamCandidateCascade:
    """
    B3 Candidate Cascade Classifier:
    - Coarse Evaluation: Candidate Model A (517 features, tau_A = 0.5100)
    - Cascade Gate: [lower_gate, upper_gate] (Default: [0.250, 0.550])
    - Escalated Evaluation: Candidate Model B Dual-Stream Late Fusion (528 tabular + 1D-CNN, tau_B = 0.5200)
    - Supports High-Recall Experimental Mode: Gate [0.20, 0.50], tau_B = 0.5000
    """
    def __init__(self, cand_model_A=None, cand_model_B=None, lower_gate=0.25, upper_gate=0.55, tau_A=0.5100, tau_B=0.5200):
        from src.model_a import CandidateModelAWrapper
        from src.model_b import CandidateModelBWrapper
        
        self.cand_model_A = cand_model_A if cand_model_A is not None else CandidateModelAWrapper()
        self.cand_model_B = cand_model_B if cand_model_B is not None else CandidateModelBWrapper()
        self.lower_gate = lower_gate
        self.upper_gate = upper_gate
        self.tau_A = tau_A
        self.tau_B = tau_B

    def set_policy(self, policy="candidate_standard"):
        """Configure cascade policy (standard candidate vs experimental high-recall)."""
        if policy == "high_recall":
            self.lower_gate = 0.20
            self.upper_gate = 0.50
            self.tau_A = 0.5100
            self.tau_B = 0.5000
        else: # candidate_standard
            self.lower_gate = 0.25
            self.upper_gate = 0.55
            self.tau_A = 0.5100
            self.tau_B = 0.5200

    def predict_cascade(self, X_A_cand, X_B_cand, block_matrix):
        """
        Execute Candidate Cascade Routing:
        - Confident pass: p_A < lower_gate -> decision = 0
        - Confident fail: p_A > upper_gate -> decision = (p_A >= tau_A)
        - Escalated zone: lower_gate <= p_A <= upper_gate -> evaluated by Dual-Stream Model B
        """
        p_A = self.cand_model_A.predict_proba(X_A_cand)
        n = len(p_A)
        
        esc_mask = (p_A >= self.lower_gate) & (p_A <= self.upper_gate)
        p_cascade = np.copy(p_A)
        preds_cascade = np.zeros(n, dtype=int)
        
        # Fast path passes
        pass_mask = (p_A < self.lower_gate)
        preds_cascade[pass_mask] = 0
        
        # Fast path fails
        fail_mask = (p_A > self.upper_gate)
        preds_cascade[fail_mask] = (p_A[fail_mask] >= self.tau_A).astype(int)
        
        # Escalated cohort evaluated by Model B
        if esc_mask.any():
            # Model B inference on escalated cohort only
            p_B_esc = self.cand_model_B.predict_proba(X_B_cand[esc_mask], block_matrix[esc_mask])
            p_cascade[esc_mask] = p_B_esc
            preds_cascade[esc_mask] = (p_B_esc >= self.tau_B).astype(int)
            
        return {
            'p_A': p_A,
            'p_cascade': p_cascade,
            'preds_cascade': preds_cascade,
            'escalated_mask': esc_mask,
            'escalation_count': int(esc_mask.sum()),
            'escalation_rate_pct': float(esc_mask.mean() * 100.0),
            'compute_saved_pct': float((1.0 - esc_mask.mean()) * 100.0)
        }
