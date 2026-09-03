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
