import os
# Prevent macOS Python 3.14 OpenMP SIGSEGV crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

class ModelAWrapper:
    """Wrapper for locked Model A baseline."""
    def __init__(self, model_dir='models'):
        self.model_txt = os.path.join(model_dir, 'model_A.txt')
        self.model_pkl = os.path.join(model_dir, 'model_A.pkl')
        self.meta_path = os.path.join(model_dir, 'feature_metadata.json')
        
        with open(self.meta_path, 'r') as f:
            self.meta = json.load(f)
            
        self.feature_names = self.meta['all_features']
        self.optimal_threshold = self.meta['optimal_threshold']
        
        # Load native LightGBM text model to prevent OpenMP C++ pickle crashes
        if os.path.exists(self.model_txt):
            self.model = lgb.Booster(model_file=self.model_txt)
            self.is_booster = True
        elif os.path.exists(self.model_pkl):
            with open(self.model_pkl, 'rb') as f:
                self.model = pickle.load(f)
            self.is_booster = False
        else:
            raise FileNotFoundError(f"Neither model_A.txt nor model_A.pkl found in {model_dir}")
        
    def predict_proba(self, X_df):
        """Predict failure probabilities using Model A."""
        X_sub = X_df[self.feature_names]
        if self.is_booster:
            # lgb.Booster.predict() returns 1D probability array directly
            return self.model.predict(X_sub, num_threads=1)
        else:
            return self.model.predict_proba(X_sub)[:, 1]
        
    def predict(self, X_df, threshold=None):
        """Predict binary failure class."""
        if threshold is None:
            threshold = self.optimal_threshold
        probs = self.predict_proba(X_df)
        return (probs >= threshold).astype(int)


class CandidateModelAWrapper:
    """
    Wrapper for Validated Candidate Model A (B3 Candidate: 517 Features).
    Combines 500 parametric electrical features + 10 polar/probe spatial features + 7 PCA moment features.
    """
    def __init__(self, candidates_dir=None):
        if candidates_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidates_dir = os.path.join(base_dir, 'models', 'candidates')
            
        self.model_path = os.path.join(candidates_dir, 'model_A_enhanced.pkl')
        self.pca_path = os.path.join(candidates_dir, 'pca_pipeline.pkl')
        self.optimal_threshold = 0.5100
        self.n_features = 517
        
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
            self.feature_names = list(self.model.feature_name_) if hasattr(self.model, 'feature_name_') else [f'f_{i}' for i in range(517)]
            
        with open(self.pca_path, 'rb') as f:
            self.pca_bundle = pickle.load(f)
            self.pca = self.pca_bundle['pca']
            self.pass_mean = self.pca_bundle['pass_mean']
            self.pass_std = self.pca_bundle['pass_std']

    def extract_features(self, df_subset, full_df=None):
        """Extract all 517 candidate features for Model A."""
        from src.spatial import compute_enhanced_spatial_features
        feat_cols = [f'feature_{i}' for i in range(1, 501)]
        
        # 1. Spatial features
        sp_df = compute_enhanced_spatial_features(df_subset, full_df)
        
        # 2. Parametric standardized moments & PCA residual
        X_raw = df_subset[feat_cols].values
        Z = (X_raw - self.pass_mean) / self.pass_std
        abs_z = np.abs(Z)
        Z_proj = self.pca.transform(Z)
        Z_recon = self.pca.inverse_transform(Z_proj)
        
        pstats = pd.DataFrame({
            'param_max_abs_z': np.max(abs_z, axis=1),
            'param_mean_abs_z': np.mean(abs_z, axis=1),
            'param_std_z': np.std(Z, axis=1),
            'param_sum_sq_z': np.sum(Z**2, axis=1),
            'param_cnt_gt_2sigma': np.sum(abs_z > 2.0, axis=1),
            'param_cnt_gt_3sigma': np.sum(abs_z > 3.0, axis=1),
            'pca_recon_error': np.sum((Z - Z_recon)**2, axis=1)
        }, index=df_subset.index)
        
        return pd.concat([df_subset[feat_cols], sp_df, pstats], axis=1)

    def predict_proba(self, X_df):
        """Predict failure probability using Candidate Model A."""
        return self.model.predict_proba(X_df)[:, 1]

    def predict(self, X_df, threshold=None):
        """Predict binary failure class using Candidate Model A."""
        if threshold is None:
            threshold = self.optimal_threshold
        probs = self.predict_proba(X_df)
        return (probs >= threshold).astype(int)
