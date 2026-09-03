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
