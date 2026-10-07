import os
# Prevent macOS Python 3.14 OpenMP SIGSEGV crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import lightgbm as lgb

SEED = 42

class Block1DCNN(nn.Module):
    """
    Compact 1D Convolutional Neural Network for 2,000 sub-die block readings.
    Receptive field >= 100 blocks to detect contiguous anomaly clusters.
    """
    def __init__(self, in_channels=1, seq_len=2000, embedding_dim=32):
        super(Block1DCNN, self).__init__()
        
        # Layer 1: Conv1D (k=15, stride=2, padding=7) -> receptive field ~15
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=15, stride=2, padding=7)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2) # seq_len / 4 = 500
        
        # Layer 2: Conv1D (k=15, stride=2, padding=7) -> receptive field ~45
        self.conv2 = nn.Conv1d(16, 32, kernel_size=15, stride=2, padding=7)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(2) # seq_len / 16 = 125
        
        # Layer 3: Conv1D (k=15, stride=1, padding=7) -> receptive field > 100
        self.conv3 = nn.Conv1d(32, 64, kernel_size=15, stride=1, padding=7)
        self.bn3 = nn.BatchNorm1d(64)
        self.relu3 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1) # [batch, 64, 1]
        
        # FC Embedding & Output Head
        self.fc_embed = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.out_head = nn.Linear(embedding_dim, 1)
        
    def extract_embedding(self, x):
        """Input x: [batch, 1, 2000] -> Returns 32-dim embedding."""
        out = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out = self.pool2(self.relu2(self.bn2(self.conv2(out))))
        out = self.gap(self.relu3(self.bn3(self.conv3(out))))
        out = out.squeeze(-1)
        embed = self.fc_embed(out)
        return embed

    def forward(self, x):
        embed = self.extract_embedding(x)
        logits = self.out_head(embed)
        return logits.squeeze(-1)


class MultiScaleDilatedTCN(nn.Module):
    """
    Multi-Scale Dilated Temporal Convolutional Network.
    Combines parallel receptive fields with dilations d=1, d=2, d=4, d=8
    and residual skip connections for 2,000-block sequence inspection.
    """
    def __init__(self, in_channels=1, embedding_dim=32):
        super(MultiScaleDilatedTCN, self).__init__()
        
        # Stem Conv
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2) # 2000 -> 500
        )
        
        # Multi-Scale Dilated Inception Block
        self.conv_d1 = nn.Conv1d(16, 16, kernel_size=3, padding=1, dilation=1)
        self.conv_d2 = nn.Conv1d(16, 16, kernel_size=3, padding=2, dilation=2)
        self.conv_d4 = nn.Conv1d(16, 16, kernel_size=3, padding=4, dilation=4)
        self.conv_pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(16, 16, kernel_size=1)
        )
        self.bn_ms = nn.BatchNorm1d(64)
        self.relu_ms = nn.ReLU()
        self.res_proj = nn.Conv1d(16, 64, kernel_size=1)
        
        # Stage 2: Temporal compression
        self.stage2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, stride=2, padding=2), # 500 -> 250
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, dilation=2), # 250 -> 125
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Global Pooling & Projection Head
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_embed = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.out_head = nn.Linear(embedding_dim, 1)

    def extract_embedding(self, x):
        out = self.stem(x)
        branch1 = self.conv_d1(out)
        branch2 = self.conv_d2(out)
        branch3 = self.conv_d4(out)
        branch4 = self.conv_pool(out)
        ms_cat = torch.cat([branch1, branch2, branch3, branch4], dim=1)
        res = self.res_proj(out)
        out = self.relu_ms(self.bn_ms(ms_cat) + res)
        
        out = self.stage2(out)
        out = self.gap(out).squeeze(-1)
        embed = self.fc_embed(out)
        return embed

    def forward(self, x):
        embed = self.extract_embedding(x)
        logits = self.out_head(embed).squeeze(-1)
        return logits


def _get_model_outputs_in_batches(model, tensor_x, batch_size=512, device='cpu'):
    """Batched forward pass to prevent GPU/MPS out-of-memory errors."""
    model.eval()
    all_embeds = []
    all_probs = []
    n = len(tensor_x)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            bx = tensor_x[start:start+batch_size].to(device)
            embed = model.extract_embedding(bx)
            logits = model.out_head(embed).squeeze(-1)
            probs = torch.sigmoid(logits)
            
            all_embeds.append(embed.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            
    return np.vstack(all_embeds), np.concatenate(all_probs)


def train_block_cnn(X_tr_blocks, y_tr, X_val_blocks, y_val, epochs=15, batch_size=256, lr=0.001, device='cpu'):
    """Train 1D CNN on 2,000 block arrays with early stopping."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    # Normalize block readings around baseline 100.0
    norm_X_tr = (X_tr_blocks - 100.0) / 15.0
    norm_X_val = (X_val_blocks - 100.0) / 15.0
    
    tensor_tr_x = torch.tensor(norm_X_tr, dtype=torch.float32).unsqueeze(1) # [N, 1, 2000]
    tensor_tr_y = torch.tensor(y_tr, dtype=torch.float32)
    
    tensor_val_x = torch.tensor(norm_X_val, dtype=torch.float32).unsqueeze(1)
    tensor_val_y = torch.tensor(y_val, dtype=torch.float32)
    
    train_dataset = TensorDataset(tensor_tr_x, tensor_tr_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    model = Block1DCNN().to(device)
    
    pos_weight_val = torch.tensor([(len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)], dtype=torch.float32).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_val)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    best_weights = None
    
    val_loader = DataLoader(TensorDataset(tensor_val_x, tensor_val_y), batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(by)
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                v_loss = criterion(v_logits, vy)
                val_loss += v_loss.item() * len(vy)
            val_loss /= len(tensor_val_y)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict()
            
    if best_weights is not None:
        model.load_state_dict(best_weights)
        
    tr_embeds, tr_probs = _get_model_outputs_in_batches(model, tensor_tr_x, batch_size=batch_size, device=device)
    val_embeds, val_probs = _get_model_outputs_in_batches(model, tensor_val_x, batch_size=batch_size, device=device)
        
    return model, tr_probs, val_probs, tr_embeds, val_embeds


class MultiResolutionModelB:
    """
    Dual-Stream Multi-Resolution Model B.
    Combines Model A (506 tabular/spatial features) + Model B 1D CNN Block Embeddings.
    """
    def __init__(self, scale_pos_weight=10.0):
        self.scale_pos_weight = scale_pos_weight
        self.fusion_model = None
        self.optimal_threshold = 0.65
        
    def fit(self, X_tr_A, tr_cnn_embeds, y_tr):
        """Fit LightGBM Fusion Classifier on Model A features + 32-dim CNN block embeddings."""
        X_tr_fused = np.hstack([X_tr_A, tr_cnn_embeds])
        
        self.fusion_model = lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.7,
            scale_pos_weight=self.scale_pos_weight,
            random_state=SEED,
            verbose=-1,
            n_jobs=1
        )
        self.fusion_model.fit(X_tr_fused, y_tr)
        
    def predict_proba(self, X_val_A, val_cnn_embeds):
        X_val_fused = np.hstack([X_val_A, val_cnn_embeds])
        return self.fusion_model.predict_proba(X_val_fused)[:, 1]
        
    def predict(self, X_val_A, val_cnn_embeds, threshold=None):
        if threshold is None:
            threshold = self.optimal_threshold
        probs = self.predict_proba(X_val_A, val_cnn_embeds)
        return (probs >= threshold).astype(int)


class CandidateModelBWrapper:
    """
    Wrapper for Validated Candidate Model B (B3 Candidate: 528 Tabular Features + 1D-CNN Late Fusion).
    Architecture:
      p_final = 0.90 * p_tabular(528 features) + 0.10 * p_cnn(2,000 blocks)
    Threshold: tau_B = 0.5200 (Default Candidate Operating Threshold)
    """
    def __init__(self, candidates_dir=None, models_dir=None, device=None):
        import pickle
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if candidates_dir is None:
            candidates_dir = os.path.join(base_dir, 'models', 'candidates')
        if models_dir is None:
            models_dir = os.path.join(base_dir, 'models')
            
        self.tab_model_path = os.path.join(candidates_dir, 'model_B_tab_enhanced.pkl')
        self.cnn_path = os.path.join(models_dir, 'cnn_1d_block_model.pt')
        self.optimal_threshold = 0.5200
        self.n_tabular_features = 528
        
        # Load tabular LightGBM
        with open(self.tab_model_path, 'rb') as f:
            self.tab_model = pickle.load(f)
            self.feature_names = list(self.tab_model.feature_name_) if hasattr(self.tab_model, 'feature_name_') else [f'f_{i}' for i in range(528)]
            
        # Load 1D-CNN
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        self.device = device
        self.cnn_model = Block1DCNN().to(self.device)
        self.cnn_model.load_state_dict(torch.load(self.cnn_path, map_location=self.device))
        self.cnn_model.eval()

    def extract_features(self, X_A_cand, block_matrix):
        """Combine 517 Model A features with 11 physics-informed block burst features."""
        from src.block_parser import extract_B3_candidate_block_features
        bstats = extract_B3_candidate_block_features(block_matrix)
        return pd.concat([X_A_cand, bstats], axis=1)

    def predict_proba(self, X_B_cand, block_matrix):
        """
        Dual-Stream Late Probability Fusion:
        p_final = 0.90 * p_tab + 0.10 * p_cnn
        """
        # 1. Tabular prediction
        p_tab = self.tab_model.predict_proba(X_B_cand)[:, 1]
        
        # 2. 1D CNN prediction
        norm_blocks = (block_matrix - 100.0) / 15.0
        tensor_x = torch.tensor(norm_blocks, dtype=torch.float32).unsqueeze(1)
        _, p_cnn = _get_model_outputs_in_batches(self.cnn_model, tensor_x, batch_size=512, device=self.device)
        
        # 3. Late Fusion Blend
        p_final = 0.90 * p_tab + 0.10 * p_cnn
        return p_final

    def predict(self, X_B_cand, block_matrix, threshold=None):
        """Predict binary failure class using Candidate Model B."""
        if threshold is None:
            threshold = self.optimal_threshold
        probs = self.predict_proba(X_B_cand, block_matrix)
        return (probs >= threshold).astype(int)
