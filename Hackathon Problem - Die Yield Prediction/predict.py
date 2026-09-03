import os
# Prevent macOS Python 3.14 OpenMP SIGSEGV crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import pickle
import json
import torch
import numpy as np
import pandas as pd

from src.data_loader import load_eligible_data
from src.spatial import compute_spatial_features
from src.block_parser import parse_block_matrix
from src.model_a import ModelAWrapper
from src.model_b import Block1DCNN, _get_model_outputs_in_batches
from src.cascade import SelectiveCascadeClassifier

base_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(base_dir, 'models')

def main():
    parser = argparse.ArgumentParser(description="SanDisk Die Yield Prediction - Selective Multi-Resolution Cascade CLI")
    parser.add_argument('--input', type=str, required=True, help="Path to input wafer dataset CSV")
    parser.add_argument('--output', type=str, required=True, help="Path to write prediction results CSV")
    parser.add_argument('--lower_gate', type=float, default=0.30, help="Lower uncertainty gate for Model A (default: 0.30)")
    parser.add_argument('--upper_gate', type=float, default=0.60, help="Upper uncertainty gate for Model A (default: 0.60)")
    args = parser.parse_args()

    print(f"Loading input dataset from {args.input}...")
    df = load_eligible_data(args.input)

    print("Loading Model A & 1D ConvNet Block Encoder...")
    model_A = ModelAWrapper(model_dir=models_dir)

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    cnn_model = Block1DCNN().to(device)
    cnn_model.load_state_dict(torch.load(os.path.join(models_dir, 'cnn_1d_block_model.pt'), map_location=device))
    cnn_model.eval()

    print("Loading Selective Cascade Meta-Classifier...")
    cascade = SelectiveCascadeClassifier(
        model_A, cnn_model, lower_gate=args.lower_gate, upper_gate=args.upper_gate,
        opt_threshold=model_A.optimal_threshold, device=device
    )
    meta_path = os.path.join(models_dir, 'selective_cascade_meta.pkl')
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            cascade.meta_classifier = pickle.load(f)

    print("Computing spatial context features...")
    spatial_df = compute_spatial_features(df)
    feat_cols = [f'feature_{i}' for i in range(1, 501)]
    spatial_names = ['radial_dist', 'density_3x3', 'density_5x5', 'edge_dist', 'die_row', 'die_col']
    df_A_input = pd.concat([df[feat_cols], spatial_df[spatial_names]], axis=1)

    print("Parsing 2,000 block readings & running 1D ConvNet feature extractor...")
    block_mat = parse_block_matrix(df['block_readings'])
    norm_blocks = (block_mat - 100.0) / 15.0
    tensor_x = torch.tensor(norm_blocks, dtype=torch.float32).unsqueeze(1)

    cnn_embeds, cnn_probs = _get_model_outputs_in_batches(cnn_model, tensor_x, batch_size=512, device=device)

    print("Running Selective Multi-Resolution Cascade Inference...")
    cascade_res = cascade.predict_cascade(df_A_input, tensor_x, cnn_probs, cnn_embeds)

    output_df = pd.DataFrame({
        'wafer_id': df['wafer_id'],
        'die_row': df['die_row'],
        'die_col': df['die_col'],
        'model_A_risk_prob': np.round(cascade_res['probs_A'], 4),
        'cascade_final_prob': np.round(cascade_res['final_probs'], 4),
        'model_A_decision': (cascade_res['probs_A'] >= model_A.optimal_threshold).astype(int),
        'cascade_final_decision': cascade_res['final_preds'],
        'resolution_used': cascade_res['resolution_used'],
        'escalation_reason': cascade_res['escalation_reasons'],
        'risk_category': cascade_res['risk_categories']
    })

    if 'label' in df.columns:
        output_df['actual_label'] = df['label']

    output_df.to_csv(args.output, index=False)
    print(f"Successfully generated selective cascade inference predictions for {len(output_df):,} dies at {args.output}")

if __name__ == '__main__':
    main()
