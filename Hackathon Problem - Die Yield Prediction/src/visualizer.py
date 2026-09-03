import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_wafer_prediction_comparison(wafer_df, wafer_id, save_path=None):
    """
    Plot 4-panel Wafer Map Comparison:
    1. Actual Failure Map
    2. Model A Risk Heatmap
    3. Model B Risk Heatmap
    4. Model B Final Pass/Fail Classification
    """
    r_max = wafer_df['die_row'].max() + 1
    c_max = wafer_df['die_col'].max() + 1

    actual_grid = np.full((r_max, c_max), np.nan)
    prob_A_grid = np.full((r_max, c_max), np.nan)
    prob_B_grid = np.full((r_max, c_max), np.nan)
    pred_B_grid = np.full((r_max, c_max), np.nan)

    actual_grid[wafer_df['die_row'], wafer_df['die_col']] = wafer_df['label']
    prob_A_grid[wafer_df['die_row'], wafer_df['die_col']] = wafer_df['prob_A']
    prob_B_grid[wafer_df['die_row'], wafer_df['die_col']] = wafer_df['prob_B']
    pred_B_grid[wafer_df['die_row'], wafer_df['die_col']] = wafer_df['pred_B']

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    # 1. Actual
    im0 = axes[0].imshow(actual_grid, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title(f"Actual Post-Test Status\nWafer {wafer_id}", fontsize=11)
    axes[0].set_xlabel("Die Col")
    axes[0].set_ylabel("Die Row")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # 2. Model A Prob
    im1 = axes[1].imshow(prob_A_grid, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title(f"Model A Risk (Parametric+Spatial)\nWafer {wafer_id}", fontsize=11)
    axes[1].set_xlabel("Die Col")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # 3. Model B Prob
    im2 = axes[2].imshow(prob_B_grid, cmap='magma', interpolation='nearest', vmin=0, vmax=1)
    axes[2].set_title(f"Model B Risk (Multi-Resolution)\nWafer {wafer_id}", fontsize=11)
    axes[2].set_xlabel("Die Col")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Model B Pred
    im3 = axes[3].imshow(pred_B_grid, cmap='YlOrRd', interpolation='nearest', vmin=0, vmax=1)
    axes[3].set_title(f"Model B Predictions\nWafer {wafer_id}", fontsize=11)
    axes[3].set_xlabel("Die Col")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    return fig


def plot_block_anomaly_attribution(explanation_dict, save_path=None):
    """
    Plot 2,000 block readings with highlighted 1D anomaly attribution.
    """
    readings = explanation_dict['block_readings']
    attributions = explanation_dict['attributions']
    wid = explanation_dict['wafer_id']
    r, c = explanation_dict['die_row'], explanation_dict['die_col']

    fig, ax1 = plt.subplots(figsize=(12, 4))

    color = 'tab:blue'
    ax1.set_xlabel('Sub-Die Block Index (0 - 1999)', fontsize=11)
    ax1.set_ylabel('Test Reading Value', color=color, fontsize=11)
    ax1.plot(readings, color=color, alpha=0.7, linewidth=1.0, label='Block Reading Value')
    ax1.axhline(100.0, color='gray', linestyle='--', label='Baseline (100.0)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([40, 160])

    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('1D CNN Anomaly Attribution', color=color, fontsize=11)
    ax2.fill_between(range(len(attributions)), attributions, color=color, alpha=0.35, label='Occlusion Attribution')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 1.2])

    plt.title(f"Block Anomaly Attribution — Wafer {wid}, Die ({r}, {c})\nModel A Risk: {explanation_dict['prob_A']*100:.1f}% | Model B Risk: {explanation_dict['prob_B']*100:.1f}% ({explanation_dict['risk_category']})", fontsize=12)
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200)
        plt.close()
    return fig
