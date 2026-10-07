from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score

from backend.config import settings
from backend.deps import (
    get_auth_metrics,
    get_benchmark_df,
    get_gate_sweep_df,
    get_diagnostics_df,
    get_model_release
)
from backend.api.schemas.models import (
    ModelMetricsResponse,
    BenchmarkTableResponse
)

router = APIRouter(prefix="/models", tags=["Model Analytics & Benchmarks"])

@router.get("/release", summary="Get Official B3 Model Release Metadata")
def get_release(release_meta: Optional[Dict[str, Any]] = Depends(get_model_release)) -> Dict[str, Any]:
    """
    Returns the comprehensive model release metadata, active production configuration,
    and finalized independent validation metrics (EXP-04).
    """
    if release_meta is None:
        raise HTTPException(status_code=404, detail={"code": "RELEASE_NOT_FOUND", "message": "Model release metadata not found"})
    res = dict(release_meta)
    res["evaluation_scope"] = "Active B3 — Independent EXP-04 Validation"
    res["dataset_label"] = "Independent Test Set (40 Wafers, 32,598 Dies)"
    return res

@router.get("/metrics", response_model=ModelMetricsResponse, summary="Get Authoritative Clean Verification Metrics")
def get_metrics(auth_metrics: Dict[str, Any] = Depends(get_auth_metrics)) -> ModelMetricsResponse:
    """
    Returns the single source of truth benchmarks evaluated on the canonical 32 holdout wafers (27,972 dies).
    Directly reflects AUTHORITATIVE_CLEAN_METRICS.json.
    """
    return ModelMetricsResponse(
        metadata=auth_metrics.get("metadata", {}),
        model_A=auth_metrics.get("model_A", {}),
        model_B=auth_metrics.get("model_B", {}),
        cascade_gate_B=auth_metrics.get("cascade_gate_B", {}),
        cascade_gate_C=auth_metrics.get("cascade_gate_C", {})
    )

@router.get("/benchmark", response_model=BenchmarkTableResponse, summary="Get Comparative Benchmark Table & Gate Sweeps")
def get_benchmark(
    bench_df: Optional[pd.DataFrame] = Depends(get_benchmark_df),
    gate_sweep_df: Optional[pd.DataFrame] = Depends(get_gate_sweep_df)
) -> BenchmarkTableResponse:
    """
    Returns the comprehensive architectural comparison matrix (Model A vs Standalone Model B vs Gate B vs Gate C)
    and the uncertainty gating sweep curve data (Gate A through Gate F).
    """
    bench_cols = list(bench_df.columns) if bench_df is not None else []
    bench_rows = bench_df.to_dict(orient="records") if bench_df is not None else []

    sweep_cols = list(gate_sweep_df.columns) if gate_sweep_df is not None else []
    sweep_rows = gate_sweep_df.to_dict(orient="records") if gate_sweep_df is not None else []

    return BenchmarkTableResponse(
        benchmark_columns=bench_cols,
        benchmark_rows=bench_rows,
        gate_sweep_columns=sweep_cols,
        gate_sweep_rows=sweep_rows
    )

@router.get("/error-drilldown", summary="Forensic Error Population Drill-Down")
def get_error_drilldown(
    group: str = Query(
        "recovered_fns",
        description="Group: recovered_fns (37 dies), false_positives (89 dies), remaining_fns (667 dies), true_positives (389 dies)"
    ),
    limit: int = Query(50, ge=1, le=500),
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> Dict[str, Any]:
    """
    Drill down into specific error populations under Cascade Gate B.
    Allows inspecting recovered failures, remaining false negatives, and false alarms with exact telemetry.
    """
    g = group.lower()
    if g == "recovered_fns":
        filtered = diag_df[diag_df["recovered_FN_Gate_B"] == 1]
        title = "Model A False Negatives Recovered by Cascade Gate B (37 Dies)"
    elif g == "false_positives":
        filtered = diag_df[(diag_df["actual_label"] == 0) & (diag_df["decision_Cascade_B"] == 1)]
        title = "False Alarms Under Gate B (89 Dies)"
    elif g == "remaining_fns":
        filtered = diag_df[(diag_df["actual_label"] == 1) & (diag_df["decision_Cascade_B"] == 0)]
        title = "Remaining False Negatives (667 Dies)"
    elif g == "true_positives":
        filtered = diag_df[(diag_df["actual_label"] == 1) & (diag_df["decision_Cascade_B"] == 1)]
        title = "True Positives Detected Under Gate B (389 Dies)"
    else:
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_GROUP", "message": f"Unknown error group '{group}'"}
        )

    cols = [
        "wafer_id", "die_row", "die_col", "actual_label",
        "prob_Model_A", "prob_Model_B", "prob_Cascade_B", "delta_prob",
        "decision_Model_A", "decision_Cascade_B", "b_max", "cnt_104"
    ]
    present_cols = [c for c in cols if c in filtered.columns]
    
    sample_df = filtered[present_cols].head(limit)

    return {
        "group": g,
        "title": title,
        "total_in_group": len(filtered),
        "returned_count": len(sample_df),
        "records": sample_df.to_dict(orient="records")
    }

@router.get("/probability-distribution", summary="Get Binned Probability Distribution for Model A and Cascade")
def get_probability_distribution(
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> Dict[str, Any]:
    """Returns binned probability histograms across all 27,972 dies comparing Model A vs Cascade Gate B."""
    bins = np.linspace(0.0, 1.0, 21)
    hist_a, _ = np.histogram(diag_df["prob_Model_A"], bins=bins)
    hist_cas, _ = np.histogram(diag_df["prob_Cascade_B"], bins=bins)

    bin_labels = [f"{bins[i]:.2f}-{bins[i+1]:.2f}" for i in range(len(bins)-1)]
    bin_centers = [round(float((bins[i] + bins[i+1]) / 2), 3) for i in range(len(bins)-1)]

    return {
        "bin_labels": bin_labels,
        "bin_centers": bin_centers,
        "model_a_counts": [int(x) for x in hist_a],
        "cascade_counts": [int(x) for x in hist_cas],
        "gate_window": {
            "lower": settings.gate_b_lower,
            "upper": settings.gate_b_upper
        },
        "model_a_threshold": settings.model_a_threshold,
        "model_b_threshold": settings.model_b_threshold,
        "total_dies": len(diag_df)
    }

@router.get("/roc-pr-curves", summary="Get Mathematically Verified ROC and PR Curve Points")
def get_roc_pr_curves(
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> Dict[str, Any]:
    """Returns exact empirical ROC and Precision-Recall curve coordinate points for Model A and Cascade Gate B."""
    y_true = diag_df["actual_label"].values
    p_a = diag_df["prob_Model_A"].values
    p_cas = diag_df["prob_Cascade_B"].values

    fpr_a, tpr_a, _ = roc_curve(y_true, p_a)
    fpr_cas, tpr_cas, _ = roc_curve(y_true, p_cas)

    prec_a, rec_a, _ = precision_recall_curve(y_true, p_a)
    prec_cas, rec_cas, _ = precision_recall_curve(y_true, p_cas)

    # Downsample cleanly to ~45 points for fast, lightweight network transfer
    n_pts = 45
    idx_roc_a = np.linspace(0, len(fpr_a) - 1, n_pts).astype(int)
    idx_roc_cas = np.linspace(0, len(fpr_cas) - 1, n_pts).astype(int)

    idx_pr_a = np.linspace(0, len(prec_a) - 1, n_pts).astype(int)
    idx_pr_cas = np.linspace(0, len(prec_cas) - 1, n_pts).astype(int)

    return {
        "roc": {
            "model_A": {
                "fpr": [round(float(x), 4) for x in fpr_a[idx_roc_a]],
                "tpr": [round(float(x), 4) for x in tpr_a[idx_roc_a]],
                "auc": round(float(roc_auc_score(y_true, p_a)), 4)
            },
            "cascade": {
                "fpr": [round(float(x), 4) for x in fpr_cas[idx_roc_cas]],
                "tpr": [round(float(x), 4) for x in tpr_cas[idx_roc_cas]],
                "auc": round(float(roc_auc_score(y_true, p_cas)), 4)
            }
        },
        "pr": {
            "model_A": {
                "recall": [round(float(x), 4) for x in rec_a[idx_pr_a]],
                "precision": [round(float(x), 4) for x in prec_a[idx_pr_a]],
                "auc_pr": round(float(average_precision_score(y_true, p_a)), 4)
            },
            "cascade": {
                "recall": [round(float(x), 4) for x in rec_cas[idx_pr_cas]],
                "precision": [round(float(x), 4) for x in prec_cas[idx_pr_cas]],
                "auc_pr": round(float(average_precision_score(y_true, p_cas)), 4)
            }
        }
    }

@router.get("/high-risk-dies", summary="Get Highest Failure Probability Dies across Validation Cohort")
def get_high_risk_dies(
    limit: int = Query(50, ge=1, le=200),
    wafer_id: Optional[str] = Query(None, description="Optional wafer filter"),
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> Dict[str, Any]:
    """Returns top dies sorted by Cascade failure probability descending."""
    df = diag_df
    if wafer_id:
        df = df[df["wafer_id"] == wafer_id]

    top_df = df.sort_values(by="prob_Cascade_B", ascending=False).head(limit)
    cols = [
        "wafer_id", "die_row", "die_col", "actual_label",
        "prob_Model_A", "prob_Model_B", "prob_Cascade_B", "delta_prob",
        "decision_Model_A", "decision_Cascade_B", "in_gate_B",
        "recovered_FN_Gate_B", "b_max", "cnt_104"
    ]
    present_cols = [c for c in cols if c in top_df.columns]
    records = top_df[present_cols].to_dict(orient="records")
    return {
        "limit": limit,
        "wafer_filter": wafer_id,
        "count": len(records),
        "records": records
    }

@router.get("/candidates", summary="Get Audited Experimental Candidates & Head-to-Head Benchmarks")
def get_candidates() -> Dict[str, Any]:
    """
    Returns officially audited experimental candidate models (e.g. B3-DualStream)
    and their comparative evaluation against the frozen B2 production baseline.
    """
    import json
    from backend.config import CANDIDATES_META_PATH, CANDIDATES_EXP04_PATH

    candidates_meta = {}
    if CANDIDATES_META_PATH.exists():
        with open(CANDIDATES_META_PATH, "r") as f:
            candidates_meta = json.load(f)

    exp04_results = {}
    if CANDIDATES_EXP04_PATH.exists():
        with open(CANDIDATES_EXP04_PATH, "r") as f:
            exp04_results = json.load(f)

    return {
        "status": "PROMOTED_TO_PRODUCTION",
        "active_production_release": "B3",
        "candidate_model_version": "B3-DualStream",
        "production_architecture": "LightGBM + 1D-CNN Late Probability Fusion (0.90 / 0.10)",
        "candidate_architecture": "LightGBM + 1D-CNN Late Probability Fusion (0.90 / 0.10)",
        "features": {
            "model_A_features": 517,
            "model_B_tabular_features": 528,
            "cnn_input_modality": "2,000 blocks raw sub-die sequence (separate modality)",
            "novel_features": [
                "Polar spatial coordinates (radial_dist, sin_theta, cos_theta)",
                "Pre-test spatial probe defect densities (3x3, 5x5, 7x7)",
                "Multivariate parametric PCA reconstruction residual (k=25)",
                "Parametric pass-distribution sigma excursions (gt_2sigma, gt_3sigma)",
                "Block burst anomaly streaks (>104, >106, longest_run_104)"
            ]
        },
        "artifacts": candidates_meta,
        "evaluation_summary": exp04_results
    }
