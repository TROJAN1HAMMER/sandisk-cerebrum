import os
import json
import logging
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, matthews_corrcoef
from fastapi import APIRouter, Depends, HTTPException

from backend.config import RUNTIME_EVAL_JSON_PATH, settings
from backend.deps import (
    get_model_a,
    get_model_b,
    get_diagnostics_df,
    get_auth_metrics
)
from backend.api.schemas.models import (
    EvaluationRunResponse,
    PolicySimulationRequest,
    PolicySimulationResponse
)

# Reused ML engine function
from src.live_inference import execute_full_live_evaluation

logger = logging.getLogger("cerebrum.routes.evaluation")
router = APIRouter(prefix="/evaluation", tags=["Dynamic Live Evaluation Engine"])

@router.post("/run", response_model=EvaluationRunResponse, summary="Execute Live Full Evaluation (27,972 Dies)")
def run_full_evaluation(
    model_A=Depends(get_model_a),
    model_B=Depends(get_model_b),
    diag_df=Depends(get_diagnostics_df),
    auth_metrics=Depends(get_auth_metrics)
) -> EvaluationRunResponse:
    """
    Executes live model evaluation dynamically across all 27,972 canonical holdout dies.
    Verifies that freshly computed metrics match the Authoritative Clean Metrics with exact 0.000000 delta.
    Saves outputs to results/runtime_evaluation_latest.json.
    """
    try:
        runtime_res = execute_full_live_evaluation(
            model_A,
            model_B,
            diag_df,
            auth_metrics_dict=auth_metrics
        )
    except Exception as e:
        logger.exception("Full evaluation execution failed")
        raise HTTPException(
            status_code=500,
            detail={"code": "EVALUATION_ERROR", "message": f"Full evaluation failed: {str(e)}"}
        )

    # Persist runtime evaluation artifact
    try:
        with open(RUNTIME_EVAL_JSON_PATH, "w") as f:
            json.dump(runtime_res, f, indent=2)
    except Exception as e:
        logger.warning("Could not write runtime evaluation artifact: %s", e)

    meta = runtime_res.get("metadata", {})
    return EvaluationRunResponse(
        status="COMPLETED",
        evaluated_at=str(meta.get("evaluated_at", "")),
        validation_dies_evaluated=int(meta.get("validation_dies_evaluated", len(diag_df))),
        validation_wafers_count=int(meta.get("validation_wafers_count", 32)),
        runtimes=meta.get("runtimes", {}),
        model_A=runtime_res.get("model_A", {}),
        model_B=runtime_res.get("model_B", {}),
        cascade_gate_B=runtime_res.get("cascade_gate_B", {}),
        cascade_gate_C=runtime_res.get("cascade_gate_C", {}),
        integrity_checks=runtime_res.get("integrity_checks", {})
    )

@router.get("/latest", response_model=EvaluationRunResponse, summary="Get Latest Evaluation Run Result")
def get_latest_evaluation(
    model_A=Depends(get_model_a),
    model_B=Depends(get_model_b),
    diag_df=Depends(get_diagnostics_df),
    auth_metrics=Depends(get_auth_metrics)
) -> EvaluationRunResponse:
    """
    Returns the most recent evaluation run artifact.
    If no previous run is found on disk, executes the evaluation on demand.
    """
    if RUNTIME_EVAL_JSON_PATH.exists():
        try:
            with open(RUNTIME_EVAL_JSON_PATH, "r") as f:
                runtime_res = json.load(f)
            meta = runtime_res.get("metadata", {})
            return EvaluationRunResponse(
                status="CACHED",
                evaluated_at=str(meta.get("evaluated_at", "")),
                validation_dies_evaluated=int(meta.get("validation_dies_evaluated", len(diag_df))),
                validation_wafers_count=int(meta.get("validation_wafers_count", 32)),
                runtimes=meta.get("runtimes", {}),
                model_A=runtime_res.get("model_A", {}),
                model_B=runtime_res.get("model_B", {}),
                cascade_gate_B=runtime_res.get("cascade_gate_B", {}),
                cascade_gate_C=runtime_res.get("cascade_gate_C", {}),
                integrity_checks=runtime_res.get("integrity_checks", {})
            )
        except Exception as e:
            logger.warning("Could not read runtime evaluation artifact: %s", e)

    # Run on demand if not found
    return run_full_evaluation(model_A, model_B, diag_df, auth_metrics)


@router.post("/simulate-policy", response_model=PolicySimulationResponse, summary="Execute In-Memory Policy Simulation")
def simulate_policy(
    req: PolicySimulationRequest,
    diag_df=Depends(get_diagnostics_df),
    auth_metrics=Depends(get_auth_metrics)
) -> PolicySimulationResponse:
    """
    Simulates custom routing and threshold policies dynamically on holdout diagnostics (27,972 dies).
    CRITICAL: Operates strictly in-memory. Does NOT modify the frozen production configuration.
    """
    tau_A = float(req.tau_A)
    gate_lower = float(req.gate_lower)
    gate_upper = float(req.gate_upper)
    tau_B = float(req.tau_B)

    p_A = diag_df["prob_Model_A"].values
    p_B = diag_df["prob_Model_B"].values
    y_true = diag_df["actual_label"].values

    in_gate = (p_A >= gate_lower) & (p_A <= gate_upper)
    total_dies = len(y_true)

    # Optional secondary inspection budget constraint
    if req.secondary_budget_pct is not None:
        budget_cap = int(total_dies * (req.secondary_budget_pct / 100.0))
        gate_indices = np.where(in_gate)[0]
        if len(gate_indices) > budget_cap:
            # Prioritize dies with highest uncertainty (closest to tau_A)
            uncertainty = np.abs(p_A[gate_indices] - tau_A)
            prioritized = gate_indices[np.argsort(uncertainty)[:budget_cap]]
            in_gate_filtered = np.zeros_like(in_gate, dtype=bool)
            in_gate_filtered[prioritized] = True
            in_gate = in_gate_filtered

    esc_count = int(np.sum(in_gate))
    esc_rate = float(esc_count / total_dies * 100.0)
    comp_saved = float(100.0 - esc_rate)

    pred_A = (p_A >= tau_A).astype(int)
    pred_B = (p_B >= tau_B).astype(int)
    pred_casc = np.where(in_gate, pred_B, pred_A)

    tn, fp, fn, tp = confusion_matrix(y_true, pred_casc).ravel()
    prec = float(precision_score(y_true, pred_casc, zero_division=0))
    rec = float(recall_score(y_true, pred_casc, zero_division=0))
    f1 = float(f1_score(y_true, pred_casc, zero_division=0))
    f2 = float((5 * prec * rec) / (4 * prec + rec)) if (4 * prec + rec) > 0 else 0.0
    mcc = float(matthews_corrcoef(y_true, pred_casc))

    # Rescue mechanics under simulated policy
    ma_fn = (pred_A == 0) & (y_true == 1)
    fn_escalated = int(np.sum(ma_fn & in_gate))
    fn_rescued = int(np.sum(ma_fn & in_gate & (pred_B == 1)))
    tp_lost = int(np.sum((pred_A == 1) & (y_true == 1) & in_gate & (pred_B == 0)))
    new_fp = int(np.sum((pred_A == 0) & (y_true == 0) & in_gate & (pred_B == 1)))
    net_tp = fn_rescued - tp_lost

    base_b = auth_metrics.get("cascade_gate_B", {})

    return PolicySimulationResponse(
        status="SIMULATION_COMPLETED",
        simulation_mode=True,
        production_policy_modified=False,
        disclaimer="WHAT-IF SIMULATION ONLY — Does not modify production policy configuration.",
        parameters={
            "simulated_tau_A": tau_A,
            "simulated_gate_lower": gate_lower,
            "simulated_gate_upper": gate_upper,
            "simulated_tau_B": tau_B,
            "secondary_budget_pct": req.secondary_budget_pct
        },
        simulated_metrics={
            "f1": round(f1, 5),
            "f2": round(f2, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
            "mcc": round(mcc, 5),
            "escalation_count": esc_count,
            "escalation_rate_pct": round(esc_rate, 4),
            "compute_saved_pct": round(comp_saved, 4),
            "tp": int(tp),
            "fn": int(fn),
            "fp": int(fp),
            "tn": int(tn),
            "fn_rescued": fn_rescued,
            "fn_escalated": fn_escalated,
            "tp_lost": tp_lost,
            "net_tp_gain": net_tp,
            "new_fp": new_fp
        },
        baseline_comparison={
            "production_f1": float(base_b.get("f1", 0.5254)),
            "production_precision": float(base_b.get("precision", 0.8932)),
            "production_recall": float(base_b.get("recall", 0.3722)),
            "production_escalation_pct": float(base_b.get("escalation_rate_pct", 5.98)),
            "delta_f1": round(f1 - float(base_b.get("f1", 0.5254)), 5),
            "delta_precision": round(prec - float(base_b.get("precision", 0.8932)), 5),
            "delta_recall": round(rec - float(base_b.get("recall", 0.3722)), 5),
            "delta_escalation_pct": round(esc_rate - float(base_b.get("escalation_rate_pct", 5.98)), 4),
            "delta_compute_saved_pct": round(comp_saved - float(base_b.get("compute_saved_pct", 94.02)), 4)
        }
    )
