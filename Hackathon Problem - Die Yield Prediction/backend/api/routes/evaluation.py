import os
import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from backend.config import RUNTIME_EVAL_JSON_PATH
from backend.deps import (
    get_model_a,
    get_model_b,
    get_diagnostics_df,
    get_auth_metrics
)
from backend.api.schemas.models import EvaluationRunResponse

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
