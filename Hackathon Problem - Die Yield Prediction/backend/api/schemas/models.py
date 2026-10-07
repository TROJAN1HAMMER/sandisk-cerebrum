from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class ConfusionMatrixSchema(BaseModel):
    tp: int
    fn: int
    fp: int
    tn: int

class ModelMetricsItem(BaseModel):
    model_name: str
    threshold: float
    f1: float
    auc_pr: float
    roc_auc: float
    precision: float
    recall: float
    brier_score: Optional[float] = None
    confusion_matrix: ConfusionMatrixSchema
    extra_metrics: Optional[Dict[str, Any]] = None

class CascadeGateMetricsItem(BaseModel):
    gate_name: str
    gating_window: str
    model_A_threshold: float
    model_B_threshold: float
    escalation_count: int
    escalation_rate_pct: float
    compute_saved_pct: float
    f1: float
    auc_pr: float
    roc_auc: float
    precision: float
    recall: float
    brier_score: Optional[float] = None
    confusion_matrix: ConfusionMatrixSchema
    fn_recovery_metrics: Optional[Dict[str, Any]] = None
    delta_vs_model_A: Optional[Dict[str, Any]] = None

class ModelMetricsResponse(BaseModel):
    evaluation_scope: str = "Historical B0 — Canonical Holdout"
    dataset_label: str = "Canonical Holdout (32 Wafers, 27,972 Dies)"
    metadata: Dict[str, Any]
    model_A: Dict[str, Any]
    model_B: Dict[str, Any]
    cascade_gate_B: Dict[str, Any]
    cascade_gate_C: Dict[str, Any]

class BenchmarkTableResponse(BaseModel):
    benchmark_columns: List[str]
    benchmark_rows: List[Dict[str, Any]]
    gate_sweep_columns: List[str]
    gate_sweep_rows: List[Dict[str, Any]]

class EvaluationRunResponse(BaseModel):
    status: str
    evaluated_at: str
    validation_dies_evaluated: int
    validation_wafers_count: int
    runtimes: Dict[str, float]
    model_A: Dict[str, Any]
    model_B: Dict[str, Any]
    cascade_gate_B: Dict[str, Any]
    cascade_gate_C: Dict[str, Any]
    integrity_checks: Dict[str, Any]

class PolicySimulationRequest(BaseModel):
    tau_A: float = 0.5100
    gate_lower: float = 0.250
    gate_upper: float = 0.550
    tau_B: float = 0.5200
    secondary_budget_pct: Optional[float] = None

class PolicySimulationResponse(BaseModel):
    status: str = "SIMULATION_COMPLETED"
    simulation_mode: bool = True
    production_policy_modified: bool = False
    disclaimer: str = "WHAT-IF SIMULATION ONLY — Does not modify production policy configuration."
    parameters: Dict[str, Any]
    simulated_metrics: Dict[str, Any]
    baseline_comparison: Dict[str, Any]
