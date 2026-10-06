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
