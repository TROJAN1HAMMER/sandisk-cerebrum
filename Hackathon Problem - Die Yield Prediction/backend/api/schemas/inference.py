from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class SingleDieRequest(BaseModel):
    wafer_id: Optional[str] = "W_TEST_0001"
    die_row: int = Field(..., ge=0, le=50, description="Die row coordinate")
    die_col: int = Field(..., ge=0, le=50, description="Die column coordinate")
    features: Optional[Dict[str, float]] = Field(default=None, description="Parametric features (feature_1 .. feature_500)")
    spatial: Optional[Dict[str, float]] = Field(default=None, description="Spatial signals (radial_dist, density_3x3, etc.)")
    block_readings: Optional[str] = Field(default=None, description="Space-separated 2,000 sub-die block readings")
    prob_Model_A: Optional[float] = Field(default=None, description="Pre-computed Model A probability override")
    prob_Model_B: Optional[float] = Field(default=None, description="Pre-computed Model B probability override")
    actual_label: Optional[int] = Field(default=None, description="Ground truth post-test label (0=Pass, 1=Fail)")
    gate_lower: Optional[float] = Field(default=None, description="Uncertainty lower gate override (default 0.25)")
    gate_upper: Optional[float] = Field(default=None, description="Uncertainty upper gate override (default 0.55)")

class DieInferenceResponse(BaseModel):
    wafer_id: str
    die_row: int
    die_col: int
    prob_Model_A: float
    threshold_Model_A: float
    decision_Model_A: int
    in_gate: bool
    gate_status: str
    gate_message: str
    model_B_invoked: bool
    prob_Model_B: Optional[float] = None
    threshold_Model_B: float
    decision_Model_B: Optional[int] = None
    final_probability: float
    final_decision: int
    decision_text: str
    compute_path: str
    actual_label: Optional[int] = None
    is_correct: Optional[bool] = None
    spatial_signals: Dict[str, float]
    block_signals: Dict[str, float]
    latency_ms: Optional[float] = None

class BatchInferenceSummary(BaseModel):
    total_dies: int
    predicted_fails: int
    predicted_passes: int
    predicted_fail_rate_pct: float
    escalated_dies: int
    escalation_rate_pct: float
    compute_saved_pct: float
    total_inference_time_s: float
    latency_per_die_ms: float

class BatchInferenceItem(BaseModel):
    wafer_id: str
    die_row: int
    die_col: int
    prob_Model_A: float
    decision_Model_A: int
    escalated: str
    prob_Model_B: Optional[float] = None
    decision_Model_B: Optional[int] = None
    final_probability: float
    final_decision: int
    decision_label: str
    compute_path: str
    actual_label: Optional[int] = None

class BatchInferenceResponse(BaseModel):
    summary: BatchInferenceSummary
    results: List[BatchInferenceItem]

class DemoSampleItem(BaseModel):
    sample_id: str
    wafer_id: str
    die_row: int
    die_col: int
    actual_label: int
    model_A_prob: float
    cascade_final_prob: float
    model_A_decision: int
    cascade_final_decision: int
    resolution_used: str
    escalation_reason: str
    description: str

class BlockAttributionResponse(BaseModel):
    wafer_id: str
    die_row: int
    die_col: int
    base_prob: float
    block_count: int
    readings: List[float]
    attributions: List[float]
    peak_attribution_idx: int
    peak_attribution_value: float
    anomalous_blocks_count: int
