from pydantic import BaseModel
from typing import List, Optional

class WaferSummaryItem(BaseModel):
    wafer_id: str
    total_dies: int
    actual_failures: int
    fail_rate_pct: float
    model_a_detections: int
    cascade_detections: int
    escalated_dies: int
    recovered_fns: int
    mean_probability: Optional[float] = None
    high_risk_dies: Optional[int] = None

class WaferListResponse(BaseModel):
    total_wafers: int
    total_holdout_dies: int
    wafers: List[WaferSummaryItem]

class DiePoint(BaseModel):
    wafer_id: str
    die_row: int
    die_col: int
    actual_label: int
    prob_Model_A: float
    prob_Model_B: float
    prob_Cascade_B: float
    decision_Model_A: int
    decision_Cascade_B: int
    in_gate_B: bool
    recovered_FN_Gate_B: int
    delta_prob: float
    radial_dist: Optional[float] = None
    edge_dist: Optional[float] = None
    b_max: Optional[float] = None
    cnt_104: Optional[float] = None

class WaferDetailResponse(BaseModel):
    summary: WaferSummaryItem
    dies: List[DiePoint]
