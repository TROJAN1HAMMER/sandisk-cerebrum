from pydantic import BaseModel
from typing import Optional

class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str

class StatusResponse(BaseModel):
    status: str
    models_loaded: bool
    model_a_active: bool
    model_b_active: bool
    cnn_active: bool
    diagnostics_loaded: bool
    total_validation_dies: int
    validation_wafers_count: int
    canonical_split: str
    gate_b_window: str
    gate_c_window: str
    active_model_version: Optional[str] = "B2"
    production_gate_b_window: Optional[str] = "[0.295, 0.550]"
    startup_time_s: float
