from fastapi import APIRouter
from backend.config import settings
from backend.deps import state
from backend.api.schemas.system import HealthResponse, StatusResponse

router = APIRouter(prefix="/system", tags=["System"])

@router.get("/health", response_model=HealthResponse, summary="System Health Check")
def get_health() -> HealthResponse:
    """Returns liveness status of the API."""
    return HealthResponse(
        status="online",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment
    )

@router.get("/status", response_model=StatusResponse, summary="Detailed System & Engine Status")
def get_status() -> StatusResponse:
    """Returns runtime verification status of loaded ML models, dataset integrity, and active gates."""
    diag_len = len(state.diagnostics_df) if state.diagnostics_df is not None else 0
    wafers_count = state.diagnostics_df["wafer_id"].nunique() if state.diagnostics_df is not None else 0
    
    return StatusResponse(
        status="online" if state.is_initialized else "initializing",
        models_loaded=state.is_initialized,
        model_a_active=state.model_A is not None,
        model_b_active=state.model_B is not None,
        cnn_active=state.cnn_model is not None,
        diagnostics_loaded=state.diagnostics_df is not None,
        total_validation_dies=diag_len,
        validation_wafers_count=wafers_count,
        canonical_split="128 Train / 32 Val (0.0% Overlap)",
        gate_b_window=f"[{settings.gate_b_lower:.2f}, {settings.gate_b_upper:.2f}]",
        gate_c_window=f"[{settings.gate_c_lower:.2f}, {settings.gate_c_upper:.2f}]",
        active_model_version=settings.active_model_version,
        production_gate_b_window=f"[{settings.production_gate_lower:.3f}, {settings.production_gate_upper:.3f}]",
        startup_time_s=float(round(state.startup_time_s, 4))
    )
