import os
from pathlib import Path
from typing import List
from pydantic import BaseModel

# -------------------------------------------------------------------------
# Thread & Concurrency Safeguards (Prevent OpenMP / BLAS SIGSEGV on macOS)
# -------------------------------------------------------------------------
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Base repository paths derived dynamically (no hardcoded machine paths)
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
INPUT_DIR = BASE_DIR / "input"

AUTH_METRICS_PATH = RESULTS_DIR / "AUTHORITATIVE_CLEAN_METRICS.json"
BENCHMARK_PATH = RESULTS_DIR / "final_clean_benchmark.csv"
GATE_SWEEP_PATH = RESULTS_DIR / "cascade_gate_sweep.csv"
DIAGNOSTICS_PATH = RESULTS_DIR / "val_die_diagnostics.csv"
PREDICTIONS_PATH = RESULTS_DIR / "clean_model_B_predictions.csv"
DEMO_SAMPLES_PATH = RESULTS_DIR / "demo_samples.json"
RUNTIME_EVAL_JSON_PATH = RESULTS_DIR / "runtime_evaluation_latest.json"
MODEL_RELEASE_PATH = MODELS_DIR / "model_release.json"
CANDIDATES_META_PATH = MODELS_DIR / "candidates" / "candidate_metadata.json"
CANDIDATES_EXP04_PATH = RESULTS_DIR / "teammate_candidate_exp04_results.json"

class Settings(BaseModel):
    app_name: str = "SanDisk Cerebrum API"
    app_version: str = "1.0.0"
    environment: str = os.getenv("ENVIRONMENT", "development")
    host: str = os.getenv("HOST", "127.0.0.1")
    port: int = int(os.getenv("PORT", "8000"))
    
    # CORS Origins
    cors_allowed_origins: List[str] = [
        origin.strip()
        for origin in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000"
        ).split(",")
        if origin.strip()
    ]
    
    # Model Thresholds & Cascade Defaults (matching validated engine)
    model_a_threshold: float = 0.5100
    model_b_threshold: float = 0.5376
    gate_b_lower: float = 0.25
    gate_b_upper: float = 0.55
    gate_c_lower: float = 0.30
    gate_c_upper: float = 0.60
    
    # Active Validated Production Configuration (EXP-03B confirmed / EXP-04 validated)
    active_model_version: str = "B2"
    production_gate_lower: float = 0.295
    production_gate_upper: float = 0.550

settings = Settings()
