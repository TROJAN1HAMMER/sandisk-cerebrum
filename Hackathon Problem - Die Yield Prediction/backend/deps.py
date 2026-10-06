import os
import time
import json
import pickle
import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import torch

from backend.config import (
    MODELS_DIR,
    AUTH_METRICS_PATH,
    BENCHMARK_PATH,
    GATE_SWEEP_PATH,
    DIAGNOSTICS_PATH,
    PREDICTIONS_PATH,
    DEMO_SAMPLES_PATH,
    RUNTIME_EVAL_JSON_PATH,
    MODEL_RELEASE_PATH,
    settings
)

# Imports from existing ML engine
from src.model_a import ModelAWrapper
from src.model_b import Block1DCNN

logger = logging.getLogger("cerebrum.deps")

class AppState:
    """Singleton container for loaded production models and datasets."""
    def __init__(self):
        self.model_A: Optional[ModelAWrapper] = None
        self.model_B: Optional[Any] = None
        self.cnn_model: Optional[Block1DCNN] = None
        self.device: str = "cpu"
        self.diagnostics_df: Optional[pd.DataFrame] = None
        self.auth_metrics: Optional[Dict[str, Any]] = None
        self.benchmark_df: Optional[pd.DataFrame] = None
        self.gate_sweep_df: Optional[pd.DataFrame] = None
        self.demo_samples: Optional[Dict[str, Any]] = None
        self.model_release: Optional[Dict[str, Any]] = None
        self.startup_time_s: float = 0.0
        self.is_initialized: bool = False

state = AppState()

def init_app_state() -> None:
    """Load production models and static data artifacts once at startup."""
    if state.is_initialized:
        return

    t0 = time.time()
    logger.info("Initializing SanDisk Cerebrum ML engine and data artifacts...")

    # 1. Load Model A (LightGBM Booster wrapper)
    logger.info("Loading Model A from %s", MODELS_DIR)
    state.model_A = ModelAWrapper(model_dir=str(MODELS_DIR))

    # 2. Load Model B (LightGBM multi-resolution model)
    model_b_path = MODELS_DIR / "model_B.pkl"
    logger.info("Loading Model B from %s", model_b_path)
    with open(model_b_path, "rb") as f:
        state.model_B = pickle.load(f)

    # 3. Load 1D CNN checkpoint for block occlusion / embeddings
    cnn_path = MODELS_DIR / "cnn_1d_block_model.pt"
    if cnn_path.exists():
        state.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        cnn = Block1DCNN().to(state.device)
        cnn.load_state_dict(torch.load(cnn_path, map_location=state.device))
        cnn.eval()
        state.cnn_model = cnn
        logger.info("Loaded PyTorch 1D CNN block model on %s", state.device)

    # 4. Load Authoritative Clean Metrics
    if AUTH_METRICS_PATH.exists():
        with open(AUTH_METRICS_PATH, "r") as f:
            state.auth_metrics = json.load(f)
        logger.info("Loaded authoritative clean metrics")

    # 4b. Load Model Release Metadata (Final Validated Candidate B2)
    if MODEL_RELEASE_PATH.exists():
        with open(MODEL_RELEASE_PATH, "r") as f:
            state.model_release = json.load(f)
        logger.info("Loaded model release metadata (Version: %s)", state.model_release.get("model_version", "unknown"))

    # 5. Load Benchmark Data Table
    if BENCHMARK_PATH.exists():
        state.benchmark_df = pd.read_csv(BENCHMARK_PATH)
        logger.info("Loaded benchmark table (%d rows)", len(state.benchmark_df))

    # 6. Load Cascade Gate Sweep Table
    if GATE_SWEEP_PATH.exists():
        state.gate_sweep_df = pd.read_csv(GATE_SWEEP_PATH)
        logger.info("Loaded cascade gate sweep table (%d rows)", len(state.gate_sweep_df))

    # 7. Load Canonical Holdout Diagnostics (27,972 dies)
    diag_file = DIAGNOSTICS_PATH if DIAGNOSTICS_PATH.exists() else PREDICTIONS_PATH
    if diag_file.exists():
        logger.info("Loading holdout diagnostics dataset from %s", diag_file)
        df = pd.read_csv(diag_file)
        
        # Pre-compute cascade decisions for instant slicing
        in_gate_b = (df["prob_Model_A"] >= settings.gate_b_lower) & (df["prob_Model_A"] <= settings.gate_b_upper)
        df["in_gate_B"] = in_gate_b
        df["prob_Cascade_B"] = np.where(in_gate_b, df["prob_Model_B"], df["prob_Model_A"])
        df["decision_Cascade_B"] = np.where(in_gate_b, (df["prob_Model_B"] >= settings.model_b_threshold).astype(int), df["decision_Model_A"])
        df["recovered_FN_Gate_B"] = ((df["actual_label"] == 1) & (df["decision_Model_A"] == 0) & (df["decision_Cascade_B"] == 1)).astype(int)
        
        in_gate_c = (df["prob_Model_A"] >= settings.gate_c_lower) & (df["prob_Model_A"] <= settings.gate_c_upper)
        df["in_gate_C"] = in_gate_c
        df["prob_Cascade_C"] = np.where(in_gate_c, df["prob_Model_B"], df["prob_Model_A"])
        df["decision_Cascade_C"] = np.where(in_gate_c, (df["prob_Model_B"] >= settings.model_b_threshold).astype(int), df["decision_Model_A"])
        
        state.diagnostics_df = df
        logger.info("Loaded and annotated diagnostics dataset (%d dies)", len(df))

    # 8. Load Demo Samples
    if DEMO_SAMPLES_PATH.exists():
        with open(DEMO_SAMPLES_PATH, "r") as f:
            state.demo_samples = json.load(f)
        logger.info("Loaded verified demo samples")

    state.startup_time_s = time.time() - t0
    state.is_initialized = True
    logger.info("Cerebrum backend initialized in %.3f seconds", state.startup_time_s)

def get_model_a() -> ModelAWrapper:
    if state.model_A is None:
        raise RuntimeError("Model A is not initialized")
    return state.model_A

def get_model_b() -> Any:
    if state.model_B is None:
        raise RuntimeError("Model B is not initialized")
    return state.model_B

def get_cnn_model() -> Optional[Block1DCNN]:
    return state.cnn_model

def get_diagnostics_df() -> pd.DataFrame:
    if state.diagnostics_df is None:
        raise RuntimeError("Diagnostics dataset is not initialized")
    return state.diagnostics_df

def get_auth_metrics() -> Dict[str, Any]:
    if state.auth_metrics is None:
        raise RuntimeError("Authoritative metrics are not initialized")
    return state.auth_metrics

def get_benchmark_df() -> Optional[pd.DataFrame]:
    return state.benchmark_df

def get_gate_sweep_df() -> Optional[pd.DataFrame]:
    return state.gate_sweep_df

def get_demo_samples() -> Optional[Dict[str, Any]]:
    return state.demo_samples

def get_model_release() -> Optional[Dict[str, Any]]:
    return state.model_release
