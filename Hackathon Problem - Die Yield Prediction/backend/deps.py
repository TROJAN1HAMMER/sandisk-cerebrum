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
    ALL_WAFERS_PATH,
    DEMO_SAMPLES_PATH,
    RUNTIME_EVAL_JSON_PATH,
    MODEL_RELEASE_PATH,
    settings
)

# Imports from existing ML engine
from src.model_a import ModelAWrapper, CandidateModelAWrapper
from src.model_b import Block1DCNN, MultiScaleDilatedTCN, CandidateModelBWrapper

logger = logging.getLogger("cerebrum.deps")

class AppState:
    """Singleton container for loaded production models and datasets."""
    def __init__(self):
        self.model_A: Optional[Any] = None
        self.model_B: Optional[Any] = None
        self.legacy_model_A: Optional[ModelAWrapper] = None
        self.legacy_model_B: Optional[Any] = None
        self.pca_pipeline: Optional[Any] = None
        self.cnn_model: Optional[Block1DCNN] = None
        self.tcn_model: Optional[MultiScaleDilatedTCN] = None
        self.calibrator: Optional[Any] = None
        self.device: str = "cpu"
        self.diagnostics_df: Optional[pd.DataFrame] = None
        self.wafers_df: Optional[pd.DataFrame] = None
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

    # 1. Load B3 Production Model A (517 features: 500 parametric + 10 spatial + 7 PCA)
    logger.info("Loading B3 Production Model A (517 features) from candidates")
    state.model_A = CandidateModelAWrapper()
    state.pca_pipeline = state.model_A.pca_bundle

    # 2. Load B3 Production Model B (528 tabular + 1D CNN late fusion 0.90/0.10)
    logger.info("Loading B3 Production Model B (528 tabular + 1D-CNN)")
    state.model_B = CandidateModelBWrapper()
    state.cnn_model = state.model_B.cnn_model
    state.device = state.model_B.device

    # 2b. Retain legacy B2 models for rollback verification
    try:
        state.legacy_model_A = ModelAWrapper(model_dir=str(MODELS_DIR))
        with open(MODELS_DIR / "model_B.pkl", "rb") as f:
            state.legacy_model_B = pickle.load(f)
    except Exception as e:
        logger.warning("Could not load legacy B2 models: %s", e)

    # 3b. Load Multi-Scale Dilated TCN for Advanced Forensic Sequence Attribution
    tcn_path = MODELS_DIR / "multiscale_tcn_model.pt"
    if tcn_path.exists():
        try:
            tcn = MultiScaleDilatedTCN().to(state.device)
            tcn.load_state_dict(torch.load(tcn_path, map_location=state.device))
            tcn.eval()
            state.tcn_model = tcn
            logger.info("Loaded PyTorch Multi-Scale Dilated TCN forensic model on %s", state.device)
        except Exception as e:
            logger.warning("Could not load TCN forensic model: %s", e)

    # 3c. Load Isotonic Probability Calibrator for Risk Reporting
    cal_path = MODELS_DIR / "isotonic_calibrator.pkl"
    if cal_path.exists():
        try:
            with open(cal_path, "rb") as f:
                state.calibrator = pickle.load(f)
            logger.info("Loaded Isotonic Probability Calibrator for reporting layer")
        except Exception as e:
            logger.warning("Could not load Isotonic calibrator: %s", e)

    # 4. Load Authoritative Clean Metrics
    if AUTH_METRICS_PATH.exists():
        with open(AUTH_METRICS_PATH, "r") as f:
            state.auth_metrics = json.load(f)
        logger.info("Loaded authoritative clean metrics")

    # 4b. Load Model Release Metadata (Active Production B3)
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
        
        # Pre-compute cascade decisions for historical holdout diagnostics (27,972 dies)
        in_gate_b = (df["prob_Model_A"] >= 0.25) & (df["prob_Model_A"] <= 0.55)
        df["in_gate_B"] = in_gate_b
        df["prob_Cascade_B"] = np.where(in_gate_b, df["prob_Model_B"], df["prob_Model_A"])
        # Preserves historical B0 holdout decision threshold (0.5376) for authoritative clean metrics audit
        df["decision_Cascade_B"] = np.where(in_gate_b, (df["prob_Model_B"] >= 0.5376).astype(int), df["decision_Model_A"])
        df["recovered_FN_Gate_B"] = ((df["actual_label"] == 1) & (df["decision_Model_A"] == 0) & (df["decision_Cascade_B"] == 1)).astype(int)
        
        # Pre-compute calibrated risk for reporting (EXP-09)
        if state.calibrator is not None:
            try:
                df["calibrated_risk"] = np.round(state.calibrator.predict(df["prob_Cascade_B"].values), 4)
            except Exception:
                df["calibrated_risk"] = df["prob_Cascade_B"]
        else:
            df["calibrated_risk"] = df["prob_Cascade_B"]

        in_gate_c = (df["prob_Model_A"] >= settings.gate_c_lower) & (df["prob_Model_A"] <= settings.gate_c_upper)
        df["in_gate_C"] = in_gate_c
        df["prob_Cascade_C"] = np.where(in_gate_c, df["prob_Model_B"], df["prob_Model_A"])
        df["decision_Cascade_C"] = np.where(in_gate_c, (df["prob_Model_B"] >= 0.5376).astype(int), df["decision_Model_A"])
        
        state.diagnostics_df = df
        logger.info("Loaded and annotated diagnostics dataset (%d dies)", len(df))

        # 7b. Load Complete 160-Wafer Dataset for Wafer Intelligence (128 train + 32 val)
        if ALL_WAFERS_PATH.exists():
            try:
                cli_df = pd.read_csv(ALL_WAFERS_PATH)
                val_wafers = set(df["wafer_id"].unique())
                extra = cli_df[~cli_df["wafer_id"].isin(val_wafers)].copy()
                if len(extra) > 0:
                    in_gate = (extra["resolution_used"].str.contains("Fine-Resolution", case=False, na=False)) | (
                        (extra["model_A_risk_prob"] >= 0.25) & (extra["model_A_risk_prob"] <= 0.55)
                    )
                    extra["prob_Model_A"] = extra["model_A_risk_prob"]
                    extra["prob_Cascade_B"] = extra["cascade_final_prob"]
                    extra["decision_Model_A"] = extra["model_A_decision"]
                    extra["decision_Cascade_B"] = extra["cascade_final_decision"]
                    extra["in_gate_B"] = in_gate
                    extra["prob_Model_B"] = np.where(in_gate, extra["cascade_final_prob"], extra["model_A_risk_prob"])
                    extra["delta_prob"] = extra["cascade_final_prob"] - extra["model_A_risk_prob"]
                    extra["recovered_FN_Gate_B"] = ((extra["actual_label"] == 1) & (extra["model_A_decision"] == 0) & (extra["cascade_final_decision"] == 1)).astype(int)
                    extra["calibrated_risk"] = extra["cascade_final_prob"]

                    # Spatial coordinates normalization
                    r_max = extra.groupby("wafer_id")["die_row"].transform("max")
                    c_max = extra.groupby("wafer_id")["die_col"].transform("max")
                    r_cen = r_max / 2.0
                    c_cen = c_max / 2.0
                    rad_norm = np.maximum(1.0, np.sqrt(r_cen**2 + c_cen**2))
                    r_diff = extra["die_row"] - r_cen
                    c_diff = extra["die_col"] - c_cen
                    radial_dist = np.minimum(1.0, np.sqrt(r_diff**2 + c_diff**2) / rad_norm)
                    extra["radial_dist"] = radial_dist
                    extra["edge_dist"] = np.maximum(0.0, 1.0 - radial_dist)
                    extra["b_max"] = 0.0
                    extra["cnt_104"] = 0.0

                    common_cols = [c for c in df.columns if c in extra.columns]
                    state.wafers_df = pd.concat([df, extra[common_cols]], ignore_index=True)
                    logger.info("Loaded complete canonical dataset: %d wafers (%d dies)", state.wafers_df["wafer_id"].nunique(), len(state.wafers_df))
                else:
                    state.wafers_df = df
            except Exception as e:
                logger.warning("Could not build unified wafers dataset: %s", e)
                state.wafers_df = df
        else:
            state.wafers_df = df

    # 8. Load Demo Samples
    if DEMO_SAMPLES_PATH.exists():
        with open(DEMO_SAMPLES_PATH, "r") as f:
            state.demo_samples = json.load(f)
        logger.info("Loaded verified demo samples")

    state.startup_time_s = time.time() - t0
    state.is_initialized = True
    logger.info("Cerebrum backend initialized in %.3f seconds", state.startup_time_s)

def get_model_a() -> Any:
    if state.model_A is None:
        raise RuntimeError("Model A is not initialized")
    return state.model_A

def get_model_b() -> Any:
    if state.model_B is None:
        raise RuntimeError("Model B is not initialized")
    return state.model_B

def get_pca_pipeline() -> Optional[Any]:
    return state.pca_pipeline

def get_legacy_model_a() -> Optional[ModelAWrapper]:
    return state.legacy_model_A

def get_legacy_model_b() -> Optional[Any]:
    return state.legacy_model_B

def get_cnn_model() -> Optional[Block1DCNN]:
    return state.cnn_model

def get_diagnostics_df() -> pd.DataFrame:
    if state.diagnostics_df is None:
        raise RuntimeError("Diagnostics dataset is not initialized")
    return state.diagnostics_df

def get_wafers_df() -> pd.DataFrame:
    if state.wafers_df is not None:
        return state.wafers_df
    if state.diagnostics_df is not None:
        return state.diagnostics_df
    raise RuntimeError("Wafers dataset is not initialized")

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

def get_tcn_model() -> Optional[MultiScaleDilatedTCN]:
    return state.tcn_model

def get_calibrator() -> Optional[Any]:
    return state.calibrator
