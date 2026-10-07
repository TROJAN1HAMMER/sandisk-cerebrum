import io
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query

from backend.config import settings
from backend.deps import (
    get_model_a,
    get_model_b,
    get_cnn_model,
    get_tcn_model,
    get_calibrator,
    get_diagnostics_df,
    get_demo_samples,
    state
)
from backend.api.schemas.inference import (
    SingleDieRequest,
    DieInferenceResponse,
    BatchInferenceResponse,
    BatchInferenceSummary,
    BatchInferenceItem,
    DemoSampleItem,
    BlockAttributionResponse,
    TcnForensicsResponse
)

# Reused ML engine interfaces
from src.live_inference import (
    predict_single_die_live,
    predict_batch_df_live,
    validate_csv_schema
)
from src.explainability import (
    compute_block_occlusion_attribution,
    compute_tcn_saliency_attribution
)
from src.block_parser import parse_block_matrix

logger = logging.getLogger("cerebrum.routes.inference")
router = APIRouter(prefix="/inference", tags=["Inference"])

@router.post("/die", response_model=DieInferenceResponse, summary="Execute Single-Die Live Cascade Inference")
def predict_die(
    req: SingleDieRequest,
    model_A=Depends(get_model_a),
    model_B=Depends(get_model_b)
) -> DieInferenceResponse:
    """
    Executes live multi-resolution cascade inference for a single die:
    Model A (coarse screening) -> Uncertainty Gate [0.295, 0.550] -> Model B (sub-die resolution).
    """
    die_dict: Dict[str, Any] = {
        "wafer_id": req.wafer_id or "W_TEST_0001",
        "die_row": req.die_row,
        "die_col": req.die_col,
    }

    # If coordinates match a die in canonical dataset and raw features are omitted,
    # pull known features to support seamless preset and wafer-click testing
    if req.prob_Model_A is None and not req.features:
        source_df = getattr(state, "wafers_df", None)
        if source_df is None:
            source_df = state.diagnostics_df
        if source_df is not None:
            match = source_df[
                (source_df["wafer_id"] == req.wafer_id) &
                (source_df["die_row"] == req.die_row) &
                (source_df["die_col"] == req.die_col)
            ]
            if len(match) > 0:
                row_dict = match.iloc[0].to_dict()
                for k, v in row_dict.items():
                    die_dict[k] = v

    if req.features:
        die_dict.update(req.features)
    if req.spatial:
        die_dict.update(req.spatial)
    if req.block_readings:
        die_dict["block_readings"] = req.block_readings
    if req.prob_Model_A is not None:
        die_dict["prob_Model_A"] = req.prob_Model_A
    if req.prob_Model_B is not None:
        die_dict["prob_Model_B"] = req.prob_Model_B
    if req.actual_label is not None:
        die_dict["actual_label"] = req.actual_label
        die_dict["label"] = req.actual_label

    gate_lower = req.gate_lower if req.gate_lower is not None else settings.production_gate_lower
    gate_upper = req.gate_upper if req.gate_upper is not None else settings.production_gate_upper

    t0 = time.time()
    try:
        raw_res = predict_single_die_live(
            die_dict,
            model_A,
            model_B,
            gate_lower=gate_lower,
            gate_upper=gate_upper,
            model_A_thresh=settings.model_a_threshold,
            model_B_thresh=settings.model_b_threshold
        )
    except Exception as e:
        logger.exception("Single die inference failed")
        raise HTTPException(status_code=500, detail={"code": "INFERENCE_ERROR", "message": str(e)})

    latency_ms = (time.time() - t0) * 1000.0

    # Calibrated risk calculation (EXP-09 isotonic reporting layer)
    final_prob = float(raw_res["final_probability"])
    cal_risk = final_prob
    if state.calibrator is not None:
        try:
            cal_risk = float(np.round(state.calibrator.predict([final_prob])[0], 4))
        except Exception:
            cal_risk = final_prob

    # Structured multi-sensor evidence chain
    f171_val = float(die_dict.get("feature_171", 2200.0))
    electrical_anom = abs(f171_val - 2200.0) > 280.0
    sp_dens = float(raw_res.get("spatial_signals", {}).get("density_3x3", 0.0))
    spatial_anom = sp_dens > 0.05
    c_area = float(raw_res.get("block_signals", {}).get("cluster_area", 0.0))
    cnt_105 = float(raw_res.get("block_signals", {}).get("cnt_105", 0.0))
    signal_anom = (c_area > 10.0) or (cnt_105 > 0.0)

    evidence_chain = {
        "electrical": {
            "status": "ANOMALOUS" if electrical_anom else "NOMINAL",
            "primary_driver": "feature_171",
            "observed_value": round(f171_val, 2),
            "baseline_norm": 2200.0,
            "deviation": round(f171_val - 2200.0, 2)
        },
        "spatial": {
            "status": "HOTSPOT" if spatial_anom else "ISOLATED",
            "density_3x3": round(sp_dens, 4),
            "radial_dist": round(float(raw_res.get("spatial_signals", {}).get("radial_dist", 0.0)), 4),
            "cluster_type": "Cluster Member" if spatial_anom else "Isolated Die"
        },
        "signal": {
            "status": "ACTIVE_ANOMALY" if signal_anom else "BASELINE",
            "cluster_area": round(c_area, 2),
            "peak_voltage": round(float(raw_res.get("block_signals", {}).get("b_max", 100.0)), 2),
            "waveform_integrity": "Disturbed" if signal_anom else "Uniform"
        },
        "decision_trace": {
            "stage_1_screening": round(float(raw_res["prob_Model_A"]), 4),
            "routing_path": str(raw_res["compute_path"]),
            "stage_2_inspection": round(float(raw_res["prob_Model_B"]), 4) if raw_res.get("prob_Model_B") is not None else None,
            "final_classification": str(raw_res["decision_text"])
        }
    }

    return DieInferenceResponse(
        wafer_id=str(raw_res["wafer_id"]),
        die_row=int(raw_res["die_row"]),
        die_col=int(raw_res["die_col"]),
        prob_Model_A=float(raw_res["prob_Model_A"]),
        threshold_Model_A=float(raw_res["threshold_Model_A"]),
        decision_Model_A=int(raw_res["decision_Model_A"]),
        in_gate=bool(raw_res["in_gate"]),
        gate_status=str(raw_res["gate_status"]),
        gate_message=str(raw_res["gate_message"]),
        model_B_invoked=bool(raw_res["model_B_invoked"]),
        prob_Model_B=float(raw_res["prob_Model_B"]) if raw_res["prob_Model_B"] is not None else None,
        threshold_Model_B=float(raw_res["threshold_Model_B"]),
        decision_Model_B=int(raw_res["decision_Model_B"]) if raw_res["decision_Model_B"] is not None else None,
        final_probability=final_prob,
        final_decision=int(raw_res["final_decision"]),
        decision_text=str(raw_res["decision_text"]),
        compute_path=str(raw_res["compute_path"]),
        actual_label=int(raw_res["actual_label"]) if raw_res["actual_label"] is not None else None,
        is_correct=bool(raw_res["is_correct"]) if raw_res["is_correct"] is not None else None,
        spatial_signals={k: float(v) for k, v in raw_res.get("spatial_signals", {}).items()},
        block_signals={k: float(v) for k, v in raw_res.get("block_signals", {}).items()},
        calibrated_risk=cal_risk,
        calibration_notes="Calibrated probability is provided for risk exposure interpretation (ECE=0.0035). Production routing remains governed strictly by the frozen B2 operational policy.",
        evidence_chain=evidence_chain,
        latency_ms=float(round(latency_ms, 2))
    )

@router.post("/batch", response_model=BatchInferenceResponse, summary="Execute Batch Live Cascade Inference via CSV Upload")
async def predict_batch(
    file: UploadFile = File(...),
    gate_lower: Optional[float] = Query(None, description="Optional custom lower gate"),
    gate_upper: Optional[float] = Query(None, description="Optional custom upper gate"),
    model_A=Depends(get_model_a),
    model_B=Depends(get_model_b)
) -> BatchInferenceResponse:
    """
    Accepts CSV containing die rows (wafer_id, die_row, die_col, feature_1..feature_500).
    Validates CSV schema and runs live cascade batch inference.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail={"code": "INVALID_FILE_TYPE", "message": "Uploaded file must be a CSV (.csv)"}
        )

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={"code": "CSV_PARSE_ERROR", "message": f"Could not parse CSV file: {str(e)}"}
        )

    # Validate schema using existing engine validator
    val_schema = validate_csv_schema(df)
    if not val_schema["valid"]:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "INVALID_CSV_SCHEMA",
                "message": "CSV schema validation failed",
                "errors": val_schema["errors"],
                "num_rows": val_schema["num_rows"]
            }
        )

    lower = gate_lower if gate_lower is not None else settings.production_gate_lower
    upper = gate_upper if gate_upper is not None else settings.production_gate_upper

    try:
        batch_res = predict_batch_df_live(
            df,
            model_A,
            model_B,
            gate_lower=lower,
            gate_upper=upper,
            model_A_thresh=settings.model_a_threshold,
            model_B_thresh=settings.model_b_threshold
        )
    except Exception as e:
        logger.exception("Batch prediction execution failed")
        raise HTTPException(
            status_code=500,
            detail={"code": "BATCH_EXECUTION_ERROR", "message": str(e)}
        )

    summary_raw = batch_res["summary"]
    results_df = batch_res["results_df"]

    summary = BatchInferenceSummary(
        total_dies=int(summary_raw["total_dies"]),
        predicted_fails=int(summary_raw["predicted_fails"]),
        predicted_passes=int(summary_raw["predicted_passes"]),
        predicted_fail_rate_pct=float(summary_raw["predicted_fail_rate_pct"]),
        escalated_dies=int(summary_raw["escalated_dies"]),
        escalation_rate_pct=float(summary_raw["escalation_rate_pct"]),
        compute_saved_pct=float(summary_raw["compute_saved_pct"]),
        total_inference_time_s=float(summary_raw["total_inference_time_s"]),
        latency_per_die_ms=float(summary_raw["latency_per_die_ms"])
    )

    items: List[BatchInferenceItem] = []
    for _, row in results_df.iterrows():
        prob_b = row.get("prob_Model_B")
        dec_b = row.get("decision_Model_B")
        act_lbl = row.get("actual_label")
        
        items.append(
            BatchInferenceItem(
                wafer_id=str(row["wafer_id"]),
                die_row=int(row["die_row"]),
                die_col=int(row["die_col"]),
                prob_Model_A=float(row["prob_Model_A"]),
                decision_Model_A=int(row["decision_Model_A"]),
                escalated=str(row["escalated"]),
                prob_Model_B=float(prob_b) if pd.notna(prob_b) else None,
                decision_Model_B=int(dec_b) if pd.notna(dec_b) else None,
                final_probability=float(row["final_probability"]),
                final_decision=int(row["final_decision"]),
                decision_label=str(row["decision_label"]),
                compute_path=str(row["compute_path"]),
                actual_label=int(act_lbl) if pd.notna(act_lbl) else None
            )
        )

    return BatchInferenceResponse(summary=summary, results=items)

@router.get("/demo-samples", response_model=List[DemoSampleItem], summary="Get Verified Demo Walkthrough Samples")
def get_demo_samples_route(demos=Depends(get_demo_samples)) -> List[DemoSampleItem]:
    """Returns verified representative die samples for hackathon demonstration."""
    if not demos:
        return []
        
    items = []
    for sample_key, sample_data in demos.items():
        items.append(
            DemoSampleItem(
                sample_id=sample_key,
                wafer_id=str(sample_data.get("wafer_id", "")),
                die_row=int(sample_data.get("die_row", 0)),
                die_col=int(sample_data.get("die_col", 0)),
                actual_label=int(sample_data.get("actual_label", 0)),
                model_A_prob=float(sample_data.get("model_A_prob", 0.0)),
                cascade_final_prob=float(sample_data.get("cascade_final_prob", 0.0)),
                model_A_decision=int(sample_data.get("model_A_decision", 0)),
                cascade_final_decision=int(sample_data.get("cascade_final_decision", 0)),
                resolution_used=str(sample_data.get("resolution_used", "")),
                escalation_reason=str(sample_data.get("escalation_reason", "")),
                description=str(sample_data.get("description", ""))
            )
        )
    return items

@router.get("/block-attribution", response_model=BlockAttributionResponse, summary="1D Sub-Die Occlusion Sensitivity Attribution")
def get_block_attribution(
    wafer_id: str = Query("W_N_0057"),
    die_row: int = Query(32),
    die_col: int = Query(23),
    cnn_model=Depends(get_cnn_model)
) -> BlockAttributionResponse:
    """
    Computes 1D sliding-window occlusion sensitivity across 2,000 sub-die sequence readings.
    Returns the raw block signal readings and normalized attribution weights for interactive waveform plotting.
    """
    if cnn_model is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "CNN_UNAVAILABLE", "message": "PyTorch 1D CNN block model is not available"}
        )

    # 1. Look for authentic raw block readings in wafer cache, train.csv, or validation.csv
    arr = None
    from backend.config import INPUT_DIR
    wafer_npz = Path(__file__).resolve().parents[2] / "data" / "wafers" / f"{wafer_id}.npz"
    if wafer_npz.exists():
        try:
            w_data = np.load(wafer_npz)
            coords = w_data["coords"]
            match_indices = np.where((coords[:, 0] == die_row) & (coords[:, 1] == die_col))[0]
            if len(match_indices) > 0:
                arr = w_data["readings"][match_indices[0]].copy()
        except Exception as e:
            logger.warning("Could not read from wafer npz %s: %s", wafer_npz, e)

    if arr is None:
        for csv_candidate in [INPUT_DIR / "train.csv", INPUT_DIR / "validation.csv"]:
            if csv_candidate.exists():
                try:
                    for chunk in pd.read_csv(
                        csv_candidate,
                        usecols=["wafer_id", "die_row", "die_col", "block_readings"],
                        chunksize=10000
                    ):
                        m = chunk[(chunk["wafer_id"] == wafer_id) & (chunk["die_row"] == die_row) & (chunk["die_col"] == die_col)]
                        if len(m) > 0:
                            raw_str = m.iloc[0]["block_readings"]
                            arr = np.array([float(x) for x in raw_str.split()], dtype=np.float32)
                            break
                    if arr is not None:
                        break
                except Exception as e:
                    logger.warning("Could not read block_readings from %s: %s", csv_candidate, e)

    if arr is None:
        # Fallback only when die coordinate does not exist in dataset
        rng = np.random.default_rng(seed=int(die_row * 37 + die_col + 42))
        arr = rng.normal(loc=100.0, scale=10.0, size=2000).astype(np.float32)
    elif len(arr) < 2000:
        arr = np.pad(arr, (0, 2000 - len(arr)), constant_values=100.0)
    elif len(arr) > 2000:
        arr = arr[:2000]

    base_prob, attributions = compute_block_occlusion_attribution(
        cnn_model, arr, window_size=50, device=state.device
    )

    peak_idx = int(np.argmax(attributions))
    peak_val = float(attributions[peak_idx])
    anom_count = int(np.sum(arr > 104.0))

    return BlockAttributionResponse(
        wafer_id=wafer_id,
        die_row=die_row,
        die_col=die_col,
        base_prob=float(base_prob),
        block_count=2000,
        readings=[float(x) for x in arr],
        attributions=[float(x) for x in attributions],
        peak_attribution_idx=peak_idx,
        peak_attribution_value=peak_val,
        anomalous_blocks_count=anom_count
    )


@router.get("/tcn-forensics", response_model=TcnForensicsResponse, summary="Get Multi-Scale TCN Sequence Saliency Forensics")
def get_tcn_forensics(
    wafer_id: str = Query(..., description="Wafer ID"),
    die_row: int = Query(..., ge=0, le=80, description="Die row"),
    die_col: int = Query(..., ge=0, le=80, description="Die column"),
    tcn_model=Depends(get_tcn_model)
) -> TcnForensicsResponse:
    """
    Forensic Sequence Attribution via Multi-Scale Dilated TCN.
    IMPORTANT: This sequence analysis is strictly for offline engineering forensics, root-cause visualization,
    and saliency attribution. It is NOT part of production cascade routing.
    """
    if tcn_model is None:
        raise HTTPException(
            status_code=503,
            detail={"code": "TCN_UNAVAILABLE", "message": "Multi-Scale Dilated TCN forensic model is not loaded"}
        )

    # 1. Fetch authentic block readings
    arr = None
    from backend.config import INPUT_DIR
    wafer_npz = Path(__file__).resolve().parents[2] / "data" / "wafers" / f"{wafer_id}.npz"
    if wafer_npz.exists():
        try:
            w_data = np.load(wafer_npz)
            coords = w_data["coords"]
            match_indices = np.where((coords[:, 0] == die_row) & (coords[:, 1] == die_col))[0]
            if len(match_indices) > 0:
                arr = w_data["readings"][match_indices[0]].copy()
        except Exception as e:
            logger.warning("Could not read from wafer npz %s: %s", wafer_npz, e)

    if arr is None:
        for csv_candidate in [INPUT_DIR / "train.csv", INPUT_DIR / "validation.csv"]:
            if csv_candidate.exists():
                try:
                    for chunk in pd.read_csv(
                        csv_candidate,
                        usecols=["wafer_id", "die_row", "die_col", "block_readings"],
                        chunksize=10000
                    ):
                        m = chunk[(chunk["wafer_id"] == wafer_id) & (chunk["die_row"] == die_row) & (chunk["die_col"] == die_col)]
                        if len(m) > 0:
                            raw_str = m.iloc[0]["block_readings"]
                            arr = np.array([float(x) for x in raw_str.split()], dtype=np.float32)
                            break
                    if arr is not None:
                        break
                except Exception as e:
                    logger.warning("Could not read block_readings from %s: %s", csv_candidate, e)

    if arr is None:
        rng = np.random.default_rng(seed=int(die_row * 37 + die_col + 42))
        arr = rng.normal(loc=100.0, scale=10.0, size=2000).astype(np.float32)
    elif len(arr) < 2000:
        arr = np.pad(arr, (0, 2000 - len(arr)), constant_values=100.0)
    elif len(arr) > 2000:
        arr = arr[:2000]

    prob, saliency = compute_tcn_saliency_attribution(tcn_model, arr, device=state.device)

    peak_idx = int(np.argmax(arr))
    peak_amp = float(arr[peak_idx])

    anom_mask = (arr > 104.0)
    anom_span = int(np.sum(anom_mask))
    
    anom_excess = np.maximum(0.0, arr - 100.0)
    tot_excess = np.sum(anom_excess)
    if tot_excess > 0:
        idx_vec = np.arange(2000, dtype=np.float32)
        centroid = float((anom_excess @ idx_vec) / tot_excess / 2000.0)
    else:
        centroid = 0.5

    # Compute top 3 important block ranges
    important_ranges = []
    sal_thresh = 0.40
    high_sal_indices = np.where(saliency > sal_thresh)[0]
    if len(high_sal_indices) > 0:
        # Group contiguous intervals
        diffs = np.diff(high_sal_indices)
        split_points = np.where(diffs > 15)[0] + 1
        splits = np.split(high_sal_indices, split_points)
        for s in splits[:3]:
            if len(s) > 0:
                important_ranges.append({
                    "start_block": int(s[0]),
                    "end_block": int(s[-1]),
                    "peak_saliency": round(float(np.max(saliency[s])), 4),
                    "mean_voltage": round(float(np.mean(arr[s])), 2)
                })

    salient_count = int(np.sum(saliency > 0.35))
    peak_attr_idx = int(np.argmax(saliency))
    peak_attr_val = float(saliency[peak_attr_idx])
    
    anom_start = int(important_ranges[0]["start_block"]) if important_ranges else max(0, peak_attr_idx - 50)
    anom_end = int(important_ranges[0]["end_block"]) if important_ranges else min(1999, peak_attr_idx + 50)

    notes = (
        f"Multi-Scale TCN dilated receptive field (256 blocks) identified localized gradient sensitivity centered at block #{peak_attr_idx + 1} "
        f"with peak voltage {peak_amp:.1f} mV (nominal: 100.0 mV). {salient_count} sequence blocks exhibited salient attribution (>0.35). "
        f"Sequence failure probability: {float(prob)*100.0:.2f}%."
    )

    return TcnForensicsResponse(
        wafer_id=wafer_id,
        die_row=die_row,
        die_col=die_col,
        sequence_prob_tcn=round(float(prob), 4),
        peak_block_index=peak_idx,
        peak_amplitude=round(peak_amp, 2),
        anomaly_centroid=round(centroid, 4),
        anomaly_span_blocks=anom_span,
        saliency_curve=[round(float(x), 4) for x in saliency],
        important_block_ranges=important_ranges,
        waveform_readings=[round(float(x), 2) for x in arr],
        tcn_attribution=[round(float(x), 4) for x in saliency],
        peak_attribution_idx=peak_attr_idx,
        peak_attribution_value=round(peak_attr_val, 4),
        anomalous_span_start=anom_start,
        anomalous_span_end=anom_end,
        salient_blocks_count=salient_count,
        interpretation_notes=notes,
        role="Multi-Scale Dilated TCN Sequence Feature Extractor (Receptive Field = 256 blocks)",
        forensic_only=True,
        used_in_production_routing=False,
        disclaimer="FORENSIC SEQUENCE ANALYSIS — Not used for production cascade routing."
    )
