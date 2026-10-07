from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
import numpy as np
import pandas as pd

from backend.deps import get_wafers_df
from backend.api.schemas.wafers import (
    WaferSummaryItem,
    WaferListResponse,
    DiePoint,
    WaferDetailResponse,
    HotspotSummaryItem,
    WaferHotspotsResponse
)

router = APIRouter(prefix="/wafers", tags=["Wafer Intelligence"])

def _summarize_wafer_group(wid: str, wdf: pd.DataFrame) -> WaferSummaryItem:
    tot = len(wdf)
    act_fails = int(wdf["actual_label"].sum())
    fail_pct = float(round(act_fails / max(1, tot) * 100, 2))
    ma_det = int(wdf["decision_Model_A"].sum())
    cas_det = int(wdf["decision_Cascade_B"].sum())
    esc = int(wdf["in_gate_B"].sum())
    rec_fn = int(wdf["recovered_FN_Gate_B"].sum())
    mean_prob = float(round(wdf["prob_Cascade_B"].mean(), 4)) if "prob_Cascade_B" in wdf.columns else None
    # High risk die: prob_Cascade_B >= 0.50
    high_risk = int((wdf["prob_Cascade_B"] >= 0.50).sum()) if "prob_Cascade_B" in wdf.columns else None
    
    return WaferSummaryItem(
        wafer_id=str(wid),
        total_dies=tot,
        actual_failures=act_fails,
        fail_rate_pct=fail_pct,
        model_a_detections=ma_det,
        cascade_detections=cas_det,
        escalated_dies=esc,
        recovered_fns=rec_fn,
        mean_probability=mean_prob,
        high_risk_dies=high_risk
    )


@router.get("", response_model=WaferListResponse, summary="List All Canonical Dataset Wafers")
def list_wafers(wafers_df: pd.DataFrame = Depends(get_wafers_df)) -> WaferListResponse:
    """Returns inventory and aggregate defect/cascade telemetry across all available canonical dataset wafers."""
    wafer_ids = sorted(wafers_df["wafer_id"].unique())
    items: List[WaferSummaryItem] = []
    
    for wid in wafer_ids:
        wdf = wafers_df[wafers_df["wafer_id"] == wid]
        items.append(_summarize_wafer_group(wid, wdf))
        
    return WaferListResponse(
        total_wafers=len(items),
        total_holdout_dies=len(wafers_df),
        wafers=items
    )

@router.get("/{wafer_id}", response_model=WaferSummaryItem, summary="Get Wafer Telemetry Summary")
def get_wafer_summary(
    wafer_id: str,
    wafers_df: pd.DataFrame = Depends(get_wafers_df)
) -> WaferSummaryItem:
    """Returns defect rate, escalation count, and recovery telemetry for a specific wafer."""
    wdf = wafers_df[wafers_df["wafer_id"] == wafer_id]
    if len(wdf) == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": "WAFER_NOT_FOUND", "message": f"Wafer ID '{wafer_id}' not found in canonical dataset"}
        )
    return _summarize_wafer_group(wafer_id, wdf)

@router.get("/{wafer_id}/dies", response_model=WaferDetailResponse, summary="Get Die Coordinates and Predictions for Wafer Map")
def get_wafer_dies(
    wafer_id: str,
    filter_mode: Optional[str] = Query(
        "all",
        description="Filter population: all, predicted_fails, predicted_passes, escalated, recovered_fns, false_alarms"
    ),
    limit: Optional[int] = Query(3000, ge=1, le=5000),
    wafers_df: pd.DataFrame = Depends(get_wafers_df)
) -> WaferDetailResponse:
    """
    Returns spatial coordinate grid and model predictions for interactive wafer visualization.
    Filters allow visualizing defect clusters, Model A missed failures recovered by Cascade, and false alarms.
    """
    wdf = wafers_df[wafers_df["wafer_id"] == wafer_id]
    if len(wdf) == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": "WAFER_NOT_FOUND", "message": f"Wafer ID '{wafer_id}' not found"}
        )

    summary = _summarize_wafer_group(wafer_id, wdf)

    # Apply filter mode
    f_mode = filter_mode.lower() if filter_mode else "all"
    if f_mode in ["predicted_fails", "fails", "fail", "defect"]:
        filtered_df = wdf[wdf["decision_Cascade_B"] == 1]
    elif f_mode in ["predicted_passes", "passes", "pass", "yield"]:
        filtered_df = wdf[wdf["decision_Cascade_B"] == 0]
    elif f_mode in ["escalated"]:
        filtered_df = wdf[wdf["in_gate_B"]]
    elif f_mode in ["bypassed"]:
        filtered_df = wdf[~wdf["in_gate_B"]]
    elif f_mode in ["high_risk"]:
        filtered_df = wdf[wdf["prob_Cascade_B"] >= 0.50]
    elif f_mode in ["recovered_fns", "recovered", "rescued_fn"]:
        filtered_df = wdf[wdf["recovered_FN_Gate_B"] == 1]
    elif f_mode in ["missed_fns", "missed_fn", "fn"]:
        filtered_df = wdf[(wdf["actual_label"] == 1) & (wdf["decision_Cascade_B"] == 0)]
    elif f_mode in ["false_alarms", "fp", "fps"]:
        filtered_df = wdf[(wdf["actual_label"] == 0) & (wdf["decision_Cascade_B"] == 1)]
    else:
        filtered_df = wdf

    if limit and len(filtered_df) > limit:
        filtered_df = filtered_df.iloc[:limit]

    die_points: List[DiePoint] = []
    for _, row in filtered_df.iterrows():
        die_points.append(
            DiePoint(
                wafer_id=str(row["wafer_id"]),
                die_row=int(row["die_row"]),
                die_col=int(row["die_col"]),
                actual_label=int(row["actual_label"]),
                prob_Model_A=float(round(row["prob_Model_A"], 4)),
                prob_Model_B=float(round(row["prob_Model_B"], 4)),
                prob_Cascade_B=float(round(row["prob_Cascade_B"], 4)),
                decision_Model_A=int(row["decision_Model_A"]),
                decision_Cascade_B=int(row["decision_Cascade_B"]),
                in_gate_B=bool(row["in_gate_B"]),
                recovered_FN_Gate_B=int(row["recovered_FN_Gate_B"]),
                delta_prob=float(round(row["delta_prob"], 4)),
                radial_dist=float(round(row["radial_dist"], 4)) if "radial_dist" in row and pd.notna(row["radial_dist"]) else None,
                edge_dist=float(round(row["edge_dist"], 4)) if "edge_dist" in row and pd.notna(row["edge_dist"]) else None,
                b_max=float(round(row["b_max"], 2)) if "b_max" in row and pd.notna(row["b_max"]) else None,
                cnt_104=float(row["cnt_104"]) if "cnt_104" in row and pd.notna(row["cnt_104"]) else None,
                calibrated_risk=float(round(row["calibrated_risk"], 4)) if "calibrated_risk" in row and pd.notna(row["calibrated_risk"]) else float(round(row["prob_Cascade_B"], 4))
            )
        )

    return WaferDetailResponse(
        summary=summary,
        dies=die_points
    )

@router.get("/{wafer_id}/dies/{die_row}/{die_col}", response_model=DiePoint, summary="Get Exact Die Telemetry Record")
def get_single_wafer_die(
    wafer_id: str,
    die_row: int,
    die_col: int,
    wafers_df: pd.DataFrame = Depends(get_wafers_df)
) -> DiePoint:
    """Returns telemetry and model inference records for a specific physical die on a wafer."""
    match = wafers_df[
        (wafers_df["wafer_id"] == wafer_id) &
        (wafers_df["die_row"] == die_row) &
        (wafers_df["die_col"] == die_col)
    ]
    if len(match) == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": "DIE_NOT_FOUND", "message": f"Die [{die_row}, {die_col}] not found on wafer '{wafer_id}'"}
        )
    row = match.iloc[0]
    return DiePoint(
        wafer_id=str(row["wafer_id"]),
        die_row=int(row["die_row"]),
        die_col=int(row["die_col"]),
        actual_label=int(row["actual_label"]),
        prob_Model_A=float(round(row["prob_Model_A"], 4)),
        prob_Model_B=float(round(row["prob_Model_B"], 4)),
        prob_Cascade_B=float(round(row["prob_Cascade_B"], 4)),
        decision_Model_A=int(row["decision_Model_A"]),
        decision_Cascade_B=int(row["decision_Cascade_B"]),
        in_gate_B=bool(row["in_gate_B"]),
        recovered_FN_Gate_B=int(row["recovered_FN_Gate_B"]),
        delta_prob=float(round(row["delta_prob"], 4)),
        radial_dist=float(round(row["radial_dist"], 4)) if "radial_dist" in row and pd.notna(row["radial_dist"]) else None,
        edge_dist=float(round(row["edge_dist"], 4)) if "edge_dist" in row and pd.notna(row["edge_dist"]) else None,
        b_max=float(round(row["b_max"], 2)) if "b_max" in row and pd.notna(row["b_max"]) else None,
        cnt_104=float(row["cnt_104"]) if "cnt_104" in row and pd.notna(row["cnt_104"]) else None,
        calibrated_risk=float(round(row["calibrated_risk"], 4)) if "calibrated_risk" in row and pd.notna(row["calibrated_risk"]) else float(round(row["prob_Cascade_B"], 4))
    )


@router.get("/{wafer_id}/hotspots", response_model=WaferHotspotsResponse, summary="Get Wafer Spatial Hotspot Summaries")
def get_wafer_hotspots(
    wafer_id: str,
    wafers_df: pd.DataFrame = Depends(get_wafers_df)
) -> WaferHotspotsResponse:
    """
    Identifies high-risk spatial clusters and hotspot regions on a wafer for targeted engineering inspection.
    Enables instant Wafer -> Hotspot -> Die drilldown navigation.
    """
    wdf = wafers_df[wafers_df["wafer_id"] == wafer_id]
    if len(wdf) == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": "WAFER_NOT_FOUND", "message": f"Wafer ID '{wafer_id}' not found"}
        )

    tot = len(wdf)
    high_risk_df = wdf[wdf["prob_Cascade_B"] >= 0.50]
    total_hr = len(high_risk_df)

    hotspots: List[HotspotSummaryItem] = []

    if total_hr > 0:
        # Spatial partitioning: Edge/Perimeter ring vs Center core vs Quadrant clusters
        edge_hr = high_risk_df[high_risk_df["edge_dist"] < 0.20] if "edge_dist" in high_risk_df.columns else pd.DataFrame()
        center_hr = high_risk_df[high_risk_df["radial_dist"] < 0.35] if "radial_dist" in high_risk_df.columns else pd.DataFrame()
        quad_hr = high_risk_df[~high_risk_df.index.isin(edge_hr.index) & ~high_risk_df.index.isin(center_hr.index)]

        # 1. Edge Bevel Hotspot
        if len(edge_hr) >= 3:
            r_c = int(edge_hr["die_row"].median())
            c_c = int(edge_hr["die_col"].median())
            obs_def = int(edge_hr["actual_label"].sum())
            hotspots.append(HotspotSummaryItem(
                hotspot_id=f"{wafer_id}_HS_EDGE",
                region_name="Edge Bevel Ring Cluster",
                center_row=r_c,
                center_col=c_c,
                radius_dies=5,
                die_count=len(wdf[(np.abs(wdf['die_row'] - r_c) <= 5) & (np.abs(wdf['die_col'] - c_c) <= 5)]),
                high_risk_die_count=len(edge_hr),
                observed_defect_count=obs_def,
                mean_risk_prob=round(float(edge_hr["prob_Cascade_B"].mean()), 4),
                defect_density_pct=round(float(len(edge_hr) / max(1, len(wdf)) * 100.0), 2),
                primary_morphology="Edge Ring Arc / CMP Exclusion",
                suspicious_signature="High edge gradient with abnormal parametric resistance"
            ))

        # 2. Center Core Hotspot
        if len(center_hr) >= 3:
            r_c = int(center_hr["die_row"].median())
            c_c = int(center_hr["die_col"].median())
            obs_def = int(center_hr["actual_label"].sum())
            hotspots.append(HotspotSummaryItem(
                hotspot_id=f"{wafer_id}_HS_CENTER",
                region_name="Center Reticle Core",
                center_row=r_c,
                center_col=c_c,
                radius_dies=4,
                die_count=len(wdf[(np.abs(wdf['die_row'] - r_c) <= 4) & (np.abs(wdf['die_col'] - c_c) <= 4)]),
                high_risk_die_count=len(center_hr),
                observed_defect_count=obs_def,
                mean_risk_prob=round(float(center_hr["prob_Cascade_B"].mean()), 4),
                defect_density_pct=round(float(len(center_hr) / max(1, len(wdf)) * 100.0), 2),
                primary_morphology="Radial Concentric Cluster",
                suspicious_signature="Central thermal excursion and analog voltage collapse"
            ))

        # 3. Main General Hotspot (if edge or center didn't trigger, or dominant cluster)
        if len(hotspots) == 0:
            r_c = int(high_risk_df["die_row"].median())
            c_c = int(high_risk_df["die_col"].median())
            obs_def = int(high_risk_df["actual_label"].sum())
            hotspots.append(HotspotSummaryItem(
                hotspot_id=f"{wafer_id}_HS_MAIN",
                region_name="Primary Defect Epicenter",
                center_row=r_c,
                center_col=c_c,
                radius_dies=5,
                die_count=len(wdf[(np.abs(wdf['die_row'] - r_c) <= 5) & (np.abs(wdf['die_col'] - c_c) <= 5)]),
                high_risk_die_count=total_hr,
                observed_defect_count=obs_def,
                mean_risk_prob=round(float(high_risk_df["prob_Cascade_B"].mean()), 4),
                defect_density_pct=round(float(total_hr / max(1, tot) * 100.0), 2),
                primary_morphology="Localized Spatial Cluster",
                suspicious_signature="Correlated electrical and sub-die block anomalies"
            ))
    else:
        # Low risk wafer
        hotspots.append(HotspotSummaryItem(
            hotspot_id=f"{wafer_id}_HS_CLEAN",
            region_name="Uniform Baseline (Low Risk)",
            center_row=int(wdf["die_row"].median()),
            center_col=int(wdf["die_col"].median()),
            radius_dies=3,
            die_count=tot,
            high_risk_die_count=0,
            observed_defect_count=int(wdf["actual_label"].sum()),
            mean_risk_prob=round(float(wdf["prob_Cascade_B"].mean()), 4),
            defect_density_pct=0.0,
            primary_morphology="Dispersed / Nominal",
            suspicious_signature="No high-risk spatial clusters detected"
        ))

    return WaferHotspotsResponse(
        wafer_id=wafer_id,
        total_dies=tot,
        total_high_risk_dies=total_hr,
        hotspots_count=len(hotspots),
        hotspots=hotspots
    )
