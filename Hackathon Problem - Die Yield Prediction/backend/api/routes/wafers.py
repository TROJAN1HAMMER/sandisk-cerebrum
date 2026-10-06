from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd

from backend.deps import get_diagnostics_df
from backend.api.schemas.wafers import (
    WaferSummaryItem,
    WaferListResponse,
    DiePoint,
    WaferDetailResponse
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


@router.get("", response_model=WaferListResponse, summary="List All 32 Canonical Validation Wafers")
def list_wafers(diag_df: pd.DataFrame = Depends(get_diagnostics_df)) -> WaferListResponse:
    """Returns inventory and aggregate defect/cascade telemetry across all 32 clean holdout wafers."""
    wafer_ids = sorted(diag_df["wafer_id"].unique())
    items: List[WaferSummaryItem] = []
    
    for wid in wafer_ids:
        wdf = diag_df[diag_df["wafer_id"] == wid]
        items.append(_summarize_wafer_group(wid, wdf))
        
    return WaferListResponse(
        total_wafers=len(items),
        total_holdout_dies=len(diag_df),
        wafers=items
    )

@router.get("/{wafer_id}", response_model=WaferSummaryItem, summary="Get Wafer Telemetry Summary")
def get_wafer_summary(
    wafer_id: str,
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> WaferSummaryItem:
    """Returns defect rate, escalation count, and recovery telemetry for a specific wafer."""
    wdf = diag_df[diag_df["wafer_id"] == wafer_id]
    if len(wdf) == 0:
        raise HTTPException(
            status_code=404,
            detail={"code": "WAFER_NOT_FOUND", "message": f"Wafer ID '{wafer_id}' not found in canonical validation set"}
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
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> WaferDetailResponse:
    """
    Returns spatial coordinate grid and model predictions for interactive wafer visualization.
    Filters allow visualizing defect clusters, Model A missed failures recovered by Cascade, and false alarms.
    """
    wdf = diag_df[diag_df["wafer_id"] == wafer_id]
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
                cnt_104=float(row["cnt_104"]) if "cnt_104" in row and pd.notna(row["cnt_104"]) else None
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
    diag_df: pd.DataFrame = Depends(get_diagnostics_df)
) -> DiePoint:
    """Returns telemetry and model inference records for a specific physical die on a wafer."""
    match = diag_df[
        (diag_df["wafer_id"] == wafer_id) &
        (diag_df["die_row"] == die_row) &
        (diag_df["die_col"] == die_col)
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
        cnt_104=float(row["cnt_104"]) if "cnt_104" in row and pd.notna(row["cnt_104"]) else None
    )
