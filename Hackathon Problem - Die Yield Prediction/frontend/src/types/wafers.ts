export interface WaferSummaryItem {
  wafer_id: string;
  total_dies: number;
  actual_failures: number;
  fail_rate_pct: number;
  model_a_detections: number;
  cascade_detections: number;
  escalated_dies: number;
  recovered_fns: number;
  mean_probability?: number | null;
  high_risk_dies?: number | null;
}


export interface WaferListResponse {
  total_wafers: number;
  total_holdout_dies: number;
  wafers: WaferSummaryItem[];
}

export interface DiePoint {
  wafer_id: string;
  die_row: number;
  die_col: number;
  actual_label: number;
  prob_Model_A: number;
  prob_Model_B: number;
  prob_Cascade_B: number;
  decision_Model_A: number;
  decision_Cascade_B: number;
  in_gate_B: boolean;
  recovered_FN_Gate_B: number;
  delta_prob: number;
  radial_dist?: number | null;
  edge_dist?: number | null;
  b_max?: number | null;
  cnt_104?: number | null;
  calibrated_risk?: number | null;
}

export interface WaferDetailResponse {
  summary: WaferSummaryItem;
  dies: DiePoint[];
}

export interface HotspotSummaryItem {
  hotspot_id: string;
  region_name: string;
  center_row: number;
  center_col: number;
  radius_dies: number;
  die_count: number;
  high_risk_die_count: number;
  observed_defect_count: number;
  mean_risk_prob: number;
  defect_density_pct: number;
  primary_morphology: string;
  suspicious_signature: string;
}

export interface WaferHotspotsResponse {
  wafer_id: string;
  total_dies: number;
  total_high_risk_dies: number;
  hotspots_count: number;
  hotspots: HotspotSummaryItem[];
}
