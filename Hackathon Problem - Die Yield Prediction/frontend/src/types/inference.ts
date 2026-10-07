export interface SingleDieRequest {
  wafer_id?: string;
  die_row: number;
  die_col: number;
  features?: Record<string, number>;
  spatial?: Record<string, number>;
  block_readings?: string;
  prob_Model_A?: number;
  prob_Model_B?: number;
  actual_label?: number;
  gate_lower?: number;
  gate_upper?: number;
}

export interface DieInferenceResponse {
  wafer_id: string;
  die_row: number;
  die_col: number;
  prob_Model_A: number;
  threshold_Model_A: number;
  decision_Model_A: number;
  in_gate: boolean;
  gate_status: 'ESCALATED' | 'BYPASSED' | string;
  gate_message: string;
  model_B_invoked: boolean;
  prob_Model_B: number | null;
  threshold_Model_B: number;
  decision_Model_B: number | null;
  final_probability: number;
  final_decision: number;
  decision_text: string;
  compute_path: 'FAST PATH' | 'ESCALATED PATH' | string;
  actual_label: number | null;
  is_correct: boolean | null;
  spatial_signals: Record<string, number>;
  block_signals: Record<string, number>;
  latency_ms: number | null;
  calibrated_risk?: number | null;
  calibration_notes?: string;
  evidence_chain?: {
    electrical: {
      status: string;
      detected: boolean;
      b_max: number;
      cnt_104: number;
      nominal_baseline: number;
      deviation_pct: number;
      details: string;
    };
    spatial: {
      status: string;
      detected: boolean;
      radial_dist: number;
      edge_dist: number;
      density_5x5: number;
      is_edge_die: boolean;
      details: string;
    };
    signal: {
      status: string;
      detected: boolean;
      cluster_area: number;
      peak_voltage: number;
      details: string;
    };
    neighbourhood: {
      status: string;
      detected: boolean;
      defect_neighbours: number;
      escalated_neighbours: number;
      details: string;
    };
    model_decision: {
      status: string;
      stage1_prob: number;
      stage1_threshold: number;
      gated: boolean;
      stage2_prob: number | null;
      stage2_threshold: number;
      final_decision: number;
      details: string;
    };
  };
}

export interface BatchInferenceSummary {
  total_dies: number;
  predicted_fails: number;
  predicted_passes: number;
  predicted_fail_rate_pct: number;
  escalated_dies: number;
  escalation_rate_pct: number;
  compute_saved_pct: number;
  total_inference_time_s: number;
  latency_per_die_ms: number;
}

export interface BatchInferenceItem {
  wafer_id: string;
  die_row: number;
  die_col: number;
  prob_Model_A: number;
  decision_Model_A: number;
  escalated: 'YES' | 'NO' | string;
  prob_Model_B: number | null;
  decision_Model_B: number | null;
  final_probability: number;
  final_decision: number;
  decision_label: string;
  compute_path: string;
  actual_label: number | null;
}

export interface BatchInferenceResponse {
  summary: BatchInferenceSummary;
  results: BatchInferenceItem[];
}

export interface DemoSampleItem {
  sample_id: string;
  wafer_id: string;
  die_row: number;
  die_col: number;
  actual_label: number;
  model_A_prob: number;
  cascade_final_prob: number;
  model_A_decision: number;
  cascade_final_decision: number;
  resolution_used: string;
  escalation_reason: string;
  description: string;
}

export interface BlockAttributionResponse {
  wafer_id: string;
  die_row: number;
  die_col: number;
  base_prob: number;
  block_count: number;
  readings: number[];
  attributions: number[];
  peak_attribution_idx: number;
  peak_attribution_value: number;
  anomalous_blocks_count: number;
}

export interface TcnForensicsResponse {
  wafer_id: string;
  die_row: number;
  die_col: number;
  role: string;
  block_count?: number;
  sequence_prob_tcn?: number;
  peak_block_index?: number;
  peak_amplitude?: number;
  anomaly_centroid?: number;
  anomaly_span_blocks?: number;
  saliency_curve?: number[];
  important_block_ranges?: Array<{
    start_block: number;
    end_block: number;
    peak_saliency: number;
    mean_voltage: number;
  }>;
  waveform_readings: number[];
  tcn_attribution: number[];
  peak_attribution_idx: number;
  peak_attribution_value: number;
  anomalous_span_start: number;
  anomalous_span_end: number;
  salient_blocks_count: number;
  interpretation_notes: string;
  forensic_only?: boolean;
  used_in_production_routing?: boolean;
  disclaimer?: string;
}
