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
