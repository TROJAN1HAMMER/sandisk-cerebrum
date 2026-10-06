export interface HealthResponse {
  status: string;
  app_name: string;
  version: string;
  environment: string;
}

export interface StatusResponse {
  status: string;
  models_loaded: boolean;
  model_a_active: boolean;
  model_b_active: boolean;
  cnn_active: boolean;
  diagnostics_loaded: boolean;
  total_validation_dies: number;
  validation_wafers_count: number;
  canonical_split: string;
  gate_b_window: string;
  gate_c_window: string;
  active_model_version?: string;
  production_gate_b_window?: string;
  startup_time_s: number;
}
