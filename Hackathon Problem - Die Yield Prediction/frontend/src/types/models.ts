export interface ConfusionMatrix {
  tp: number;
  fn: number;
  fp: number;
  tn: number;
}

export interface ModelMetricsProfile {
  model_name: string;
  threshold: number;
  f1: number;
  auc_pr: number;
  roc_auc: number;
  precision: number;
  recall: number;
  brier_score?: number;
  confusion_matrix: ConfusionMatrix;
  extra_metrics?: Record<string, unknown>;
  delta_vs_model_A?: Record<string, unknown>;
}

export interface CascadeGateMetricsProfile extends ModelMetricsProfile {
  gate_name: string;
  gating_window: string;
  model_A_threshold: number;
  model_B_threshold: number;
  escalation_count: number;
  escalation_rate_pct: number;
  compute_saved_pct: number;
  fn_recovery_metrics?: {
    model_A_fn_total: number;
    gross_fn_recovered: number;
    gross_fn_recovery_rate_pct: number;
    tp_lost_by_model_B: number;
    net_tp_gain: number;
    new_false_positives: number;
  };
}

export interface ModelMetricsResponse {
  metadata: {
    version: string;
    created_at: string;
    description: string;
    dataset: {
      file: string;
      eligible_filter: string;
      split_file: string;
      total_validation_dies: number;
      validation_wafers_count: number;
      positive_failures_count: number;
      negative_passes_count: number;
      fail_rate_pct: number;
    };
  };
  model_A: ModelMetricsProfile;
  model_B: ModelMetricsProfile;
  cascade_gate_B: CascadeGateMetricsProfile;
  cascade_gate_C: CascadeGateMetricsProfile;
}

export interface BenchmarkTableResponse {
  benchmark_columns: string[];
  benchmark_rows: Record<string, unknown>[];
  gate_sweep_columns: string[];
  gate_sweep_rows: Record<string, unknown>[];
}

export interface EvaluationRunResponse {
  status: string;
  evaluated_at: string;
  validation_dies_evaluated: number;
  validation_wafers_count: number;
  runtimes: {
    model_A_time_s: number;
    model_B_time_s: number;
    total_eval_time_s: number;
  };
  model_A: Record<string, unknown>;
  model_B: Record<string, unknown>;
  cascade_gate_B: Record<string, unknown>;
  cascade_gate_C: Record<string, unknown>;
  integrity_checks: {
    canonical_split: string;
    delta_f1_vs_authoritative: number;
    prediction_mismatch_count: number;
    status_message: string;
  };
}

export interface ErrorDrilldownResponse {
  group: string;
  title: string;
  total_in_group: number;
  returned_count: number;
  records: Array<{
    wafer_id: string;
    die_row: number;
    die_col: number;
    actual_label: number;
    prob_Model_A: number;
    prob_Model_B: number;
    prob_Cascade_B: number;
    delta_prob: number;
    decision_Model_A: number;
    decision_Cascade_B: number;
    b_max: number;
    cnt_104: number;
  }>;
}

export interface ProbabilityDistributionResponse {
  bin_labels: string[];
  bin_centers: number[];
  model_a_counts: number[];
  cascade_counts: number[];
  gate_window: {
    lower: number;
    upper: number;
  };
  model_a_threshold: number;
  model_b_threshold: number;
  total_dies: number;
}

export interface RocPrCurvesResponse {
  roc: {
    model_A: { fpr: number[]; tpr: number[]; auc: number };
    cascade: { fpr: number[]; tpr: number[]; auc: number };
  };
  pr: {
    model_A: { recall: number[]; precision: number[]; auc_pr: number };
    cascade: { recall: number[]; precision: number[]; auc_pr: number };
  };
}

export interface HighRiskDieRecord {
  wafer_id: string;
  die_row: number;
  die_col: number;
  actual_label: number;
  prob_Model_A: number;
  prob_Model_B: number;
  prob_Cascade_B: number;
  delta_prob: number;
  decision_Model_A: number;
  decision_Cascade_B: number;
  in_gate_B: boolean | number;
  recovered_FN_Gate_B?: boolean | number;
  b_max: number;
  cnt_104: number;
}

export interface HighRiskDiesResponse {
  limit: number;
  wafer_filter: string | null;
  count: number;
  records: HighRiskDieRecord[];
}

export interface ModelReleaseResponse {
  model_name: string;
  model_version: string;
  model_status: string;
  model_family: string;
  release_timestamp: string;
  active_configuration: {
    model_A: {
      version: string;
      threshold: number;
      feature_count: number;
    };
    model_B: {
      version: string;
      threshold: number;
      feature_count: number;
      active_block_features: number;
      engineered_block_candidates: number;
      restored_spatial_features: number;
    };
    cascade_gate: {
      name: string;
      lower: number;
      upper: number;
      historical_baseline_lower: number;
      historical_baseline_upper: number;
    };
    cnn_auxiliary: {
      status: string;
      model: string;
    };
  };
  evaluations: {
    final_independent_validation: {
      population: string;
      wafers_count: number;
      dies_count: number;
      actual_defects: number;
      defect_rate_pct: number;
      cascade_metrics: {
        f1: number;
        f2: number;
        mcc: number;
        pr_auc: number;
        roc_auc: number;
        precision: number;
        recall: number;
        accuracy: number;
        balanced_accuracy: number;
        tp: number;
        fn: number;
        fp: number;
        tn: number;
        escalation_rate_pct: number;
        compute_saved_pct: number;
      };
      model_A_alone: {
        f1: number;
        f2: number;
        precision: number;
        recall: number;
        tp: number;
        fn: number;
        fp: number;
      };
      rescue_gain: {
        net_tp_gain: number;
        fn_reduction: number;
        fn_rescued: number;
        fn_sent: number;
        rescue_rate_pct: number;
        additional_fp: number;
      };
      bootstrap_95_ci: {
        f1: [number, number];
        f2: [number, number];
        mcc: [number, number];
        pr_auc: [number, number];
      };
    };
    confirmed_canonical_holdout: {
      population: string;
      wafers_count: number;
      dies_count: number;
      actual_defects: number;
      defect_rate_pct: number;
      cascade_metrics: {
        f1: number;
        f2: number;
        mcc: number;
        pr_auc: number;
        roc_auc: number;
        precision: number;
        recall: number;
        tp: number;
        fn: number;
        fp: number;
        tn: number;
        escalation_rate_pct: number;
        compute_saved_pct: number;
      };
    };
    historical_production_baseline: {
      population: string;
      wafers_count: number;
      dies_count: number;
      actual_defects: number;
      cascade_metrics: {
        f1: number;
        f2: number;
        mcc: number;
        pr_auc: number;
        roc_auc: number;
        precision: number;
        recall: number;
        tp: number;
        fn: number;
        fp: number;
        tn: number;
        escalation_rate_pct: number;
        compute_saved_pct: number;
      };
    };
  };
}

export interface PolicySimulationRequest {
  tau_A?: number;
  gate_lower?: number;
  gate_upper?: number;
  tau_B?: number;
  secondary_budget_pct?: number | null;
}

export interface PolicySimulationResponse {
  status: string;
  simulation_mode: boolean;
  production_policy_modified: boolean;
  disclaimer: string;
  parameters: {
    simulated_tau_A: number;
    simulated_gate_lower: number;
    simulated_gate_upper: number;
    simulated_tau_B: number;
    secondary_budget_pct?: number | null;
  };
  simulated_metrics: {
    f1: number;
    f2: number;
    precision: number;
    recall: number;
    mcc: number;
    escalation_count: number;
    escalation_rate_pct: number;
    compute_saved_pct: number;
    tp: number;
    fn: number;
    fp: number;
    tn: number;
    fn_rescued: number;
    fn_escalated: number;
    tp_lost: number;
    net_tp_gain: number;
    new_fp: number;
  };
  baseline_comparison: {
    production_f1: number;
    production_precision: number;
    production_recall: number;
    production_escalation_pct: number;
    delta_f1: number;
    delta_precision: number;
    delta_recall: number;
    delta_escalation_pct: number;
    delta_compute_saved_pct: number;
  };
}
