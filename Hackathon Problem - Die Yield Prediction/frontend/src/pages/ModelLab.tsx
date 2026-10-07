import React, { useState } from 'react';
import { useModelBenchmark, useModelRelease } from '../hooks/useModels';
import { useRunEvaluation } from '../hooks/useEvaluation';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Panel } from '../components/ui/Panel';
import { DataTable, Column } from '../components/data/DataTable';
import { LoadingSkeleton } from '../components/ui/LoadingSkeleton';
import { ErrorState } from '../components/ui/ErrorState';
import { Play, CheckCircle2, ShieldCheck, Flame, GitCommit, Layers, Cpu, Zap, Award } from 'lucide-react';
import { EvaluationRunResponse } from '../types/models';
import { Tooltip, InfoTooltip } from '../components/ui/Tooltip';

export const ModelLab: React.FC = () => {
  const { data: benchmark, isLoading, error, refetch } = useModelBenchmark();
  const { data: release } = useModelRelease();
  const runEvaluationMutation = useRunEvaluation();
  const [evalResult, setEvalResult] = useState<EvaluationRunResponse | null>(null);

  const handleRunEvaluation = async () => {
    try {
      const res = await runEvaluationMutation.mutateAsync();
      setEvalResult(res);
    } catch (e) {
      console.error('Evaluation run failed:', e);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <LoadingSkeleton className="h-8 w-64" />
        <LoadingSkeleton className="h-48" />
        <LoadingSkeleton className="h-64" />
      </div>
    );
  }

  if (error || !benchmark) {
    return (
      <ErrorState
        title="Could not load benchmark data"
        message="Failed to retrieve benchmark matrix from /api/v1/models/benchmark."
        onRetry={refetch}
      />
    );
  }

  // Format benchmark columns
  const benchCols: Column<Record<string, unknown>>[] = [
    { key: 'model', header: 'Model Architecture', className: 'font-semibold' },
    { key: 'gate', header: 'Gating Strategy' },
    { key: 'F1', header: 'Fail F1', align: 'right' },
    { key: 'AUC_PR', header: 'AUC-PR', align: 'right' },
    { key: 'precision', header: 'Precision', align: 'right' },
    { key: 'recall', header: 'Recall', align: 'right' },
    { key: 'escalation_rate', header: 'Escalation %', align: 'right' },
    { key: 'net_TP_gain', header: 'Net TP Gain', align: 'right' },
  ];

  // Format gate sweep columns
  const sweepCols: Column<Record<string, unknown>>[] = [
    { key: 'Gate_Name', header: 'Gate Window', className: 'font-semibold' },
    { key: 'Escalation_Pct', header: 'Escalation Rate', align: 'right' },
    { key: 'Compute_Saved_Pct', header: 'Compute Saved', align: 'right' },
    { key: 'Cascade_F1', header: 'Cascade F1', align: 'right' },
    { key: 'Cascade_AUC_PR', header: 'AUC-PR', align: 'right' },
    { key: 'Cascade_Precision', header: 'Precision', align: 'right' },
    { key: 'FN_Recovered', header: 'FN Recovered', align: 'right' },
  ];

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Architectural Model Lab & Benchmark Matrix"
        subtitle="Forensic head-to-head comparison across 27,972 canonical holdout dies with dynamic evaluation engine."
      />

      {/* Model Evolution: B0 vs B2 vs B3 Specification Panel */}
      <Panel
        title="Active Model Architecture: B3 Dual-Stream vs B2 Checkpoint vs B0 Baseline"
        subtitle="B3 Multimodal Dual-Stream Cascade active in production. B2 preserved as immutable rollback baseline."
      >
        <div className="space-y-4 p-2">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Historical B0 Baseline */}
            <div className="p-4 rounded-lg bg-silicon-elevated border border-silicon-border space-y-3 font-mono text-xs">
              <div className="flex items-center justify-between border-b border-silicon-border/80 pb-2">
                <span className="font-bold text-content-secondary flex items-center gap-1.5">
                  <GitCommit className="w-4 h-4 text-telemetry-amber" />
                  HISTORICAL BASELINE (B0)
                </span>
                <span className="px-2 py-0.5 rounded bg-silicon-canvas text-content-muted text-[10px]">
                  ARCHIVAL BACKUP
                </span>
              </div>
              <div className="space-y-1.5 text-content-secondary">
                <div className="flex justify-between">
                  <span className="text-content-muted">Model Family:</span>
                  <span className="text-content-primary">LightGBM Cascade</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Model B Features:</span>
                  <span className="text-content-primary">526 (20 Block Feats)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Spatial Defect Density:</span>
                  <span className="text-telemetry-red">Constant Zero (Buggy)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Gate Window:</span>
                  <span className="text-content-primary">[0.25, 0.55]</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Thresholds:</span>
                  <span className="text-content-primary">τ_A = 0.5100, τ_B = 0.5376</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Eval Population:</span>
                  <span className="text-content-primary">Canonical 32 Wafers</span>
                </div>
                <div className="flex justify-between border-t border-silicon-border/60 pt-1.5 font-bold">
                  <span className="text-content-muted">Cascade F1:</span>
                  <span className="text-content-primary">0.5072 (88.14% Saved)</span>
                </div>
              </div>
            </div>

            {/* Preserved B2 Rollback Checkpoint */}
            <div className="p-4 rounded-lg bg-silicon-elevated border border-telemetry-cyan/40 space-y-3 font-mono text-xs shadow-sm">
              <div className="flex items-center justify-between border-b border-silicon-border/80 pb-2">
                <span className="font-bold text-telemetry-cyan flex items-center gap-1.5">
                  <GitCommit className="w-4 h-4 text-telemetry-cyan" />
                  HISTORICAL CHECKPOINT (B2)
                </span>
                <span className="px-2 py-0.5 rounded bg-telemetry-cyan/15 text-telemetry-cyan text-[10px] font-bold">
                  ROLLBACK READY
                </span>
              </div>
              <div className="space-y-1.5 text-content-secondary">
                <div className="flex justify-between">
                  <span className="text-content-muted">Model Family:</span>
                  <span className="text-content-primary">LightGBM Cascade</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Model B Features:</span>
                  <span className="text-content-primary font-bold">536 (30 Block Feats)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Spatial Defect Density:</span>
                  <span className="text-telemetry-green font-semibold">EXP-01 Pre-Test Restored</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Gate Window:</span>
                  <span className="text-content-primary font-bold">[0.295, 0.550] (Asym)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Thresholds:</span>
                  <span className="text-content-primary">τ_A = 0.5100, τ_B = 0.5376</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Eval Population:</span>
                  <span className="text-content-primary">Independent 40 Wafers</span>
                </div>
                <div className="flex justify-between border-t border-silicon-border/60 pt-1.5 font-bold">
                  <span className="text-content-muted">Cascade F1:</span>
                  <span className="text-telemetry-cyan">0.5177 (93.78% Saved)</span>
                </div>
              </div>
            </div>

            {/* Promoted B3 Active Production */}
            <div className="p-4 rounded-lg bg-silicon-elevated border border-telemetry-green/50 space-y-3 font-mono text-xs shadow-sm ring-1 ring-telemetry-green/30">
              <div className="flex items-center justify-between border-b border-silicon-border/80 pb-2">
                <span className="font-bold text-telemetry-green flex items-center gap-1.5">
                  <Award className="w-4 h-4 text-telemetry-green" />
                  ACTIVE PRODUCTION (B3)
                </span>
                <span className="px-2 py-0.5 rounded bg-telemetry-green/15 text-telemetry-green text-[10px] font-bold">
                  DUAL-STREAM PROMOTED
                </span>
              </div>
              <div className="space-y-1.5 text-content-secondary">
                <div className="flex justify-between">
                  <span className="text-content-muted">Model Family:</span>
                  <span className="text-telemetry-green font-bold">Multimodal Dual-Stream</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Model A Features:</span>
                  <span className="text-telemetry-purple font-bold">517 Features</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Model B Features:</span>
                  <span className="text-telemetry-purple font-bold">528 Tab + 1D-CNN</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-content-muted flex items-center">
                    CNN Modality:
                    <InfoTooltip text="An additional deep neural model that looks at the die's 2,000-point signal pattern for abnormal behavior." />
                  </span>
                  <span className="text-content-primary">2,000-Block (Escalated Only)</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-content-muted flex items-center">
                    Late Fusion:
                    <InfoTooltip text="Final probability combines 90% tabular tree evidence with 10% raw signal pattern evidence." />
                  </span>
                  <span className="text-telemetry-green font-semibold">0.90 Tabular + 0.10 CNN</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-content-muted flex items-center">
                    Gate Window:
                    <InfoTooltip text="Decides whether this die needs a deeper inspection. Dies with uncertain Model A predictions (0.250 to 0.550) are sent to the second stage." />
                  </span>
                  <span className="text-telemetry-cyan font-bold">[0.250, 0.550] (Standard)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Thresholds:</span>
                  <span className="text-content-primary font-bold">τ_A = 0.5100, τ_B = 0.5200</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Eval Population:</span>
                  <span className="text-telemetry-cyan">Independent 40 Wafers</span>
                </div>
                <div className="flex justify-between border-t border-silicon-border/60 pt-1.5 font-bold">
                  <span className="text-content-muted">Cascade F1:</span>
                  <span className="text-telemetry-green font-bold">0.5264 (96.05% Saved)</span>
                </div>
              </div>
            </div>
          </div>

          {/* Model B Feature Anatomy Badge Row */}
          <div className="p-3 rounded bg-silicon-canvas border border-silicon-border/80 font-mono text-[11px] flex flex-wrap items-center justify-between gap-2">
            <span className="text-content-secondary">
              B2 ENHANCED FEATURE SUITE:
            </span>
            <div className="flex flex-wrap gap-1.5">
              <span className="px-2 py-0.5 rounded bg-telemetry-blue/15 text-telemetry-blue">500 Parametric Electrical</span>
              <span className="px-2 py-0.5 rounded bg-telemetry-green/15 text-telemetry-green">6 Wafer Spatial (Restored Density)</span>
              <span className="px-2 py-0.5 rounded bg-silicon-elevated text-content-primary border border-silicon-border">20 Fixed Block Baseline</span>
              <span className="px-2 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple font-bold border border-telemetry-purple/40">10 Engineered Block Anomaly Candidates</span>
            </div>
          </div>
        </div>
      </Panel>

      {/* Dynamic Model Evaluation Engine Card */}
      <Panel
        title="Dynamic Live Evaluation Engine (All 27,972 Canonical Dies)"
        subtitle="Execute dynamic full-split evaluation across both models to verify exact 0.000000 metric reproducibility in real-time."
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-2">
          <div>
            <div className="text-xs font-mono text-content-primary font-semibold">
              EVALUATION PROTOCOL: HOLD-OUT CANONICAL SPLIT (32 WAFERS)
            </div>
            <div className="text-[11px] text-content-muted mt-0.5">
              Wraps src/live_inference.py: execute_full_live_evaluation() via POST /api/v1/evaluation/run
            </div>
          </div>

          <button
            onClick={handleRunEvaluation}
            disabled={runEvaluationMutation.isPending}
            className="inline-flex items-center gap-2 px-4 py-2 rounded bg-telemetry-green/20 hover:bg-telemetry-green/30 text-telemetry-green border border-telemetry-green/40 font-mono text-xs font-bold transition-all disabled:opacity-50"
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            {runEvaluationMutation.isPending ? 'EVALUATING 27,972 DIES...' : 'RUN FULL LIVE EVALUATION'}
          </button>
        </div>

        {evalResult && (
          <div className="mt-4 p-4 rounded bg-silicon-elevated border border-telemetry-green/40 font-mono text-xs space-y-2">
            <div className="flex items-center gap-2 text-telemetry-green font-bold">
              <CheckCircle2 className="w-4 h-4" />
              {evalResult.integrity_checks.status_message}
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px] text-content-secondary pt-1">
              <div>Dies: {evalResult.validation_dies_evaluated.toLocaleString()}</div>
              <div>Delta vs Auth: {evalResult.integrity_checks.delta_f1_vs_authoritative.toFixed(6)}</div>
              <div>Mismatches: {evalResult.integrity_checks.prediction_mismatch_count}</div>
              <div>Runtime: {evalResult.runtimes.total_eval_time_s.toFixed(3)}s</div>
            </div>
          </div>
        )}
      </Panel>

      {/* Benchmark Matrix */}
      <Panel
        title="Comprehensive Architecture Benchmark Table"
        subtitle="Authoritative comparison of Model A Baseline, Standalone Model B, and Cascade Gates B and C"
      >
        <DataTable
          columns={benchCols}
          data={benchmark.benchmark_rows}
          keyExtractor={(_, i) => i}
        />
      </Panel>

      {/* Uncertainty Gating Sweeps */}
      <Panel
        title="Uncertainty Gating Sweeps (Pareto Trade-Off Curve Data)"
        subtitle="Evaluation of escalation thresholds from narrow to wide gating windows"
      >
        <DataTable
          columns={sweepCols}
          data={benchmark.gate_sweep_rows}
          keyExtractor={(_, i) => i}
        />
      </Panel>

      {/* Model Evolution & Evidence-Driven Model Selection Panel */}
      <Panel
        title="Evidence-Driven Model Evolution & Architecture Decisions"
        subtitle="Rigorous empirical testing trajectory: retained only components with measurable, validated gains."
      >
        <div className="space-y-4 p-2 font-mono text-xs">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3.5">
            <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-content-primary">EXP-05: Late Probability Stacking</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-telemetry-red/20 text-telemetry-red border border-telemetry-red/40 font-bold">
                  REJECTED
                </span>
              </div>
              <p className="text-[11px] text-content-secondary leading-relaxed font-sans">
                Logistic regression probability stacking on holdout caused probability distortion and destroyed cascade gating separation without improving F1.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-content-primary">EXP-06: XGBoost + CatBoost Ensemble</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-telemetry-red/20 text-telemetry-red border border-telemetry-red/40 font-bold">
                  REJECTED
                </span>
              </div>
              <p className="text-[11px] text-content-secondary leading-relaxed font-sans">
                Heavy multi-library ensemble increased per-die inference latency by 3.8× while F1 showed negligible variation (+0.0011). Did not justify production complexity.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-content-primary">EXP-07: FFT / Kurtosis / Run-Length</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-telemetry-red/20 text-telemetry-red border border-telemetry-red/40 font-bold">
                  REJECTED
                </span>
              </div>
              <p className="text-[11px] text-content-secondary leading-relaxed font-sans">
                FFT spectral harmonics and run-length features added 42 dimensions but caused feature redundancy, slightly degrading F1 (-0.0038).
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-content-primary">EXP-08: Multi-Scale Dilated TCN</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/40 font-bold">
                  RETAINED FOR FORENSICS
                </span>
              </div>
              <p className="text-[11px] text-content-secondary leading-relaxed font-sans">
                Replacing GBDT tabular decisioning failed to match B2 precision, but multi-scale TCN saliency gradient attribution proved superior for 2,000-block waveform forensics.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-content-primary">EXP-09: Isotonic Risk Calibration</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/40 font-bold">
                  RETAINED FOR REPORTING
                </span>
              </div>
              <p className="text-[11px] text-content-secondary leading-relaxed font-sans">
                Reduced Expected Calibration Error (ECE) from 0.1250 to 0.0035 (-97.2%). Retained strictly for risk interpretation while keeping production routing thresholds frozen.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border space-y-2">
              <div className="flex items-center justify-between">
                <span className="font-bold text-content-primary">EXP-10: Context-Aware Intelligence</span>
                <span className="px-2 py-0.5 rounded text-[10px] bg-telemetry-red/20 text-telemetry-red border border-telemetry-red/40 font-bold">
                  REJECTED
                </span>
              </div>
              <p className="text-[11px] text-content-secondary leading-relaxed font-sans">
                Spatial ring cluster aggregations and reticle neighborhood features introduced wafer-level variance overfitting without exceeding the frozen B2 validation baseline.
              </p>
            </div>
          </div>
        </div>
      </Panel>

      {/* Production Architecture & Dataflow Diagram Panel */}
      <Panel
        title="Technical Architecture: Dual-Stream Production & Forensic Pipeline"
        subtitle="Strict separation between frozen production decisioning, forensic waveform attribution, and calibrated risk reporting."
      >
        <div className="p-4 bg-silicon-elevated rounded-lg border border-silicon-border space-y-4 font-mono text-xs">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Stream 1 */}
            <div className="p-3 rounded bg-silicon-canvas border border-silicon-border space-y-2">
              <div className="text-telemetry-cyan font-bold flex items-center gap-1.5">
                <Cpu className="w-4 h-4" /> 1. PRODUCTION DECISION STREAM
              </div>
              <div className="space-y-1 text-content-secondary text-[11px]">
                <div>• Stage 1: LightGBM Model A (517 feats)</div>
                <div>• Standard Gate: [0.250, 0.550]</div>
                <div>• Stage 2: Dual-Stream Model B (528 feats + 1D-CNN)</div>
                <div>• Production Thresholds: τ_A=0.5100, τ_B=0.5200</div>
                <div className="text-telemetry-green font-semibold">• 96.05% compute saved • 87.67% precision • F1=0.5264</div>
              </div>
            </div>

            {/* Stream 2 */}
            <div className="p-3 rounded bg-silicon-canvas border border-silicon-border space-y-2">
              <div className="text-telemetry-purple font-bold flex items-center gap-1.5">
                <Zap className="w-4 h-4" /> 2. FORENSIC WAVEFORM STREAM
              </div>
              <div className="space-y-1 text-content-secondary text-[11px]">
                <div>• 2,000-Block Voltage Readings</div>
                <div>• Multi-Scale Dilated TCN (rf=256)</div>
                <div>• Input-Gradient Saliency Attribution</div>
                <div>• Anomaly Span Localization</div>
                <div className="text-telemetry-purple font-semibold">• Strictly forensic (never alters routing)</div>
              </div>
            </div>

            {/* Stream 3 */}
            <div className="p-3 rounded bg-silicon-canvas border border-silicon-border space-y-2">
              <div className="text-telemetry-green font-bold flex items-center gap-1.5">
                <ShieldCheck className="w-4 h-4" /> 3. RISK REPORTING STREAM
              </div>
              <div className="space-y-1 text-content-secondary text-[11px]">
                <div>• Isotonic Calibrator (EXP-09)</div>
                <div>• ECE: 0.0035 (-97.2% error)</div>
                <div>• Calibrated Risk Indicator for Fab Planning</div>
                <div>• Yield-at-Risk Spatial Concentration</div>
                <div className="text-telemetry-green font-semibold">• Planning & yield exposure quantification</div>
              </div>
            </div>
          </div>
        </div>
      </Panel>
    </div>
  );
};
