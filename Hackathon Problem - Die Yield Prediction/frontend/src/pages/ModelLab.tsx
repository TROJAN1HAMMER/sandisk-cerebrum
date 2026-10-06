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

      {/* Final Production Candidate: B2 vs Baseline B0 Specification Panel */}
      <Panel
        title="Active Model Architecture: B2 Enhanced vs Production Baseline B0"
        subtitle="Final validated configuration with selective asymmetric gate and enhanced sub-die representation."
      >
        <div className="space-y-4 p-2">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                  <span className="text-content-primary">LightGBM Selective Cascade</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Model B Features:</span>
                  <span className="text-content-primary">526 (Legacy 20 Block Features)</span>
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
                  <span className="text-content-primary">Canonical 32 Wafers (27,972 dies)</span>
                </div>
                <div className="flex justify-between border-t border-silicon-border/60 pt-1.5 font-bold">
                  <span className="text-content-muted">Cascade F1:</span>
                  <span className="text-content-primary">0.5072 (88.14% Compute Saved)</span>
                </div>
              </div>
            </div>

            {/* Frozen B2 Candidate */}
            <div className="p-4 rounded-lg bg-silicon-elevated border border-telemetry-green/40 space-y-3 font-mono text-xs shadow-sm">
              <div className="flex items-center justify-between border-b border-silicon-border/80 pb-2">
                <span className="font-bold text-telemetry-green flex items-center gap-1.5">
                  <Award className="w-4 h-4 text-telemetry-green" />
                  ACTIVE PRODUCTION CANDIDATE (B2)
                </span>
                <span className="px-2 py-0.5 rounded bg-telemetry-green/15 text-telemetry-green text-[10px] font-bold">
                  FROZEN PROMOTION
                </span>
              </div>
              <div className="space-y-1.5 text-content-secondary">
                <div className="flex justify-between">
                  <span className="text-content-muted">Model Family:</span>
                  <span className="text-content-primary">LightGBM Selective Cascade</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Model B Features:</span>
                  <span className="text-telemetry-green font-bold">536 (30 Block Features)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Spatial Defect Density:</span>
                  <span className="text-telemetry-green font-semibold">EXP-01 Pre-Test Restored</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Gate Window:</span>
                  <span className="text-telemetry-cyan font-bold">[0.295, 0.550] (Asymmetric)</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Thresholds:</span>
                  <span className="text-content-primary">τ_A = 0.5100, τ_B = 0.5376</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-content-muted">Eval Population:</span>
                  <span className="text-telemetry-cyan">Independent 40 Wafers (32,598 dies)</span>
                </div>
                <div className="flex justify-between border-t border-silicon-border/60 pt-1.5 font-bold">
                  <span className="text-content-muted">Cascade F1:</span>
                  <span className="text-telemetry-green">0.5177 (93.78% Compute Saved)</span>
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
    </div>
  );
};
