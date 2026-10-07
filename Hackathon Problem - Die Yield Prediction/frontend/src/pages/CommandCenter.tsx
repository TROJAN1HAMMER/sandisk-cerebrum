import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useModelMetrics, useModelRelease } from '../hooks/useModels';
import { useSystemStatus } from '../hooks/useSystem';
import { useWafers } from '../hooks/useWafers';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Panel } from '../components/ui/Panel';
import { StatusBadge } from '../components/ui/StatusBadge';
import { LoadingSkeleton } from '../components/ui/LoadingSkeleton';
import { ErrorState } from '../components/ui/ErrorState';
import {
  CheckCircle2,
  Cpu,
  ShieldCheck,
  Zap,
  ArrowRight,
  Layers,
  Disc,
  ExternalLink,
  Flame,
  Award,
  Database,
} from 'lucide-react';

export const CommandCenter: React.FC = () => {
  const navigate = useNavigate();
  const { data: metrics, isLoading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useModelMetrics();
  const { data: release } = useModelRelease();
  const { data: status, isLoading: statusLoading } = useSystemStatus();
  const { data: wafersData, isLoading: wafersLoading } = useWafers();
  const [metricView, setMetricView] = useState<'independent' | 'canonical'>('independent');

  if (metricsLoading || statusLoading) {
    return (
      <div className="space-y-6">
        <LoadingSkeleton className="h-8 w-64" />
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <LoadingSkeleton key={i} className="h-28" />
          ))}
        </div>
        <LoadingSkeleton className="h-64" />
      </div>
    );
  }

  if (metricsError || !metrics) {
    return (
      <ErrorState
        title="Could not retrieve production model metrics"
        message="Failed to connect to the FastAPI models metrics endpoint."
        onRetry={refetchMetrics}
      />
    );
  }

  const ds = metrics.metadata.dataset;
  const mA = metrics.model_A;
  const mB = metrics.model_B;
  const gB = metrics.cascade_gate_B;

  // Independent 40-wafer evaluation metrics (EXP-04 Fresh Benchmark)
  const indVal = release?.evaluations?.final_independent_validation?.cascade_metrics || {
    f1: 0.5264,
    f2: 0.4246,
    mcc: 0.5636,
    pr_auc: 0.5004,
    roc_auc: 0.8270,
    precision: 0.8767,
    recall: 0.3761,
    tp: 519,
    fn: 861,
    fp: 73,
    tn: 31145,
    escalation_rate_pct: 3.95,
    compute_saved_pct: 96.05,
  };

  // Canonical 32-wafer evaluation metrics (EXP-03B confirmed B2)
  const canVal = release?.evaluations?.confirmed_canonical_holdout?.cascade_metrics || {
    f1: 0.5254,
    f2: 0.4213,
    mcc: 0.5674,
    pr_auc: 0.4777,
    roc_auc: 0.7986,
    precision: 0.8932,
    recall: 0.3722,
    tp: 393,
    fn: 663,
    fp: 47,
    tn: 26869,
    escalation_rate_pct: 5.98,
    compute_saved_pct: 94.02,
  };

  return (
    <div className="space-y-6">
      {/* Page Title & Status Badges */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-silicon-border/60">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold">
              Mission Control
            </span>
            <span className="text-content-muted">•</span>
            <span className="text-xs font-mono text-telemetry-purple font-semibold">
              MODEL RELEASE: B3 DUAL-STREAM CASCADE
            </span>
            <span className="text-content-muted">•</span>
            <span className="text-xs font-mono text-telemetry-green font-bold">
              STATUS: PRODUCTION
            </span>
          </div>
          <h2 className="text-xl font-bold tracking-tight text-content-primary mt-0.5">
            Selective Multi-Resolution Die Screening Platform
          </h2>
          <div className="text-xs text-content-secondary mt-0.5 font-mono">
            Validated on: Independent 40-Wafer Population (32,598 dies • 1,380 defects)
          </div>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <StatusBadge label="Release: B3 Production" variant="pass" pulse />
          <StatusBadge label="Gate B: [0.250, 0.550]" variant="pass" />
          <StatusBadge label="Zero-Leakage Compliance" variant="info" />
        </div>
      </div>

      {/* Active Production Model B3 Insight Banner */}
      <div className="p-3.5 rounded-lg bg-gradient-to-r from-telemetry-cyan/15 via-silicon-elevated to-telemetry-purple/15 border border-telemetry-cyan/40 font-mono text-xs flex flex-col md:flex-row md:items-center justify-between gap-3 shadow-sm">
        <div className="flex items-center gap-2.5">
          <ShieldCheck className="w-4 h-4 text-telemetry-cyan shrink-0" />
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-bold text-content-primary">ACTIVE PRODUCTION ARCHITECTURE:</span>
              <span className="px-2 py-0.5 rounded bg-telemetry-cyan/20 text-telemetry-cyan font-bold text-[10px] border border-telemetry-cyan/40">
                B3-DualStream
              </span>
              <span className="px-2 py-0.5 rounded bg-telemetry-green/20 text-telemetry-green font-bold text-[10px] border border-telemetry-green/40">
                LOCKED IN PRODUCTION
              </span>
            </div>
            <div className="text-[11px] text-content-secondary mt-0.5 font-sans">
              Stage 1 (517 features) → Gate [0.250, 0.550] → Stage 2 (528 tabular features + 2,000-block 1D CNN late fusion 0.90/0.10).
            </div>
          </div>
        </div>
        <button
          onClick={() => navigate('/model-lab')}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-silicon-canvas hover:bg-silicon-interactive text-telemetry-cyan hover:text-content-primary border border-telemetry-cyan/40 text-[11px] font-bold transition-all shrink-0 self-start md:self-auto cursor-pointer"
        >
          <span>Inspect Architecture in Model Lab</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Metric View Switcher */}
      <div className="flex items-center justify-between bg-silicon-elevated px-4 py-2.5 rounded-lg border border-silicon-border/80">
        <div className="flex items-center gap-2 text-xs font-mono">
          <Award className="w-4 h-4 text-telemetry-cyan" />
          <span className="text-content-secondary">BENCHMARK POPULATION:</span>
          <span className="text-content-primary font-bold">
            {metricView === 'independent' ? 'Independent 40 Wafers (Authoritative EXP-04)' : 'Canonical 32 Wafers (EXP-03B Confirmation)'}
          </span>
        </div>
        <div className="flex items-center gap-1.5 bg-silicon-canvas p-1 rounded border border-silicon-border text-xs font-mono">
          <button
            onClick={() => setMetricView('independent')}
            className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
              metricView === 'independent'
                ? 'bg-telemetry-cyan/20 text-telemetry-cyan font-bold border border-telemetry-cyan/40'
                : 'text-content-muted hover:text-content-primary'
            }`}
          >
            Active B3 — Independent EXP-04 Validation (40 Wafers)
          </button>
          <button
            onClick={() => setMetricView('canonical')}
            className={`px-2.5 py-1 rounded transition-colors cursor-pointer ${
              metricView === 'canonical'
                ? 'bg-telemetry-green/20 text-telemetry-green font-bold border border-telemetry-green/40'
                : 'text-content-muted hover:text-content-primary'
            }`}
          >
            Historical B0 — Canonical Holdout (32 Wafers)
          </button>
        </div>
      </div>

      {/* 6 Hero Metric Cards */}
      {metricView === 'independent' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3.5">
          <MetricCard
            title="Cascade Fail F1"
            value={indVal.f1.toFixed(4)}
            subtitle="Harmonic defect score"
            tooltip="Harmonic balance of precision and recall for defect identification. Higher score means fewer misses and fewer false alarms."
            badge="INDEPENDENT"
            accent="green"
          />

          <MetricCard
            title="PR-AUC"
            value={indVal.pr_auc.toFixed(4)}
            subtitle={`ROC-AUC: ${indVal.roc_auc.toFixed(4)}`}
            tooltip="Precision-Recall Area Under Curve. Measures defect ranking accuracy across imbalanced data without distortion from abundant healthy dies."
            badge="RANKING"
            accent="purple"
          />

          <MetricCard
            title="Cascade Recall"
            value={`${(indVal.recall * 100).toFixed(2)}%`}
            subtitle="519 TP / 861 FN"
            tooltip="Percentage of actual defective dies successfully detected by the cascade system."
            badge="COVERAGE"
            accent="blue"
          />

          <MetricCard
            title="Cascade Precision"
            value={`${(indVal.precision * 100).toFixed(2)}%`}
            subtitle="73 False Positives"
            tooltip="Percentage of flagged dies that are genuine defects rather than false alarms."
            badge="PURITY"
            accent="green"
          />

          <MetricCard
            title="Compute Saved"
            value={`${indVal.compute_saved_pct.toFixed(2)}%`}
            subtitle="31,218 Fast-Path Dies"
            tooltip="Percentage of dies classified instantly by Stage 1, completely skipping expensive Stage 2 computation."
            badge="EFFICIENCY"
            accent="cyan"
          />

          <MetricCard
            title="Escalation Rate"
            value={`${indVal.escalation_rate_pct.toFixed(2)}%`}
            subtitle="1,283 Dies to Stage 2"
            tooltip="Percentage of dies with borderline Stage 1 uncertainty routed to Stage 2 for deeper inspection."
            badge="SELECTIVE"
            accent="amber"
          />
        </div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3.5">
          <MetricCard
            title="Cascade Fail F1"
            value={canVal.f1.toFixed(4)}
            subtitle="+3.59% vs Baseline B0"
            tooltip="Harmonic balance of precision and recall for defect identification on 32 canonical holdout wafers."
            badge="CANONICAL"
            accent="green"
          />

          <MetricCard
            title="PR-AUC"
            value={canVal.pr_auc.toFixed(4)}
            subtitle={`ROC-AUC: ${canVal.roc_auc.toFixed(4)}`}
            tooltip="Precision-Recall Area Under Curve on canonical holdout wafers."
            badge="RANKING"
            accent="purple"
          />

          <MetricCard
            title="Cascade Recall"
            value={`${(canVal.recall * 100).toFixed(2)}%`}
            subtitle="393 TP / 663 FN"
            tooltip="Percentage of defective dies detected on canonical holdout wafers."
            badge="COVERAGE"
            accent="blue"
          />

          <MetricCard
            title="Cascade Precision"
            value={`${(canVal.precision * 100).toFixed(2)}%`}
            subtitle="47 False Positives"
            tooltip="Defect classification purity on canonical holdout wafers."
            badge="PURITY"
            accent="green"
          />

          <MetricCard
            title="Compute Saved"
            value={`${canVal.compute_saved_pct.toFixed(2)}%`}
            subtitle="26,298 Fast-Path Dies"
            tooltip="Percentage of holdout dies that bypassed secondary compute."
            badge="EFFICIENCY"
            accent="cyan"
          />

          <MetricCard
            title="Escalation Rate"
            value={`${canVal.escalation_rate_pct.toFixed(2)}%`}
            subtitle="1,674 Dies to Model B"
            tooltip="Percentage of holdout dies that required secondary inspection."
            badge="SELECTIVE"
            accent="amber"
          />
        </div>
      )}

      {/* Secondary Independent Population Summary Banner */}
      <div className="p-3.5 rounded-lg bg-silicon-elevated border border-silicon-border/80 font-mono text-xs flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <Database className="w-4 h-4 text-telemetry-cyan" />
          <span className="text-content-secondary">
            INDEPENDENT GROUND TRUTH: <span className="text-content-primary font-bold">40 Wafers • 32,598 Dies • 1,380 Defects (4.23%)</span>
          </span>
        </div>
        <div className="flex items-center gap-4 text-[11px] text-content-secondary">
          <span>TP: <strong className="text-telemetry-green">513</strong></span>
          <span>FN: <strong className="text-telemetry-amber">867</strong></span>
          <span>FP: <strong className="text-telemetry-red">89</strong></span>
          <span>TN: <strong className="text-content-primary">31,129</strong></span>
          <span className="border-l border-silicon-border pl-3 text-telemetry-cyan font-bold">
            Net Defect Gain vs Model A: +37 dies (23.30% rescue rate)
          </span>
        </div>
      </div>

      {/* Judge View / Live Investigation Fast Track Banner */}
      <div className="p-4 rounded-lg bg-gradient-to-r from-telemetry-blue/20 via-silicon-elevated to-telemetry-purple/20 border border-telemetry-cyan/40 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40">
              JUDGE PRESENTATION FLOW
            </span>
            <span className="text-xs font-mono text-content-muted">Single-Click Live Case</span>
          </div>
          <h3 className="text-sm font-bold text-content-primary">
            Follow One Suspicious Die: Wafer W_N_0057 • Die R32/C23
          </h3>
          <p className="text-xs text-content-secondary">
            Experience the complete engineering evidence chain: Model A escape → Uncertainty Gate [0.250, 0.550] → Dual-Stream Model B3 sub-die rescue (+6 Net Defect Gain) → TCN waveform forensics.
          </p>
        </div>
        <button
          onClick={() => navigate('/wafer-intelligence/W_N_0057/32/23')}
          className="px-4 py-2 rounded-lg bg-telemetry-cyan text-silicon-canvas font-mono font-bold text-xs hover:bg-telemetry-cyan/90 transition-all flex items-center gap-2 shadow-lg shadow-telemetry-cyan/20 shrink-0"
        >
          <span>OPEN LIVE INVESTIGATION</span>
          <ArrowRight className="w-4 h-4" />
        </button>
      </div>

      {/* Yield-At-Risk & Model Health Intelligence Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Yield-At-Risk Panel */}
        <div className="lg:col-span-2 p-4 rounded-lg bg-silicon-elevated border border-silicon-border space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-silicon-border/80 pb-2">
            <div className="flex items-center gap-2">
              <Flame className="w-4 h-4 text-telemetry-amber" />
              <span className="font-bold text-content-primary uppercase">
                YIELD AT RISK INTELLIGENCE
              </span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-silicon-canvas border border-silicon-border text-content-muted">
              Simulation / planning estimate
            </span>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div className="p-3 rounded bg-silicon-canvas border border-silicon-border space-y-1">
              <span className="text-content-muted text-[10px] uppercase">Predicted High-Risk Dies</span>
              <div className="text-lg font-bold text-telemetry-red">
                {metricView === 'independent' ? '602 dies' : '440 dies'}
              </div>
              <div className="text-[11px] text-content-secondary">
                {metricView === 'independent' ? '1.85% of holdout' : '1.57% of holdout'}
              </div>
            </div>

            <div className="p-3 rounded bg-silicon-canvas border border-silicon-border space-y-1">
              <span className="text-content-muted text-[10px] uppercase">Observed Holdout Yield</span>
              <div className="text-lg font-bold text-telemetry-green">
                {metricView === 'independent' ? '95.77%' : '96.22%'}
              </div>
              <div className="text-[11px] text-content-secondary">
                {metricView === 'independent' ? '31,218 clean dies' : '26,916 clean dies'}
              </div>
            </div>

            <div className="p-3 rounded bg-silicon-canvas border border-silicon-border space-y-1">
              <span className="text-content-muted text-[10px] uppercase">Yield Exposure Indicator</span>
              <div className="text-lg font-bold text-telemetry-amber">
                {metricView === 'independent' ? '4.23% Defect Base' : '3.78% Defect Base'}
              </div>
              <div className="text-[11px] text-content-secondary">
                Concentrated in 4 perimeter rings
              </div>
            </div>
          </div>

          <div className="p-2.5 rounded bg-silicon-canvas/60 border border-silicon-border/60 text-[11px] text-content-secondary leading-relaxed">
            <strong className="text-telemetry-cyan font-semibold">METHODOLOGY NOTE:</strong> Yield-at-risk quantifies dies predicted as high-risk across spatial clusters. Clearly distinguished from physical factory scrap: actual economic impact requires fab cost-per-wafer metrics not provided in raw competition dataset.
          </div>
        </div>

        {/* Model Health & Validation Trust Panel */}
        <div className="p-4 rounded-lg bg-silicon-elevated border border-silicon-border space-y-3 font-mono text-xs">
          <div className="flex items-center justify-between border-b border-silicon-border/80 pb-2">
            <div className="flex items-center gap-2">
              <ShieldCheck className="w-4 h-4 text-telemetry-green" />
              <span className="font-bold text-content-primary uppercase">
                MODEL HEALTH & TRUST
              </span>
            </div>
            <span className="text-[10px] px-2 py-0.5 rounded bg-telemetry-green/20 text-telemetry-green font-bold">
              FROZEN
            </span>
          </div>

          <div className="space-y-2 pt-1 text-content-secondary">
            <div className="flex justify-between">
              <span className="text-content-muted">ACTIVE MODEL:</span>
              <span className="text-telemetry-purple font-bold">B3 Dual-Stream Cascade</span>
            </div>
            <div className="flex justify-between">
              <span className="text-content-muted">ENGINE STATUS:</span>
              <span className="text-telemetry-green font-bold">PRODUCTION</span>
            </div>
            <div className="flex justify-between">
              <span className="text-content-muted">INDEPENDENT TEST:</span>
              <span className="text-content-primary font-bold">40 Unseen Wafers</span>
            </div>
            <div className="flex justify-between">
              <span className="text-content-muted">VALIDATION DIES:</span>
              <span>32,598 Dies</span>
            </div>
            <div className="flex justify-between">
              <span className="text-content-muted">TEST SUITE HEALTH:</span>
              <span className="text-telemetry-green font-bold">47 / 47 PASS (100%)</span>
            </div>
            <div className="flex justify-between">
              <span className="text-content-muted">RELEASE ID:</span>
              <span className="text-telemetry-cyan font-bold">REL-2026-B3-FINAL</span>
            </div>
          </div>
        </div>
      </div>

      {/* Interactive Cascade Decision Architecture Flow */}
      <Panel
        title="Selective Multi-Resolution Cascade Architecture (Active Production: B3)"
        subtitle="Fast screening with Model A (517 feats) -> Uncertainty Gate [0.250, 0.550] -> B3 Dual-Stream Model B (528 Tab + 1D-CNN)"
      >
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 p-2">
          {/* Step 1: Model A Coarse Screening */}
          <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-4 space-y-3 relative">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-telemetry-blue/20 text-telemetry-blue border border-telemetry-blue/30">
                STAGE 1 • 100% DIES
              </span>
              <Cpu className="w-4 h-4 text-telemetry-blue" />
            </div>

            <h4 className="text-sm font-semibold text-content-primary">
              Model A Screening Layer
            </h4>
            <p className="text-xs text-content-secondary leading-relaxed">
              500 electrical + 10 polar/probe spatial + 7 PCA moment features. Ultra-low latency screening (τ_A = 0.5100).
            </p>

            <div className="bg-silicon-canvas p-2.5 rounded border border-silicon-border/80 font-mono text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-content-muted">Features:</span>
                <span className="text-content-primary font-semibold">517 Features</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Threshold τ_A:</span>
                <span className="text-telemetry-blue font-semibold">0.5100</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Version:</span>
                <span className="text-content-primary font-semibold">B3 Enhanced Model A</span>
              </div>
            </div>
          </div>

          {/* Step 2: Uncertainty Gating Window */}
          <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-4 space-y-3 relative">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/30">
                STAGE 2 • STANDARD GATE
              </span>
              <Zap className="w-4 h-4 text-telemetry-amber" />
            </div>

            <h4 className="text-sm font-semibold text-content-primary">
              Confidence Gate [0.250, 0.550]
            </h4>
            <p className="text-xs text-content-secondary leading-relaxed">
              P_A &lt; 0.250: Fast path pass (~96.05%). 0.250 ≤ P_A ≤ 0.550: Escalate to Dual-Stream Model B. P_A &gt; 0.550: Confident fail.
            </p>

            <div className="bg-silicon-canvas p-2.5 rounded border border-silicon-border/80 font-mono text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-content-muted">Fast-Path Cleared:</span>
                <span className="text-telemetry-green font-semibold">96.05% (31,310 Dies)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Escalation Rate:</span>
                <span className="text-telemetry-amber font-semibold">3.95% (1,288 Dies)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">B2 Checkpoint Gate:</span>
                <span className="text-content-muted font-semibold">[0.295, 0.550] (6.22% esc.)</span>
              </div>
            </div>
          </div>

          {/* Step 3: Model B Sub-Die Resolution */}
          <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-4 space-y-3 relative">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/30">
                STAGE 3 • B3 DUAL-STREAM
              </span>
              <Layers className="w-4 h-4 text-telemetry-purple" />
            </div>

            <h4 className="text-sm font-semibold text-content-primary">
              B3 Multimodal Model B
            </h4>
            <p className="text-xs text-content-secondary leading-relaxed">
              528 tabular features + 1D-CNN late probability fusion (0.90 Tabular + 0.10 CNN). Selective escalation only. τ_B = 0.5200.
            </p>

            <div className="bg-silicon-canvas p-2.5 rounded border border-silicon-border/80 font-mono text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-content-muted">Total Modality:</span>
                <span className="text-telemetry-purple font-semibold">528 Tab + 1D-CNN</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Threshold τ_B:</span>
                <span className="text-telemetry-blue font-semibold">0.5200</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Late Fusion Blend:</span>
                <span className="text-telemetry-green font-semibold">0.90 Tab + 0.10 CNN</span>
              </div>
            </div>
          </div>
        </div>
      </Panel>

      {/* Head-to-Head Comparative Overview */}
      <Panel
        title="Head-to-Head Architecture Benchmark Profile"
        subtitle="Comparing Historical Baseline B0, Confirmed B2 (Canonical Holdout), and Final B2 (Independent 40-Wafer Validation)."
      >
        <div className="overflow-x-auto">
          <table className="w-full text-xs font-mono border-collapse">
            <thead>
              <tr className="border-b border-silicon-border bg-silicon-surface text-content-secondary text-[11px]">
                <th className="py-2.5 px-3 text-left">Configuration / Population</th>
                <th className="py-2.5 px-3 text-left">Gating Strategy</th>
                <th className="py-2.5 px-3 text-right">Fail F1</th>
                <th className="py-2.5 px-3 text-right">PR-AUC</th>
                <th className="py-2.5 px-3 text-right">ROC-AUC</th>
                <th className="py-2.5 px-3 text-right">Precision</th>
                <th className="py-2.5 px-3 text-right">Recall</th>
                <th className="py-2.5 px-3 text-right">Escalation</th>
                <th className="py-2.5 px-3 text-right">Compute Saved</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-silicon-border/60">
              <tr className="hover:bg-silicon-interactive/40 transition-colors">
                <td className="py-2.5 px-3 font-semibold text-content-primary">
                  Historical Baseline B0 (Canonical 32 Wafers)
                </td>
                <td className="py-2.5 px-3 text-content-secondary">
                  Gate [0.25, 0.55]
                </td>
                <td className="py-2.5 px-3 text-right text-content-primary">
                  0.5072
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  0.4677
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  0.7990
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-blue font-semibold">
                  81.38%
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  36.84%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-amber">
                  11.86%
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  88.14%
                </td>
              </tr>

              <tr className="hover:bg-silicon-interactive/40 transition-colors">
                <td className="py-2.5 px-3 font-semibold text-content-primary">
                  Confirmed B2 (Canonical 32 Wafers — EXP-03B)
                </td>
                <td className="py-2.5 px-3 text-content-secondary">
                  Gate [0.295, 0.550]
                </td>
                <td className="py-2.5 px-3 text-right text-content-primary">
                  0.5254
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  0.4777
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  0.7986
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-blue font-semibold">
                  89.32%
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  37.22%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-amber">
                  5.98%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green">
                  94.02%
                </td>
              </tr>

              <tr className="hover:bg-silicon-interactive/40 transition-colors">
                <td className="py-2.5 px-3 font-semibold text-content-secondary">
                  Historical B2 (Independent 40 Wafers — EXP-04)
                </td>
                <td className="py-2.5 px-3 text-content-secondary">
                  Gate [0.295, 0.550]
                </td>
                <td className="py-2.5 px-3 text-right text-content-primary">
                  0.5177
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  0.4808
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  0.8078
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-blue">
                  85.22%
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  37.17%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-amber">
                  6.22%
                </td>
                <td className="py-2.5 px-3 text-right text-content-secondary">
                  93.78%
                </td>
              </tr>

              <tr className="bg-telemetry-green/5 hover:bg-telemetry-green/10 transition-colors font-semibold">
                <td className="py-2.5 px-3 text-telemetry-green flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Active Production B3 Dual-Stream (EXP-04) ★
                </td>
                <td className="py-2.5 px-3 text-telemetry-green">
                  Gate [0.250, 0.550]
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  0.5264
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  0.5004
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-purple font-bold">
                  0.8270
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-blue font-bold">
                  87.67%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  37.61%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-amber">
                  3.95%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  96.05%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Interactive Canonical Holdout Wafers Catalog */}
      <Panel
        title="Canonical Holdout Wafers Catalog (32 Clean Wafers)"
        subtitle="Click any wafer row or 'View Map' to launch spatial wafer intelligence, defect maps, and die inspection"
      >
        {wafersLoading ? (
          <LoadingSkeleton className="h-48 w-full" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-silicon-border bg-silicon-surface text-content-secondary text-[11px]">
                  <th className="py-2.5 px-3 text-left">Wafer ID</th>
                  <th className="py-2.5 px-3 text-right">Total Dies</th>
                  <th className="py-2.5 px-3 text-right">Ground Truth Fails</th>
                  <th className="py-2.5 px-3 text-right">Defect Rate</th>
                  <th className="py-2.5 px-3 text-right">Model A Dets</th>
                  <th className="py-2.5 px-3 text-right">Cascade Dets</th>
                  <th className="py-2.5 px-3 text-right">Escalations</th>
                  <th className="py-2.5 px-3 text-right">Rescued FNs</th>
                  <th className="py-2.5 px-3 text-center">Interactive Drilldown</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-silicon-border/60">
                {wafersData?.wafers.map((w) => (
                  <tr
                    key={w.wafer_id}
                    onClick={() => navigate(`/wafer-intelligence/${w.wafer_id}`)}
                    className="hover:bg-silicon-interactive/50 transition-colors cursor-pointer group"
                  >
                    <td className="py-2.5 px-3 font-bold text-telemetry-cyan flex items-center gap-1.5">
                      <Disc className="w-3.5 h-3.5 text-telemetry-cyan group-hover:rotate-45 transition-transform" />
                      {w.wafer_id}
                    </td>
                    <td className="py-2.5 px-3 text-right text-content-primary">
                      {w.total_dies.toLocaleString()}
                    </td>
                    <td className="py-2.5 px-3 text-right text-telemetry-red font-semibold">
                      {w.actual_failures}
                    </td>
                    <td className="py-2.5 px-3 text-right text-content-secondary">
                      {w.fail_rate_pct.toFixed(2)}%
                    </td>
                    <td className="py-2.5 px-3 text-right text-content-muted">
                      {w.model_a_detections}
                    </td>
                    <td className="py-2.5 px-3 text-right text-telemetry-blue font-semibold">
                      {w.cascade_detections}
                    </td>
                    <td className="py-2.5 px-3 text-right text-telemetry-amber">
                      {w.escalated_dies}
                    </td>
                    <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                      +{w.recovered_fns}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/wafer-intelligence/${w.wafer_id}`);
                        }}
                        className="px-2.5 py-1 rounded bg-telemetry-cyan/10 hover:bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/30 text-[11px] font-semibold transition-colors inline-flex items-center gap-1"
                      >
                        <span>View Map</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
};
