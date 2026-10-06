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

  // Independent 40-wafer evaluation metrics (EXP-04)
  const indVal = release?.evaluations?.final_independent_validation?.cascade_metrics || {
    f1: 0.5177,
    f2: 0.4190,
    mcc: 0.5517,
    pr_auc: 0.4808,
    roc_auc: 0.8078,
    precision: 0.8522,
    recall: 0.3717,
    tp: 513,
    fn: 867,
    fp: 89,
    tn: 31129,
    escalation_rate_pct: 6.22,
    compute_saved_pct: 93.78,
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
              MODEL RELEASE: B2 ENHANCED CASCADE
            </span>
            <span className="text-content-muted">•</span>
            <span className="text-xs font-mono text-telemetry-green">
              STATUS: FINAL VALIDATED CANDIDATE
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
          <StatusBadge label="Release: B2 Candidate" variant="purple" pulse />
          <StatusBadge label="Gate B: [0.295, 0.550]" variant="pass" />
          <StatusBadge label="Zero-Leakage Compliance" variant="info" />
        </div>
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
            className={`px-2.5 py-1 rounded transition-colors ${
              metricView === 'independent'
                ? 'bg-telemetry-cyan/20 text-telemetry-cyan font-bold border border-telemetry-cyan/40'
                : 'text-content-muted hover:text-content-primary'
            }`}
          >
            Independent 40 Wafers (EXP-04)
          </button>
          <button
            onClick={() => setMetricView('canonical')}
            className={`px-2.5 py-1 rounded transition-colors ${
              metricView === 'canonical'
                ? 'bg-telemetry-green/20 text-telemetry-green font-bold border border-telemetry-green/40'
                : 'text-content-muted hover:text-content-primary'
            }`}
          >
            Canonical 32 Wafers (EXP-03B)
          </button>
        </div>
      </div>

      {/* 6 Hero Metric Cards */}
      {metricView === 'independent' ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3.5">
          <MetricCard
            title="Cascade Fail F1"
            value={indVal.f1.toFixed(4)}
            subtitle="95% CI: [0.4948, 0.5369]"
            badge="INDEPENDENT"
            accent="green"
          />

          <MetricCard
            title="PR-AUC"
            value={indVal.pr_auc.toFixed(4)}
            subtitle={`ROC-AUC: ${indVal.roc_auc.toFixed(4)}`}
            badge="RANKING"
            accent="purple"
          />

          <MetricCard
            title="Cascade Recall"
            value={`${(indVal.recall * 100).toFixed(2)}%`}
            subtitle="513 TP / 867 FN"
            badge="COVERAGE"
            accent="blue"
          />

          <MetricCard
            title="Cascade Precision"
            value={`${(indVal.precision * 100).toFixed(2)}%`}
            subtitle="89 False Positives"
            badge="PURITY"
            accent="green"
          />

          <MetricCard
            title="Compute Saved"
            value={`${indVal.compute_saved_pct.toFixed(2)}%`}
            subtitle="30,571 Fast-Path Dies"
            badge="EFFICIENCY"
            accent="cyan"
          />

          <MetricCard
            title="Escalation Rate"
            value={`${indVal.escalation_rate_pct.toFixed(2)}%`}
            subtitle="2,027 Dies to Model B"
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
            badge="CANONICAL"
            accent="green"
          />

          <MetricCard
            title="PR-AUC"
            value={canVal.pr_auc.toFixed(4)}
            subtitle={`ROC-AUC: ${canVal.roc_auc.toFixed(4)}`}
            badge="RANKING"
            accent="purple"
          />

          <MetricCard
            title="Cascade Recall"
            value={`${(canVal.recall * 100).toFixed(2)}%`}
            subtitle="393 TP / 663 FN"
            badge="COVERAGE"
            accent="blue"
          />

          <MetricCard
            title="Cascade Precision"
            value={`${(canVal.precision * 100).toFixed(2)}%`}
            subtitle="47 False Positives"
            badge="PURITY"
            accent="green"
          />

          <MetricCard
            title="Compute Saved"
            value={`${canVal.compute_saved_pct.toFixed(2)}%`}
            subtitle="26,298 Fast-Path Dies"
            badge="EFFICIENCY"
            accent="cyan"
          />

          <MetricCard
            title="Escalation Rate"
            value={`${canVal.escalation_rate_pct.toFixed(2)}%`}
            subtitle="1,674 Dies to Model B"
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

      {/* Interactive Cascade Decision Architecture Flow */}
      <Panel
        title="Selective Multi-Resolution Cascade Architecture (Active Production: B2)"
        subtitle="Fast screening with Model A -> Uncertainty Gate [0.295, 0.550] -> B2 Enhanced Sub-Die Model B"
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
              500 electrical features + 6 spatial features (restored pre-test spatial defect density). Ultra-low latency GBDT classifier (τ_A = 0.5100).
            </p>

            <div className="bg-silicon-canvas p-2.5 rounded border border-silicon-border/80 font-mono text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-content-muted">Features:</span>
                <span className="text-content-primary font-semibold">506 Features</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Threshold τ_A:</span>
                <span className="text-telemetry-blue font-semibold">0.5100</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Version:</span>
                <span className="text-content-primary font-semibold">EXP-01 Spatial-Restored</span>
              </div>
            </div>
          </div>

          {/* Step 2: Uncertainty Gating Window */}
          <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-4 space-y-3 relative">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/30">
                STAGE 2 • ASYMMETRIC GATE
              </span>
              <Zap className="w-4 h-4 text-telemetry-amber" />
            </div>

            <h4 className="text-sm font-semibold text-content-primary">
              Confidence Gate [0.295, 0.550]
            </h4>
            <p className="text-xs text-content-secondary leading-relaxed">
              P_A &lt; 0.295: Fast path bypass (~93.78%). 0.295 ≤ P_A &lt; 0.550: Escalate to B2. P_A ≥ 0.550: Model A confident failure.
            </p>

            <div className="bg-silicon-canvas p-2.5 rounded border border-silicon-border/80 font-mono text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-content-muted">Fast-Path Cleared:</span>
                <span className="text-telemetry-green font-semibold">93.78% (30,571 Dies)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Escalation Rate:</span>
                <span className="text-telemetry-amber font-semibold">6.22% (2,027 Dies)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Historical Baseline:</span>
                <span className="text-content-muted font-semibold">[0.25, 0.55] (11.86% esc.)</span>
              </div>
            </div>
          </div>

          {/* Step 3: Model B Sub-Die Resolution */}
          <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-4 space-y-3 relative">
            <div className="flex items-center justify-between">
              <span className="text-[11px] font-mono font-semibold px-2 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/30">
                STAGE 3 • FROZEN B2 MODEL
              </span>
              <Layers className="w-4 h-4 text-telemetry-purple" />
            </div>

            <h4 className="text-sm font-semibold text-content-primary">
              B2 Enhanced Model B
            </h4>
            <p className="text-xs text-content-secondary leading-relaxed">
              536 features: 500 electrical + 6 spatial + 20 legacy block + 10 engineered candidates (cnt_110/115/120, cluster_area, roll20, prominence, top10). τ_B = 0.5376.
            </p>

            <div className="bg-silicon-canvas p-2.5 rounded border border-silicon-border/80 font-mono text-xs space-y-1">
              <div className="flex justify-between">
                <span className="text-content-muted">Total Features:</span>
                <span className="text-telemetry-purple font-semibold">536 Features</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Threshold τ_B:</span>
                <span className="text-telemetry-blue font-semibold">0.5376</span>
              </div>
              <div className="flex justify-between">
                <span className="text-content-muted">Rescued FN (EXP-04):</span>
                <span className="text-telemetry-green font-semibold">41 / 176 (23.30% rescue)</span>
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

              <tr className="bg-telemetry-green/5 hover:bg-telemetry-green/10 transition-colors font-semibold">
                <td className="py-2.5 px-3 text-telemetry-green flex items-center gap-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Final B2 (Independent 40 Wafers — EXP-04) ★
                </td>
                <td className="py-2.5 px-3 text-telemetry-green">
                  Gate [0.295, 0.550]
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  0.5177
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  0.4808
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-purple font-bold">
                  0.8078
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-blue">
                  85.22%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green">
                  37.17%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-amber">
                  6.22%
                </td>
                <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                  93.78%
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
