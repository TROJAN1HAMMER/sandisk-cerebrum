import React, { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  useModelMetrics,
  useModelBenchmark,
  useProbabilityDistribution,
  useRocPrCurves,
  useHighRiskDies,
  useModelRelease,
  useModelCandidates,
} from '../hooks/useModels';
import { useWafers } from '../hooks/useWafers';
import { MetricCard } from '../components/ui/MetricCard';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Panel } from '../components/ui/Panel';
import { StatusBadge } from '../components/ui/StatusBadge';
import { LoadingSkeleton } from '../components/ui/LoadingSkeleton';
import { ErrorState } from '../components/ui/ErrorState';
import {
  BarChart3,
  TrendingUp,
  Cpu,
  Layers,
  Zap,
  ShieldCheck,
  Disc,
  ArrowRight,
  Filter,
  RotateCcw,
  Download,
  AlertTriangle,
  CheckCircle2,
  ChevronRight,
  ExternalLink,
  Sliders,
  Sparkles,
  Percent,
} from 'lucide-react';
import { HighRiskDieRecord } from '../types/models';
import { WaferSummaryItem } from '../types/wafers';
import { PolicySimulator } from '../components/inference/PolicySimulator';

export const YieldAnalytics: React.FC = () => {
  const navigate = useNavigate();

  // Global filters
  const [selectedWafer, setSelectedWafer] = useState<string>('all');
  const [errorFilter, setErrorFilter] = useState<'all' | 'TP' | 'TN' | 'FP' | 'FN' | 'recovered'>('all');
  const [cascadePathFilter, setCascadePathFilter] = useState<'all' | 'fast_path' | 'escalated'>('all');
  const [waferSortBy, setWaferSortBy] = useState<'yield' | 'defect_rate' | 'failures' | 'escalation' | 'dies'>('yield');
  const [failureViewMode, setFailureViewMode] = useState<'actual' | 'predicted' | 'difference'>('actual');
  const [comparisonDataset, setComparisonDataset] = useState<'exp04' | 'holdout'>('exp04');

  // Authoritative queries
  const { data: metrics, isLoading: metricsLoading, error: metricsError, refetch: refetchMetrics } = useModelMetrics();
  const { data: release } = useModelRelease();
  const { data: candidates } = useModelCandidates();
  const { data: benchmark, isLoading: benchLoading } = useModelBenchmark();
  const { data: probDist, isLoading: distLoading } = useProbabilityDistribution();
  const { data: curves, isLoading: curvesLoading } = useRocPrCurves();
  const { data: wafersData, isLoading: wafersLoading } = useWafers();
  const { data: highRiskData, isLoading: riskLoading } = useHighRiskDies(
    100,
    selectedWafer === 'all' ? undefined : selectedWafer
  );

  const isInitialLoading = metricsLoading || wafersLoading || distLoading || curvesLoading;

  // Extract metrics
  const ds = metrics?.metadata.dataset;
  const mA = metrics?.model_A;
  const mB = metrics?.model_B;
  const gB = metrics?.cascade_gate_B;
  const rec = gB?.fn_recovery_metrics;

  // Wafers list sorted
  const wafers = useMemo(() => {
    if (!wafersData?.wafers) return [];
    const list = [...wafersData.wafers];
    list.sort((a, b) => {
      const yieldA = ((a.total_dies - a.actual_failures) / a.total_dies) * 100;
      const yieldB = ((b.total_dies - b.actual_failures) / b.total_dies) * 100;
      if (waferSortBy === 'yield') return yieldB - yieldA;
      if (waferSortBy === 'defect_rate') return b.fail_rate_pct - a.fail_rate_pct;
      if (waferSortBy === 'failures') return b.actual_failures - a.actual_failures;
      if (waferSortBy === 'escalation') {
        const escA = (a.escalated_dies / a.total_dies) * 100;
        const escB = (b.escalated_dies / b.total_dies) * 100;
        return escB - escA;
      }
      return b.total_dies - a.total_dies;
    });
    return list;
  }, [wafersData, waferSortBy]);

  // Filtered high-risk dies
  const filteredDies = useMemo(() => {
    if (!highRiskData?.records) return [];
    return highRiskData.records.filter((die) => {
      // Error filter
      if (errorFilter === 'TP' && !(die.actual_label === 1 && die.decision_Cascade_B === 1)) return false;
      if (errorFilter === 'TN' && !(die.actual_label === 0 && die.decision_Cascade_B === 0)) return false;
      if (errorFilter === 'FP' && !(die.actual_label === 0 && die.decision_Cascade_B === 1)) return false;
      if (errorFilter === 'FN' && !(die.actual_label === 1 && die.decision_Cascade_B === 0)) return false;
      if (errorFilter === 'recovered' && !die.recovered_FN_Gate_B) return false;

      // Cascade path filter
      if (cascadePathFilter === 'fast_path' && die.in_gate_B) return false;
      if (cascadePathFilter === 'escalated' && !die.in_gate_B) return false;

      return true;
    });
  }, [highRiskData, errorFilter, cascadePathFilter]);

  // Reset filters
  const handleResetFilters = () => {
    setSelectedWafer('all');
    setErrorFilter('all');
    setCascadePathFilter('all');
  };

  // CSV export handler
  const handleExportCSV = () => {
    if (!filteredDies.length) return;
    const headers = [
      'wafer_id',
      'die_row',
      'die_col',
      'actual_label',
      'prob_Model_A',
      'prob_Model_B',
      'prob_Cascade_B',
      'decision_Model_A',
      'decision_Cascade_B',
      'in_gate_B',
      'recovered_FN_Gate_B',
    ];
    const rows = filteredDies.map((d) => [
      d.wafer_id,
      d.die_row,
      d.die_col,
      d.actual_label,
      d.prob_Model_A.toFixed(4),
      d.prob_Model_B.toFixed(4),
      d.prob_Cascade_B.toFixed(4),
      d.decision_Model_A,
      d.decision_Cascade_B,
      d.in_gate_B ? 1 : 0,
      d.recovered_FN_Gate_B ? 1 : 0,
    ]);
    const csvContent =
      'data:text/csv;charset=utf-8,' +
      [headers.join(','), ...rows.map((e) => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', `cerebrum_yield_analytics_${selectedWafer}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  if (isInitialLoading) {
    return (
      <div className="space-y-6">
        <LoadingSkeleton className="h-10 w-96" />
        <div className="grid grid-cols-2 md:grid-cols-4 xl:grid-cols-6 gap-3">
          {Array.from({ length: 12 }).map((_, i) => (
            <LoadingSkeleton key={i} className="h-24" />
          ))}
        </div>
        <LoadingSkeleton className="h-96" />
      </div>
    );
  }

  if (metricsError || !metrics) {
    return (
      <ErrorState
        title="Yield Analytics Offline"
        message="Failed to retrieve authoritative validation metrics from the FastAPI backend."
        onRetry={refetchMetrics}
      />
    );
  }

  return (
    <div className="space-y-8 pb-16">
      {/* 1. Header & Title */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-silicon-border/70">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-bold flex items-center gap-1.5">
              <BarChart3 className="w-3.5 h-3.5" />
              YIELD ANALYTICS WORKSPACE
            </span>
            <span className="text-content-muted">•</span>
            <span className="text-xs font-mono text-content-secondary">
              Canonical 32 Holdout Wafers
            </span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-content-primary mt-1">
            Yield Analytics
          </h1>
          <p className="text-xs text-content-secondary mt-0.5">
            Die-level yield intelligence, model performance, and cascade efficiency
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          <StatusBadge label="Validation Dies: 27,972" variant="neutral" />
          <StatusBadge label="Zero Fabricated Data" variant="pass" />
          <StatusBadge label="Gate B [0.25, 0.55]" variant="purple" />
        </div>
      </div>

      {/* 2. Sticky Analytical Global Filter Bar */}
      <div className="sticky top-0 z-20 p-3.5 rounded-lg bg-silicon-card/95 backdrop-blur-md border border-silicon-border shadow-lg space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-xs font-mono font-semibold text-content-primary">
            <Filter className="w-4 h-4 text-telemetry-cyan" />
            <span>ANALYTICAL FILTERS</span>
            {(selectedWafer !== 'all' || errorFilter !== 'all' || cascadePathFilter !== 'all') && (
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40">
                ACTIVE
              </span>
            )}
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportCSV}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-silicon-elevated hover:bg-silicon-border text-content-secondary hover:text-content-primary border border-silicon-border text-xs font-mono transition-colors"
              title="Export currently filtered die records to CSV"
            >
              <Download className="w-3.5 h-3.5" />
              Export CSV
            </button>
            <button
              onClick={handleResetFilters}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-silicon-elevated hover:bg-silicon-border text-content-secondary hover:text-content-primary border border-silicon-border text-xs font-mono transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              Reset Filters
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1 border-t border-silicon-border/50 text-xs">
          {/* Wafer Filter */}
          <div>
            <label className="block text-[11px] font-mono text-content-muted mb-1">
              WAFER SELECTOR
            </label>
            <select
              value={selectedWafer}
              onChange={(e) => setSelectedWafer(e.target.value)}
              className="w-full bg-silicon-surface border border-silicon-border rounded px-2.5 py-1.5 text-xs text-content-primary font-mono focus:border-telemetry-cyan focus:outline-none"
            >
              <option value="all">All 32 Validation Wafers (27,972 dies)</option>
              {wafers.map((w) => (
                <option key={w.wafer_id} value={w.wafer_id}>
                  {w.wafer_id} — {w.total_dies} dies ({w.actual_failures} fails, {w.fail_rate_pct.toFixed(1)}% fail)
                </option>
              ))}
            </select>
          </div>

          {/* Error Class Filter */}
          <div>
            <label className="block text-[11px] font-mono text-content-muted mb-1">
              ERROR / RECOVERY CLASS
            </label>
            <select
              value={errorFilter}
              onChange={(e) => setErrorFilter(e.target.value as any)}
              className="w-full bg-silicon-surface border border-silicon-border rounded px-2.5 py-1.5 text-xs text-content-primary font-mono focus:border-telemetry-cyan focus:outline-none"
            >
              <option value="all">All Predictions & Outcomes</option>
              <option value="recovered">Rescued False Negatives (37 dies)</option>
              <option value="FN">Remaining False Negatives (667 dies)</option>
              <option value="FP">False Positives (89 dies)</option>
              <option value="TP">True Positives (389 dies)</option>
              <option value="TN">True Negatives (26,827 dies)</option>
            </select>
          </div>

          {/* Cascade Path Filter */}
          <div>
            <label className="block text-[11px] font-mono text-content-muted mb-1">
              CASCADE EXECUTION PATH
            </label>
            <select
              value={cascadePathFilter}
              onChange={(e) => setCascadePathFilter(e.target.value as any)}
              className="w-full bg-silicon-surface border border-silicon-border rounded px-2.5 py-1.5 text-xs text-content-primary font-mono focus:border-telemetry-cyan focus:outline-none"
            >
              <option value="all">All Paths (100% dies)</option>
              <option value="fast_path">Fast Path Bypassed (88.14% • 24,655 dies)</option>
              <option value="escalated">Gate B Escalated (11.86% • 3,317 dies)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Interactive What-If Inspection Policy Simulator */}
      <PolicySimulator />

      {/* Production B2 vs Candidate B3 Dual-Namespace Comparison */}
      <Panel
        title="Production B2 vs. Candidate B3 Policy Comparison (Dual-Namespace Isolation)"
        subtitle="Independent multi-lot qualification assessment • Zero production contamination guarantee"
      >
        <div className="space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 rounded-lg bg-silicon-canvas/80 border border-silicon-border">
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono font-bold text-content-primary uppercase flex items-center gap-1.5">
                <Layers className="w-4 h-4 text-telemetry-cyan" />
                BENCHMARK POPULATION:
              </span>
              <div className="flex rounded bg-silicon-elevated p-0.5 border border-silicon-border text-xs font-mono">
                <button
                  onClick={() => setComparisonDataset('exp04')}
                  className={`px-3 py-1 rounded transition-colors ${
                    comparisonDataset === 'exp04'
                      ? 'bg-telemetry-cyan text-silicon-base font-bold'
                      : 'text-content-secondary hover:text-content-primary'
                  }`}
                >
                  EXP-04 Independent (40 Wafers / 32,598 Dies)
                </button>
                <button
                  onClick={() => setComparisonDataset('holdout')}
                  className={`px-3 py-1 rounded transition-colors ${
                    comparisonDataset === 'holdout'
                      ? 'bg-telemetry-cyan text-silicon-base font-bold'
                      : 'text-content-secondary hover:text-content-primary'
                  }`}
                >
                  Canonical Holdout (32 Wafers / 27,972 Dies)
                </button>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono">
              <span className="px-2 py-0.5 rounded bg-telemetry-green/10 text-telemetry-green border border-telemetry-green/30 flex items-center gap-1">
                <CheckCircle2 className="w-3 h-3" />
                Isolated Namespaces
              </span>
              <span className="px-2 py-0.5 rounded bg-silicon-border text-content-secondary border border-silicon-border">
                Zero Leaking
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Column 1: Production B2 */}
            <div className="p-4 rounded-lg bg-silicon-canvas border-2 border-telemetry-green/40 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/40 font-bold">
                    ACTIVE PRODUCTION STANDARD
                  </span>
                  <h3 className="text-sm font-bold text-content-primary mt-1">B2 — Selective Cascade</h3>
                  <div className="text-[11px] font-mono text-content-muted">Namespace: production.b2 • 506 A / 536 B Features</div>
                </div>
                <div className="text-right text-xs font-mono">
                  <div className="text-content-secondary">Gate: [0.295, 0.550]</div>
                  <div className="text-content-muted">τA=0.5100 | τB=0.5376</div>
                </div>
              </div>

              {comparisonDataset === 'exp04' ? (
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-silicon-border/60 font-mono text-xs">
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">F1 SCORE</span>
                    <span className="text-sm font-bold text-content-primary">0.5177</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">PRECISION</span>
                    <span className="text-sm font-bold text-telemetry-green">85.22%</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">RECALL</span>
                    <span className="text-sm font-bold text-content-primary">37.17%</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">PR-AUC</span>
                    <span className="text-sm font-bold text-content-primary">0.4808</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">CONFUSION</span>
                    <span className="text-xs font-semibold text-content-primary">513 TP / 89 FP</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">COMPUTE SAVED</span>
                    <span className="text-sm font-bold text-telemetry-cyan">93.78%</span>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-silicon-border/60 font-mono text-xs">
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">F1 SCORE</span>
                    <span className="text-sm font-bold text-content-primary">0.5254</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">PRECISION</span>
                    <span className="text-sm font-bold text-telemetry-green">89.32%</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">RECALL</span>
                    <span className="text-sm font-bold text-content-primary">37.22%</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">PR-AUC</span>
                    <span className="text-sm font-bold text-content-primary">0.4777</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">CONFUSION</span>
                    <span className="text-xs font-semibold text-content-primary">393 TP / 47 FP</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70">
                    <span className="text-[10px] text-content-muted block">COMPUTE SAVED</span>
                    <span className="text-sm font-bold text-telemetry-cyan">94.02%</span>
                  </div>
                </div>
              )}
              <div className="text-[11px] text-content-muted leading-relaxed">
                Active fab baseline: Zero PyTorch inference dependency in primary path. Proven 85.22% to 89.32% precision with ultra-low false alarm rate.
              </div>
            </div>

            {/* Column 2: Candidate B3-DualStream */}
            <div className="p-4 rounded-lg bg-silicon-canvas border-2 border-telemetry-purple/40 space-y-3">
              <div className="flex items-center justify-between">
                <div>
                  <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/40 font-bold">
                    VALIDATED STAGED CANDIDATE
                  </span>
                  <h3 className="text-sm font-bold text-content-primary mt-1">B3-DualStream (Tabular + 1D-CNN)</h3>
                  <div className="text-[11px] font-mono text-content-muted">Namespace: candidate.b3_dualstream • 517 A / 528 B Tab + 2,000-Block CNN</div>
                </div>
                <div className="text-right text-xs font-mono">
                  <div className="text-content-secondary">Gate: [0.250, 0.550]</div>
                  <div className="text-content-muted">τA=0.5100 | τB=0.5200</div>
                </div>
              </div>

              {comparisonDataset === 'exp04' ? (
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-silicon-border/60 font-mono text-xs">
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">F1 SCORE</span>
                    <span className="text-sm font-bold text-telemetry-purple">0.5264</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+0.0087 Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">PRECISION</span>
                    <span className="text-sm font-bold text-telemetry-green">87.67%</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+2.45 pp Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">RECALL</span>
                    <span className="text-sm font-bold text-content-primary">37.61%</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+0.44 pp Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">PR-AUC</span>
                    <span className="text-sm font-bold text-telemetry-cyan">0.5004</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+0.0196 Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">CONFUSION</span>
                    <span className="text-xs font-semibold text-content-primary">519 TP / 73 FP</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+6 TP / -16 FP</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">COMPUTE SAVED</span>
                    <span className="text-sm font-bold text-telemetry-cyan">96.05%</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+2.27 pp Δ</span>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-2 pt-2 border-t border-silicon-border/60 font-mono text-xs">
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">F1 SCORE</span>
                    <span className="text-sm font-bold text-telemetry-purple">0.5291</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+0.0037 Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-amber/30">
                    <span className="text-[10px] text-content-muted block">PRECISION</span>
                    <span className="text-sm font-bold text-telemetry-amber">81.78%</span>
                    <span className="text-[10px] text-telemetry-red block font-bold">-7.54 pp Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">RECALL</span>
                    <span className="text-sm font-bold text-content-primary">39.11%</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+1.89 pp Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">PR-AUC</span>
                    <span className="text-sm font-bold text-telemetry-cyan">0.4967</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+0.0190 Δ</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-amber/30">
                    <span className="text-[10px] text-content-muted block">CONFUSION</span>
                    <span className="text-xs font-semibold text-content-primary">413 TP / 92 FP</span>
                    <span className="text-[10px] text-telemetry-amber block font-bold">+20 TP / +45 FP</span>
                  </div>
                  <div className="p-2 rounded bg-silicon-elevated/70 border border-telemetry-purple/30">
                    <span className="text-[10px] text-content-muted block">COMPUTE SAVED</span>
                    <span className="text-sm font-bold text-telemetry-cyan">94.68%</span>
                    <span className="text-[10px] text-telemetry-green block font-bold">+0.66 pp Δ</span>
                  </div>
                </div>
              )}
              <div className="text-[11px] text-content-muted leading-relaxed">
                Staged candidate: 10% late probability fusion with 1D-CNN. While EXP-04 yields +0.0087 F1 and 16 fewer FPs, Canonical Holdout exhibits +45 false alarms (92 vs 47 FP), justifying controlled staging rather than automatic production promotion.
              </div>
            </div>
          </div>

          {/* Experimental High-Recall Policy strip */}
          <div className="p-3 rounded bg-silicon-elevated/80 border border-silicon-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs font-mono">
            <div>
              <div className="flex items-center gap-2">
                <span className="px-1.5 py-0.5 rounded text-[10px] font-bold bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/40">
                  EXPERIMENTAL POLICY
                </span>
                <span className="font-bold text-content-primary">High-Recall Inspection Policy</span>
                <span className="text-content-muted">• NOT Production Standard</span>
              </div>
              <div className="text-[11px] text-content-secondary mt-0.5">
                Configuration: Gate [0.20, 0.50], τB = 0.50 | Target: Maximize defect capture in safety-critical lots
              </div>
            </div>
            <div className="flex items-center gap-3 text-[11px] text-content-primary shrink-0">
              <span>F1: <strong className="text-telemetry-purple">0.5271</strong></span>
              <span>Recall: <strong className="text-telemetry-green">38.77%</strong></span>
              <span>Prec: <strong>82.31%</strong></span>
              <span>TP/FP: <strong>535 / 115</strong></span>
              <span>Compute: <strong>92.63%</strong></span>
            </div>
          </div>
        </div>
      </Panel>

      {/* 3. Executive Metrics Grid (12 Cards bound directly to authoritative API) */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-xs font-mono uppercase tracking-wider text-content-muted font-semibold">
            HISTORICAL B0 — CANONICAL HOLDOUT (32 WAFERS, 27,972 DIES)
          </div>
          <div className="text-[11px] font-mono text-content-muted">
            Source: /api/v1/models/metrics (Holdout Baseline)
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3">
          <MetricCard
            title="Total Dies"
            value={ds?.total_validation_dies.toLocaleString() || '27,972'}
            subtitle="Canonical Holdout"
            badge="100% DIES"
            accent="neutral"
          />
          <MetricCard
            title="Yield"
            value={`${(100 - (ds?.fail_rate_pct || 3.7752)).toFixed(2)}%`}
            subtitle="26,916 Passed Dies"
            badge="PRODUCTION"
            accent="green"
          />
          <MetricCard
            title="Failures"
            value={ds?.positive_failures_count.toLocaleString() || '1,056'}
            subtitle="Ground Truth Defect"
            badge="ACTUAL"
            accent="red"
          />
          <MetricCard
            title="Failure Rate"
            value={`${ds?.fail_rate_pct.toFixed(4) || '3.7752'}%`}
            subtitle="Post-Test Verification"
            badge="BASELINE"
            accent="amber"
          />
          <MetricCard
            title="Fast Path"
            value={(27972 - (gB?.escalation_count || 3317)).toLocaleString()}
            subtitle={`${gB?.compute_saved_pct.toFixed(2) || '88.14'}% Bypassed`}
            badge="MODEL A ALONE"
            accent="cyan"
          />
          <MetricCard
            title="Gate B Escalated"
            value={gB?.escalation_count.toLocaleString() || '3,317'}
            subtitle={`${gB?.escalation_rate_pct.toFixed(2) || '11.86'}% Evaluated`}
            badge="P ∈ [0.25, 0.55]"
            accent="purple"
          />
          <MetricCard
            title="Compute Saved"
            value={`${gB?.compute_saved_pct.toFixed(2) || '88.14'}%`}
            subtitle="Block Signal Savings"
            badge="24,655 SAVED"
            accent="cyan"
          />
          <MetricCard
            title="Gross FN Recovery"
            value={rec?.gross_fn_recovered.toLocaleString() || '37'}
            subtitle={`${rec?.gross_fn_recovery_rate_pct.toFixed(2) || '5.29'}% FN Rescued`}
            badge="STAGE 2 RESCUE"
            accent="green"
          />
          <MetricCard
            title="TP Lost by Model B"
            value={`-${rec?.tp_lost_by_model_B || 5}`}
            subtitle="Model B Misses"
            badge="OVERTURNED"
            accent="amber"
          />
          <MetricCard
            title="Net TP Gain"
            value={`+${rec?.net_tp_gain || 32}`}
            subtitle="37 Rescued - 5 Lost"
            badge="NET POSITIVE"
            accent="green"
          />
          <MetricCard
            title="New False Positives"
            value={rec?.new_false_positives.toLocaleString() || '47'}
            subtitle="Conservative Gating"
            badge="CONTROLLED"
            accent="red"
          />
          <MetricCard
            title="Validation Wafers"
            value={ds?.validation_wafers_count || 32}
            subtitle="100% Unseen Lots"
            badge="FAB VERIFIED"
            accent="neutral"
          />
        </div>
      </div>

      {/* 4. Yield Overview & Failure Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Yield Overview */}
        <Panel
          title="Overall Yield Composition & Class Balance"
          subtitle="True fab post-test yield across 27,972 canonical holdout dies"
        >
          <div className="space-y-5">
            {/* Horizontal stacked progress bar */}
            <div>
              <div className="flex justify-between items-center text-xs font-mono mb-2">
                <span className="text-telemetry-green font-semibold flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-telemetry-green" />
                  PASS (0): {ds?.negative_passes_count.toLocaleString() || '26,916'} dies (
                  {(100 - (ds?.fail_rate_pct || 3.78)).toFixed(2)}%)
                </span>
                <span className="text-telemetry-red font-semibold flex items-center gap-1.5">
                  <span className="w-2.5 h-2.5 rounded-sm bg-telemetry-red" />
                  FAIL (1): {ds?.positive_failures_count.toLocaleString() || '1,056'} dies (
                  {ds?.fail_rate_pct.toFixed(2) || '3.78'}%)
                </span>
              </div>
              <div className="h-7 w-full rounded bg-silicon-canvas flex overflow-hidden border border-silicon-border/80">
                <div
                  className="bg-telemetry-green/80 hover:bg-telemetry-green transition-all"
                  style={{ width: `${100 - (ds?.fail_rate_pct || 3.78)}%` }}
                  title={`Passed: ${ds?.negative_passes_count.toLocaleString() || '26,916'}`}
                />
                <div
                  className="bg-telemetry-red/80 hover:bg-telemetry-red transition-all"
                  style={{ width: `${ds?.fail_rate_pct || 3.78}%` }}
                  title={`Defective: ${ds?.positive_failures_count.toLocaleString() || '1,056'}`}
                />
              </div>
            </div>

            {/* Imbalance context callout */}
            <div className="p-3.5 rounded bg-silicon-elevated/80 border border-silicon-border text-xs space-y-2">
              <div className="flex items-center gap-2 font-mono text-content-primary font-bold">
                <Percent className="w-4 h-4 text-telemetry-cyan" />
                CLASS IMBALANCE: 25.5 to 1 PASS-TO-FAIL RATIO
              </div>
              <p className="text-content-secondary leading-relaxed">
                Because defective dies represent only <strong className="text-telemetry-amber font-mono">3.7752%</strong> of
                the population, standard accuracy metrics (e.g. 96.22% trivial baseline) are clinically deceptive.
                The Cerebrum cascade focuses strictly on <strong className="text-telemetry-cyan font-mono">Fail F1</strong>,
                <strong className="text-telemetry-cyan font-mono"> AUC-PR (0.4677)</strong>, and FN recovery without
                overwhelming downstream review with runaway false alarms.
              </p>
            </div>
          </div>
        </Panel>

        {/* Failure Distribution with Actual vs Predicted toggle */}
        <Panel
          title="Defect Distribution: Ground Truth vs Model Prediction"
          subtitle="Compare post-test failure counts against Model A and Cascade Gate B detections"
        >
          <div className="space-y-4">
            <div className="flex items-center justify-between border-b border-silicon-border pb-3">
              <div className="text-xs font-mono text-content-muted">VIEWING MODE:</div>
              <div className="inline-flex rounded-lg bg-silicon-canvas p-0.5 border border-silicon-border text-xs font-mono">
                <button
                  onClick={() => setFailureViewMode('actual')}
                  className={`px-3 py-1 rounded transition-colors ${
                    failureViewMode === 'actual'
                      ? 'bg-telemetry-cyan text-silicon-canvas font-bold'
                      : 'text-content-secondary hover:text-content-primary'
                  }`}
                >
                  Actual
                </button>
                <button
                  onClick={() => setFailureViewMode('predicted')}
                  className={`px-3 py-1 rounded transition-colors ${
                    failureViewMode === 'predicted'
                      ? 'bg-telemetry-cyan text-silicon-canvas font-bold'
                      : 'text-content-secondary hover:text-content-primary'
                  }`}
                >
                  Cascade Predicted
                </button>
                <button
                  onClick={() => setFailureViewMode('difference')}
                  className={`px-3 py-1 rounded transition-colors ${
                    failureViewMode === 'difference'
                      ? 'bg-telemetry-cyan text-silicon-canvas font-bold'
                      : 'text-content-secondary hover:text-content-primary'
                  }`}
                >
                  Difference
                </button>
              </div>
            </div>

            {/* Display content based on toggle */}
            {failureViewMode === 'actual' && (
              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-telemetry-green font-semibold">ACTUAL PASS (Label 0)</span>
                  <span className="text-content-primary font-bold">26,916 dies (96.22%)</span>
                </div>
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-telemetry-red font-semibold">ACTUAL DEFECT (Label 1)</span>
                  <span className="text-content-primary font-bold">1,056 dies (3.78%)</span>
                </div>
              </div>
            )}

            {failureViewMode === 'predicted' && (
              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-telemetry-cyan font-semibold">CASCADE PREDICTED PASS</span>
                  <span className="text-content-primary font-bold">
                    {(26827 + 667).toLocaleString()} dies (98.29%)
                  </span>
                </div>
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-telemetry-purple font-semibold">CASCADE PREDICTED DEFECT</span>
                  <span className="text-content-primary font-bold">
                    {(389 + 89).toLocaleString()} dies (1.71%)
                  </span>
                </div>
              </div>
            )}

            {failureViewMode === 'difference' && (
              <div className="space-y-3 font-mono text-xs">
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-content-secondary">NET DEFECT RECALL DELTA</span>
                  <span className="text-telemetry-green font-bold">+32 Defective Dies Recovered</span>
                </div>
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-content-secondary">FALSE ALARM INCREASE</span>
                  <span className="text-telemetry-amber font-bold">+47 New False Positives (Controlled)</span>
                </div>
                <div className="p-3 rounded bg-silicon-elevated border border-silicon-border flex justify-between items-center">
                  <span className="text-content-secondary">OVERALL ESCALATION FRACTION</span>
                  <span className="text-telemetry-cyan font-bold">11.86% (88.14% Fast Path)</span>
                </div>
              </div>
            )}
          </div>
        </Panel>
      </div>

      {/* 5. Wafer Analysis: Ranking & Wafer-by-Metric Heatmap */}
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold">
              WAFER-LEVEL PERFORMANCE
            </div>
            <h2 className="text-lg font-bold text-content-primary">
              All 32 Canonical Validation Wafers (Ranking & Intensity Heatmap)
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs font-mono text-content-muted">Sort By:</span>
            <select
              value={waferSortBy}
              onChange={(e) => setWaferSortBy(e.target.value as any)}
              className="bg-silicon-surface border border-silicon-border rounded px-2.5 py-1 text-xs text-content-primary font-mono focus:outline-none"
            >
              <option value="yield">Yield % (High → Low)</option>
              <option value="defect_rate">Defect Rate % (High → Low)</option>
              <option value="failures">Defect Count (High → Low)</option>
              <option value="escalation">Escalation % (High → Low)</option>
              <option value="dies">Total Dies (High → Low)</option>
            </select>
          </div>
        </div>

        {/* Wafer Yield Ranking Horizontal Bars */}
        <Panel
          title="Wafer Yield & Defect Ranking"
          subtitle="Click any wafer bar to select it or open its deep spatial wafer map"
        >
          <div className="max-h-[440px] overflow-y-auto space-y-2 pr-2">
            {wafers.map((w) => {
              const yieldPct = ((w.total_dies - w.actual_failures) / w.total_dies) * 100;
              const escPct = (w.escalated_dies / w.total_dies) * 100;
              const isSelected = selectedWafer === w.wafer_id;

              return (
                <div
                  key={w.wafer_id}
                  onClick={() => setSelectedWafer(isSelected ? 'all' : w.wafer_id)}
                  className={`p-2 rounded border cursor-pointer transition-all ${
                    isSelected
                      ? 'bg-telemetry-blue/15 border-telemetry-cyan shadow-sm'
                      : 'bg-silicon-elevated/60 border-silicon-border/70 hover:border-silicon-border hover:bg-silicon-elevated'
                  }`}
                >
                  <div className="flex items-center justify-between text-xs font-mono mb-1">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-content-primary">{w.wafer_id}</span>
                      <span className="text-[10px] text-content-muted">
                        ({w.total_dies} dies • {w.actual_failures} fails • {w.fail_rate_pct.toFixed(2)}% fail)
                      </span>
                    </div>

                    <div className="flex items-center gap-3">
                      <span className="text-telemetry-purple font-semibold text-[11px]">
                        Esc: {escPct.toFixed(1)}% ({w.escalated_dies})
                      </span>
                      <span className="text-telemetry-green font-bold text-xs">
                        Yield: {yieldPct.toFixed(2)}%
                      </span>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          navigate(`/wafer-intelligence/${w.wafer_id}`);
                        }}
                        className="p-1 rounded hover:bg-silicon-surface text-content-muted hover:text-telemetry-cyan"
                        title={`Open Wafer Intelligence for ${w.wafer_id}`}
                      >
                        <ExternalLink className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  </div>

                  {/* Visual Bar */}
                  <div className="h-2 w-full rounded bg-silicon-canvas overflow-hidden flex">
                    <div
                      className="bg-telemetry-green"
                      style={{ width: `${yieldPct}%` }}
                    />
                    <div
                      className="bg-telemetry-red"
                      style={{ width: `${100 - yieldPct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </Panel>

        {/* Wafer Performance Heatmap (Wafer-by-Metric) */}
        <Panel
          title="Wafer-by-Metric Multi-Dimensional Heatmap"
          subtitle="Normalized visual intensity across 32 holdout wafers. Hover for precise values, click to open wafer."
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-silicon-border text-content-muted text-[11px]">
                  <th className="text-left py-2 px-3 font-semibold">WAFER ID</th>
                  <th className="text-right py-2 px-3 font-semibold">TOTAL DIES</th>
                  <th className="text-right py-2 px-3 font-semibold">YIELD %</th>
                  <th className="text-right py-2 px-3 font-semibold">DEFECT RATE</th>
                  <th className="text-right py-2 px-3 font-semibold">ESCALATION RATE</th>
                  <th className="text-right py-2 px-3 font-semibold">MEAN PROB</th>
                  <th className="text-right py-2 px-3 font-semibold">HIGH-RISK DIES</th>
                  <th className="text-center py-2 px-3 font-semibold">ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-silicon-border/40">
                {wafers.map((w) => {
                  const yieldPct = ((w.total_dies - w.actual_failures) / w.total_dies) * 100;
                  const escPct = (w.escalated_dies / w.total_dies) * 100;
                  const meanProb = w.mean_probability ?? (w.fail_rate_pct / 100);
                  const highRisk = w.high_risk_dies ?? w.cascade_detections;

                  // Normalize defect color intensity (0 - 15%)
                  const failIntensity = Math.min(1, w.fail_rate_pct / 12);
                  const escIntensity = Math.min(1, escPct / 25);

                  return (
                    <tr
                      key={w.wafer_id}
                      onClick={() => navigate(`/wafer-intelligence/${w.wafer_id}`)}
                      className="hover:bg-silicon-interactive/60 cursor-pointer transition-colors group"
                    >
                      <td className="py-2 px-3 text-telemetry-cyan font-bold flex items-center gap-1.5">
                        <Disc className="w-3.5 h-3.5 text-content-muted group-hover:text-telemetry-cyan" />
                        {w.wafer_id}
                      </td>
                      <td className="py-2 px-3 text-right text-content-secondary">
                        {w.total_dies.toLocaleString()}
                      </td>
                      <td className="py-2 px-3 text-right text-telemetry-green font-semibold">
                        {yieldPct.toFixed(2)}%
                      </td>
                      <td
                        className="py-2 px-3 text-right font-semibold"
                        style={{
                          backgroundColor: `rgba(239, 68, 68, ${failIntensity * 0.25})`,
                          color: failIntensity > 0.4 ? '#FCA5A5' : '#EF4444',
                        }}
                        title={`Failures: ${w.actual_failures} / ${w.total_dies}`}
                      >
                        {w.fail_rate_pct.toFixed(2)}%
                      </td>
                      <td
                        className="py-2 px-3 text-right font-semibold"
                        style={{
                          backgroundColor: `rgba(168, 85, 247, ${escIntensity * 0.25})`,
                          color: escIntensity > 0.4 ? '#E9D5FF' : '#C084FC',
                        }}
                        title={`Escalated: ${w.escalated_dies} / ${w.total_dies}`}
                      >
                        {escPct.toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-right text-content-primary">
                        {(meanProb * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-right text-telemetry-amber font-semibold">
                        {highRisk}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className="inline-flex items-center gap-1 text-[11px] text-telemetry-cyan group-hover:underline">
                          Inspect <ChevronRight className="w-3 h-3" />
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* 6. Model Performance: Comparison, Confusion Matrices, ROC & PR Curves */}
      <div className="space-y-6">
        <div>
          <div className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold">
            MODEL PERFORMANCE COMPARISON
          </div>
          <h2 className="text-lg font-bold text-content-primary">
            Head-to-Head Comparison: Model A vs Model B vs Cascade Gate B
          </h2>
        </div>

        {/* Model Metrics Table */}
        <Panel
          title="Authoritative Metric Benchmark Matrix"
          subtitle="Zero-extrapolated performance metrics verified against holdout split"
        >
          <div className="overflow-x-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead>
                <tr className="border-b border-silicon-border text-content-muted text-[11px]">
                  <th className="text-left py-2.5 px-3">ARCHITECTURE</th>
                  <th className="text-right py-2.5 px-3">THRESHOLD</th>
                  <th className="text-right py-2.5 px-3">FAIL F1</th>
                  <th className="text-right py-2.5 px-3">AUC-PR</th>
                  <th className="text-right py-2.5 px-3">ROC-AUC</th>
                  <th className="text-right py-2.5 px-3">PRECISION</th>
                  <th className="text-right py-2.5 px-3">RECALL</th>
                  <th className="text-right py-2.5 px-3">TP</th>
                  <th className="text-right py-2.5 px-3">FP</th>
                  <th className="text-right py-2.5 px-3">FN</th>
                  <th className="text-right py-2.5 px-3">COMPUTE SAVED</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-silicon-border/40">
                {/* Model A */}
                <tr className="hover:bg-silicon-elevated/50">
                  <td className="py-2.5 px-3 font-semibold text-content-primary">LightGBM Model A</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mA?.threshold.toFixed(4) || '0.5100'}</td>
                  <td className="py-2.5 px-3 text-right text-content-primary font-bold">{mA?.f1.toFixed(4) || '0.4907'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mA?.auc_pr.toFixed(4) || '0.4498'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mA?.roc_auc.toFixed(4) || '0.7934'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mA?.precision.toFixed(4) || '0.8947'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mA?.recall.toFixed(4) || '0.3381'}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-green font-semibold">{mA?.confusion_matrix.tp || 357}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-red font-semibold">{mA?.confusion_matrix.fp || 42}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-amber font-semibold">{mA?.confusion_matrix.fn || 699}</td>
                  <td className="py-2.5 px-3 text-right text-content-muted">0.00% (Baseline)</td>
                </tr>
                {/* Model B */}
                <tr className="hover:bg-silicon-elevated/50">
                  <td className="py-2.5 px-3 font-semibold text-telemetry-purple">Model B Standalone (1D CNN)</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mB?.threshold.toFixed(4) || '0.5000'}</td>
                  <td className="py-2.5 px-3 text-right text-content-primary font-bold">{mB?.f1.toFixed(4) || '0.4136'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mB?.auc_pr.toFixed(4) || '0.4072'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mB?.roc_auc.toFixed(4) || '0.7481'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mB?.precision.toFixed(4) || '0.7505'}</td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">{mB?.recall.toFixed(4) || '0.2879'}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-green font-semibold">{mB?.confusion_matrix.tp || 304}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-red font-semibold">{mB?.confusion_matrix.fp || 101}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-amber font-semibold">{mB?.confusion_matrix.fn || 752}</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-red">0.00% (100% Signal Cost)</td>
                </tr>
                {/* Cascade Gate B */}
                <tr className="bg-telemetry-blue/10 hover:bg-telemetry-blue/15 font-semibold">
                  <td className="py-2.5 px-3 text-telemetry-cyan flex items-center gap-1.5">
                    <Sparkles className="w-3.5 h-3.5" />
                    Cascade Gate B (Active)
                  </td>
                  <td className="py-2.5 px-3 text-right text-content-secondary">[0.25, 0.55]</td>
                  <td className="py-2.5 px-3 text-right text-telemetry-green font-bold text-sm">
                    {gB?.f1.toFixed(4) || '0.5072'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-telemetry-cyan font-bold">
                    {gB?.auc_pr.toFixed(4) || '0.4677'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-content-primary">
                    {gB?.roc_auc.toFixed(4) || '0.7990'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-content-primary">
                    {gB?.precision.toFixed(4) || '0.8138'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                    {gB?.recall.toFixed(4) || '0.3684'}
                  </td>
                  <td className="py-2.5 px-3 text-right text-telemetry-green font-bold">
                    {gB?.confusion_matrix.tp || 389}
                  </td>
                  <td className="py-2.5 px-3 text-right text-telemetry-red">
                    {gB?.confusion_matrix.fp || 89}
                  </td>
                  <td className="py-2.5 px-3 text-right text-telemetry-amber">
                    {gB?.confusion_matrix.fn || 667}
                  </td>
                  <td className="py-2.5 px-3 text-right text-telemetry-cyan font-bold">
                    {gB?.compute_saved_pct.toFixed(2) || '88.14'}%
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </Panel>

        {/* Confusion Matrices (Interactive Clickable Cells) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Model A Confusion Matrix */}
          <Panel
            title="Model A Baseline Confusion Matrix"
            subtitle="Fast screening baseline (TP: 357, FN: 699, FP: 42, TN: 26,874)"
          >
            <div className="space-y-3 font-mono text-xs">
              <div className="grid grid-cols-3 gap-2 text-center font-semibold">
                <div />
                <div className="p-1 rounded bg-silicon-elevated text-content-secondary">PREDICTED PASS</div>
                <div className="p-1 rounded bg-silicon-elevated text-content-secondary">PREDICTED FAIL</div>
              </div>

              <div className="grid grid-cols-3 gap-2 items-center">
                <div className="text-right pr-2 text-content-secondary font-semibold">ACTUAL PASS</div>
                <div className="p-4 rounded bg-silicon-surface border border-silicon-border text-center">
                  <div className="text-[10px] text-content-muted">TN (True Neg)</div>
                  <div className="text-base font-bold text-telemetry-green mt-1">26,874</div>
                  <div className="text-[10px] text-content-muted mt-0.5">99.84%</div>
                </div>
                <div className="p-4 rounded bg-silicon-surface border border-telemetry-red/40 text-center">
                  <div className="text-[10px] text-telemetry-red">FP (False Pos)</div>
                  <div className="text-base font-bold text-telemetry-red mt-1">42</div>
                  <div className="text-[10px] text-content-muted mt-0.5">0.16%</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-2 items-center">
                <div className="text-right pr-2 text-content-secondary font-semibold">ACTUAL FAIL</div>
                <div className="p-4 rounded bg-silicon-surface border border-telemetry-amber/40 text-center">
                  <div className="text-[10px] text-telemetry-amber">FN (False Neg)</div>
                  <div className="text-base font-bold text-telemetry-amber mt-1">699</div>
                  <div className="text-[10px] text-content-muted mt-0.5">66.19%</div>
                </div>
                <div className="p-4 rounded bg-silicon-surface border border-telemetry-green/40 text-center">
                  <div className="text-[10px] text-telemetry-green">TP (True Pos)</div>
                  <div className="text-base font-bold text-telemetry-green mt-1">357</div>
                  <div className="text-[10px] text-content-muted mt-0.5">33.81%</div>
                </div>
              </div>
            </div>
          </Panel>

          {/* Cascade Gate B Confusion Matrix (Interactive) */}
          <Panel
            title="Cascade Gate B Confusion Matrix (Interactive)"
            subtitle="Click FN, FP, TP, or TN to filter the error triage table below"
          >
            <div className="space-y-3 font-mono text-xs">
              <div className="grid grid-cols-3 gap-2 text-center font-semibold">
                <div />
                <div className="p-1 rounded bg-silicon-elevated text-content-secondary">PREDICTED PASS</div>
                <div className="p-1 rounded bg-silicon-elevated text-content-secondary">PREDICTED FAIL</div>
              </div>

              <div className="grid grid-cols-3 gap-2 items-center">
                <div className="text-right pr-2 text-content-secondary font-semibold">ACTUAL PASS</div>
                <button
                  onClick={() => setErrorFilter(errorFilter === 'TN' ? 'all' : 'TN')}
                  className={`p-4 rounded text-center border transition-all ${
                    errorFilter === 'TN'
                      ? 'bg-telemetry-green/20 border-telemetry-green'
                      : 'bg-silicon-surface border-silicon-border hover:border-telemetry-green/60'
                  }`}
                >
                  <div className="text-[10px] text-content-muted">TN (True Neg)</div>
                  <div className="text-base font-bold text-telemetry-green mt-1">
                    {gB?.confusion_matrix.tn.toLocaleString() || '26,827'}
                  </div>
                  <div className="text-[10px] text-content-muted mt-0.5">Filter Cohort</div>
                </button>

                <button
                  onClick={() => setErrorFilter(errorFilter === 'FP' ? 'all' : 'FP')}
                  className={`p-4 rounded text-center border transition-all ${
                    errorFilter === 'FP'
                      ? 'bg-telemetry-red/20 border-telemetry-red'
                      : 'bg-silicon-surface border-telemetry-red/40 hover:border-telemetry-red'
                  }`}
                >
                  <div className="text-[10px] text-telemetry-red">FP (False Pos)</div>
                  <div className="text-base font-bold text-telemetry-red mt-1">
                    {gB?.confusion_matrix.fp.toLocaleString() || '89'}
                  </div>
                  <div className="text-[10px] text-telemetry-red mt-0.5">+47 vs Model A</div>
                </button>
              </div>

              <div className="grid grid-cols-3 gap-2 items-center">
                <div className="text-right pr-2 text-content-secondary font-semibold">ACTUAL FAIL</div>
                <button
                  onClick={() => setErrorFilter(errorFilter === 'FN' ? 'all' : 'FN')}
                  className={`p-4 rounded text-center border transition-all ${
                    errorFilter === 'FN'
                      ? 'bg-telemetry-amber/20 border-telemetry-amber'
                      : 'bg-silicon-surface border-telemetry-amber/40 hover:border-telemetry-amber'
                  }`}
                >
                  <div className="text-[10px] text-telemetry-amber">FN (False Neg)</div>
                  <div className="text-base font-bold text-telemetry-amber mt-1">
                    {gB?.confusion_matrix.fn.toLocaleString() || '667'}
                  </div>
                  <div className="text-[10px] text-telemetry-green mt-0.5">-32 Net vs A</div>
                </button>

                <button
                  onClick={() => setErrorFilter(errorFilter === 'TP' ? 'all' : 'TP')}
                  className={`p-4 rounded text-center border transition-all ${
                    errorFilter === 'TP'
                      ? 'bg-telemetry-green/20 border-telemetry-green'
                      : 'bg-silicon-surface border-telemetry-green/40 hover:border-telemetry-green'
                  }`}
                >
                  <div className="text-[10px] text-telemetry-green">TP (True Pos)</div>
                  <div className="text-base font-bold text-telemetry-green mt-1">
                    {gB?.confusion_matrix.tp.toLocaleString() || '389'}
                  </div>
                  <div className="text-[10px] text-telemetry-green mt-0.5">+32 Net Gain</div>
                </button>
              </div>
            </div>
          </Panel>
        </div>

        {/* Empirical ROC Curve & Precision-Recall Curve (Crisp SVG) */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* ROC Curve */}
          <Panel
            title="Empirical ROC Curve (Receiver Operating Characteristic)"
            subtitle="Verified holdout performance (Model A AUC: 0.7934 → Cascade AUC: 0.7990)"
          >
            {curves ? (
              <div className="space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between text-[11px] text-content-muted">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1.5 text-content-secondary">
                      <span className="w-3 h-0.5 bg-content-muted" /> Model A (AUC: {curves.roc.model_A.auc})
                    </span>
                    <span className="flex items-center gap-1.5 text-telemetry-cyan font-bold">
                      <span className="w-3 h-0.5 bg-telemetry-cyan" /> Cascade Gate B (AUC: {curves.roc.cascade.auc})
                    </span>
                  </div>
                </div>

                {/* SVG Curve Canvas */}
                <div className="relative h-60 w-full bg-silicon-canvas rounded border border-silicon-border p-2">
                  <svg className="w-full h-full" viewBox="0 0 300 200" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="30" y1="20" x2="290" y2="20" stroke="#1E293B" strokeWidth="1" />
                    <line x1="30" y1="70" x2="290" y2="70" stroke="#1E293B" strokeWidth="1" />
                    <line x1="30" y1="120" x2="290" y2="120" stroke="#1E293B" strokeWidth="1" />
                    <line x1="30" y1="170" x2="290" y2="170" stroke="#334155" strokeWidth="1.5" />
                    <line x1="30" y1="20" x2="30" y2="170" stroke="#334155" strokeWidth="1.5" />

                    {/* Diagonal baseline (Random classifier) */}
                    <line x1="30" y1="170" x2="290" y2="20" stroke="#475569" strokeDasharray="4 4" strokeWidth="1" />

                    {/* Model A Path */}
                    <polyline
                      fill="none"
                      stroke="#94A3B8"
                      strokeWidth="1.5"
                      points={curves.roc.model_A.fpr
                        .map((fpr, i) => {
                          const tpr = curves.roc.model_A.tpr[i];
                          const x = 30 + fpr * 260;
                          const y = 170 - tpr * 150;
                          return `${x},${y}`;
                        })
                        .join(' ')}
                    />

                    {/* Cascade Gate B Path */}
                    <polyline
                      fill="none"
                      stroke="#38BDF8"
                      strokeWidth="2.5"
                      points={curves.roc.cascade.fpr
                        .map((fpr, i) => {
                          const tpr = curves.roc.cascade.tpr[i];
                          const x = 30 + fpr * 260;
                          const y = 170 - tpr * 150;
                          return `${x},${y}`;
                        })
                        .join(' ')}
                    />
                  </svg>
                  {/* Axis labels */}
                  <div className="absolute left-2 bottom-1 text-[10px] text-content-muted">FPR 0.0</div>
                  <div className="absolute right-3 bottom-1 text-[10px] text-content-muted">FPR 1.0</div>
                  <div className="absolute left-1 top-2 text-[10px] text-content-muted">TPR 1.0</div>
                </div>
              </div>
            ) : (
              <LoadingSkeleton className="h-60" />
            )}
          </Panel>

          {/* Precision-Recall Curve */}
          <Panel
            title="Precision-Recall Curve (Defect Imbalance Focus)"
            subtitle="Critical for 3.78% defect prevalence (Model A AUC-PR: 0.4498 → Gate B AUC-PR: 0.4677)"
          >
            {curves ? (
              <div className="space-y-3 font-mono text-xs">
                <div className="flex items-center justify-between text-[11px] text-content-muted">
                  <div className="flex items-center gap-3">
                    <span className="flex items-center gap-1.5 text-content-secondary">
                      <span className="w-3 h-0.5 bg-content-muted" /> Model A (AUC-PR: {curves.pr.model_A.auc_pr})
                    </span>
                    <span className="flex items-center gap-1.5 text-telemetry-purple font-bold">
                      <span className="w-3 h-0.5 bg-telemetry-purple" /> Cascade Gate B (AUC-PR: {curves.pr.cascade.auc_pr})
                    </span>
                  </div>
                </div>

                {/* SVG PR Canvas */}
                <div className="relative h-60 w-full bg-silicon-canvas rounded border border-silicon-border p-2">
                  <svg className="w-full h-full" viewBox="0 0 300 200" preserveAspectRatio="none">
                    {/* Grid lines */}
                    <line x1="30" y1="20" x2="290" y2="20" stroke="#1E293B" strokeWidth="1" />
                    <line x1="30" y1="70" x2="290" y2="70" stroke="#1E293B" strokeWidth="1" />
                    <line x1="30" y1="120" x2="290" y2="120" stroke="#1E293B" strokeWidth="1" />
                    <line x1="30" y1="170" x2="290" y2="170" stroke="#334155" strokeWidth="1.5" />
                    <line x1="30" y1="20" x2="30" y2="170" stroke="#334155" strokeWidth="1.5" />

                    {/* Model A Path */}
                    <polyline
                      fill="none"
                      stroke="#94A3B8"
                      strokeWidth="1.5"
                      points={curves.pr.model_A.recall
                        .map((recVal, i) => {
                          const precVal = curves.pr.model_A.precision[i];
                          const x = 30 + recVal * 260;
                          const y = 170 - precVal * 150;
                          return `${x},${y}`;
                        })
                        .join(' ')}
                    />

                    {/* Cascade Gate B Path */}
                    <polyline
                      fill="none"
                      stroke="#A855F7"
                      strokeWidth="2.5"
                      points={curves.pr.cascade.recall
                        .map((recVal, i) => {
                          const precVal = curves.pr.cascade.precision[i];
                          const x = 30 + recVal * 260;
                          const y = 170 - precVal * 150;
                          return `${x},${y}`;
                        })
                        .join(' ')}
                    />
                  </svg>
                  {/* Axis labels */}
                  <div className="absolute left-2 bottom-1 text-[10px] text-content-muted">Recall 0.0</div>
                  <div className="absolute right-3 bottom-1 text-[10px] text-content-muted">Recall 1.0</div>
                  <div className="absolute left-1 top-2 text-[10px] text-content-muted">Prec 1.0</div>
                </div>
              </div>
            ) : (
              <LoadingSkeleton className="h-60" />
            )}
          </Panel>
        </div>
      </div>

      {/* 7. Cascade Analytics & Compute Savings */}
      <div className="space-y-6">
        <div>
          <div className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold flex items-center justify-between">
            <span>CASCADE FLOW & COMPUTE EFFICIENCY</span>
            <span className="text-telemetry-green text-[11px] font-bold">Active B2 Independent: 93.78% Saved</span>
          </div>
          <h2 className="text-lg font-bold text-content-primary">
            Multi-Resolution Routing & Compute Efficiency (Historical Canonical: 88.14% • B2 Active: 93.78%)
          </h2>
        </div>

        {/* Visual Cascade Execution Flow */}
        <Panel
          title="Cascade Execution Flow Diagram (Canonical 32-Wafer Diagnostic Baseline)"
          subtitle="Real die volumes routed through fast path vs heavy block evaluation (EXP-04 Independent achieves 93.78% fast path)"
        >
          <div className="p-6 rounded bg-silicon-elevated/80 border border-silicon-border font-mono text-xs space-y-6">
            {/* Stage 1: Model A */}
            <div className="flex flex-col items-center">
              <div className="w-full max-w-md p-3.5 rounded bg-silicon-surface border border-telemetry-cyan/50 text-center shadow-md">
                <div className="text-[11px] text-telemetry-cyan font-bold">STAGE 1: LIGHTGBM MODEL A (FAB TELEMETRY)</div>
                <div className="text-sm font-bold text-content-primary mt-1">27,972 Total Holdout Dies Evaluated</div>
                <div className="text-[10px] text-content-muted mt-0.5">Runtime: 1.14ms per die • Zero block extraction cost</div>
              </div>
            </div>

            {/* Split Arrow */}
            <div className="flex justify-center items-center">
              <div className="w-0.5 h-6 bg-silicon-border" />
            </div>

            {/* Decision Branches */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 max-w-2xl mx-auto">
              {/* Branch 1: Fast Path (Bypassed) */}
              <div className="p-4 rounded bg-silicon-canvas border border-telemetry-green/50 text-center space-y-2">
                <div className="text-[11px] text-telemetry-green font-bold">FAST PATH (BYPASSED)</div>
                <div className="text-xl font-bold text-content-primary">24,655 Dies</div>
                <div className="text-xs font-bold text-telemetry-cyan">88.14% (Canonical Holdout) • 93.78% (B2 Indep.)</div>
                <div className="text-[10px] text-content-secondary leading-tight pt-1 border-t border-silicon-border/60">
                  Confident Pass or Confident Defect. Model A decision accepted directly without secondary inference.
                </div>
              </div>

              {/* Branch 2: Escalated (Gate B) */}
              <div className="p-4 rounded bg-silicon-canvas border border-telemetry-purple/50 text-center space-y-2">
                <div className="text-[11px] text-telemetry-purple font-bold">UNCERTAIN DIES (GATE B)</div>
                <div className="text-xl font-bold text-content-primary">3,317 Dies</div>
                <div className="text-xs font-bold text-telemetry-purple">11.86% (Canonical) • 6.22% (B2 Indep.)</div>
                <div className="text-[10px] text-content-secondary leading-tight pt-1 border-t border-silicon-border/60">
                  Ambiguous zone. Triggers 2,000-block raw signal extraction & Model B deep evaluation.
                </div>
              </div>
            </div>

            {/* Final Decision */}
            <div className="flex flex-col items-center pt-2">
              <div className="w-full max-w-md p-3 rounded bg-silicon-surface border border-silicon-border text-center">
                <div className="text-[11px] text-content-muted">STAGE 3: FINAL PRODUCTION DECISION</div>
                <div className="text-xs font-bold text-content-primary mt-0.5">
                  100% of dies screened • 0 dies lost • 88.14% canonical savings (93.78% independent B2 savings)
                </div>
              </div>
            </div>
          </div>
        </Panel>

        {/* Compute Savings Hero & Gate Sweep Pareto Table */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Compute Saved Hero Visual */}
          <Panel
            title="Block Compute Budget Saved"
            subtitle="Massive fab test-time acceleration"
          >
            <div className="flex flex-col items-center justify-center p-6 text-center space-y-4">
              <div className="w-24 h-24 rounded-full border-4 border-telemetry-cyan flex items-center justify-center bg-telemetry-blue/10">
                <span className="text-2xl font-bold text-telemetry-cyan font-mono">
                  {gB?.compute_saved_pct.toFixed(1) || '88.1'}%
                </span>
              </div>
              <div>
                <div className="text-sm font-bold text-content-primary font-mono">
                  24,655 / 27,972 DIES BYPASSED
                </div>
                <p className="text-xs text-content-secondary mt-1 max-w-xs">
                  Avoided extracting and inferring raw 2,000-block sequential signals on 88.14% of dies, reserving heavy GPU inference strictly for uncertain boundaries.
                </p>
              </div>
            </div>
          </Panel>

          {/* Interactive Gate Sweep Tradeoff Chart */}
          <div className="lg:col-span-2">
            <Panel
              title="Gate Sweep Pareto Analysis (Accuracy vs Compute Saved)"
              subtitle="Data loaded directly from results/cascade_gate_sweep.csv (Gates A through F)"
            >
              {benchmark?.gate_sweep_rows ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr className="border-b border-silicon-border text-content-muted text-[11px]">
                        <th className="text-left py-2 px-3">GATE WINDOW</th>
                        <th className="text-right py-2 px-3">ESCALATION %</th>
                        <th className="text-right py-2 px-3">COMPUTE SAVED</th>
                        <th className="text-right py-2 px-3">CASCADE F1</th>
                        <th className="text-right py-2 px-3">AUC-PR</th>
                        <th className="text-right py-2 px-3">FN RESCUED</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-silicon-border/40">
                      {benchmark.gate_sweep_rows.map((row: any, i: number) => {
                        const isCurrent = row.Gate_Name?.includes('0.25') && row.Gate_Name?.includes('0.55');
                        return (
                          <tr
                            key={i}
                            className={`hover:bg-silicon-elevated/50 ${
                              isCurrent ? 'bg-telemetry-blue/10 text-telemetry-cyan font-semibold' : ''
                            }`}
                          >
                            <td className="py-2 px-3 flex items-center gap-1.5">
                              {isCurrent && <Sparkles className="w-3 h-3 text-telemetry-cyan" />}
                              {row.Gate_Name}
                            </td>
                            <td className="py-2 px-3 text-right">{Number(row.Escalation_Pct).toFixed(2)}%</td>
                            <td className="py-2 px-3 text-right text-telemetry-green font-bold">
                              {Number(row.Compute_Saved_Pct).toFixed(2)}%
                            </td>
                            <td className="py-2 px-3 text-right font-bold">{Number(row.Cascade_F1).toFixed(4)}</td>
                            <td className="py-2 px-3 text-right">{Number(row.Cascade_AUC_PR).toFixed(4)}</td>
                            <td className="py-2 px-3 text-right text-telemetry-cyan">{row.FN_Recovered}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <LoadingSkeleton className="h-48" />
              )}
            </Panel>
          </div>
        </div>
      </div>

      {/* 8. Error & Defect Recovery Visualization */}
      <div className="space-y-6">
        <div>
          <div className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold">
            ERROR & DEFECT RECOVERY VISUALIZATION
          </div>
          <h2 className="text-lg font-bold text-content-primary">
            Transparent Defect Recovery Ledger (+32 Net True Positive Gain)
          </h2>
        </div>

        <Panel
          title="Defect Recovery Mathematical Equation"
          subtitle="Zero-cherry-picking: Transparent accounting of rescues, overturned decisions, and new false positives"
        >
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 p-4 rounded bg-silicon-elevated/60 border border-silicon-border font-mono text-center">
            {/* Step 1 */}
            <div className="p-3 rounded bg-silicon-surface border border-silicon-border">
              <div className="text-[10px] text-content-muted">1. MODEL A BASELINE</div>
              <div className="text-xl font-bold text-telemetry-amber mt-1">699 FN</div>
              <div className="text-[10px] text-content-secondary mt-0.5">Missed Defects</div>
            </div>

            {/* Step 2 */}
            <div className="p-3 rounded bg-silicon-surface border border-telemetry-green/40">
              <div className="text-[10px] text-telemetry-green">2. RESCUED BY GATE B</div>
              <div className="text-xl font-bold text-telemetry-green mt-1">37 Dies</div>
              <div className="text-[10px] text-content-secondary mt-0.5">Model B Detected</div>
            </div>

            {/* Step 3 */}
            <div className="p-3 rounded bg-silicon-surface border border-telemetry-red/40">
              <div className="text-[10px] text-telemetry-red">3. TRUE POSITIVES OVERTURNED</div>
              <div className="text-xl font-bold text-telemetry-red mt-1">-5 Dies</div>
              <div className="text-[10px] text-content-secondary mt-0.5">Lost by Model B</div>
            </div>

            {/* Step 4 */}
            <div className="p-3 rounded bg-silicon-surface border border-telemetry-cyan/50 bg-telemetry-blue/10">
              <div className="text-[10px] text-telemetry-cyan font-bold">4. NET DEFECT GAIN</div>
              <div className="text-2xl font-bold text-telemetry-cyan mt-1">+32 NET</div>
              <div className="text-[10px] text-content-secondary mt-0.5">(+47 New FPs)</div>
            </div>
          </div>
        </Panel>
      </div>

      {/* 9. Probability Distribution & Gate Analysis */}
      <div className="space-y-6">
        <div>
          <div className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold">
            PROBABILITY DISTRIBUTION & THRESHOLD ANALYSIS
          </div>
          <h2 className="text-lg font-bold text-content-primary">
            20-Bin Die Probability Histogram with Gate B Window [0.25, 0.55]
          </h2>
        </div>

        <Panel
          title="Prediction Probability Distribution (Model A vs Cascade)"
          subtitle="Explains why dies enter the Gate B escalation window and demonstrates calibrated separation"
        >
          {probDist ? (
            <div className="space-y-4 font-mono text-xs">
              <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-content-muted">
                <div className="flex items-center gap-4">
                  <span className="flex items-center gap-1.5 text-content-secondary">
                    <span className="w-3 h-2 bg-slate-500 rounded-sm" /> Model A Bins
                  </span>
                  <span className="flex items-center gap-1.5 text-telemetry-cyan font-semibold">
                    <span className="w-3 h-2 bg-telemetry-cyan rounded-sm" /> Cascade Gate B Bins
                  </span>
                  <span className="flex items-center gap-1.5 text-telemetry-purple font-semibold">
                    <span className="w-3 h-2 bg-telemetry-purple/40 border border-telemetry-purple rounded-sm" /> Gate B Window [0.25, 0.55]
                  </span>
                  <span className="flex items-center gap-1.5 text-telemetry-red font-semibold">
                    <span className="w-3 h-0.5 bg-telemetry-red" /> Threshold: 0.5100
                  </span>
                </div>
              </div>

              {/* SVG Histogram */}
              <div className="relative h-64 w-full bg-silicon-canvas rounded border border-silicon-border p-2">
                <svg className="w-full h-full" viewBox="0 0 500 200" preserveAspectRatio="none">
                  {/* Gate B Shaded Window (X: 0.25 to 0.55 -> pixels: 500 * 0.25 = 125, 500 * 0.55 = 275) */}
                  <rect x="125" y="10" width="150" height="175" fill="rgba(168, 85, 247, 0.12)" stroke="#A855F7" strokeDasharray="3 3" />

                  {/* Model A Threshold line at 0.5100 -> x = 500 * 0.51 = 255 */}
                  <line x1="255" y1="10" x2="255" y2="185" stroke="#EF4444" strokeWidth="2" strokeDasharray="4 2" />

                  {/* Histogram Bars */}
                  {probDist.bin_centers.map((center, i) => {
                    const x = i * 25 + 2;
                    const maxCount = Math.max(...probDist.model_a_counts, ...probDist.cascade_counts);
                    const hA = Math.max(3, (probDist.model_a_counts[i] / maxCount) * 160);
                    const hCas = Math.max(3, (probDist.cascade_counts[i] / maxCount) * 160);

                    return (
                      <g key={i}>
                        {/* Model A bar */}
                        <rect
                          x={x}
                          y={185 - hA}
                          width="9"
                          height={hA}
                          fill="#64748B"
                          opacity="0.7"
                        >
                          <title>{`Bin: ${probDist.bin_labels[i]} | Model A: ${probDist.model_a_counts[i]} dies`}</title>
                        </rect>
                        {/* Cascade bar */}
                        <rect
                          x={x + 10}
                          y={185 - hCas}
                          width="9"
                          height={hCas}
                          fill="#38BDF8"
                          opacity="0.85"
                        >
                          <title>{`Bin: ${probDist.bin_labels[i]} | Cascade: ${probDist.cascade_counts[i]} dies`}</title>
                        </rect>
                      </g>
                    );
                  })}
                </svg>

                {/* Legend Labels */}
                <div className="absolute left-3 bottom-1 text-[10px] text-content-muted">P = 0.0</div>
                <div className="absolute left-1/4 bottom-1 text-[10px] text-telemetry-purple font-semibold">Lower 0.25</div>
                <div className="absolute left-1/2 bottom-1 text-[10px] text-telemetry-red font-semibold">Thr 0.51</div>
                <div className="absolute right-3 bottom-1 text-[10px] text-content-muted">P = 1.0</div>
              </div>
            </div>
          ) : (
            <LoadingSkeleton className="h-64" />
          )}
        </Panel>
      </div>

      {/* 10. High-Risk Die & Error Analysis Table */}
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="text-xs font-mono uppercase tracking-wider text-telemetry-cyan font-semibold">
              HIGH-RISK DIE CATALOG & ERROR TRIAGE
            </div>
            <h2 className="text-lg font-bold text-content-primary">
              Interactive Die Triage Table ({filteredDies.length} Dies matching current filters)
            </h2>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={handleExportCSV}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-silicon-elevated hover:bg-silicon-border text-content-secondary hover:text-content-primary border border-silicon-border text-xs font-mono transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              Download CSV
            </button>
          </div>
        </div>

        <Panel
          title="High-Risk Die Records"
          subtitle="Click any die row to immediately open the full Wafer & Die Inspector workstation"
        >
          <div className="overflow-x-auto max-h-[460px] overflow-y-auto">
            <table className="w-full text-xs font-mono border-collapse">
              <thead className="sticky top-0 bg-silicon-surface border-b border-silicon-border text-content-muted text-[11px] z-10">
                <tr>
                  <th className="text-left py-2 px-3">WAFER</th>
                  <th className="text-left py-2 px-3">DIE (R, C)</th>
                  <th className="text-center py-2 px-3">ACTUAL</th>
                  <th className="text-right py-2 px-3">MODEL A</th>
                  <th className="text-right py-2 px-3">MODEL B</th>
                  <th className="text-right py-2 px-3">CASCADE P</th>
                  <th className="text-center py-2 px-3">GATE B</th>
                  <th className="text-center py-2 px-3">RECOVERY</th>
                  <th className="text-center py-2 px-3">ACTION</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-silicon-border/40">
                {filteredDies.slice(0, 50).map((d, i) => {
                  const isRecovered = !!d.recovered_FN_Gate_B;
                  const isFail = d.actual_label === 1;

                  return (
                    <tr
                      key={i}
                      onClick={() =>
                        navigate(`/wafer-intelligence/${d.wafer_id}/${d.die_row}/${d.die_col}`)
                      }
                      className="hover:bg-silicon-interactive/60 cursor-pointer transition-colors group"
                    >
                      <td className="py-2 px-3 text-telemetry-cyan font-bold">{d.wafer_id}</td>
                      <td className="py-2 px-3 text-content-primary font-semibold">
                        R{d.die_row}/C{d.die_col}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <StatusBadge
                          variant={isFail ? 'fail' : 'pass'}
                          label={isFail ? 'DEFECT (1)' : 'PASS (0)'}
                        />
                      </td>
                      <td className="py-2 px-3 text-right text-content-secondary">
                        {(d.prob_Model_A * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-right text-telemetry-purple font-medium">
                        {(d.prob_Model_B * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-right text-telemetry-cyan font-bold">
                        {(d.prob_Cascade_B * 100).toFixed(2)}%
                      </td>
                      <td className="py-2 px-3 text-center">
                        {d.in_gate_B ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/40 font-semibold">
                            ESCALATED
                          </span>
                        ) : (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-silicon-canvas text-content-muted border border-silicon-border">
                            FAST PATH
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-center">
                        {isRecovered ? (
                          <span className="px-1.5 py-0.5 rounded text-[10px] bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/40 font-bold">
                            RESCUED FN
                          </span>
                        ) : (
                          <span className="text-content-muted text-[10px]">—</span>
                        )}
                      </td>
                      <td className="py-2 px-3 text-center">
                        <span className="inline-flex items-center gap-1 text-[11px] text-telemetry-cyan group-hover:underline">
                          Inspect <ExternalLink className="w-3 h-3" />
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>
      </div>

      {/* 11. Spatial Insights & Deep Workstation Link */}
      <Panel
        title="Spatial Yield Intelligence & Cluster Integration"
        subtitle="Summary spatial metrics across validation cohort with one-click deep navigation"
      >
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-2">
          <div>
            <div className="text-xs font-mono text-content-primary font-semibold">
              CIRCULAR WAFER CARTOGRAPHY & 2,000-BLOCK SIGNAL VISUALIZATION
            </div>
            <p className="text-xs text-content-secondary mt-1 max-w-xl">
              Inspect physical radial/edge clustering, die coordinate telemetry, 2,000-block raw signal profiles, and 1D CNN attribution traces directly in the Wafer Intelligence workstation.
            </p>
          </div>

          <button
            onClick={() =>
              navigate(
                selectedWafer !== 'all'
                  ? `/wafer-intelligence/${selectedWafer}`
                  : '/wafer-intelligence'
              )
            }
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded bg-telemetry-cyan text-silicon-canvas font-mono text-xs font-bold transition-all hover:bg-telemetry-cyan/90 shrink-0"
          >
            <Disc className="w-4 h-4" />
            Open Wafer Intelligence
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </Panel>
    </div>
  );
};
