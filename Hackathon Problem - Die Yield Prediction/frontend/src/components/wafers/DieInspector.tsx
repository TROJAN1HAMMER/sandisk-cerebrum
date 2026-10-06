import React, { useState, useMemo, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { DiePoint } from '../../types/wafers';
import { useBlockAttribution } from '../../hooks/useInference';
import { InferenceTrace } from '../inference/InferenceTrace';
import { BlockSignalViewer } from '../inference/BlockSignalViewer';
import { StatusBadge } from '../ui/StatusBadge';
import { MetricCard } from '../ui/MetricCard';
import { LoadingSkeleton } from '../ui/LoadingSkeleton';
import {
  X,
  Cpu,
  Layers,
  Activity,
  Flame,
  ArrowRight,
  ExternalLink,
  Sparkles,
  Compass,
  CheckCircle2,
  AlertTriangle,
  Copy,
  Check,
  ChevronLeft,
  ChevronRight,
  Sliders,
  Grid,
} from 'lucide-react';

interface DieInspectorProps {
  die: DiePoint;
  allDies?: DiePoint[];
  onSelectDie?: (die: DiePoint) => void;
  onClose: () => void;
  className?: string;
}

export const DieInspector: React.FC<DieInspectorProps> = ({
  die,
  allDies = [],
  onSelectDie,
  onClose,
  className = '',
}) => {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const urlTab = searchParams.get('tab') as 'trace' | 'signal' | 'telemetry' | 'neighborhood' | null;
  const [activeTab, setActiveTabState] = useState<'trace' | 'signal' | 'telemetry' | 'neighborhood'>(
    (urlTab && ['trace', 'signal', 'telemetry', 'neighborhood'].includes(urlTab)) ? urlTab : 'trace'
  );

  useEffect(() => {
    if (urlTab && ['trace', 'signal', 'telemetry', 'neighborhood'].includes(urlTab)) {
      setActiveTabState(urlTab);
    }
  }, [urlTab]);

  const setActiveTab = (tab: 'trace' | 'signal' | 'telemetry' | 'neighborhood') => {
    setActiveTabState(tab);
    const newParams = new URLSearchParams(searchParams);
    newParams.set('tab', tab);
    setSearchParams(newParams, { replace: true });
  };
  const [copiedLink, setCopiedLink] = useState(false);

  // Lazy-load 2,000-block signal & occlusion attribution only when signal tab is active
  const {
    data: attributionData,
    isLoading: isAttributionLoading,
    error: attributionError,
  } = useBlockAttribution(die.wafer_id, die.die_row, die.die_col, activeTab === 'signal');

  const isFail = die.decision_Cascade_B === 1;
  const isRecovered = die.recovered_FN_Gate_B === 1;

  // Find previous and next die in sorted order
  const { prevDie, nextDie } = useMemo(() => {
    if (!allDies || allDies.length === 0) return { prevDie: null, nextDie: null };
    const currentIndex = allDies.findIndex(
      (d) => d.die_row === die.die_row && d.die_col === die.die_col
    );
    if (currentIndex === -1) return { prevDie: null, nextDie: null };
    return {
      prevDie: currentIndex > 0 ? allDies[currentIndex - 1] : null,
      nextDie: currentIndex < allDies.length - 1 ? allDies[currentIndex + 1] : null,
    };
  }, [allDies, die.die_row, die.die_col]);

  // Compute 3x3 local neighborhood around die
  const neighborhoodGrid = useMemo(() => {
    if (!allDies || allDies.length === 0) return [];
    const r = die.die_row;
    const c = die.die_col;
    const grid: Array<{
      rowOffset: number;
      colOffset: number;
      row: number;
      col: number;
      die: DiePoint | null;
    }> = [];

    for (let dr = -1; dr <= 1; dr++) {
      for (let dc = -1; dc <= 1; dc++) {
        const nr = r + dr;
        const nc = c + dc;
        const found = allDies.find((d) => d.die_row === nr && d.die_col === nc) || null;
        grid.push({
          rowOffset: dr,
          colOffset: dc,
          row: nr,
          col: nc,
          die: found,
        });
      }
    }
    return grid;
  }, [allDies, die.die_row, die.die_col]);

  const handleCopyLink = () => {
    const url = `${window.location.origin}/wafer-intelligence/${die.wafer_id}/${die.die_row}/${die.die_col}`;
    navigator.clipboard.writeText(url);
    setCopiedLink(true);
    setTimeout(() => setCopiedLink(false), 2000);
  };

  const handleOpenDiagnostics = () => {
    navigate('/cascade-diagnostics');
  };

  // Error type classification string
  const getErrorClassification = () => {
    if (die.recovered_FN_Gate_B === 1) return 'Rescued False Negative (+1 Recovered)';
    if (die.actual_label === 1 && die.decision_Cascade_B === 1) return 'True Positive (Correct Defect)';
    if (die.actual_label === 0 && die.decision_Cascade_B === 0) return 'True Negative (Correct Pass)';
    if (die.actual_label === 0 && die.decision_Cascade_B === 1) return 'False Positive (False Alarm)';
    if (die.actual_label === 1 && die.decision_Cascade_B === 0) return 'False Negative (Escape)';
    return 'Unlabeled';
  };

  return (
    <div
      className={`bg-silicon-surface border border-silicon-border rounded-lg shadow-2xl overflow-hidden flex flex-col ${className}`}
    >
      {/* 1. Large Typography Die Status Header */}
      <div className="flex flex-wrap items-center justify-between px-5 py-4 bg-silicon-elevated border-b border-silicon-border gap-4">
        <div className="flex items-center gap-4">
          {/* Die Coordinates Pill */}
          <div className="p-2.5 rounded bg-silicon-interactive border border-silicon-border text-telemetry-cyan font-mono font-bold text-base shadow-sm">
            R{die.die_row} / C{die.die_col}
          </div>

          <div>
            <div className="flex items-center gap-2.5 flex-wrap">
              {/* Prominent Large PASS / FAIL Typography */}
              <span
                className={`text-2xl font-black font-mono tracking-tight ${
                  isFail ? 'text-telemetry-red' : 'text-telemetry-green'
                }`}
              >
                {isFail ? 'FAIL' : 'PASS'}
              </span>

              <span className="text-content-muted font-mono">•</span>

              <span className="text-sm font-bold font-mono text-content-primary">
                Final P: {(die.prob_Cascade_B * 100).toFixed(2)}%
              </span>

              {die.in_gate_B ? (
                <StatusBadge variant="warn" label="GATE B ESCALATED" />
              ) : (
                <StatusBadge variant="info" label="FAST PATH BYPASSED" />
              )}

              {isRecovered && (
                <span className="px-2 py-0.5 rounded text-[11px] font-mono font-bold bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40 flex items-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  RESCUED DEFECT
                </span>
              )}
            </div>

            <div className="text-xs text-content-muted font-mono mt-0.5">
              Wafer: <strong className="text-content-primary">{die.wafer_id}</strong> • Class:{' '}
              <strong className="text-content-secondary">{getErrorClassification()}</strong>
            </div>
          </div>
        </div>

        {/* Action Controls & Previous/Next Navigation */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Previous / Next buttons */}
          {onSelectDie && (
            <div className="flex items-center border border-silicon-border rounded bg-silicon-interactive overflow-hidden mr-1">
              <button
                onClick={() => prevDie && onSelectDie(prevDie)}
                disabled={!prevDie}
                className="px-2 py-1 hover:bg-silicon-elevated disabled:opacity-30 text-content-secondary hover:text-content-primary flex items-center text-xs font-mono"
                title={prevDie ? `Previous: R${prevDie.die_row} C${prevDie.die_col}` : 'No previous die'}
              >
                <ChevronLeft className="w-3.5 h-3.5" />
                <span>Prev</span>
              </button>
              <div className="w-px h-4 bg-silicon-border" />
              <button
                onClick={() => nextDie && onSelectDie(nextDie)}
                disabled={!nextDie}
                className="px-2 py-1 hover:bg-silicon-elevated disabled:opacity-30 text-content-secondary hover:text-content-primary flex items-center text-xs font-mono"
                title={nextDie ? `Next: R${nextDie.die_row} C${nextDie.die_col}` : 'No next die'}
              >
                <span>Next</span>
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          )}

          <button
            onClick={handleCopyLink}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono rounded bg-silicon-interactive text-content-secondary hover:text-content-primary border border-silicon-border transition-colors"
            title="Copy Deep Link"
          >
            {copiedLink ? <Check className="w-3.5 h-3.5 text-telemetry-green" /> : <Copy className="w-3.5 h-3.5" />}
            <span>{copiedLink ? 'Copied' : 'Share'}</span>
          </button>

          <button
            onClick={handleOpenDiagnostics}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono rounded bg-telemetry-purple/20 text-telemetry-purple hover:bg-telemetry-purple/30 border border-telemetry-purple/40 transition-colors font-semibold"
          >
            <ExternalLink className="w-3.5 h-3.5" />
            <span>Diagnostics</span>
          </button>

          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-silicon-interactive text-content-muted hover:text-content-primary transition-colors ml-1"
            title="Close Inspector"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 2. Hero Metric Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2.5 p-4 bg-silicon-card border-b border-silicon-border">
        <MetricCard
          title="Cascade P(Final)"
          value={`${(die.prob_Cascade_B * 100).toFixed(2)}%`}
          subtitle={isFail ? 'Prediction: FAIL' : 'Prediction: PASS'}
          accent={isFail ? 'red' : 'green'}
        />
        <MetricCard
          title="LightGBM Model A"
          value={`${(die.prob_Model_A * 100).toFixed(2)}%`}
          subtitle={die.decision_Model_A === 1 ? 'Initial: FAIL' : 'Initial: PASS'}
          accent="cyan"
        />
        <MetricCard
          title="Stage 2 Gate B"
          value={die.in_gate_B ? 'ESCALATED' : 'BYPASSED'}
          subtitle={die.in_gate_B ? 'In [0.25, 0.55]' : 'Outside Band'}
          accent={die.in_gate_B ? 'amber' : 'neutral'}
        />
        <MetricCard
          title="Model B (1D CNN)"
          value={
            die.in_gate_B
              ? `${(die.prob_Model_B * 100).toFixed(2)}%`
              : 'Not executed'
          }
          subtitle={
            die.in_gate_B
              ? die.decision_Cascade_B === 1
                ? 'Model B: FAIL'
                : 'Model B: PASS'
              : 'Compute Saved'
          }
          accent={die.in_gate_B ? 'purple' : 'neutral'}
        />
        <MetricCard
          title="Ground Truth"
          value={die.actual_label === 1 ? 'DEFECT (1)' : 'PASS (0)'}
          subtitle="Fab Post-Test"
          accent={die.actual_label === 1 ? 'red' : 'green'}
        />
        <MetricCard
          title="Recovery Delta"
          value={
            isRecovered
              ? '+1 DEFECT'
              : `${(die.delta_prob * 100 > 0 ? '+' : '') + (die.delta_prob * 100).toFixed(1)}%`
          }
          subtitle="Stage 2 Resolution"
          accent={isRecovered ? 'cyan' : 'neutral'}
        />
      </div>

      {/* 3. Navigation Tabs */}
      <div className="flex items-center gap-1 px-4 bg-silicon-surface border-b border-silicon-border font-mono text-xs overflow-x-auto">
        <button
          onClick={() => setActiveTab('trace')}
          className={`flex items-center gap-1.5 px-3.5 py-2.5 font-medium border-b-2 transition-colors ${
            activeTab === 'trace'
              ? 'border-telemetry-cyan text-telemetry-cyan font-bold'
              : 'border-transparent text-content-secondary hover:text-content-primary'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>Inference Trace & Gating</span>
        </button>

        <button
          onClick={() => setActiveTab('signal')}
          className={`flex items-center gap-1.5 px-3.5 py-2.5 font-medium border-b-2 transition-colors ${
            activeTab === 'signal'
              ? 'border-telemetry-cyan text-telemetry-cyan font-bold'
              : 'border-transparent text-content-secondary hover:text-content-primary'
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          <span>2,000-Block Signal & Attribution</span>
          {attributionData && (
            <span className="w-1.5 h-1.5 rounded-full bg-telemetry-cyan ml-0.5" />
          )}
        </button>

        <button
          onClick={() => setActiveTab('neighborhood')}
          className={`flex items-center gap-1.5 px-3.5 py-2.5 font-medium border-b-2 transition-colors ${
            activeTab === 'neighborhood'
              ? 'border-telemetry-cyan text-telemetry-cyan font-bold'
              : 'border-transparent text-content-secondary hover:text-content-primary'
          }`}
        >
          <Grid className="w-3.5 h-3.5" />
          <span>Local 3×3 Neighborhood</span>
        </button>

        <button
          onClick={() => setActiveTab('telemetry')}
          className={`flex items-center gap-1.5 px-3.5 py-2.5 font-medium border-b-2 transition-colors ${
            activeTab === 'telemetry'
              ? 'border-telemetry-cyan text-telemetry-cyan font-bold'
              : 'border-transparent text-content-secondary hover:text-content-primary'
          }`}
        >
          <Sliders className="w-3.5 h-3.5" />
          <span>Spatial & Electrical Features</span>
        </button>
      </div>

      {/* 4. Tab Content */}
      <div className="p-4 bg-silicon-canvas">
        {/* TAB 1: Inference Trace */}
        {activeTab === 'trace' && (
          <InferenceTrace
            waferId={die.wafer_id}
            dieRow={die.die_row}
            dieCol={die.die_col}
            probModelA={die.prob_Model_A}
            decisionModelA={die.decision_Model_A}
            inGateB={die.in_gate_B}
            probModelB={die.in_gate_B ? die.prob_Model_B : null}
            decisionModelB={die.in_gate_B ? die.decision_Cascade_B : null}
            probCascade={die.prob_Cascade_B}
            finalDecision={die.decision_Cascade_B}
            actualLabel={die.actual_label}
            recoveredFn={die.recovered_FN_Gate_B}
            radialDist={die.radial_dist}
            edgeDist={die.edge_dist}
          />
        )}

        {/* TAB 2: 2,000-Block Signal & Attribution */}
        {activeTab === 'signal' && (
          <div>
            {isAttributionLoading ? (
              <div className="space-y-4 p-4">
                <LoadingSkeleton className="h-6 w-80" />
                <LoadingSkeleton className="h-44 w-full" />
                <LoadingSkeleton className="h-20 w-full" />
              </div>
            ) : attributionError ? (
              <div className="p-6 rounded bg-silicon-elevated border border-silicon-border text-center space-y-2">
                <AlertTriangle className="w-6 h-6 text-telemetry-amber mx-auto" />
                <div className="text-sm font-semibold text-content-primary">
                  Sequential Block Signal Unavailable
                </div>
                <p className="text-xs text-content-secondary max-w-md mx-auto">
                  The selected die does not contain an active raw 2,000-block signal sequence in the
                  current dataset slice. Inference and parametric telemetry remain fully accessible.
                </p>
              </div>
            ) : attributionData ? (
              <BlockSignalViewer
                key={`${die.wafer_id}-${die.die_row}-${die.die_col}`}
                data={attributionData}
              />
            ) : null}
          </div>
        )}

        {/* TAB 3: Local 3×3 Neighborhood */}
        {activeTab === 'neighborhood' && (
          <div className="p-4 bg-silicon-elevated rounded-lg border border-silicon-border space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <h4 className="text-sm font-bold text-content-primary font-mono">
                  Local 3×3 Die Neighborhood Context
                </h4>
                <p className="text-xs text-content-secondary mt-0.5">
                  Inspect adjacent dies to determine if die [{die.die_row}, {die.die_col}] is an isolated anomaly or part of a nearby spatial defect pattern. Click any neighbor to inspect.
                </p>
              </div>
              <StatusBadge
                label={`Center: [${die.die_row}, ${die.die_col}]`}
                variant={isFail ? 'fail' : 'pass'}
              />
            </div>

            <div className="grid grid-cols-3 gap-3 max-w-lg mx-auto font-mono text-xs pt-2">
              {neighborhoodGrid.map((cell, idx) => {
                const isCenter = cell.rowOffset === 0 && cell.colOffset === 0;
                const nDie = cell.die;
                const nFail = nDie?.decision_Cascade_B === 1;

                if (!nDie) {
                  return (
                    <div
                      key={idx}
                      className="p-3 rounded bg-silicon-canvas/60 border border-silicon-border/50 text-center opacity-40"
                    >
                      <div className="text-[10px] text-content-muted">
                        [{cell.row}, {cell.col}]
                      </div>
                      <div className="text-[11px] text-content-muted mt-1">Empty / Kerf</div>
                    </div>
                  );
                }

                return (
                  <button
                    key={idx}
                    onClick={() => onSelectDie && onSelectDie(nDie)}
                    className={`p-3 rounded text-center border transition-all ${
                      isCenter
                        ? 'border-telemetry-cyan bg-telemetry-blue/20 ring-2 ring-telemetry-cyan shadow-md'
                        : nFail
                        ? 'bg-telemetry-red/15 border-telemetry-red/40 hover:border-telemetry-red'
                        : 'bg-silicon-surface border-silicon-border hover:border-telemetry-cyan/60'
                    }`}
                  >
                    <div className="flex items-center justify-between text-[10px] text-content-muted">
                      <span>R{cell.row}/C{cell.col}</span>
                      {isCenter && <span className="text-telemetry-cyan font-bold">FOCUS</span>}
                    </div>
                    <div
                      className={`text-sm font-bold mt-1 ${
                        nFail ? 'text-telemetry-red' : 'text-telemetry-green'
                      }`}
                    >
                      {nFail ? 'FAIL' : 'PASS'}
                    </div>
                    <div className="text-[10px] text-content-secondary mt-0.5">
                      P: {(nDie.prob_Cascade_B * 100).toFixed(1)}%
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* TAB 4: Spatial & Electrical Features */}
        {activeTab === 'telemetry' && (
          <div className="bg-silicon-elevated rounded-lg border border-silicon-border p-4 space-y-4">
            <h4 className="text-sm font-bold text-content-primary font-mono">
              Parametric Die Telemetry & Physical Coordinates
            </h4>

            <div className="overflow-x-auto">
              <table className="w-full text-xs font-mono border-collapse">
                <tbody className="divide-y divide-silicon-border/50">
                  <tr>
                    <td className="py-2 px-3 text-content-muted">Physical Coordinates</td>
                    <td className="py-2 px-3 font-semibold text-content-primary">
                      Row: {die.die_row}, Column: {die.die_col}
                    </td>
                    <td className="py-2 px-3 text-content-muted">Wafer grid location</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-content-muted">Normalized Radial Distance</td>
                    <td className="py-2 px-3 font-semibold text-content-primary">
                      {die.radial_dist !== undefined && die.radial_dist !== null
                        ? die.radial_dist.toFixed(4)
                        : '0.4500 (Center)'}
                    </td>
                    <td className="py-2 px-3 text-content-muted">Radius from wafer geometric center [0, 1]</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-content-muted">Edge Distance Factor</td>
                    <td className="py-2 px-3 font-semibold text-content-primary">
                      {die.edge_dist !== undefined && die.edge_dist !== null
                        ? die.edge_dist.toFixed(4)
                        : '0.5500'}
                    </td>
                    <td className="py-2 px-3 text-content-muted">Distance to physical wafer edge boundary</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-content-muted">Peak Block Signal (b_max)</td>
                    <td className="py-2 px-3 font-semibold text-telemetry-purple">
                      {die.b_max !== undefined && die.b_max !== null
                        ? die.b_max.toFixed(2)
                        : '100.00 (Nominal)'}
                    </td>
                    <td className="py-2 px-3 text-content-muted">Maximum sequential reading (Defect &gt; 104.0)</td>
                  </tr>
                  <tr>
                    <td className="py-2 px-3 text-content-muted">Anomalous Blocks (cnt_104)</td>
                    <td className="py-2 px-3 font-semibold text-telemetry-amber">
                      {die.cnt_104 !== undefined && die.cnt_104 !== null
                        ? die.cnt_104
                        : '0'}
                    </td>
                    <td className="py-2 px-3 text-content-muted">Total sequential readings exceeding 104.0 threshold</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
