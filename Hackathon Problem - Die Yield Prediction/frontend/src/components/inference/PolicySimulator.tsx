import React, { useState } from 'react';
import { useMutation } from '@tanstack/react-query';
import { evaluationService } from '../../services/evaluation';
import { PolicySimulationRequest, PolicySimulationResponse } from '../../types/models';
import { Panel } from '../ui/Panel';
import { MetricCard } from '../ui/MetricCard';
import { Sliders, RotateCcw, AlertTriangle, ShieldCheck, Zap, ArrowRight, TrendingUp } from 'lucide-react';

export const PolicySimulator: React.FC = () => {
  // Frozen baseline settings
  const defaultParams = {
    tau_A: 0.5100,
    gate_lower: 0.2950,
    gate_upper: 0.5500,
    tau_B: 0.5376,
    secondary_budget_pct: null,
  };

  const [tauA, setTauA] = useState<number>(defaultParams.tau_A);
  const [gateLower, setGateLower] = useState<number>(defaultParams.gate_lower);
  const [gateUpper, setGateUpper] = useState<number>(defaultParams.gate_upper);
  const [tauB, setTauB] = useState<number>(defaultParams.tau_B);
  const [budgetPct, setBudgetPct] = useState<number | ''>('');

  const [simResult, setSimResult] = useState<PolicySimulationResponse | null>(null);

  const mutation = useMutation({
    mutationFn: (req: PolicySimulationRequest) => evaluationService.simulatePolicy(req),
    onSuccess: (data) => {
      setSimResult(data);
    },
  });

  // Run simulation on mount or param change
  const handleSimulate = (override?: Partial<PolicySimulationRequest>) => {
    const payload: PolicySimulationRequest = {
      tau_A: override?.tau_A ?? tauA,
      gate_lower: override?.gate_lower ?? gateLower,
      gate_upper: override?.gate_upper ?? gateUpper,
      tau_B: override?.tau_B ?? tauB,
      secondary_budget_pct: override?.secondary_budget_pct !== undefined 
        ? override.secondary_budget_pct 
        : (budgetPct === '' ? null : Number(budgetPct)),
    };
    mutation.mutate(payload);
  };

  const handleReset = () => {
    setTauA(defaultParams.tau_A);
    setGateLower(defaultParams.gate_lower);
    setGateUpper(defaultParams.gate_upper);
    setTauB(defaultParams.tau_B);
    setBudgetPct('');
    handleSimulate({
      tau_A: defaultParams.tau_A,
      gate_lower: defaultParams.gate_lower,
      gate_upper: defaultParams.gate_upper,
      tau_B: defaultParams.tau_B,
      secondary_budget_pct: null,
    });
  };

  // Initial run if no result
  React.useEffect(() => {
    handleSimulate();
  }, []);

  const res = simResult?.simulated_metrics;
  const delta = simResult?.baseline_comparison;

  return (
    <Panel
      title="What-If Inspection Policy Simulator"
      subtitle="Evaluate trade-offs between secondary analysis budget, defect recall, and compute savings."
    >
      <div className="space-y-6">
        {/* Prominent Mandatory Safety Label */}
        <div className="p-3.5 rounded-lg bg-telemetry-amber/10 border border-telemetry-amber/30 font-mono text-xs flex items-center justify-between gap-3 text-telemetry-amber">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 shrink-0 text-telemetry-amber" />
            <span className="font-bold tracking-wide">
              WHAT-IF SIMULATION — DOES NOT CHANGE PRODUCTION POLICY
            </span>
          </div>
          <span className="text-[11px] text-content-secondary hidden md:inline">
            In-memory evaluation across 27,972 canonical validation dies
          </span>
        </div>

        {/* Operating Policy Presets Bar */}
        <div className="flex flex-wrap items-center gap-2 p-3 rounded-lg bg-silicon-canvas border border-silicon-border/80 font-mono text-xs">
          <span className="text-content-secondary font-semibold flex items-center gap-1.5 mr-2">
            <Sliders className="w-3.5 h-3.5 text-telemetry-cyan" />
            POLICY PRESETS:
          </span>
          <button
            onClick={() => {
              setTauA(0.5100); setGateLower(0.2950); setGateUpper(0.5500); setTauB(0.5376);
              handleSimulate({ tau_A: 0.5100, gate_lower: 0.2950, gate_upper: 0.5500, tau_B: 0.5376 });
            }}
            className="px-2.5 py-1 rounded bg-telemetry-green/15 text-telemetry-green border border-telemetry-green/40 hover:bg-telemetry-green/25 font-bold transition-all text-[11px]"
          >
            B2 Production Standard [0.295, 0.550]
          </button>
          <button
            onClick={() => {
              setTauA(0.5100); setGateLower(0.2500); setGateUpper(0.5500); setTauB(0.5200);
              handleSimulate({ tau_A: 0.5100, gate_lower: 0.2500, gate_upper: 0.5500, tau_B: 0.5200 });
            }}
            className="px-2.5 py-1 rounded bg-telemetry-purple/15 text-telemetry-purple border border-telemetry-purple/40 hover:bg-telemetry-purple/25 font-bold transition-all text-[11px]"
          >
            B3 Staged Candidate [0.250, 0.550]
          </button>
          <button
            onClick={() => {
              setTauA(0.5100); setGateLower(0.2000); setGateUpper(0.5000); setTauB(0.5000);
              handleSimulate({ tau_A: 0.5100, gate_lower: 0.2000, gate_upper: 0.5000, tau_B: 0.5000 });
            }}
            className="px-2.5 py-1 rounded bg-telemetry-amber/15 text-telemetry-amber border border-telemetry-amber/40 hover:bg-telemetry-amber/25 font-bold transition-all text-[11px]"
          >
            Experimental High-Recall [0.200, 0.500]
          </button>
        </div>

        {/* Controls Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 p-4 rounded-lg bg-silicon-elevated border border-silicon-border">
          {/* Tau A Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-content-secondary">Model A Threshold (τ_A):</span>
              <span className="text-telemetry-cyan font-bold">{tauA.toFixed(4)}</span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.70"
              step="0.005"
              value={tauA}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setTauA(val);
                handleSimulate({ tau_A: val });
              }}
              className="w-full accent-telemetry-cyan cursor-pointer"
            />
            <div className="text-[10px] text-content-muted font-mono flex justify-between">
              <span>0.3000 (Sensitive)</span>
              <span>0.7000 (Conservative)</span>
            </div>
          </div>

          {/* Gate Range Sliders */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-content-secondary">Gate Lower / Upper:</span>
              <span className="text-telemetry-amber font-bold">[{gateLower.toFixed(3)}, {gateUpper.toFixed(3)}]</span>
            </div>
            <div className="grid grid-cols-2 gap-2">
              <input
                type="range"
                min="0.15"
                max="0.45"
                step="0.005"
                value={gateLower}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setGateLower(val);
                  handleSimulate({ gate_lower: val });
                }}
                className="w-full accent-telemetry-amber cursor-pointer"
              />
              <input
                type="range"
                min="0.45"
                max="0.75"
                step="0.005"
                value={gateUpper}
                onChange={(e) => {
                  const val = parseFloat(e.target.value);
                  setGateUpper(val);
                  handleSimulate({ gate_upper: val });
                }}
                className="w-full accent-telemetry-amber cursor-pointer"
              />
            </div>
            <div className="text-[10px] text-content-muted font-mono flex justify-between">
              <span>Wider Gate = More Escalation</span>
            </div>
          </div>

          {/* Tau B Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-content-secondary">Model B2 Threshold (τ_B):</span>
              <span className="text-telemetry-purple font-bold">{tauB.toFixed(4)}</span>
            </div>
            <input
              type="range"
              min="0.30"
              max="0.70"
              step="0.005"
              value={tauB}
              onChange={(e) => {
                const val = parseFloat(e.target.value);
                setTauB(val);
                handleSimulate({ tau_B: val });
              }}
              className="w-full accent-telemetry-purple cursor-pointer"
            />
            <div className="text-[10px] text-content-muted font-mono flex justify-between">
              <span>0.3000 (Aggressive)</span>
              <span>0.7000 (Strict)</span>
            </div>
          </div>

          {/* Secondary Budget Cap & Reset */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs font-mono">
              <span className="text-content-secondary">Escalation Budget Cap (%):</span>
              <span className="text-content-primary font-bold">{budgetPct === '' ? 'Uncapped' : `${budgetPct}%`}</span>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min="1"
                max="30"
                step="0.5"
                placeholder="e.g. 8.0"
                value={budgetPct}
                onChange={(e) => {
                  const val = e.target.value === '' ? '' : parseFloat(e.target.value);
                  setBudgetPct(val);
                  handleSimulate({ secondary_budget_pct: val === '' ? null : Number(val) });
                }}
                className="w-full px-2.5 py-1.5 rounded bg-silicon-canvas border border-silicon-border text-xs font-mono text-content-primary focus:outline-none focus:border-telemetry-cyan"
              />
              <button
                onClick={handleReset}
                title="Reset to Frozen Production Policy"
                className="px-2.5 py-1.5 rounded bg-silicon-canvas hover:bg-silicon-interactive border border-silicon-border text-xs font-mono text-content-secondary hover:text-content-primary flex items-center gap-1 transition-colors shrink-0"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                <span>Reset</span>
              </button>
            </div>
            <div className="text-[10px] text-content-muted font-mono">
              Simulates fab tester throughput quota limits
            </div>
          </div>
        </div>

        {/* Dynamic Trade-Off Visualization Card */}
        <div className="p-4 rounded-lg bg-silicon-surface border border-silicon-border">
          <div className="flex items-center justify-between mb-3">
            <div className="text-xs font-mono uppercase tracking-wider text-content-secondary font-semibold flex items-center gap-2">
              <Zap className="w-4 h-4 text-telemetry-cyan" />
              <span>Inspection Policy Trade-Off Dynamics</span>
            </div>
            <div className="text-xs font-mono">
              {simResult?.baseline_comparison?.delta_f1 === 0 ? (
                <span className="px-2 py-0.5 rounded bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/40 font-bold">
                  MATCHES PRODUCTION BASELINE (B2)
                </span>
              ) : (
                <span className="px-2 py-0.5 rounded bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/40 font-bold">
                  CUSTOM POLICY SIMULATION
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-3">
            <MetricCard
              title="Escalation Rate"
              value={res ? `${res.escalation_rate_pct.toFixed(2)}%` : '—'}
              subtitle={res ? `${res.escalation_count.toLocaleString()} dies escalated` : '—'}
              badge={delta && delta.delta_escalation_pct !== 0 ? `${delta.delta_escalation_pct > 0 ? '+' : ''}${delta.delta_escalation_pct.toFixed(2)}%` : 'BASELINE'}
              accent={res && res.escalation_rate_pct > 10 ? 'amber' : 'blue'}
            />
            <MetricCard
              title="Compute Saved"
              value={res ? `${res.compute_saved_pct.toFixed(2)}%` : '—'}
              subtitle={res ? `${(27972 - res.escalation_count).toLocaleString()} fast-path dies` : '—'}
              badge={delta && delta.delta_compute_saved_pct !== 0 ? `${delta.delta_compute_saved_pct > 0 ? '+' : ''}${delta.delta_compute_saved_pct.toFixed(2)}%` : 'BASELINE'}
              accent="cyan"
            />
            <MetricCard
              title="Precision"
              value={res ? `${(res.precision * 100).toFixed(2)}%` : '—'}
              subtitle={res ? `${res.fp} False Alarms` : '—'}
              badge={delta && delta.delta_precision !== 0 ? `${delta.delta_precision > 0 ? '+' : ''}${(delta.delta_precision * 100).toFixed(2)}%` : 'BASELINE'}
              accent="green"
            />
            <MetricCard
              title="Recall"
              value={res ? `${(res.recall * 100).toFixed(2)}%` : '—'}
              subtitle={res ? `${res.tp} TP / ${res.fn} Missed` : '—'}
              badge={delta && delta.delta_recall !== 0 ? `${delta.delta_recall > 0 ? '+' : ''}${(delta.delta_recall * 100).toFixed(2)}%` : 'BASELINE'}
              accent="blue"
            />
            <MetricCard
              title="Cascade F1"
              value={res ? res.f1.toFixed(4) : '—'}
              subtitle={`F2: ${res ? res.f2.toFixed(4) : '—'}`}
              badge={delta && delta.delta_f1 !== 0 ? `${delta.delta_f1 > 0 ? '+' : ''}${delta.delta_f1.toFixed(4)}` : 'BASELINE'}
              accent="purple"
            />
            <MetricCard
              title="MCC"
              value={res ? res.mcc.toFixed(4) : '—'}
              subtitle={`Net TP: ${res && res.net_tp_gain > 0 ? '+' : ''}${res ? res.net_tp_gain : 0}`}
              badge="CORRELATION"
              accent="green"
            />
          </div>

          {/* Trade-off summary text */}
          <div className="mt-4 p-3 rounded bg-silicon-elevated/80 border border-silicon-border/60 font-mono text-xs text-content-secondary flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-telemetry-cyan font-bold">INSIGHT:</span>
              <span>
                {res && res.escalation_rate_pct > 8
                  ? 'Higher secondary analysis budget yields greater defect recall at the cost of additional sub-die computation.'
                  : 'Lower secondary analysis budget maximizes tester throughput while risking missed borderline defects.'}
              </span>
            </div>
            <div className="text-[11px] text-content-muted">
              Authoritative population: 27,972 dies
            </div>
          </div>
        </div>
      </div>
    </Panel>
  );
};
