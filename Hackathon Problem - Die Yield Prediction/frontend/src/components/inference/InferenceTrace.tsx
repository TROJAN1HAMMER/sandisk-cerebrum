import React from 'react';
import {
  Cpu,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Sparkles,
  Info,
  Sliders,
} from 'lucide-react';

interface InferenceTraceProps {
  waferId: string;
  dieRow: number;
  dieCol: number;
  probModelA: number;
  thresholdModelA?: number;
  decisionModelA: number;
  inGateB: boolean;
  gateLower?: number;
  gateUpper?: number;
  probModelB?: number | null;
  thresholdModelB?: number;
  decisionModelB?: number | null;
  probCascade: number;
  finalDecision: number;
  actualLabel?: number | null;
  recoveredFn?: boolean | number;
  radialDist?: number | null;
  edgeDist?: number | null;
  className?: string;
}

export const InferenceTrace: React.FC<InferenceTraceProps> = ({
  waferId,
  dieRow,
  dieCol,
  probModelA,
  thresholdModelA = 0.51,
  decisionModelA,
  inGateB,
  gateLower = 0.25,
  gateUpper = 0.55,
  probModelB,
  thresholdModelB = 0.5376,
  decisionModelB,
  probCascade,
  finalDecision,
  actualLabel,
  recoveredFn,
  radialDist,
  edgeDist,
  className = '',
}) => {
  const isRecovered = Boolean(recoveredFn);
  const isFail = finalDecision === 1;

  // Percentage position of probModelA on [0, 1] scale for the threshold gauge
  const pAPct = Math.max(0, Math.min(100, probModelA * 100));
  const lowerPct = gateLower * 100;
  const upperPct = gateUpper * 100;
  const threshAPct = thresholdModelA * 100;

  return (
    <div className={`space-y-4 ${className}`}>
      {/* 1. Visual Pipeline Flow - Distinct for Escalated vs Bypassed dies */}
      <div
        className={`grid gap-3 relative ${
          inGateB ? 'grid-cols-1 md:grid-cols-5' : 'grid-cols-1 md:grid-cols-4'
        }`}
      >
        {/* Step 1: Input Die */}
        <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-3.5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-silicon-interactive text-content-muted border border-silicon-border uppercase">
                Stage 1 • Input
              </span>
              <Cpu className="w-3.5 h-3.5 text-telemetry-cyan" />
            </div>
            <div className="text-xs font-mono font-bold text-content-primary">
              [{dieRow}, {dieCol}]
            </div>
            <div className="text-[11px] font-mono text-content-muted mt-0.5 truncate">
              {waferId}
            </div>
          </div>

          <div className="mt-3 pt-2 border-t border-silicon-border text-[10px] font-mono space-y-0.5 text-content-secondary">
            {radialDist !== undefined && radialDist !== null && (
              <div className="flex justify-between">
                <span>Radial:</span>
                <span className="text-content-primary">{radialDist.toFixed(3)}</span>
              </div>
            )}
            {edgeDist !== undefined && edgeDist !== null && (
              <div className="flex justify-between">
                <span>Edge:</span>
                <span className="text-content-primary">{edgeDist.toFixed(3)}</span>
              </div>
            )}
          </div>
        </div>

        {/* Step 2: LightGBM Model A Screening */}
        <div className="bg-silicon-elevated border border-silicon-border rounded-lg p-3.5 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-telemetry-blue/20 text-telemetry-blue border border-telemetry-blue/30 uppercase font-semibold">
                Stage 2 • LightGBM
              </span>
              <Zap className="w-3.5 h-3.5 text-telemetry-blue" />
            </div>
            <div className="text-xs font-mono font-bold text-content-primary">
              P_A = {(probModelA * 100).toFixed(2)}%
            </div>
            <div className="text-[10px] font-mono text-content-muted mt-0.5">
              τ_A = {(thresholdModelA * 100).toFixed(1)}%
            </div>
          </div>

          <div className="mt-3 pt-2 border-t border-silicon-border flex items-center justify-between text-[11px] font-mono">
            <span className="text-content-muted">Decision:</span>
            <span
              className={`font-semibold ${
                decisionModelA === 1 ? 'text-telemetry-red' : 'text-telemetry-green'
              }`}
            >
              {decisionModelA === 1 ? 'FAIL (1)' : 'PASS (0)'}
            </span>
          </div>
        </div>

        {/* Step 3: Gate B Routing */}
        <div
          className={`border rounded-lg p-3.5 flex flex-col justify-between transition-colors ${
            inGateB
              ? 'bg-telemetry-amber/10 border-telemetry-amber/40 shadow-sm'
              : 'bg-telemetry-blue/10 border-telemetry-blue/40 shadow-sm'
          }`}
        >
          <div>
            <div className="flex items-center justify-between mb-2">
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase font-bold ${
                  inGateB
                    ? 'bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/40'
                    : 'bg-telemetry-blue/20 text-telemetry-blue border border-telemetry-blue/30'
                }`}
              >
                Stage 3 • Gate B
              </span>
              <Layers
                className={`w-3.5 h-3.5 ${
                  inGateB ? 'text-telemetry-amber' : 'text-telemetry-blue'
                }`}
              />
            </div>
            <div className="text-xs font-mono font-semibold text-content-primary">
              [{gateLower.toFixed(2)}, {gateUpper.toFixed(2)}]
            </div>
            <div className="text-[10px] font-mono text-content-muted mt-0.5">
              {inGateB ? 'Inside uncertainty band' : 'Outside band → Bypassed'}
            </div>
          </div>

          <div className="mt-3 pt-2 border-t border-silicon-border flex items-center justify-between text-[11px] font-mono">
            <span className="text-content-muted">Routing:</span>
            <span
              className={`font-bold uppercase ${
                inGateB ? 'text-telemetry-amber' : 'text-telemetry-blue'
              }`}
            >
              {inGateB ? 'ESCALATED' : 'FAST PATH'}
            </span>
          </div>
        </div>

        {/* Step 4: Model B (ONLY RENDERED IF ESCALATED!) */}
        {inGateB && (
          <div className="bg-silicon-elevated border border-telemetry-purple/40 rounded-lg p-3.5 flex flex-col justify-between shadow-sm">
            <div>
              <div className="flex items-center justify-between mb-2">
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/40 uppercase font-bold">
                  Stage 4 • Model B (CNN)
                </span>
                <ShieldCheck className="w-3.5 h-3.5 text-telemetry-purple" />
              </div>

              {probModelB !== undefined && probModelB !== null ? (
                <>
                  <div className="text-xs font-mono font-bold text-telemetry-purple">
                    P_B = {(probModelB * 100).toFixed(2)}%
                  </div>
                  <div className="text-[10px] font-mono text-content-muted mt-0.5">
                    τ_B = {(thresholdModelB * 100).toFixed(2)}%
                  </div>
                </>
              ) : (
                <>
                  <div className="text-xs font-mono text-content-muted">Executing...</div>
                  <div className="text-[10px] font-mono text-content-muted mt-0.5">1D CNN Inference</div>
                </>
              )}
            </div>

            <div className="mt-3 pt-2 border-t border-silicon-border flex items-center justify-between text-[11px] font-mono">
              <span className="text-content-muted">Model B Dec:</span>
              <span className="font-semibold text-content-primary">
                {decisionModelB !== undefined && decisionModelB !== null
                  ? decisionModelB === 1
                    ? 'FAIL (1)'
                    : 'PASS (0)'
                  : '—'}
              </span>
            </div>
          </div>
        )}

        {/* Step 5 / Step 4: Final Disposition */}
        <div
          className={`border rounded-lg p-3.5 flex flex-col justify-between ${
            isFail
              ? 'bg-telemetry-red/10 border-telemetry-red/50 shadow-sm'
              : 'bg-telemetry-green/10 border-telemetry-green/50 shadow-sm'
          }`}
        >
          <div>
            <div className="flex items-center justify-between mb-2">
              <span
                className={`text-[10px] font-mono px-1.5 py-0.5 rounded uppercase font-bold ${
                  isFail
                    ? 'bg-telemetry-red/20 text-telemetry-red border border-telemetry-red/40'
                    : 'bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/40'
                }`}
              >
                {inGateB ? 'Stage 5 • Final' : 'Stage 4 • Final'}
              </span>
              {isFail ? (
                <XCircle className="w-3.5 h-3.5 text-telemetry-red" />
              ) : (
                <CheckCircle2 className="w-3.5 h-3.5 text-telemetry-green" />
              )}
            </div>
            <div
              className={`text-sm font-mono font-bold tracking-tight ${
                isFail ? 'text-telemetry-red' : 'text-telemetry-green'
              }`}
            >
              {isFail ? 'DEFECT (FAIL)' : 'YIELD (PASS)'}
            </div>
            <div className="text-[11px] font-mono text-content-secondary mt-0.5 font-semibold">
              Final P = {(probCascade * 100).toFixed(2)}%
            </div>
          </div>

          <div className="mt-3 pt-2 border-t border-silicon-border/70 flex items-center justify-between text-[10px] font-mono">
            <span className="text-content-muted">Ground Truth:</span>
            <span
              className={`font-bold ${
                actualLabel === 1 ? 'text-telemetry-red' : 'text-telemetry-green'
              }`}
            >
              {actualLabel !== undefined && actualLabel !== null
                ? actualLabel === 1
                  ? 'ACTUAL FAIL (1)'
                  : 'ACTUAL PASS (0)'
                : 'UNLABELED'}
            </span>
          </div>
        </div>
      </div>

      {/* 2. Model Threshold & Gate Position Gauge */}
      <div className="p-3.5 bg-silicon-elevated rounded-lg border border-silicon-border space-y-2 font-mono text-xs">
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-content-muted uppercase tracking-wider font-semibold flex items-center gap-1.5">
            <Sliders className="w-3.5 h-3.5 text-telemetry-cyan" />
            Uncertainty Gate Alignment
          </span>
          <span className="text-content-secondary">
            P_A = <strong className="text-telemetry-cyan">{(probModelA * 100).toFixed(2)}%</strong>{' '}
            {inGateB ? (
              <span className="text-telemetry-amber font-bold">(Inside Gate B)</span>
            ) : (
              <span className="text-telemetry-blue font-bold">(Outside Gate B)</span>
            )}
          </span>
        </div>

        {/* Gauge Bar */}
        <div className="relative pt-4 pb-2">
          {/* Main track */}
          <div className="h-3 w-full bg-silicon-canvas rounded-full relative overflow-hidden border border-silicon-border">
            {/* Fast path low zone [0, 0.25] */}
            <div
              className="absolute left-0 top-0 bottom-0 bg-telemetry-blue/20"
              style={{ width: `${lowerPct}%` }}
              title="Fast Path Pass [0.00, 0.25]"
            />
            {/* Gate B Escalation Band [0.25, 0.55] */}
            <div
              className="absolute top-0 bottom-0 bg-telemetry-amber/30 border-x border-telemetry-amber/60"
              style={{ left: `${lowerPct}%`, width: `${upperPct - lowerPct}%` }}
              title="Gate B Uncertainty Interval [0.25, 0.55]"
            />
            {/* Fast path high zone [0.55, 1.00] */}
            <div
              className="absolute right-0 top-0 bottom-0 bg-telemetry-red/20"
              style={{ left: `${upperPct}%`, width: `${100 - upperPct}%` }}
              title="Fast Path Defect [0.55, 1.00]"
            />
          </div>

          {/* Model A Threshold line (0.5100) */}
          <div
            className="absolute top-2 bottom-0 w-0.5 bg-telemetry-red pointer-events-none"
            style={{ left: `${threshAPct}%` }}
            title="Model A Threshold = 0.5100"
          />

          {/* Current die pointer pin */}
          <div
            className="absolute top-0 -ml-2 flex flex-col items-center pointer-events-none transition-all duration-300"
            style={{ left: `${pAPct}%` }}
          >
            <div className="w-4 h-4 rounded-full bg-telemetry-cyan border-2 border-white shadow-md flex items-center justify-center animate-pulse" />
            <div className="w-0.5 h-3 bg-telemetry-cyan" />
          </div>

          {/* Labels underneath */}
          <div className="flex justify-between text-[10px] text-content-muted mt-2">
            <span>0.00</span>
            <span className="text-telemetry-blue">Bypass &lt; 0.25</span>
            <span className="text-telemetry-amber font-bold">Gate B [0.25, 0.55]</span>
            <span className="text-telemetry-red">τ_A: 0.51</span>
            <span>1.00</span>
          </div>
        </div>
      </div>

      {/* 3. Deterministic Decision Narrative */}
      <div className="p-3 bg-silicon-base rounded-lg border border-silicon-border flex flex-wrap items-center justify-between gap-3 text-xs font-mono text-content-secondary">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-telemetry-cyan shrink-0" />
          <span>
            {inGateB ? (
              <>
                LightGBM Model A produced intermediate probability{' '}
                <strong className="text-content-primary">
                  {(probModelA * 100).toFixed(2)}%
                </strong>
                , placing the die inside Gate B [{gateLower.toFixed(2)}, {gateUpper.toFixed(2)}]. The die was
                therefore escalated to Model B (1D CNN), which produced{' '}
                <strong className="text-telemetry-purple font-bold">
                  {probModelB !== undefined && probModelB !== null
                    ? `${(probModelB * 100).toFixed(2)}%`
                    : 'an evaluated probability'}
                </strong>
                , resulting in a final{' '}
                <strong className={isFail ? 'text-telemetry-red font-bold' : 'text-telemetry-green font-bold'}>
                  {isFail ? 'FAIL' : 'PASS'}
                </strong>{' '}
                prediction.
              </>
            ) : (
              <>
                LightGBM Model A produced confident probability{' '}
                <strong className="text-content-primary">
                  {(probModelA * 100).toFixed(2)}%
                </strong>
                , outside the Gate B [{gateLower.toFixed(2)}, {gateUpper.toFixed(2)}] escalation interval. Model B
                was therefore bypassed, saving heavy block signal compute, and the final decision remained{' '}
                <strong className={isFail ? 'text-telemetry-red font-bold' : 'text-telemetry-green font-bold'}>
                  {isFail ? 'FAIL' : 'PASS'}
                </strong>.
              </>
            )}
          </span>
        </div>

        {isRecovered && (
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40 font-bold">
            <Sparkles className="w-3.5 h-3.5" />
            <span>RESCUED FALSE NEGATIVE (+1 YIELD DEFECT RECOVERED)</span>
          </div>
        )}
      </div>
    </div>
  );
};
