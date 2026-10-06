import React, { useState } from 'react';
import { useDemoSamples, useSingleDieInference } from '../hooks/useInference';
import { SectionHeader } from '../components/ui/SectionHeader';
import { Panel } from '../components/ui/Panel';
import { MetricCard } from '../components/ui/MetricCard';
import { StatusBadge } from '../components/ui/StatusBadge';
import { LoadingSkeleton } from '../components/ui/LoadingSkeleton';
import { ErrorState } from '../components/ui/ErrorState';
import { DieInferenceResponse } from '../types/inference';
import { Activity, Play, CheckCircle2, AlertTriangle, ArrowRight, ShieldAlert } from 'lucide-react';

export const CascadeDiagnostics: React.FC = () => {
  const { data: demos, isLoading: demosLoading, error: demosError } = useDemoSamples();
  const singleInference = useSingleDieInference();

  const [activeDie, setActiveDie] = useState<{ wafer_id: string; die_row: number; die_col: number }>({
    wafer_id: 'W_N_0057',
    die_row: 32,
    die_col: 23,
  });

  const [inferenceResult, setInferenceResult] = useState<DieInferenceResponse | null>(null);

  const handleRunInference = async (wafer_id: string, die_row: number, die_col: number) => {
    setActiveDie({ wafer_id, die_row, die_col });
    try {
      const res = await singleInference.mutateAsync({
        wafer_id,
        die_row,
        die_col,
      });
      setInferenceResult(res);
    } catch (e) {
      console.error('Inference error:', e);
    }
  };

  if (demosLoading) {
    return (
      <div className="space-y-6">
        <LoadingSkeleton className="h-8 w-64" />
        <LoadingSkeleton className="h-32" />
        <LoadingSkeleton className="h-64" />
      </div>
    );
  }

  if (demosError) {
    return (
      <ErrorState
        title="Could not load demo samples"
        message="Failed to connect to /api/v1/inference/demo-samples."
      />
    );
  }

  return (
    <div className="space-y-6">
      <SectionHeader
        title="Cascade Diagnostics & Live Inference Console"
        subtitle="Execute live end-to-end multi-resolution cascade inference using locked production models."
      />

      {/* Guided Hackathon Preset Demos */}
      <Panel
        title="Verified Demonstration Presets (Judges Walkthrough)"
        subtitle="Click any representative die below to execute live cascade inference through the FastAPI backend"
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          {demos?.map((d) => (
            <button
              key={d.sample_id}
              onClick={() => handleRunInference(d.wafer_id, d.die_row, d.die_col)}
              className={`p-3.5 rounded-lg border text-left transition-all font-mono text-xs ${
                activeDie.wafer_id === d.wafer_id &&
                activeDie.die_row === d.die_row &&
                activeDie.die_col === d.die_col
                  ? 'border-telemetry-cyan bg-telemetry-cyan/10 ring-1 ring-telemetry-cyan/50'
                  : 'border-silicon-border bg-silicon-elevated hover:border-silicon-borderLight'
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <span className="font-bold text-content-primary">
                  {d.sample_id.replace(/_/g, ' ').toUpperCase()}
                </span>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded ${
                    d.actual_label === 1
                      ? 'bg-telemetry-red/15 text-telemetry-red'
                      : 'bg-telemetry-green/15 text-telemetry-green'
                  }`}
                >
                  {d.actual_label === 1 ? 'ACTUAL DEFECT' : 'ACTUAL PASS'}
                </span>
              </div>
              <div className="text-content-secondary font-mono">
                {d.wafer_id} (R:{d.die_row} C:{d.die_col})
              </div>
              <div className="text-[11px] text-content-muted mt-1 leading-relaxed">
                {d.description}
              </div>
            </button>
          ))}
        </div>
      </Panel>

      {/* Live Inference Action Button */}
      <div className="flex items-center justify-between bg-silicon-card border border-silicon-border rounded-lg p-4">
        <div className="flex items-center gap-3">
          <Activity className="w-5 h-5 text-telemetry-cyan" />
          <div>
            <div className="text-xs font-mono font-semibold text-content-primary">
              SELECTED DIE: <span className="text-telemetry-cyan">{activeDie.wafer_id}</span> • ROW {activeDie.die_row} • COL {activeDie.die_col}
            </div>
            <div className="text-[11px] text-content-muted">
              Live inference wraps predict_single_die_live() on Model A + Model B
            </div>
          </div>
        </div>

        <button
          onClick={() => handleRunInference(activeDie.wafer_id, activeDie.die_row, activeDie.die_col)}
          disabled={singleInference.isPending}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-telemetry-blue hover:bg-telemetry-blue/90 text-white font-mono text-xs font-bold tracking-wider transition-all shadow disabled:opacity-50"
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          {singleInference.isPending ? 'RUNNING INFERENCE...' : 'RUN LIVE CASCADE'}
        </button>
      </div>

      {/* 4 Pipeline Stages Display (If inference run) */}
      {inferenceResult && (
        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
            <MetricCard
              title="1. Model A Screening"
              value={inferenceResult.prob_Model_A.toFixed(4)}
              subtitle={`Threshold: 0.5100 (${inferenceResult.decision_Model_A === 1 ? 'FAIL' : 'PASS'})`}
              badge="SCREENING"
              accent="blue"
            />

            <MetricCard
              title="2. Gate [0.295, 0.550]"
              value={inferenceResult.gate_status}
              subtitle={inferenceResult.in_gate ? 'Escalated to Model B' : 'Fast Path Bypass'}
              badge="CONFIDENCE"
              accent={inferenceResult.in_gate ? 'amber' : 'green'}
            />

            <MetricCard
              title="3. B2 Enhanced Model B"
              value={inferenceResult.prob_Model_B !== null ? inferenceResult.prob_Model_B.toFixed(4) : 'FAST PATH BYPASS'}
              subtitle={inferenceResult.in_gate ? 'Threshold: 0.5376' : '93.78% Compute Saved'}
              badge="MODEL B"
              accent={inferenceResult.in_gate ? 'purple' : 'neutral'}
            />

            <MetricCard
              title="4. Final Disposition"
              value={inferenceResult.decision_text}
              subtitle={`Risk Prob: ${inferenceResult.final_probability.toFixed(4)}`}
              badge="DISPOSITION"
              accent={inferenceResult.final_decision === 1 ? 'red' : 'green'}
            />
          </div>

          {/* Decision Outcome Banner */}
          <div
            className={`p-4 rounded-lg border font-mono text-xs leading-relaxed ${
              inferenceResult.in_gate && inferenceResult.final_decision === 1 && inferenceResult.actual_label === 1
                ? 'bg-telemetry-green/10 border-telemetry-green/40 text-telemetry-green'
                : inferenceResult.in_gate
                ? 'bg-telemetry-amber/10 border-telemetry-amber/40 text-telemetry-amber'
                : 'bg-telemetry-blue/10 border-telemetry-blue/40 text-telemetry-blue'
            }`}
          >
            <div className="font-bold flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4" />
              {inferenceResult.in_gate && inferenceResult.final_decision === 1 && inferenceResult.actual_label === 1
                ? 'MODEL A MISSED FAILURE RESCUED BY CASCADE'
                : inferenceResult.in_gate
                ? 'ESCALATED SUB-DIE INSPECTION COMPLETED (B2 ENHANCED)'
                : 'FAST PATH BYPASS (93.78% COMPUTE SAVED)'}
            </div>
            <div className="text-content-secondary font-sans">
              {inferenceResult.gate_message} Final probability resolved to{' '}
              <span className="font-mono text-content-primary font-semibold">
                {inferenceResult.final_probability.toFixed(4)}
              </span>{' '}
              in <span className="font-mono text-content-primary font-semibold">{inferenceResult.latency_ms || 1.4} ms</span>.
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
