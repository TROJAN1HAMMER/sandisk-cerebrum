import React, { useState } from 'react';
import { WaferSummaryItem } from '../../types/wafers';
import { MetricCard } from '../ui/MetricCard';
import { StatusBadge } from '../ui/StatusBadge';
import { X, ArrowRight, Disc, Layers } from 'lucide-react';

interface CrossWaferCompareProps {
  currentWafer: WaferSummaryItem;
  allWafers: WaferSummaryItem[];
  onClose: () => void;
  onSwitchWafer: (waferId: string) => void;
}

export const CrossWaferCompare: React.FC<CrossWaferCompareProps> = ({
  currentWafer,
  allWafers,
  onClose,
  onSwitchWafer,
}) => {
  // Default to a different wafer in the list
  const otherWafers = allWafers.filter((w) => w.wafer_id !== currentWafer.wafer_id);
  const [selectedOtherId, setSelectedOtherId] = useState<string>(
    otherWafers[0]?.wafer_id || ''
  );

  const otherWafer = allWafers.find((w) => w.wafer_id === selectedOtherId) || otherWafers[0];

  if (!otherWafer) return null;

  const currentYield = 100 - currentWafer.fail_rate_pct;
  const otherYield = 100 - otherWafer.fail_rate_pct;
  const yieldDelta = currentYield - otherYield;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="bg-silicon-surface border border-silicon-border rounded-xl shadow-2xl w-full max-w-4xl overflow-hidden flex flex-col font-mono text-xs">
        {/* Modal Header */}
        <div className="flex items-center justify-between px-6 py-4 bg-silicon-elevated border-b border-silicon-border">
          <div className="flex items-center gap-3">
            <Disc className="w-5 h-5 text-telemetry-cyan" />
            <div>
              <h2 className="text-base font-bold text-content-primary">
                Cross-Wafer Comparative Telemetry
              </h2>
              <p className="text-xs text-content-muted">
                Side-by-side yield, defect concentration, and cascade routing comparison
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-silicon-interactive text-content-muted hover:text-content-primary transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Wafer Selector Bar */}
        <div className="px-6 py-3 bg-silicon-card border-b border-silicon-border flex items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="text-content-muted">Base Wafer:</span>
            <span className="text-telemetry-cyan font-bold">{currentWafer.wafer_id}</span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-content-muted">Comparison Target:</span>
            <select
              value={selectedOtherId}
              onChange={(e) => setSelectedOtherId(e.target.value)}
              className="bg-silicon-interactive border border-silicon-border text-content-primary rounded px-2.5 py-1 focus:outline-none focus:border-telemetry-cyan"
            >
              {otherWafers.map((w) => (
                <option key={w.wafer_id} value={w.wafer_id}>
                  {w.wafer_id} ({w.total_dies} dies, {w.actual_failures} defects)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Side-by-Side Comparison Grid */}
        <div className="p-6 grid grid-cols-2 gap-6 bg-silicon-canvas">
          {/* Left Column: Current Wafer */}
          <div className="bg-silicon-card border border-silicon-border rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-silicon-border pb-2">
              <span className="text-sm font-bold text-telemetry-cyan">
                {currentWafer.wafer_id}
              </span>
              <StatusBadge variant="info" label="CURRENT VIEW" />
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              <MetricCard
                title="Total Population"
                value={currentWafer.total_dies.toLocaleString()}
                subtitle="Evaluated Dies"
                accent="neutral"
              />
              <MetricCard
                title="Yield Rate"
                value={`${currentYield.toFixed(2)}%`}
                subtitle={`${currentWafer.actual_failures} defects`}
                accent="green"
              />
              <MetricCard
                title="Defect Rate"
                value={`${currentWafer.fail_rate_pct.toFixed(2)}%`}
                subtitle="Ground Truth Base"
                accent="red"
              />
              <MetricCard
                title="Stage 2 Escalations"
                value={currentWafer.escalated_dies.toLocaleString()}
                subtitle={`${((currentWafer.escalated_dies / currentWafer.total_dies) * 100).toFixed(1)}% escalated`}
                accent="amber"
              />
            </div>

            <div className="p-3 bg-silicon-elevated rounded border border-silicon-border text-[11px] space-y-1 text-content-secondary">
              <div className="flex justify-between">
                <span>Model A Detections:</span>
                <span className="text-content-primary font-semibold">{currentWafer.model_a_detections}</span>
              </div>
              <div className="flex justify-between">
                <span>Cascade Detections:</span>
                <span className="text-content-primary font-semibold">{currentWafer.cascade_detections}</span>
              </div>
              <div className="flex justify-between">
                <span>Rescued False Negatives:</span>
                <span className="text-telemetry-green font-semibold">+{currentWafer.recovered_fns}</span>
              </div>
            </div>
          </div>

          {/* Right Column: Other Wafer */}
          <div className="bg-silicon-card border border-silicon-border rounded-lg p-4 space-y-4">
            <div className="flex items-center justify-between border-b border-silicon-border pb-2">
              <span className="text-sm font-bold text-telemetry-purple">
                {otherWafer.wafer_id}
              </span>
              <button
                onClick={() => {
                  onSwitchWafer(otherWafer.wafer_id);
                  onClose();
                }}
                className="text-[11px] text-telemetry-cyan hover:underline flex items-center gap-1 font-semibold"
              >
                Switch to this wafer <ArrowRight className="w-3 h-3" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2.5">
              <MetricCard
                title="Total Population"
                value={otherWafer.total_dies.toLocaleString()}
                subtitle="Evaluated Dies"
                accent="neutral"
              />
              <MetricCard
                title="Yield Rate"
                value={`${otherYield.toFixed(2)}%`}
                subtitle={`${otherWafer.actual_failures} defects`}
                accent="green"
              />
              <MetricCard
                title="Defect Rate"
                value={`${otherWafer.fail_rate_pct.toFixed(2)}%`}
                subtitle="Ground Truth Base"
                accent="red"
              />
              <MetricCard
                title="Stage 2 Escalations"
                value={otherWafer.escalated_dies.toLocaleString()}
                subtitle={`${((otherWafer.escalated_dies / otherWafer.total_dies) * 100).toFixed(1)}% escalated`}
                accent="amber"
              />
            </div>

            <div className="p-3 bg-silicon-elevated rounded border border-silicon-border text-[11px] space-y-1 text-content-secondary">
              <div className="flex justify-between">
                <span>Model A Detections:</span>
                <span className="text-content-primary font-semibold">{otherWafer.model_a_detections}</span>
              </div>
              <div className="flex justify-between">
                <span>Cascade Detections:</span>
                <span className="text-content-primary font-semibold">{otherWafer.cascade_detections}</span>
              </div>
              <div className="flex justify-between">
                <span>Rescued False Negatives:</span>
                <span className="text-telemetry-green font-semibold">+{otherWafer.recovered_fns}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Delta Summary Footer */}
        <div className="px-6 py-3 bg-silicon-elevated border-t border-silicon-border flex items-center justify-between text-xs">
          <div>
            Yield Difference ({currentWafer.wafer_id} vs {otherWafer.wafer_id}):{' '}
            <strong className={yieldDelta >= 0 ? 'text-telemetry-green' : 'text-telemetry-red'}>
              {yieldDelta >= 0 ? '+' : ''}{yieldDelta.toFixed(2)}%
            </strong>
          </div>
          <button
            onClick={onClose}
            className="px-4 py-1.5 rounded bg-silicon-interactive hover:bg-silicon-interactive/80 text-content-primary font-semibold transition-colors"
          >
            Close Comparison
          </button>
        </div>
      </div>
    </div>
  );
};
