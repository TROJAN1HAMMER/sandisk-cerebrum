import React from 'react';
import { useLocation } from 'react-router-dom';
import { RefreshCw, Zap } from 'lucide-react';
import { useQueryClient } from '@tanstack/react-query';
import { ConnectionIndicator } from '../ui/ConnectionIndicator';

const titles: Record<string, { title: string; subtitle: string }> = {
  '/command-center': {
    title: 'Fab Operations Command Center',
    subtitle: 'Selective Multi-Resolution Die Screening & Production KPIs',
  },
  '/wafer-intelligence': {
    title: 'Wafer Spatial Intelligence',
    subtitle: 'Spatial Die Reticle Defect Analysis & 32 Holdout Wafers',
  },
  '/cascade-diagnostics': {
    title: 'Live Cascade Diagnostics Console',
    subtitle: 'Real-Time Single-Die & Sub-Die Anomaly Verification',
  },
  '/model-lab': {
    title: 'Architectural Model Lab',
    subtitle: 'Head-to-Head Benchmarking, Pareto Curves & Live Evaluation',
  },
  '/forensic-audit': {
    title: 'Forensic Error Audit & Split Integrity',
    subtitle: 'Net TP Gain Proof (+32 dies), Leakage Hardening & Verification',
  },
};

export const TopBar: React.FC = () => {
  const location = useLocation();
  const queryClient = useQueryClient();
  const [isRefreshing, setIsRefreshing] = React.useState(false);

  const active = titles[location.pathname] || {
    title: 'SanDisk Cerebrum Yield Platform',
    subtitle: 'Mission-Critical Semiconductor Intelligence',
  };

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await queryClient.invalidateQueries();
    setTimeout(() => setIsRefreshing(false), 500);
  };

  return (
    <header className="h-16 border-b border-silicon-border bg-silicon-surface/80 backdrop-blur-md px-6 flex items-center justify-between sticky top-0 z-30 shrink-0">
      <div>
        <h1 className="text-sm font-semibold tracking-tight text-content-primary flex items-center gap-2">
          {active.title}
        </h1>
        <p className="text-[11px] text-content-muted font-sans truncate">
          {active.subtitle}
        </p>
      </div>

      <div className="flex items-center gap-3">
        <div className="hidden sm:flex items-center gap-1.5 px-2.5 py-1 rounded bg-silicon-card border border-silicon-border text-xs font-mono text-content-secondary">
          <Zap className="w-3.5 h-3.5 text-telemetry-cyan" />
          <span>CASCADE:</span>
          <span className="text-telemetry-cyan font-semibold">GATE B [0.25, 0.55]</span>
        </div>

        <ConnectionIndicator />

        <button
          onClick={handleRefresh}
          disabled={isRefreshing}
          title="Refresh telemetry caches"
          className="p-1.5 rounded bg-silicon-interactive/60 hover:bg-silicon-interactive text-content-secondary hover:text-content-primary border border-silicon-border transition-colors disabled:opacity-50"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isRefreshing ? 'animate-spin text-telemetry-cyan' : ''}`} />
        </button>
      </div>
    </header>
  );
};
