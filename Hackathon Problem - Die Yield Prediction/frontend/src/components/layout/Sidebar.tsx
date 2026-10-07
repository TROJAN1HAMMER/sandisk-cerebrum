import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  BarChart3,
  Disc,
  Activity,
  FlaskConical,
  ShieldCheck,
  Cpu,
  Layers
} from 'lucide-react';
import { useSystemStatus } from '../../hooks/useSystem';
import { cn } from '../../lib/utils';

export const Sidebar: React.FC = () => {
  const { data: statusData } = useSystemStatus();

  const navItems = [
    {
      to: '/command-center',
      label: 'Command Center',
      icon: LayoutDashboard,
      description: 'Executive fab overview & KPIs',
    },
    {
      to: '/yield-analytics',
      label: 'Yield Analytics',
      icon: BarChart3,
      description: 'Yield intelligence, metrics & cascade flow',
    },
    {
      to: '/wafer-intelligence',
      label: 'Wafer Intelligence',
      icon: Disc,
      description: 'Spatial die maps & reticle telemetry',
    },
    {
      to: '/cascade-diagnostics',
      label: 'Cascade Diagnostics',
      icon: Activity,
      description: 'Live die inference & signals',
    },
    {
      to: '/model-lab',
      label: 'Model Lab',
      icon: FlaskConical,
      description: 'Architecture comparison & F1',
    },
    {
      to: '/forensic-audit',
      label: 'Forensic Audit',
      icon: ShieldCheck,
      description: 'Zero-leakage & recovery proof',
    },
  ];


  return (
    <aside className="w-64 bg-silicon-surface border-r border-silicon-border flex flex-col justify-between shrink-0 h-screen sticky top-0 select-none">
      <div>
        {/* Brand Header */}
        <div className="p-4 border-b border-silicon-border flex items-center gap-3">
          <div className="w-8 h-8 rounded bg-silicon-elevated border border-telemetry-cyan/40 flex items-center justify-center text-telemetry-cyan shadow-sm">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <div className="text-xs font-bold tracking-widest text-content-primary font-mono flex items-center gap-1.5">
              SANDISK <span className="text-telemetry-cyan">CEREBRUM</span>
            </div>
            <div className="text-[10px] text-content-muted uppercase tracking-wider font-mono">
              Die Yield Intelligence
            </div>
          </div>
        </div>

        {/* Navigation Section */}
        <div className="p-3">
          <div className="text-[10px] font-mono uppercase tracking-wider text-content-muted px-3 py-1.5 font-semibold">
            Workspaces
          </div>
          <nav className="space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.to}
                  to={item.to}
                  className={({ isActive }) =>
                    cn(
                      'flex items-center gap-3 px-3 py-2.5 rounded-md text-xs font-medium transition-all group',
                      isActive
                        ? 'bg-telemetry-blue/15 text-telemetry-cyan border border-telemetry-blue/30 font-semibold shadow-sm'
                        : 'text-content-secondary hover:text-content-primary hover:bg-silicon-interactive/50 border border-transparent'
                    )
                  }
                >
                  <Icon className="w-4 h-4 shrink-0 transition-transform group-hover:scale-105" />
                  <div className="truncate">
                    <div>{item.label}</div>
                  </div>
                </NavLink>
              );
            })}
          </nav>
        </div>
      </div>

      {/* Engine Status / System Hardware Box */}
      <div className="p-3 border-t border-silicon-border bg-silicon-canvas/60">
        <div className="p-3 rounded-lg border border-silicon-border bg-silicon-card/80 text-[11px] font-mono space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-content-muted">ENGINE:</span>
            <span className="inline-flex items-center gap-1 text-telemetry-green font-semibold">
              <span className="w-1.5 h-1.5 rounded-full bg-telemetry-green animate-pulse" />
              {statusData?.status ? statusData.status.toUpperCase() : 'ONLINE'}
            </span>
          </div>

          <div className="flex items-center justify-between text-content-secondary">
            <span className="text-content-muted">MODEL:</span>
            <span className="text-telemetry-purple font-semibold">
              {statusData?.active_model_version || 'B3'}
            </span>
          </div>

          <div className="flex items-center justify-between text-content-secondary">
            <span className="text-content-muted">GATE B:</span>
            <span className="text-telemetry-amber font-semibold">
              {statusData?.production_gate_b_window || '[0.250, 0.550]'}
            </span>
          </div>

          <div className="flex items-center justify-between text-content-secondary">
            <span className="text-content-muted">CANONICAL:</span>
            <span>128 / 32 WAFERS</span>
          </div>

          <div className="flex items-center justify-between text-content-secondary">
            <span className="text-content-muted">HOLDOUT DIES:</span>
            <span className="text-telemetry-cyan">
              {statusData?.total_validation_dies ? statusData.total_validation_dies.toLocaleString() : '27,972'}
            </span>
          </div>
        </div>
      </div>
    </aside>
  );
};
