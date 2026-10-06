import React from 'react';
import { cn } from '../../lib/utils';

export type MetricAccent = 'green' | 'red' | 'amber' | 'blue' | 'purple' | 'cyan' | 'neutral';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  badge?: string;
  accent?: MetricAccent;
  className?: string;
  icon?: React.ReactNode;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  badge,
  accent = 'neutral',
  className,
  icon,
}) => {
  const accentTextColors: Record<MetricAccent, string> = {
    green: 'text-telemetry-green',
    red: 'text-telemetry-red',
    amber: 'text-telemetry-amber',
    blue: 'text-telemetry-blue',
    purple: 'text-telemetry-purple',
    cyan: 'text-telemetry-cyan',
    neutral: 'text-content-primary',
  };

  const accentBorderHover: Record<MetricAccent, string> = {
    green: 'hover:border-telemetry-green/50',
    red: 'hover:border-telemetry-red/50',
    amber: 'hover:border-telemetry-amber/50',
    blue: 'hover:border-telemetry-blue/50',
    purple: 'hover:border-telemetry-purple/50',
    cyan: 'hover:border-telemetry-cyan/50',
    neutral: 'hover:border-silicon-borderLight',
  };

  return (
    <div
      className={cn(
        'bg-silicon-card border border-silicon-border rounded-lg p-4 transition-all duration-200',
        accentBorderHover[accent],
        className
      )}
    >
      <div className="flex items-center justify-between gap-2 mb-1.5">
        <span className="text-xs font-semibold uppercase tracking-wider text-content-muted font-sans truncate">
          {title}
        </span>
        {icon && <span className="text-content-muted">{icon}</span>}
        {badge && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-silicon-interactive text-content-secondary border border-silicon-border">
            {badge}
          </span>
        )}
      </div>

      <div
        className={cn(
          'text-2xl font-bold font-mono tabular-nums tracking-tight my-1',
          accentTextColors[accent]
        )}
      >
        {value}
      </div>

      {subtitle && (
        <div className="text-xs text-content-secondary mt-1 truncate">
          {subtitle}
        </div>
      )}
    </div>
  );
};
