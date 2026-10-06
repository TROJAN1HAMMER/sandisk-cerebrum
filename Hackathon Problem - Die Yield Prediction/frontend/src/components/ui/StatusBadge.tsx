import React from 'react';
import { cn } from '../../lib/utils';

export type BadgeVariant = 'pass' | 'fail' | 'warn' | 'info' | 'purple' | 'neutral';

interface StatusBadgeProps {
  label: string;
  variant?: BadgeVariant;
  className?: string;
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  label,
  variant = 'info',
  className,
  pulse = false,
}) => {
  const variantStyles: Record<BadgeVariant, string> = {
    pass: 'bg-telemetry-green/10 text-telemetry-green border-telemetry-green/30',
    fail: 'bg-telemetry-red/10 text-telemetry-red border-telemetry-red/30',
    warn: 'bg-telemetry-amber/10 text-telemetry-amber border-telemetry-amber/30',
    info: 'bg-telemetry-blue/10 text-telemetry-blue border-telemetry-blue/30',
    purple: 'bg-telemetry-purple/10 text-telemetry-purple border-telemetry-purple/30',
    neutral: 'bg-silicon-interactive text-content-secondary border-silicon-border',
  };

  const dotColors: Record<BadgeVariant, string> = {
    pass: 'bg-telemetry-green',
    fail: 'bg-telemetry-red',
    warn: 'bg-telemetry-amber',
    info: 'bg-telemetry-blue',
    purple: 'bg-telemetry-purple',
    neutral: 'bg-content-muted',
  };

  return (
    <span
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-semibold uppercase tracking-wider border font-mono',
        variantStyles[variant],
        className
      )}
    >
      <span
        className={cn(
          'w-1.5 h-1.5 rounded-full',
          dotColors[variant],
          pulse && 'animate-pulse'
        )}
      />
      {label}
    </span>
  );
};
