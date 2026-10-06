import React from 'react';
import { Database } from 'lucide-react';
import { cn } from '../../lib/utils';

interface EmptyStateProps {
  title?: string;
  description?: string;
  icon?: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}

export const EmptyState: React.FC<EmptyStateProps> = ({
  title = 'No records found',
  description = 'No telemetry points match the specified filtering criteria.',
  icon,
  action,
  className,
}) => {
  return (
    <div
      className={cn(
        'border border-dashed border-silicon-border rounded-lg p-8 text-center flex flex-col items-center justify-center gap-3 bg-silicon-surface/50',
        className
      )}
    >
      <div className="w-10 h-10 rounded-full bg-silicon-interactive flex items-center justify-center text-content-muted">
        {icon || <Database className="w-5 h-5" />}
      </div>
      <div>
        <h4 className="text-sm font-semibold text-content-primary">
          {title}
        </h4>
        <p className="text-xs text-content-secondary mt-1 max-w-sm">
          {description}
        </p>
      </div>
      {action && <div className="mt-2">{action}</div>}
    </div>
  );
};
