import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { cn } from '../../lib/utils';

interface ErrorStateProps {
  title?: string;
  message?: string;
  onRetry?: () => void;
  className?: string;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Failed to load telemetry data',
  message = 'Unable to establish connection with the Cerebrum backend API.',
  onRetry,
  className,
}) => {
  return (
    <div
      className={cn(
        'bg-telemetry-red/10 border border-telemetry-red/30 rounded-lg p-5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4',
        className
      )}
    >
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-telemetry-red mt-0.5 shrink-0" />
        <div>
          <h4 className="text-sm font-semibold text-content-primary">
            {title}
          </h4>
          <p className="text-xs text-content-secondary mt-0.5">
            {message}
          </p>
        </div>
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold bg-telemetry-red/20 text-content-primary hover:bg-telemetry-red/30 border border-telemetry-red/40 transition-colors shrink-0"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          Retry Request
        </button>
      )}
    </div>
  );
};
