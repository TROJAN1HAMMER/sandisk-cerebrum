import React from 'react';
import { useSystemHealth } from '../../hooks/useSystem';
import { cn } from '../../lib/utils';
import { Wifi, WifiOff } from 'lucide-react';

export const ConnectionIndicator: React.FC<{ className?: string }> = ({ className }) => {
  const { data, isSuccess, isLoading, isError } = useSystemHealth();

  if (isLoading) {
    return (
      <div
        className={cn(
          'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-silicon-interactive text-content-muted border border-silicon-border',
          className
        )}
      >
        <span className="w-1.5 h-1.5 rounded-full bg-telemetry-amber animate-ping" />
        CONNECTING...
      </div>
    );
  }

  if (isError || !isSuccess || data?.status !== 'online') {
    return (
      <div
        className={cn(
          'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-telemetry-red/10 text-telemetry-red border border-telemetry-red/30',
          className
        )}
      >
        <WifiOff className="w-3 h-3" />
        API OFFLINE
      </div>
    );
  }

  return (
    <div
      className={cn(
        'inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-mono font-medium bg-telemetry-green/10 text-telemetry-green border border-telemetry-green/30',
        className
      )}
    >
      <Wifi className="w-3 h-3" />
      API ONLINE
    </div>
  );
};
