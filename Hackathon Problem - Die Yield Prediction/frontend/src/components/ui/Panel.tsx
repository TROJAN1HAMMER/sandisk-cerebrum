import React from 'react';
import { cn } from '../../lib/utils';

interface PanelProps {
  title?: string;
  subtitle?: string;
  headerAction?: React.ReactNode;
  children: React.ReactNode;
  className?: string;
  contentClassName?: string;
  variant?: 'default' | 'elevated' | 'bordered';
}

export const Panel: React.FC<PanelProps> = ({
  title,
  subtitle,
  headerAction,
  children,
  className,
  contentClassName,
  variant = 'default',
}) => {
  const bgClasses = {
    default: 'bg-silicon-card border-silicon-border',
    elevated: 'bg-silicon-elevated border-silicon-border',
    bordered: 'bg-silicon-surface border-silicon-borderLight',
  };

  return (
    <div className={cn('rounded-lg border p-4 sm:p-5', bgClasses[variant], className)}>
      {(title || headerAction) && (
        <div className="flex items-center justify-between gap-4 pb-3 mb-4 border-b border-silicon-border">
          <div>
            {title && (
              <h3 className="text-sm font-semibold text-content-primary tracking-tight">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-xs text-content-secondary mt-0.5">
                {subtitle}
              </p>
            )}
          </div>
          {headerAction && <div>{headerAction}</div>}
        </div>
      )}
      <div className={contentClassName}>{children}</div>
    </div>
  );
};
