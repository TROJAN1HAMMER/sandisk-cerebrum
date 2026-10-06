import React from 'react';
import { cn } from '../../lib/utils';

interface SectionHeaderProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const SectionHeader: React.FC<SectionHeaderProps> = ({
  title,
  subtitle,
  badge,
  actions,
  className,
}) => {
  return (
    <div className={cn('flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4', className)}>
      <div>
        <div className="flex items-center gap-2.5">
          <h2 className="text-lg font-semibold tracking-tight text-content-primary">
            {title}
          </h2>
          {badge}
        </div>
        {subtitle && (
          <p className="text-xs text-content-secondary mt-0.5 max-w-3xl leading-relaxed">
            {subtitle}
          </p>
        )}
      </div>
      {actions && <div className="flex items-center gap-2 mt-2 md:mt-0">{actions}</div>}
    </div>
  );
};
