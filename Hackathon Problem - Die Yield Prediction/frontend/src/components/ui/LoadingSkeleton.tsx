import React from 'react';
import { cn } from '../../lib/utils';

interface LoadingSkeletonProps {
  className?: string;
  rows?: number;
}

export const LoadingSkeleton: React.FC<LoadingSkeletonProps> = ({
  className,
  rows = 1,
}) => {
  return (
    <div className="space-y-2 w-full animate-pulse">
      {Array.from({ length: rows }).map((_, i) => (
        <div
          key={i}
          className={cn('h-4 bg-silicon-interactive rounded', className)}
        />
      ))}
    </div>
  );
};
