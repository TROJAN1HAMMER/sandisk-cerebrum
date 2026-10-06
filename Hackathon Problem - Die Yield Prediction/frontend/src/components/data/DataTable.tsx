import React from 'react';
import { cn } from '../../lib/utils';
import { EmptyState } from '../ui/EmptyState';

export interface Column<T> {
  key: string;
  header: string;
  render?: (item: T, index: number) => React.ReactNode;
  align?: 'left' | 'center' | 'right';
  className?: string;
  headerClassName?: string;
}

interface DataTableProps<T> {
  columns: Column<T>[];
  data: T[];
  keyExtractor?: (item: T, index: number) => string | number;
  emptyMessage?: string;
  className?: string;
  rowClassName?: (item: T, index: number) => string;
}

export function DataTable<T extends Record<string, unknown>>({
  columns,
  data,
  keyExtractor,
  emptyMessage = 'No telemetry records available',
  className,
  rowClassName,
}: DataTableProps<T>) {
  if (!data || data.length === 0) {
    return <EmptyState description={emptyMessage} />;
  }

  const alignmentClasses = {
    left: 'text-left',
    center: 'text-center',
    right: 'text-right font-mono tabular-nums',
  };

  return (
    <div className={cn('overflow-x-auto rounded-lg border border-silicon-border bg-silicon-card', className)}>
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr className="border-b border-silicon-border bg-silicon-surface/80 text-content-secondary font-mono tracking-wider uppercase text-[11px]">
            {columns.map((col) => (
              <th
                key={col.key}
                className={cn(
                  'px-3.5 py-2.5 font-semibold',
                  alignmentClasses[col.align || 'left'],
                  col.headerClassName
                )}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-silicon-border/50 text-content-primary">
          {data.map((item, idx) => (
            <tr
              key={keyExtractor ? keyExtractor(item, idx) : idx}
              className={cn(
                'hover:bg-silicon-interactive/40 transition-colors',
                rowClassName ? rowClassName(item, idx) : ''
              )}
            >
              {columns.map((col) => (
                <td
                  key={col.key}
                  className={cn(
                    'px-3.5 py-2.5 whitespace-nowrap',
                    alignmentClasses[col.align || 'left'],
                    col.className
                  )}
                >
                  {col.render
                    ? col.render(item, idx)
                    : (item[col.key] as React.ReactNode) ?? '—'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
