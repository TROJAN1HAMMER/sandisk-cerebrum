import React, { useState, useRef, useEffect } from 'react';
import { cn } from '../../lib/utils';
import { HelpCircle } from 'lucide-react';

interface TooltipProps {
  content: string | React.ReactNode;
  children?: React.ReactNode;
  position?: 'top' | 'bottom' | 'left' | 'right';
  className?: string;
  delayMs?: number;
}

export const Tooltip: React.FC<TooltipProps> = ({
  content,
  children,
  position = 'top',
  className,
  delayMs = 120,
}) => {
  const [isVisible, setIsVisible] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const showTooltip = () => {
    timeoutRef.current = setTimeout(() => {
      setIsVisible(true);
    }, delayMs);
  };

  const hideTooltip = () => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
    }
    setIsVisible(false);
  };

  useEffect(() => {
    return () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current);
    };
  }, []);

  const positionClasses = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  };

  return (
    <span
      className={cn('relative inline-flex items-center group', className)}
      onMouseEnter={showTooltip}
      onMouseLeave={hideTooltip}
      onFocus={showTooltip}
      onBlur={hideTooltip}
      tabIndex={0}
      role="tooltip"
    >
      {children || <HelpCircle className="w-3.5 h-3.5 text-content-muted hover:text-telemetry-cyan transition-colors cursor-help" />}
      
      {isVisible && (
        <span
          className={cn(
            'absolute z-50 px-2.5 py-1.5 text-[11px] leading-snug rounded-md shadow-2xl pointer-events-none',
            'bg-[#0d1520] border border-[#2a3c50] text-[#e0e8f0] font-normal font-sans',
            'whitespace-normal max-w-xs min-w-[160px] text-left transition-opacity duration-150',
            positionClasses[position]
          )}
        >
          {content}
        </span>
      )}
    </span>
  );
};

export const InfoTooltip: React.FC<{ text: string; className?: string }> = ({ text, className }) => (
  <Tooltip content={text} className={className}>
    <HelpCircle className="w-3 h-3 text-content-muted hover:text-telemetry-cyan transition-colors cursor-help ml-1 inline-block" />
  </Tooltip>
);
