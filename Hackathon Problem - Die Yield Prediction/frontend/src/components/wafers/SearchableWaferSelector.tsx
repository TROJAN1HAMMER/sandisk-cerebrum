import React, { useState, useRef, useEffect, useMemo } from 'react';
import { WaferSummaryItem } from '../../types/wafers';
import { Search, Disc, Check, X, AlertCircle } from 'lucide-react';

interface SearchableWaferSelectorProps {
  wafers: WaferSummaryItem[];
  selectedWaferId: string;
  onSelectWafer: (waferId: string) => void;
  className?: string;
}

export const SearchableWaferSelector: React.FC<SearchableWaferSelectorProps> = ({
  wafers,
  selectedWaferId,
  onSelectWafer,
  className = '',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleOutsideClick = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleOutsideClick);
    return () => document.removeEventListener('mousedown', handleOutsideClick);
  }, []);

  // Filter wafers by search query with partial matching
  const filteredWafers = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    if (!q) return wafers;
    return wafers.filter((w) =>
      w.wafer_id.toLowerCase().includes(q)
    );
  }, [wafers, searchQuery]);

  const selectedWafer = useMemo(() => {
    return wafers.find((w) => w.wafer_id === selectedWaferId);
  }, [wafers, selectedWaferId]);

  const handleSelect = (waferId: string) => {
    onSelectWafer(waferId);
    setIsOpen(false);
    setSearchQuery('');
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSearchQuery('');
    if (inputRef.current) inputRef.current.focus();
  };

  return (
    <div ref={containerRef} className={`relative inline-block text-xs font-mono ${className}`}>
      {/* Trigger & Search Bar */}
      <div className="flex items-center gap-2">
        <div
          onClick={() => {
            setIsOpen((prev) => !prev);
            setTimeout(() => inputRef.current?.focus(), 50);
          }}
          className="flex items-center gap-2.5 px-3 py-1.5 rounded-lg border border-[#2a3c50] bg-[#0c1420] hover:border-telemetry-cyan/60 hover:bg-[#101b2b] cursor-pointer transition-all shadow-sm"
        >
          <Disc className="w-4 h-4 text-telemetry-cyan shrink-0 animate-pulse" />
          <div className="flex flex-col text-left">
            <span className="text-[10px] text-content-muted leading-none uppercase tracking-wider">
              Active Wafer
            </span>
            <span className="text-white font-bold tracking-tight text-xs mt-0.5">
              {selectedWafer ? (
                <>
                  {selectedWafer.wafer_id}
                  <span className="text-content-secondary font-normal ml-1.5 text-[11px]">
                    — {selectedWafer.total_dies} dies — {selectedWafer.actual_failures} defects
                  </span>
                </>
              ) : (
                selectedWaferId
              )}
            </span>
          </div>
          <span className="ml-2 text-[10px] px-1.5 py-0.5 rounded bg-telemetry-cyan/15 text-telemetry-cyan font-semibold border border-telemetry-cyan/30">
            {wafers.length} available
          </span>
        </div>
      </div>

      {/* Dropdown Menu & Search Input */}
      {isOpen && (
        <div className="absolute left-0 top-full mt-1.5 w-84 sm:w-96 max-w-[95vw] bg-[#09101a] border border-[#2a3c50] rounded-lg shadow-2xl z-50 overflow-hidden flex flex-col animate-in fade-in slide-in-from-top-2 duration-150">
          {/* Search Input Field Header */}
          <div className="p-2.5 bg-[#0e1724] border-b border-[#1e2c3d] flex items-center gap-2">
            <Search className="w-4 h-4 text-telemetry-cyan shrink-0" />
            <input
              ref={inputRef}
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search wafer ID..."
              className="w-full bg-[#05080e] border border-[#243549] text-white placeholder:text-gray-400 px-2.5 py-1.5 rounded text-xs font-mono focus:outline-none focus:border-telemetry-cyan focus:ring-1 focus:ring-telemetry-cyan caret-telemetry-cyan"
            />
            {searchQuery && (
              <button
                onClick={handleClear}
                className="text-gray-400 hover:text-white p-1 rounded hover:bg-[#1a2738] transition-colors"
                title="Clear search"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {/* Available Counter Header */}
          <div className="px-3 py-1.5 bg-[#0c1420] border-b border-[#1a2738] flex items-center justify-between text-[11px] text-content-muted">
            <span>
              {searchQuery ? `Matching ${filteredWafers.length} of ${wafers.length}` : `${wafers.length} wafers available`}
            </span>
            <span className="text-[10px] text-telemetry-cyan font-sans">Click to inspect</span>
          </div>

          {/* Wafer List Options */}
          <div className="max-h-64 overflow-y-auto divide-y divide-[#15202e] overscroll-contain">
            {filteredWafers.length > 0 ? (
              filteredWafers.map((w) => {
                const isSelected = w.wafer_id === selectedWaferId;
                return (
                  <button
                    key={w.wafer_id}
                    onClick={() => handleSelect(w.wafer_id)}
                    className={`w-full text-left px-3 py-2 flex items-center justify-between transition-colors ${
                      isSelected
                        ? 'bg-telemetry-cyan/15 text-white font-bold border-l-2 border-telemetry-cyan'
                        : 'text-gray-200 hover:bg-[#142030] hover:text-white'
                    }`}
                  >
                    <div className="flex flex-col">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-white tracking-wide">
                          {w.wafer_id}
                        </span>
                        {isSelected && (
                          <span className="flex items-center text-telemetry-cyan text-[10px] gap-0.5">
                            <Check className="w-3 h-3" /> Selected
                          </span>
                        )}
                      </div>
                      <span className="text-[11px] text-content-secondary mt-0.5">
                        {w.total_dies} dies — {w.actual_failures} defects{' '}
                        <span className={w.fail_rate_pct > 5 ? 'text-telemetry-red font-semibold' : 'text-telemetry-green'}>
                          ({w.fail_rate_pct.toFixed(1)}% fail)
                        </span>
                      </span>
                    </div>

                    <div className="text-right text-[10px] text-content-muted">
                      {w.escalated_dies > 0 && (
                        <span className="px-1.5 py-0.5 rounded bg-telemetry-amber/15 text-telemetry-amber border border-telemetry-amber/30">
                          {w.escalated_dies} esc
                        </span>
                      )}
                    </div>
                  </button>
                );
              })
            ) : (
              <div className="p-6 text-center text-content-muted flex flex-col items-center justify-center gap-2">
                <AlertCircle className="w-5 h-5 text-telemetry-amber" />
                <span className="text-xs font-semibold text-gray-300">No wafers found</span>
                <span className="text-[11px] text-gray-500 font-sans">
                  No wafer matches "{searchQuery}". Clear search to view all {wafers.length} wafers.
                </span>
                <button
                  onClick={handleClear}
                  className="mt-1 px-2.5 py-1 rounded bg-[#182638] text-telemetry-cyan hover:bg-[#203248] text-[11px] transition-colors"
                >
                  Clear search
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
