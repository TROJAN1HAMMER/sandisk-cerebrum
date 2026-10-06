import React, { useState, useMemo, useRef } from 'react';
import { BlockAttributionResponse } from '../../types/inference';
import {
  Activity,
  Flame,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  AlertTriangle,
  Info,
  Maximize2,
  Sliders,
  ChevronRight,
  Sparkles,
} from 'lucide-react';

interface BlockSignalViewerProps {
  data: BlockAttributionResponse;
  className?: string;
}

export const BlockSignalViewer: React.FC<BlockSignalViewerProps> = ({
  data,
  className = '',
}) => {
  const { readings, attributions, peak_attribution_idx, peak_attribution_value, anomalous_blocks_count } = data;

  const totalBlocks = readings.length || 2000;
  const [viewRange, setViewRange] = useState<[number, number]>([0, totalBlocks]);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<{ start: number; end: number } | null>(null);
  const [isSelecting, setIsSelecting] = useState<boolean>(false);
  const selectionStartRef = useRef<number | null>(null);

  const startIdx = viewRange[0];
  const endIdx = viewRange[1];
  const windowLength = Math.max(1, endIdx - startIdx);

  // Top 6 most influential blocks by attribution score
  const topInfluentialBlocks = useMemo(() => {
    if (!attributions || attributions.length === 0) return [];
    const indexed = attributions.map((val, idx) => ({
      block: idx + 1,
      idx,
      attr: val,
      reading: readings[idx] || 100.0,
    }));
    indexed.sort((a, b) => b.attr - a.attr);
    return indexed.slice(0, 6);
  }, [attributions, readings]);

  // Slice visible points for rendering
  const visibleReadings = useMemo(() => {
    return readings.slice(startIdx, endIdx);
  }, [readings, startIdx, endIdx]);

  const visibleAttributions = useMemo(() => {
    return attributions.slice(startIdx, endIdx);
  }, [attributions, startIdx, endIdx]);

  // Compute scale boundaries for readings
  const { minVal, maxVal } = useMemo(() => {
    let min = 80;
    let max = 115;
    for (const v of visibleReadings) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    return { minVal: Math.floor(min - 5), maxVal: Math.ceil(max + 5) };
  }, [visibleReadings]);

  // SVG dimensions
  const svgWidth = 800;
  const signalHeight = 160;
  const attrHeight = 75;
  const paddingLeft = 45;
  const paddingRight = 20;
  const plotWidth = svgWidth - paddingLeft - paddingRight;

  // Coordinate conversion helpers
  const getX = (idxInSlice: number) => {
    return paddingLeft + (idxInSlice / Math.max(1, windowLength - 1)) * plotWidth;
  };

  const getYSignal = (val: number) => {
    const norm = (val - minVal) / Math.max(1, maxVal - minVal);
    return signalHeight - norm * (signalHeight - 20) - 10;
  };

  const getYAttr = (val: number) => {
    const clamped = Math.max(0, Math.min(1, val));
    return attrHeight - clamped * (attrHeight - 15) - 5;
  };

  // Generate SVG polyline path strings
  const signalPath = useMemo(() => {
    if (visibleReadings.length === 0) return '';
    return visibleReadings
      .map((val, idx) => `${getX(idx).toFixed(1)},${getYSignal(val).toFixed(1)}`)
      .join(' L ');
  }, [visibleReadings, minVal, maxVal, windowLength]);

  const attrAreaPath = useMemo(() => {
    if (visibleAttributions.length === 0) return '';
    const points = visibleAttributions.map(
      (val, idx) => `${getX(idx).toFixed(1)},${getYAttr(val).toFixed(1)}`
    );
    const bottomY = attrHeight - 5;
    return `M ${getX(0)},${bottomY} L ${points.join(' L ')} L ${getX(
      visibleAttributions.length - 1
    )},${bottomY} Z`;
  }, [visibleAttributions, windowLength]);

  const threshY = getYSignal(104.0);

  // Preset window helpers
  const handleReset = () => {
    setViewRange([0, totalBlocks]);
    setSelectedRegion(null);
  };

  const handleZoomToBlock = (targetIdx: number, span: number = 150) => {
    const start = Math.max(0, targetIdx - Math.floor(span / 2));
    const end = Math.min(totalBlocks, start + span);
    setViewRange([start, end]);
    setHoveredIndex(targetIdx);
  };

  const handleZoomAnomalous = () => {
    if (peak_attribution_idx !== undefined && peak_attribution_idx !== null) {
      handleZoomToBlock(peak_attribution_idx, 300);
    }
  };

  // Drag selection handlers on SVG
  const handleSvgMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const ratio = (mouseX - paddingLeft) / plotWidth;
    if (ratio >= 0 && ratio <= 1) {
      const idx = Math.min(totalBlocks - 1, Math.max(0, startIdx + Math.round(ratio * (windowLength - 1))));
      selectionStartRef.current = idx;
      setIsSelecting(true);
      setSelectedRegion({ start: idx, end: idx });
    }
  };

  const handleSvgMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const ratio = (mouseX - paddingLeft) / plotWidth;
    if (ratio >= 0 && ratio <= 1) {
      const currentIdx = Math.min(totalBlocks - 1, Math.max(0, startIdx + Math.round(ratio * (windowLength - 1))));
      setHoveredIndex(currentIdx);

      if (isSelecting && selectionStartRef.current !== null) {
        const start = Math.min(selectionStartRef.current, currentIdx);
        const end = Math.max(selectionStartRef.current, currentIdx);
        setSelectedRegion({ start, end });
      }
    }
  };

  const handleSvgMouseUp = () => {
    if (isSelecting && selectedRegion) {
      if (selectedRegion.end - selectedRegion.start > 15) {
        // Zoom into selected region
        setViewRange([selectedRegion.start, selectedRegion.end]);
      }
    }
    setIsSelecting(false);
    selectionStartRef.current = null;
  };

  const handleSvgMouseLeave = () => {
    setHoveredIndex(null);
    if (isSelecting) {
      setIsSelecting(false);
      selectionStartRef.current = null;
    }
  };

  // Compute stats on active region
  const regionStats = useMemo(() => {
    if (!selectedRegion) return null;
    const rSlice = readings.slice(selectedRegion.start, selectedRegion.end + 1);
    if (rSlice.length === 0) return null;
    let min = Infinity, max = -Infinity, sum = 0;
    for (const v of rSlice) {
      if (v < min) min = v;
      if (v > max) max = v;
      sum += v;
    }
    const mean = sum / rSlice.length;
    let sqDiffSum = 0;
    for (const v of rSlice) {
      sqDiffSum += (v - mean) * (v - mean);
    }
    const std = Math.sqrt(sqDiffSum / rSlice.length);
    return {
      start: selectedRegion.start + 1,
      end: selectedRegion.end + 1,
      count: rSlice.length,
      min,
      max,
      mean,
      std,
    };
  }, [selectedRegion, readings]);

  const hoveredVal = hoveredIndex !== null ? readings[hoveredIndex] : null;
  const hoveredAttr = hoveredIndex !== null ? attributions[hoveredIndex] : null;

  return (
    <div className={`bg-silicon-elevated border border-silicon-border rounded-lg p-4 space-y-4 ${className}`}>
      {/* 1. Header and Telemetry Badges */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-silicon-border pb-3">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-telemetry-cyan" />
          <h3 className="text-sm font-semibold text-content-primary">
            2,000-Block Sequential Signal & CNN Occlusion Attribution
          </h3>
        </div>

        <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
          <span className="px-2 py-0.5 rounded bg-silicon-surface text-content-secondary border border-silicon-border">
            Total: {totalBlocks} Blocks
          </span>
          <span
            className={`px-2 py-0.5 rounded border ${
              anomalous_blocks_count > 0
                ? 'bg-telemetry-red/20 text-telemetry-red border-telemetry-red/30 font-bold'
                : 'bg-telemetry-green/20 text-telemetry-green border-telemetry-green/30'
            }`}
          >
            {anomalous_blocks_count} Blocks &gt; 104.0
          </span>
          <span className="px-2 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/30 font-bold flex items-center gap-1">
            <Flame className="w-3 h-3 text-telemetry-purple" />
            Peak Attribution: Block #{peak_attribution_idx + 1} (ΔP={(peak_attribution_value * 100).toFixed(1)}%)
          </span>
        </div>
      </div>

      {/* 2. Top Influential Blocks Attribution Shelf */}
      <div className="p-3 bg-silicon-card rounded-lg border border-silicon-border space-y-2">
        <div className="flex items-center justify-between text-xs font-mono">
          <span className="text-telemetry-cyan font-bold uppercase tracking-wider text-[11px] flex items-center gap-1.5">
            <Sparkles className="w-3.5 h-3.5" />
            Top Influential Blocks (PyTorch 1D CNN Attribution)
          </span>
          <span className="text-content-muted text-[11px]">
            Click any block to zoom waveform directly
          </span>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2 font-mono text-xs">
          {topInfluentialBlocks.map((b, rank) => {
            const isPeak = b.idx === peak_attribution_idx;
            const isHovered = hoveredIndex === b.idx;

            return (
              <button
                key={b.idx}
                onClick={() => handleZoomToBlock(b.idx, 200)}
                className={`p-2 rounded text-left border transition-all ${
                  isPeak
                    ? 'bg-telemetry-purple/20 border-telemetry-purple/50 text-telemetry-purple shadow-sm'
                    : isHovered
                    ? 'bg-telemetry-blue/20 border-telemetry-cyan text-telemetry-cyan'
                    : 'bg-silicon-elevated border-silicon-border/70 hover:border-silicon-border text-content-primary'
                }`}
              >
                <div className="flex items-center justify-between text-[10px] text-content-muted">
                  <span>#{rank + 1} Rank</span>
                  {isPeak && <span className="font-bold text-telemetry-purple">PEAK</span>}
                </div>
                <div className="font-bold text-xs mt-0.5 text-content-primary">
                  Block #{b.block}
                </div>
                <div className="flex items-center justify-between text-[11px] mt-1 pt-1 border-t border-silicon-border/40">
                  <span className="text-content-muted">ΔP:</span>
                  <span className="font-semibold text-telemetry-purple">
                    {(b.attr * 100).toFixed(1)}%
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* 3. Preset Window Navigation Bar & HUD */}
      <div className="flex flex-wrap items-center justify-between gap-2 text-xs font-mono">
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-content-muted text-[11px]">Range:</span>
          <button
            onClick={handleReset}
            className={`px-2 py-1 rounded text-xs transition-colors ${
              startIdx === 0 && endIdx === totalBlocks
                ? 'bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40 font-bold'
                : 'bg-silicon-interactive text-content-secondary hover:text-content-primary'
            }`}
          >
            Full (1-2000)
          </button>
          <button
            onClick={handleZoomAnomalous}
            className="px-2 py-1 rounded bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/30 hover:bg-telemetry-purple/30 transition-colors font-bold flex items-center gap-1"
          >
            <Flame className="w-3 h-3" />
            Zoom Peak (#{peak_attribution_idx + 1})
          </button>
          <button
            onClick={() => setViewRange([0, 500])}
            className="px-2 py-1 rounded bg-silicon-interactive text-content-secondary hover:text-content-primary"
          >
            1-500
          </button>
          <button
            onClick={() => setViewRange([500, 1500])}
            className="px-2 py-1 rounded bg-silicon-interactive text-content-secondary hover:text-content-primary"
          >
            500-1500
          </button>
          <button
            onClick={() => setViewRange([1500, 2000])}
            className="px-2 py-1 rounded bg-silicon-interactive text-content-secondary hover:text-content-primary"
          >
            1500-2000
          </button>
        </div>

        {/* Live Hover Readout HUD */}
        <div className="flex items-center gap-3 bg-silicon-base px-3 py-1 rounded border border-silicon-border">
          {hoveredIndex !== null ? (
            <>
              <span className="text-telemetry-cyan font-bold">
                Block #{hoveredIndex + 1}
              </span>
              <span className="text-content-muted">|</span>
              <span
                className={`font-semibold ${
                  (hoveredVal ?? 0) >= 104 ? 'text-telemetry-red font-bold' : 'text-content-primary'
                }`}
              >
                Reading: {(hoveredVal ?? 0).toFixed(2)}
              </span>
              <span className="text-content-muted">|</span>
              <span className="text-telemetry-purple font-semibold">
                Occlusion ΔP: {((hoveredAttr ?? 0) * 100).toFixed(1)}%
              </span>
            </>
          ) : (
            <span className="text-content-muted text-[11px]">
              Drag on waveform to select region • Hover for block readout
            </span>
          )}
        </div>
      </div>

      {/* 4. Region Selection Statistical Summary (When Selected) */}
      {regionStats && (
        <div className="p-3 bg-telemetry-blue/10 border border-telemetry-cyan/40 rounded-lg flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
          <div className="flex items-center gap-3 flex-wrap">
            <span className="font-bold text-telemetry-cyan">
              SELECTED REGION: Blocks #{regionStats.start} to #{regionStats.end} ({regionStats.count} Blocks)
            </span>
            <span className="text-content-muted">|</span>
            <span>Min: <strong>{regionStats.min.toFixed(2)}</strong></span>
            <span>Max: <strong className={regionStats.max >= 104 ? 'text-telemetry-red font-bold' : ''}>{regionStats.max.toFixed(2)}</strong></span>
            <span>Mean: <strong>{regionStats.mean.toFixed(2)}</strong></span>
            <span>Std: <strong>{regionStats.std.toFixed(2)}</strong></span>
          </div>

          <button
            onClick={() => setSelectedRegion(null)}
            className="px-2 py-0.5 rounded bg-silicon-surface hover:bg-silicon-border text-content-secondary text-[11px]"
          >
            Clear Selection
          </button>
        </div>
      )}

      {/* 5. Synchronized SVG Waveform + Attribution Chart */}
      <div className="bg-silicon-canvas rounded border border-silicon-border p-2 overflow-x-auto select-none">
        <svg
          viewBox={`0 0 ${svgWidth} ${signalHeight + attrHeight + 25}`}
          className="w-full h-auto min-w-[700px] cursor-crosshair"
          onMouseDown={handleSvgMouseDown}
          onMouseMove={handleSvgMouseMove}
          onMouseUp={handleSvgMouseUp}
          onMouseLeave={handleSvgMouseLeave}
        >
          <defs>
            <linearGradient id="attrGradient" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#A855F7" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#A855F7" stopOpacity="0.05" />
            </linearGradient>
          </defs>

          {/* ================= SECTION 1: SIGNAL WAVEFORM ================= */}
          {/* Y Axis Gridlines & Labels */}
          {[minVal, 95, 100, 104, maxVal].map((tick) => {
            const y = getYSignal(tick);
            const isThresh = tick === 104;
            return (
              <g key={`y-tick-${tick}`}>
                <line
                  x1={paddingLeft}
                  y1={y}
                  x2={svgWidth - paddingRight}
                  y2={y}
                  stroke={isThresh ? '#EF4444' : '#1E2638'}
                  strokeWidth={isThresh ? 1.2 : 0.8}
                  strokeDasharray={isThresh ? '4,3' : '2,2'}
                />
                <text
                  x={paddingLeft - 6}
                  y={y + 3.5}
                  textAnchor="end"
                  fontSize="9"
                  fill={isThresh ? '#EF4444' : '#64748B'}
                  fontFamily="monospace"
                  fontWeight={isThresh ? 'bold' : 'normal'}
                >
                  {tick}
                </text>
              </g>
            );
          })}

          {/* Threshold label */}
          <text
            x={svgWidth - paddingRight - 4}
            y={threshY - 4}
            textAnchor="end"
            fontSize="8"
            fill="#EF4444"
            fontFamily="monospace"
            fontWeight="bold"
          >
            Threshold (104.0)
          </text>

          {/* Selected Region Shaded Box */}
          {selectedRegion && selectedRegion.start >= startIdx && selectedRegion.end <= endIdx && (
            <rect
              x={getX(selectedRegion.start - startIdx)}
              y={5}
              width={Math.max(2, getX(selectedRegion.end - startIdx) - getX(selectedRegion.start - startIdx))}
              height={signalHeight + attrHeight + 15}
              fill="rgba(56, 189, 248, 0.15)"
              stroke="#38BDF8"
              strokeDasharray="2,2"
              pointerEvents="none"
            />
          )}

          {/* Signal Polyline Path */}
          {signalPath && (
            <path
              d={`M ${signalPath}`}
              fill="none"
              stroke="#38BDF8"
              strokeWidth="1.4"
              strokeLinejoin="round"
            />
          )}

          {/* Signal chart section boundary */}
          <line
            x1={paddingLeft}
            y1={signalHeight + 5}
            x2={svgWidth - paddingRight}
            y2={signalHeight + 5}
            stroke="#32435E"
            strokeWidth="1"
          />

          {/* ================= SECTION 2: ATTRIBUTION AREA ================= */}
          <g transform={`translate(0, ${signalHeight + 12})`}>
            {/* Attribution background area */}
            <rect
              x={paddingLeft}
              y={0}
              width={plotWidth}
              height={attrHeight}
              fill="#0E131F"
              opacity="0.5"
            />

            {/* Attribution Area Path */}
            {attrAreaPath && (
              <path d={attrAreaPath} fill="url(#attrGradient)" stroke="#A855F7" strokeWidth="1.2" />
            )}

            {/* Attribution Y axis labels */}
            <text
              x={paddingLeft - 6}
              y={getYAttr(1.0) + 3}
              textAnchor="end"
              fontSize="8"
              fill="#A855F7"
              fontFamily="monospace"
            >
              1.0
            </text>
            <text
              x={paddingLeft - 6}
              y={getYAttr(0.0) + 3}
              textAnchor="end"
              fontSize="8"
              fill="#64748B"
              fontFamily="monospace"
            >
              0.0
            </text>
            <text
              x={paddingLeft + 6}
              y={12}
              fontSize="8"
              fill="#A855F7"
              fontFamily="monospace"
              fontWeight="bold"
            >
              1D CNN Occlusion Attribution (Δ Probability)
            </text>

            {/* Peak Attribution Flag */}
            {peak_attribution_idx >= startIdx && peak_attribution_idx < endIdx && (
              <g
                transform={`translate(${getX(peak_attribution_idx - startIdx)}, ${getYAttr(
                  peak_attribution_value
                )})`}
              >
                <circle r="3.5" fill="#A855F7" stroke="#FFFFFF" strokeWidth="1.2" />
                <line x1="0" y1="0" x2="0" y2="-12" stroke="#A855F7" strokeWidth="1" />
                <text
                  x="0"
                  y="-14"
                  textAnchor="middle"
                  fontSize="8"
                  fill="#A855F7"
                  fontFamily="monospace"
                  fontWeight="bold"
                >
                  Peak #{peak_attribution_idx + 1}
                </text>
              </g>
            )}
          </g>

          {/* ================= X AXIS TICKS ================= */}
          {[0, 0.25, 0.5, 0.75, 1.0].map((frac) => {
            const blockNum = Math.round(startIdx + frac * (windowLength - 1));
            const x = paddingLeft + frac * plotWidth;
            const y = signalHeight + attrHeight + 20;
            return (
              <g key={`x-tick-${frac}`}>
                <line x1={x} y1={signalHeight + attrHeight + 10} x2={x} y2={y - 4} stroke="#32435E" strokeWidth="1" />
                <text
                  x={x}
                  y={y + 6}
                  textAnchor="middle"
                  fontSize="9"
                  fill="#64748B"
                  fontFamily="monospace"
                >
                  Block #{blockNum + 1}
                </text>
              </g>
            );
          })}

          {/* Synchronized Crosshair on Hover */}
          {hoveredIndex !== null && hoveredIndex >= startIdx && hoveredIndex < endIdx && (
            <g pointerEvents="none">
              <line
                x1={getX(hoveredIndex - startIdx)}
                y1={5}
                x2={getX(hoveredIndex - startIdx)}
                y2={signalHeight + attrHeight + 12}
                stroke="#38BDF8"
                strokeWidth="1.2"
                strokeDasharray="3,2"
              />
              {/* Highlight dot on signal */}
              <circle
                cx={getX(hoveredIndex - startIdx)}
                cy={getYSignal(readings[hoveredIndex])}
                r="3.5"
                fill="#38BDF8"
                stroke="#FFFFFF"
                strokeWidth="1.2"
              />
              {/* Highlight dot on attribution */}
              <circle
                cx={getX(hoveredIndex - startIdx)}
                cy={signalHeight + 12 + getYAttr(attributions[hoveredIndex])}
                r="3"
                fill="#A855F7"
                stroke="#FFFFFF"
                strokeWidth="1"
              />
            </g>
          )}
        </svg>
      </div>

      {/* 6. Explanatory footer */}
      <div className="flex items-center gap-2 p-2.5 bg-silicon-base rounded border border-silicon-border text-xs font-mono text-content-secondary">
        <Info className="w-4 h-4 text-telemetry-cyan shrink-0" />
        <span>
          A sliding occlusion window ($W=50$) replaces block sequences with baseline nominal readings
          to determine how severely each block interval alters the PyTorch 1D CNN failure prediction.
        </span>
      </div>
    </div>
  );
};
