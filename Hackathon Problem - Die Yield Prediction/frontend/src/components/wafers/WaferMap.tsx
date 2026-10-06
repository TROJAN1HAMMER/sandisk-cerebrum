import React, { useState, useRef, useMemo, useEffect } from 'react';
import { DiePoint } from '../../types/wafers';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Layers,
  Sparkles,
} from 'lucide-react';

export type HeatmapMode =
  | 'status'
  | 'probability'
  | 'prob_A'
  | 'prob_B'
  | 'gate'
  | 'ground_truth'
  | 'error_type';

interface WaferMapProps {
  dies: DiePoint[];
  selectedDie: DiePoint | null;
  onSelectDie: (die: DiePoint) => void;
  onHoverDie?: (die: DiePoint | null, mouseEvent?: React.MouseEvent) => void;
  activeMode: HeatmapMode;
  onChangeMode: (mode: HeatmapMode) => void;
  className?: string;
}

export const WaferMap: React.FC<WaferMapProps> = ({
  dies,
  selectedDie,
  onSelectDie,
  onHoverDie,
  activeMode,
  onChangeMode,
  className = '',
}) => {
  const containerRef = useRef<HTMLDivElement>(null);

  // Zoom and pan transform state
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [hoveredDieState, setHoveredDieState] = useState<DiePoint | null>(null);

  // Compute spatial bounding coordinates
  const bounds = useMemo(() => {
    if (!dies || dies.length === 0) {
      return { minR: 0, maxR: 40, minC: 0, maxC: 40, rows: 41, cols: 41 };
    }
    let minR = Infinity, maxR = -Infinity, minC = Infinity, maxC = -Infinity;
    for (const d of dies) {
      if (d.die_row < minR) minR = d.die_row;
      if (d.die_row > maxR) maxR = d.die_row;
      if (d.die_col < minC) minC = d.die_col;
      if (d.die_col > maxC) maxC = d.die_col;
    }
    return {
      minR,
      maxR,
      minC,
      maxC,
      rows: maxR - minR + 1,
      cols: maxC - minC + 1,
    };
  }, [dies]);

  // Die layout dimensions in SVG units
  const DIE_SIZE = 12;
  const DIE_GAP = 1.2;
  const PITCH = DIE_SIZE + DIE_GAP;

  const waferWidth = (bounds.cols + 2) * PITCH;
  const waferHeight = (bounds.rows + 2) * PITCH;
  const centerX = waferWidth / 2;
  const centerY = waferHeight / 2;
  const waferRadius = Math.max(centerX, centerY) * 1.04;

  const viewBox = `0 0 ${waferWidth} ${waferHeight}`;

  // Reset view on wafer change
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [dies[0]?.wafer_id]);

  // Center on selected die if provided
  useEffect(() => {
    if (selectedDie && zoom > 1.5) {
      const targetX = (selectedDie.die_col - bounds.minC + 1) * PITCH;
      const targetY = (selectedDie.die_row - bounds.minR + 1) * PITCH;
      const panOffsetX = (centerX - targetX) * (zoom - 1);
      const panOffsetY = (centerY - targetY) * (zoom - 1);
      setPan({ x: panOffsetX, y: panOffsetY });
    }
  }, [selectedDie?.die_row, selectedDie?.die_col]);

  // Zoom controls
  const handleZoomIn = () => setZoom((z) => Math.min(6, z * 1.3));
  const handleZoomOut = () => setZoom((z) => Math.max(0.6, z / 1.3));
  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Mouse wheel zoom
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.88;
    setZoom((z) => Math.min(6, Math.max(0.6, z * factor)));
  };

  // Dragging / panning
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  // Color mapping logic per mode
  const getDieColor = (die: DiePoint): string => {
    switch (activeMode) {
      case 'status':
        if (die.decision_Cascade_B === 1) return '#EF4444'; // Red Fail
        if (die.in_gate_B) return '#F59E0B'; // Amber Escalated
        return '#22C55E'; // Green Pass

      case 'probability': {
        const p = die.prob_Cascade_B;
        if (p >= 0.5376) return '#EF4444'; // Fail threshold
        if (p >= 0.40) return '#F97316';
        if (p >= 0.25) return '#FBBF24'; // Gate B region
        if (p >= 0.10) return '#06B6D4';
        return '#10B981'; // Confident pass
      }

      case 'prob_A': {
        const p = die.prob_Model_A;
        if (p >= 0.51) return '#EF4444'; // Model A threshold
        if (p >= 0.40) return '#F97316';
        if (p >= 0.25) return '#FBBF24';
        if (p >= 0.10) return '#38BDF8';
        return '#10B981';
      }

      case 'prob_B': {
        const p = die.prob_Model_B;
        if (p >= 0.5376) return '#A855F7';
        if (p >= 0.35) return '#EC4899';
        return '#64748B';
      }

      case 'gate':
        return die.in_gate_B ? '#F59E0B' : '#3B82F6';

      case 'ground_truth':
        return die.actual_label === 1 ? '#EF4444' : '#22C55E';

      case 'error_type': {
        const actual = die.actual_label;
        const pred = die.decision_Cascade_B;
        if (die.recovered_FN_Gate_B === 1) return '#38BDF8'; // Rescued False Negative
        if (actual === 1 && pred === 1) return '#EF4444'; // True Positive
        if (actual === 0 && pred === 1) return '#F59E0B'; // False Positive (Alarm)
        if (actual === 1 && pred === 0) return '#A855F7'; // Remaining False Negative
        return '#1E293B'; // True Negative (Pass)
      }

      default:
        return '#22C55E';
    }
  };

  // Compute counts for active legend
  const legendCounts = useMemo(() => {
    let c1 = 0, c2 = 0, c3 = 0, c4 = 0, c5 = 0;
    for (const d of dies) {
      if (activeMode === 'status') {
        if (d.decision_Cascade_B === 1) c1++;
        else if (d.in_gate_B) c2++;
        else c3++;
      } else if (activeMode === 'gate') {
        if (d.in_gate_B) c1++;
        else c2++;
      } else if (activeMode === 'ground_truth') {
        if (d.actual_label === 1) c1++;
        else c2++;
      } else if (activeMode === 'error_type') {
        if (d.recovered_FN_Gate_B === 1) c1++;
        else if (d.actual_label === 1 && d.decision_Cascade_B === 1) c2++;
        else if (d.actual_label === 0 && d.decision_Cascade_B === 1) c3++;
        else if (d.actual_label === 1 && d.decision_Cascade_B === 0) c4++;
        else c5++;
      }
    }
    return { c1, c2, c3, c4, c5 };
  }, [dies, activeMode]);

  return (
    <div
      ref={containerRef}
      className={`flex flex-col bg-silicon-card border border-silicon-border rounded-lg overflow-hidden select-none ${className}`}
    >
      {/* Top Map Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-2.5 bg-silicon-surface border-b border-silicon-border text-xs">
        {/* Mode Selector */}
        <div className="flex items-center gap-1.5 flex-wrap">
          <span className="text-content-muted font-mono uppercase text-[11px] mr-1 flex items-center gap-1">
            <Layers className="w-3.5 h-3.5 text-telemetry-cyan" />
            Mode:
          </span>
          <button
            onClick={() => onChangeMode('status')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'status'
                ? 'bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            Decision Status
          </button>
          <button
            onClick={() => onChangeMode('probability')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'probability'
                ? 'bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            P(Final)
          </button>
          <button
            onClick={() => onChangeMode('prob_A')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'prob_A'
                ? 'bg-telemetry-blue/20 text-telemetry-blue border border-telemetry-blue/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            P(Model A)
          </button>
          <button
            onClick={() => onChangeMode('prob_B')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'prob_B'
                ? 'bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            P(Model B)
          </button>
          <button
            onClick={() => onChangeMode('gate')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'gate'
                ? 'bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            Gate B State
          </button>
          <button
            onClick={() => onChangeMode('ground_truth')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'ground_truth'
                ? 'bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            Ground Truth
          </button>
          <button
            onClick={() => onChangeMode('error_type')}
            className={`px-2.5 py-1 rounded text-xs font-mono font-medium transition-colors ${
              activeMode === 'error_type'
                ? 'bg-telemetry-purple/20 text-telemetry-purple border border-telemetry-purple/40 font-semibold'
                : 'text-content-secondary hover:text-content-primary bg-silicon-interactive/50'
            }`}
          >
            Error Triage
          </button>
        </div>


        {/* Zoom & View Controls */}
        <div className="flex items-center gap-1 bg-silicon-base p-0.5 rounded border border-silicon-border">
          <span className="text-[11px] font-mono text-content-muted px-2">
            {Math.round(zoom * 100)}%
          </span>
          <button
            onClick={handleZoomIn}
            className="p-1 hover:bg-silicon-interactive rounded text-content-secondary hover:text-content-primary transition-colors"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleZoomOut}
            className="p-1 hover:bg-silicon-interactive rounded text-content-secondary hover:text-content-primary transition-colors"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleReset}
            className="p-1 hover:bg-silicon-interactive rounded text-content-secondary hover:text-content-primary transition-colors"
            title="Reset to Fit"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* SVG Canvas Area */}
      <div
        className="relative w-full h-[540px] bg-silicon-canvas cursor-grab active:cursor-grabbing overflow-hidden flex items-center justify-center"
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <svg
          viewBox={viewBox}
          className="w-full h-full transition-transform duration-75"
          style={{
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: 'center center',
          }}
        >
          <defs>
            {/* Wafer surface gradient */}
            <radialGradient id="waferSurfaceGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#151D2C" stopOpacity="0.8" />
              <stop offset="90%" stopColor="#0E131F" stopOpacity="0.95" />
              <stop offset="100%" stopColor="#080B10" stopOpacity="1" />
            </radialGradient>

            {/* Notch shadow filter */}
            <filter id="dieGlow" x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="1.5" floodColor="#38BDF8" floodOpacity="0.9" />
            </filter>
          </defs>

          {/* Wafer Spatial Boundary */}
          <circle
            cx={centerX}
            cy={centerY}
            r={waferRadius}
            fill="url(#waferSurfaceGradient)"
            stroke="#32435E"
            strokeWidth="2.5"
            strokeDasharray="none"
          />

          {/* Spatial Margin Boundary */}
          <circle
            cx={centerX}
            cy={centerY}
            r={waferRadius * 0.96}
            fill="none"
            stroke="#1E293B"
            strokeWidth="1"
            strokeDasharray="4,4"
          />

          {/* Alignment Notch at 6 o'clock (Orientation Reference) */}
          <path
            d={`M ${centerX - 8} ${centerY + waferRadius - 1} Q ${centerX} ${centerY + waferRadius - 9} ${centerX + 8} ${centerY + waferRadius - 1}`}
            fill="#080B10"
            stroke="#38BDF8"
            strokeWidth="1.5"
          />

          {/* Center alignment crosshair */}
          <line
            x1={centerX - 10}
            y1={centerY}
            x2={centerX + 10}
            y2={centerY}
            stroke="#32435E"
            strokeWidth="0.8"
            strokeDasharray="2,2"
          />
          <line
            x1={centerX}
            y1={centerY - 10}
            x2={centerX}
            y2={centerY + 10}
            stroke="#32435E"
            strokeWidth="0.8"
            strokeDasharray="2,2"
          />

          {/* Die Matrix */}
          <g id="wafer-dies">
            {dies.map((die) => {
              const x = (die.die_col - bounds.minC + 1) * PITCH;
              const y = (die.die_row - bounds.minR + 1) * PITCH;
              const isSelected =
                selectedDie &&
                selectedDie.die_row === die.die_row &&
                selectedDie.die_col === die.die_col;
              const isHovered =
                hoveredDieState &&
                hoveredDieState.die_row === die.die_row &&
                hoveredDieState.die_col === die.die_col;

              const fillColor = getDieColor(die);

              return (
                <g key={`${die.die_row}-${die.die_col}`}>
                  <rect
                    x={x}
                    y={y}
                    width={DIE_SIZE}
                    height={DIE_SIZE}
                    rx={1}
                    fill={fillColor}
                    stroke={
                      isSelected
                        ? '#FFFFFF'
                        : isHovered
                        ? '#38BDF8'
                        : '#0E131F'
                    }
                    strokeWidth={isSelected ? 2.2 : isHovered ? 1.5 : 0.6}
                    filter={isSelected ? 'url(#dieGlow)' : undefined}
                    className="cursor-pointer transition-colors duration-75"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectDie(die);
                    }}
                    onMouseEnter={(e) => {
                      setHoveredDieState(die);
                      if (onHoverDie) onHoverDie(die, e);
                    }}
                    onMouseLeave={() => {
                      setHoveredDieState(null);
                      if (onHoverDie) onHoverDie(null);
                    }}
                  />
                  {/* Subtle defect indicator dot for high-contrast accessibility */}
                  {die.decision_Cascade_B === 1 && DIE_SIZE >= 10 && (
                    <circle
                      cx={x + DIE_SIZE / 2}
                      cy={y + DIE_SIZE / 2}
                      r={1.5}
                      fill="#FFFFFF"
                      opacity={0.8}
                      pointerEvents="none"
                    />
                  )}
                </g>
              );
            })}
          </g>
        </svg>

        {/* Floating Die Tooltip Overlay */}
        {hoveredDieState && (
          <div className="absolute top-4 left-4 z-20 pointer-events-none bg-silicon-elevated/95 backdrop-blur border border-silicon-border rounded-lg shadow-xl p-3 text-xs font-mono space-y-1.5 min-w-[240px]">
            <div className="flex items-center justify-between border-b border-silicon-border pb-1">
              <div>
                <span className="text-[10px] text-content-muted block">{hoveredDieState.wafer_id}</span>
                <span className="text-telemetry-cyan font-bold tracking-tight">
                  R{hoveredDieState.die_row} / C{hoveredDieState.die_col}
                </span>
              </div>
              <span
                className={`px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${
                  hoveredDieState.decision_Cascade_B === 1
                    ? 'bg-telemetry-red/20 text-telemetry-red border border-telemetry-red/30'
                    : 'bg-telemetry-green/20 text-telemetry-green border border-telemetry-green/30'
                }`}
              >
                {hoveredDieState.decision_Cascade_B === 1 ? 'FAIL' : 'PASS'}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px]">
              <span className="text-content-muted">Final P:</span>
              <span className="text-content-primary font-bold text-right">
                {(hoveredDieState.prob_Cascade_B * 100).toFixed(2)}%
              </span>

              <span className="text-content-muted">LightGBM Model A:</span>
              <span className="text-content-secondary text-right">
                {(hoveredDieState.prob_Model_A * 100).toFixed(2)}%
              </span>

              <span className="text-content-muted">Model B:</span>
              <span className="text-right text-telemetry-purple font-medium">
                {hoveredDieState.in_gate_B
                  ? `${(hoveredDieState.prob_Model_B * 100).toFixed(2)}%`
                  : 'Not executed'}
              </span>

              <span className="text-content-muted">Gate:</span>
              <span
                className={`text-right font-semibold ${
                  hoveredDieState.in_gate_B
                    ? 'text-telemetry-amber'
                    : 'text-telemetry-blue'
                }`}
              >
                {hoveredDieState.in_gate_B ? 'ESCALATED' : 'BYPASSED'}
              </span>

              <span className="text-content-muted">Actual Ground Truth:</span>
              <span
                className={`text-right font-semibold ${
                  hoveredDieState.actual_label === 1
                    ? 'text-telemetry-red'
                    : 'text-telemetry-green'
                }`}
              >
                {hoveredDieState.actual_label === 1 ? 'FAIL (1)' : 'PASS (0)'}
              </span>

              {hoveredDieState.recovered_FN_Gate_B === 1 && (
                <div className="col-span-2 text-center py-0.5 mt-0.5 rounded bg-telemetry-cyan/20 text-telemetry-cyan font-bold text-[10px] border border-telemetry-cyan/30 flex items-center justify-center gap-1">
                  <Sparkles className="w-3 h-3" />
                  RESCUED FALSE NEGATIVE
                </div>
              )}
            </div>
          </div>
        )}

      </div>

      {/* Dynamic Bottom Legend */}
      <div className="px-4 py-2 bg-silicon-surface border-t border-silicon-border text-xs flex flex-wrap items-center justify-between gap-3 font-mono">
        {activeMode === 'status' && (
          <div className="flex items-center gap-4 flex-wrap text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#22C55E]" />
              <span className="text-content-secondary">
                Confident Pass ({legendCounts.c3})
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#F59E0B]" />
              <span className="text-content-secondary">
                Escalated Pass ({legendCounts.c2})
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#EF4444]" />
              <span className="text-content-secondary">
                Defect / Fail ({legendCounts.c1})
              </span>
            </div>
          </div>
        )}

        {activeMode === 'probability' && (
          <div className="flex items-center gap-3 text-[11px]">
            <span className="text-content-muted">Probability:</span>
            <span className="text-telemetry-green">0.0%</span>
            <div className="h-2 w-32 rounded bg-gradient-to-r from-[#10B981] via-[#FBBF24] to-[#EF4444]" />
            <span className="text-telemetry-red">100.0%</span>
            <span className="text-content-muted ml-2">
              (Gate B: [25%, 55%])
            </span>
          </div>
        )}

        {activeMode === 'gate' && (
          <div className="flex items-center gap-4 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#3B82F6]" />
              <span className="text-content-secondary">
                Fast-Path Bypassed ({legendCounts.c2})
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#F59E0B]" />
              <span className="text-content-secondary">
                Escalated to Stage 2 ({legendCounts.c1})
              </span>
            </div>
          </div>
        )}

        {activeMode === 'ground_truth' && (
          <div className="flex items-center gap-4 text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#22C55E]" />
              <span className="text-content-secondary">
                Actual Pass (0): {legendCounts.c2}
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#EF4444]" />
              <span className="text-content-secondary">
                Actual Defect (1): {legendCounts.c1}
              </span>
            </div>
          </div>
        )}

        {activeMode === 'error_type' && (
          <div className="flex items-center gap-4 flex-wrap text-[11px]">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#38BDF8]" />
              <span className="text-telemetry-cyan font-semibold">
                Rescued FN ({legendCounts.c1})
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#EF4444]" />
              <span className="text-content-secondary">
                True Positive ({legendCounts.c2})
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#F59E0B]" />
              <span className="text-content-secondary">
                False Alarm ({legendCounts.c3})
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-[#A855F7]" />
              <span className="text-content-secondary">
                Missed FN ({legendCounts.c4})
              </span>
            </div>
          </div>
        )}

        <div className="text-content-muted text-[11px]">
          Showing {dies.length.toLocaleString()} Dies
        </div>
      </div>
    </div>
  );
};
