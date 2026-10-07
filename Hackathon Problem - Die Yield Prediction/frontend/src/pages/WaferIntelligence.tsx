import React, { useState, useEffect, useMemo } from 'react';
import { useParams, useNavigate, useSearchParams, useLocation } from 'react-router-dom';
import { useWafers, useWaferSummary, useWaferDies, useWaferHotspots } from '../hooks/useWafers';
import { DiePoint, HotspotSummaryItem } from '../types/wafers';
import { WaferMap, HeatmapMode } from '../components/wafers/WaferMap';
import { DieInspector } from '../components/wafers/DieInspector';
import { CrossWaferCompare } from '../components/wafers/CrossWaferCompare';
import { SectionHeader } from '../components/ui/SectionHeader';
import { MetricCard } from '../components/ui/MetricCard';
import { Panel } from '../components/ui/Panel';
import { StatusBadge } from '../components/ui/StatusBadge';
import { LoadingSkeleton } from '../components/ui/LoadingSkeleton';
import { ErrorState } from '../components/ui/ErrorState';
import {
  Disc,
  Search,
  Layers,
  Sparkles,
  Flame,
  ArrowRight,
  TrendingUp,
  AlertTriangle,
  Compass,
  Maximize2,
  SlidersHorizontal,
  Filter,
  CheckCircle2,
  XCircle,
} from 'lucide-react';

export const WaferIntelligence: React.FC = () => {
  const { waferId: routeWaferId, row: routeRow, col: routeCol } = useParams<{
    waferId?: string;
    row?: string;
    col?: string;
  }>();
  const navigate = useNavigate();
  const location = useLocation();
  const [searchParams] = useSearchParams();

  // Load all 32 canonical validation wafers
  const { data: wafersList, isLoading: isWafersLoading, error: wafersError } = useWafers();

  // Selected wafer state (defaults to URL param, or W_N_0057, or first wafer)
  const [selectedWaferId, setSelectedWaferId] = useState<string>(
    routeWaferId || 'W_N_0057'
  );

  // Sync selected wafer from URL param
  useEffect(() => {
    if (routeWaferId && routeWaferId !== selectedWaferId) {
      setSelectedWaferId(routeWaferId);
    }
  }, [routeWaferId]);

  // If initial wafer list loads and routeWaferId was empty, set to W_N_0057 or first wafer
  useEffect(() => {
    if (!routeWaferId && wafersList?.wafers?.length) {
      const defaultId = wafersList.wafers.some((w) => w.wafer_id === 'W_N_0057')
        ? 'W_N_0057'
        : wafersList.wafers[0].wafer_id;
      setSelectedWaferId(defaultId);
    }
  }, [wafersList, routeWaferId]);

  // Quick filter state: check if passed via query param (e.g. ?filter=fn)
  const initialFilter = searchParams.get('filter') || 'all';
  const [filterMode, setFilterMode] = useState<string>(initialFilter);
  const [minProbFilter, setMinProbFilter] = useState<number>(0);

  // Load dies and summary for active wafer
  const { data: waferData, isLoading: isDiesLoading, error: diesError } = useWaferDies(
    selectedWaferId,
    'all',
    3000
  );
  const { data: waferSummary } = useWaferSummary(selectedWaferId);
  const { data: hotspotsData } = useWaferHotspots(selectedWaferId);

  // Selected die state
  const [selectedDie, setSelectedDie] = useState<DiePoint | null>(null);
  const [activeHotspotId, setActiveHotspotId] = useState<string | null>(null);
  const [activeHeatmapMode, setActiveHeatmapMode] = useState<HeatmapMode>('status');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [searchError, setSearchError] = useState<string | null>(null);
  const [showCompareModal, setShowCompareModal] = useState<boolean>(false);

  // Auto-select die from deep link params: /wafer-intelligence/:waferId/:row/:col
  useEffect(() => {
    if (waferData?.dies && routeRow !== undefined && routeCol !== undefined) {
      const r = parseInt(routeRow, 10);
      const c = parseInt(routeCol, 10);
      const matched = waferData.dies.find((d) => d.die_row === r && d.die_col === c);
      if (matched) {
        setSelectedDie(matched);
        setTimeout(() => {
          document.getElementById('selected-die-inspector')?.scrollIntoView({
            behavior: 'smooth',
            block: 'start',
          });
        }, 200);
      }
    }
  }, [waferData?.dies, routeRow, routeCol]);

  // Handle wafer selection switch
  const handleSelectWafer = (newWaferId: string) => {
    setSelectedWaferId(newWaferId);
    setSelectedDie(null);
    setActiveHotspotId(null);
    setSearchQuery('');
    setSearchError(null);
    navigate(`/wafer-intelligence/${newWaferId}`);
  };

  // Handle die selection with search params preservation and auto-scroll
  const handleSelectDie = (die: DiePoint) => {
    setSelectedDie(die);
    const search = location.search;
    navigate(`/wafer-intelligence/${die.wafer_id}/${die.die_row}/${die.die_col}${search}`, {
      replace: true,
    });
    setTimeout(() => {
      document.getElementById('selected-die-inspector')?.scrollIntoView({
        behavior: 'smooth',
        block: 'start',
      });
    }, 100);
  };

  // Handle hotspot selection & centering
  const handleSelectHotspot = (hs: HotspotSummaryItem) => {
    setActiveHotspotId(hs.hotspot_id);
    if (!waferData?.dies) return;
    const hotspotDies = waferData.dies.filter(
      (d) => Math.hypot(d.die_row - hs.center_row, d.die_col - hs.center_col) <= hs.radius_dies
    );
    const targetDie =
      hotspotDies.find((d) => d.prob_Cascade_B >= 0.50) ||
      hotspotDies.find((d) => d.die_row === hs.center_row && d.die_col === hs.center_col) ||
      waferData.dies.find((d) => d.die_row === hs.center_row && d.die_col === hs.center_col) ||
      hotspotDies[0] ||
      waferData.dies[0];

    if (targetDie) {
      handleSelectDie(targetDie);
    }
  };

  // Handle die search with multi-format coordinate parsing
  const handleSearchDie = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchError(null);
    if (!searchQuery.trim() || !waferData?.dies) return;

    const raw = searchQuery.trim();

    // Check if query contains wafer ID (e.g. W_N_0057/R32/C23 or W_N_0057/32/23)
    const waferMatch = raw.match(/(W_N_\d+)[/,\s]+(?:R?(\d+))[/,\s]+(?:C?(\d+))/i);
    if (waferMatch) {
      const targetWafer = waferMatch[1].toUpperCase();
      const targetRow = parseInt(waferMatch[2], 10);
      const targetCol = parseInt(waferMatch[3], 10);

      if (targetWafer !== selectedWaferId) {
        navigate(`/wafer-intelligence/${targetWafer}/${targetRow}/${targetCol}`);
        return;
      }

      const matched = waferData.dies.find(
        (d) => d.die_row === targetRow && d.die_col === targetCol
      );
      if (matched) {
        handleSelectDie(matched);
        return;
      }
    }

    // Parse standard coordinate formats: R32 C23, 32, 23, 32 23, 32/23, R32C23
    const clean = raw.toUpperCase().replace(/[RC,/]/g, ' ').trim();
    const parts = clean.split(/\s+/).filter(Boolean);

    if (parts.length >= 2) {
      const targetRow = parseInt(parts[0], 10);
      const targetCol = parseInt(parts[1], 10);

      const matched = waferData.dies.find(
        (d) => d.die_row === targetRow && d.die_col === targetCol
      );

      if (matched) {
        handleSelectDie(matched);
      } else {
        setSearchError(`Die [${targetRow}, ${targetCol}] not found on wafer ${selectedWaferId}`);
      }
    } else {
      setSearchError('Enter format: "R32 C23", "32, 23", or "32/23"');
    }
  };

  // Filtered dies based on quick filters
  const filteredDies = useMemo(() => {
    if (!waferData?.dies) return [];
    return waferData.dies.filter((d) => {
      // Probability threshold filter
      if (minProbFilter > 0 && d.prob_Cascade_B < minProbFilter) return false;

      // Filter mode
      if (filterMode === 'fail') return d.decision_Cascade_B === 1;
      if (filterMode === 'pass') return d.decision_Cascade_B === 0;
      if (filterMode === 'escalated') return d.in_gate_B;
      if (filterMode === 'bypassed') return !d.in_gate_B;
      if (filterMode === 'high_risk') return d.prob_Cascade_B >= 0.50;
      if (filterMode === 'recovered') return d.recovered_FN_Gate_B === 1;
      if (filterMode === 'fn') return d.actual_label === 1 && d.decision_Cascade_B === 0;
      if (filterMode === 'fp') return d.actual_label === 0 && d.decision_Cascade_B === 1;
      if (filterMode === 'hotspot' && hotspotsData?.hotspots && hotspotsData.hotspots.length > 0) {
        const targetClusters = activeHotspotId
          ? hotspotsData.hotspots.filter((h) => h.hotspot_id === activeHotspotId)
          : hotspotsData.hotspots;
        return targetClusters.some(
          (h) => Math.hypot(d.die_row - h.center_row, d.die_col - h.center_col) <= h.radius_dies
        );
      }

      return true;
    });
  }, [waferData?.dies, filterMode, minProbFilter, hotspotsData?.hotspots, activeHotspotId]);

  // Quick Action: Inspect Highest Risk Die
  const handleInspectHighestRisk = () => {
    if (!waferData?.dies || waferData.dies.length === 0) return;
    let highest = waferData.dies[0];
    for (const d of waferData.dies) {
      if (d.prob_Cascade_B > highest.prob_Cascade_B) {
        highest = d;
      }
    }
    handleSelectDie(highest);
  };

  // Compute spatial defect patterns from actual coordinates
  const spatialMetrics = useMemo(() => {
    if (!waferData?.dies) return null;
    const dies = waferData.dies;
    let edgeDefects = 0;
    let centerDefects = 0;
    let q1 = 0, q2 = 0, q3 = 0, q4 = 0;

    let minR = Infinity, maxR = -Infinity, minC = Infinity, maxC = -Infinity;
    for (const d of dies) {
      if (d.die_row < minR) minR = d.die_row;
      if (d.die_row > maxR) maxR = d.die_row;
      if (d.die_col < minC) minC = d.die_col;
      if (d.die_col > maxC) maxC = d.die_col;
    }
    const midR = (minR + maxR) / 2;
    const midC = (minC + maxC) / 2;

    for (const d of dies) {
      if (d.decision_Cascade_B === 1) {
        if ((d.edge_dist ?? 0.5) <= 0.25) edgeDefects++;
        else centerDefects++;

        if (d.die_row < midR && d.die_col >= midC) q1++;
        else if (d.die_row < midR && d.die_col < midC) q2++;
        else if (d.die_row >= midR && d.die_col < midC) q3++;
        else q4++;
      }
    }

    return {
      edgeDefects,
      centerDefects,
      q1,
      q2,
      q3,
      q4,
    };
  }, [waferData?.dies]);

  if (isWafersLoading) {
    return (
      <div className="space-y-6">
        <SectionHeader
          title="Wafer Intelligence & Spatial Defect Analysis"
          subtitle="Loading wafer catalog and spatial die telemetry..."
        />
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {Array.from({ length: 4 }).map((_, i) => (
            <LoadingSkeleton key={i} className="h-28" />
          ))}
        </div>
        <LoadingSkeleton className="h-[540px]" />
      </div>
    );
  }

  if (wafersError) {
    return (
      <ErrorState
        title="Failed to Load Wafer Catalog"
        message={(wafersError as Error).message}
      />
    );
  }

  const currentSummary =
    waferSummary ||
    wafersList?.wafers.find((w) => w.wafer_id === selectedWaferId);

  const predictedYield = currentSummary
    ? (
        ((currentSummary.total_dies - currentSummary.cascade_detections) /
          Math.max(1, currentSummary.total_dies)) *
        100
      ).toFixed(2)
    : '—';

  return (
    <div className="space-y-6">
      {/* Page Title & Navigation Actions */}
      <SectionHeader
        title="Wafer Intelligence & Spatial Defect Workstation"
        subtitle="Interactive spatial die reticle grid, multi-resolution cascade routing, and 2,000-block signal inspection"
        badge={<StatusBadge variant="info" label="SPATIAL DIE RETICLE" />}
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowCompareModal(true)}
              disabled={!currentSummary}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-mono bg-silicon-elevated hover:bg-silicon-interactive text-content-primary border border-silicon-border transition-colors font-medium disabled:opacity-50"
            >
              <SlidersHorizontal className="w-3.5 h-3.5 text-telemetry-cyan" />
              <span>Compare Wafers</span>
            </button>
          </div>
        }
      />

      {/* Wafer Selector & Search Control Strip */}
      <div className="bg-silicon-card border border-silicon-border rounded-lg p-3.5 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
        {/* Wafer Selector Dropdown */}
        <div className="flex items-center gap-2 flex-wrap">
          <Disc className="w-4 h-4 text-telemetry-cyan shrink-0" />
          <span className="text-content-muted">Wafer:</span>
          <select
            value={selectedWaferId}
            onChange={(e) => handleSelectWafer(e.target.value)}
            className="bg-silicon-interactive border border-silicon-border text-content-primary font-bold rounded px-3 py-1.5 focus:outline-none focus:border-telemetry-cyan font-mono"
          >
            {wafersList?.wafers.map((w) => (
              <option key={w.wafer_id} value={w.wafer_id}>
                {w.wafer_id} — {w.total_dies} dies ({w.actual_failures} defects,{' '}
                {w.fail_rate_pct.toFixed(1)}% fail)
              </option>
            ))}
          </select>

          {/* Quick preset links for verified demo coordinates */}
          {selectedWaferId === 'W_N_0057' && (
            <div className="flex items-center gap-1 ml-2 text-[11px]">
              <span className="text-content-muted">Demo Dies:</span>
              <button
                onClick={() => {
                  const d = waferData?.dies?.find(
                    (die) => die.die_row === 32 && die.die_col === 23
                  );
                  if (d) handleSelectDie(d);
                }}
                className="px-2 py-0.5 rounded bg-telemetry-cyan/20 text-telemetry-cyan hover:bg-telemetry-cyan/30 border border-telemetry-cyan/40 transition-colors font-bold"
              >
                R32 C23 (Rescued FN)
              </button>
              <button
                onClick={() => {
                  const d = waferData?.dies?.find(
                    (die) => die.die_row === 18 && die.die_col === 26
                  );
                  if (d) handleSelectDie(d);
                }}
                className="px-2 py-0.5 rounded bg-telemetry-purple/20 text-telemetry-purple hover:bg-telemetry-purple/30 border border-telemetry-purple/40 transition-colors font-bold"
              >
                R18 C26 (Rescued FN)
              </button>
            </div>
          )}
        </div>

        {/* Die Search Box & Quick Actions */}
        <div className="flex items-center gap-2 flex-wrap">
          <form onSubmit={handleSearchDie} className="flex items-center gap-1.5">
            <div className="relative">
              <input
                type="text"
                placeholder='Search Die: "R32 C23", "32/23"'
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  setSearchError(null);
                }}
                className="bg-silicon-base border border-silicon-border text-content-primary rounded px-2.5 py-1.5 text-xs placeholder:text-content-muted focus:outline-none focus:border-telemetry-cyan font-mono w-52"
              />
            </div>
            <button
              type="submit"
              className="px-2.5 py-1.5 rounded bg-silicon-interactive hover:bg-silicon-interactive/80 text-content-primary border border-silicon-border transition-colors flex items-center gap-1"
            >
              <Search className="w-3.5 h-3.5 text-telemetry-cyan" />
              <span>Locate</span>
            </button>
          </form>

          <button
            onClick={handleInspectHighestRisk}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded bg-telemetry-red/20 text-telemetry-red hover:bg-telemetry-red/30 border border-telemetry-red/40 transition-colors font-bold"
          >
            <Flame className="w-3.5 h-3.5" />
            <span>Highest-Risk Die</span>
          </button>
        </div>
      </div>

      {searchError && (
        <div className="p-2.5 rounded bg-telemetry-red/10 border border-telemetry-red/30 text-telemetry-red text-xs font-mono flex items-center gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0" />
          <span>{searchError}</span>
        </div>
      )}

      {/* Die Quick Filter Bar */}
      <div className="p-3 bg-silicon-card border border-silicon-border rounded-lg flex flex-wrap items-center justify-between gap-3 font-mono text-xs">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-content-muted uppercase text-[11px] flex items-center gap-1 mr-1">
            <Filter className="w-3.5 h-3.5 text-telemetry-cyan" />
            Quick Filter:
          </span>

          {[
            { id: 'all', label: 'All Dies' },
            { id: 'fail', label: 'Defects (FAIL)' },
            { id: 'pass', label: 'Yield (PASS)' },
            { id: 'escalated', label: 'Escalated' },
            { id: 'bypassed', label: 'Bypassed' },
            { id: 'high_risk', label: 'High Risk (P≥0.50)' },
            { id: 'recovered', label: 'Rescued FN' },
            { id: 'fn', label: 'Missed FN' },
            { id: 'fp', label: 'False Alarms (FP)' },
            ...(hotspotsData?.hotspots && hotspotsData.hotspots.length > 0
              ? [{ id: 'hotspot', label: `🔥 Hotspots (${hotspotsData.hotspots.length})` }]
              : []),
          ].map((f) => (
            <button
              key={f.id}
              onClick={() => setFilterMode(f.id)}
              className={`px-2.5 py-1 rounded text-xs transition-colors ${
                filterMode === f.id
                  ? 'bg-telemetry-cyan/20 text-telemetry-cyan border border-telemetry-cyan/40 font-bold'
                  : 'bg-silicon-surface text-content-secondary hover:text-content-primary border border-silicon-border/60'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>

        {/* Filter readout */}
        <div className="text-[11px] text-content-muted">
          Showing <strong className="text-telemetry-cyan">{filteredDies.length}</strong> matching dies
          {waferData?.dies && (
            <span> (out of {waferData.dies.length} total dies)</span>
          )}
        </div>
      </div>

      {/* Wafer Overview KPIs Strip */}
      {currentSummary && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <MetricCard
            title="Total Die Population"
            value={currentSummary.total_dies.toLocaleString()}
            subtitle="Silicon Reticle Matrix"
            accent="neutral"
          />
          <MetricCard
            title="Predicted Yield"
            value={`${predictedYield}%`}
            subtitle={`${currentSummary.total_dies - currentSummary.cascade_detections} Good Dies`}
            accent="green"
          />
          <MetricCard
            title="Cascade Defectives"
            value={currentSummary.cascade_detections}
            subtitle={`vs ${currentSummary.model_a_detections} Model A`}
            accent="red"
          />
          <MetricCard
            title="Ground Truth Failures"
            value={currentSummary.actual_failures}
            subtitle={`${currentSummary.fail_rate_pct.toFixed(2)}% Defect Base`}
            accent="red"
          />
          <MetricCard
            title="Stage 2 Escalations"
            value={currentSummary.escalated_dies}
            subtitle={`${((currentSummary.escalated_dies / currentSummary.total_dies) * 100).toFixed(1)}% of Wafer`}
            accent="amber"
          />
          <MetricCard
            title="Rescued Defectives"
            value={`+${currentSummary.recovered_fns}`}
            subtitle="FN Escapes Rescued"
            accent="cyan"
          />
        </div>
      )}

      {/* Interactive Wafer Map Display */}
      {isDiesLoading ? (
        <div className="space-y-4">
          <LoadingSkeleton className="h-[540px] w-full" />
        </div>
      ) : diesError ? (
        <ErrorState
          title={`Failed to load die coordinates for ${selectedWaferId}`}
          message={(diesError as Error).message}
        />
      ) : waferData?.dies ? (
        <div className="space-y-6">
          <WaferMap
            dies={filteredDies.length > 0 ? filteredDies : waferData.dies}
            selectedDie={selectedDie}
            onSelectDie={handleSelectDie}
            activeMode={activeHeatmapMode}
            onChangeMode={setActiveHeatmapMode}
            hotspots={hotspotsData?.hotspots || []}
            activeHotspotId={activeHotspotId}
            onSelectHotspot={handleSelectHotspot}
          />

          {/* Persistent Selected Die Inspector */}
          {selectedDie && (
            <div id="selected-die-inspector" className="scroll-mt-4">
              <DieInspector
                die={selectedDie}
                allDies={waferData.dies}
                onSelectDie={handleSelectDie}
                onClose={() => setSelectedDie(null)}
              />
            </div>
          )}

          {/* Spatial Hotspot Clusters Intelligence Panel */}
          {hotspotsData?.hotspots && hotspotsData.hotspots.length > 0 && (
            <Panel
              title="Spatial Hotspot Intelligence & Risk Partitioning"
              subtitle={`Wafer ${selectedWaferId}: ${hotspotsData.total_high_risk_dies} high-risk dies categorized into ${hotspotsData.hotspots_count} physical reticle zones.`}
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {hotspotsData.hotspots.map((hs) => (
                  <div
                    key={hs.hotspot_id}
                    onClick={() => handleSelectHotspot(hs)}
                    className={`p-4 rounded-lg bg-silicon-elevated border space-y-3 font-mono text-xs cursor-pointer transition-all ${
                      activeHotspotId === hs.hotspot_id
                        ? 'border-telemetry-amber ring-2 ring-telemetry-amber bg-silicon-interactive/80 shadow-lg'
                        : 'border-silicon-border hover:border-telemetry-amber/60 hover:bg-silicon-elevated/90'
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-bold text-content-primary flex items-center gap-1.5 text-sm">
                        <Flame className="w-4 h-4 text-telemetry-amber" />
                        {hs.region_name}
                      </span>
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-telemetry-amber/20 text-telemetry-amber border border-telemetry-amber/40">
                        R = {hs.radius_dies} DIES
                      </span>
                    </div>

                    <div className="space-y-1.5 text-content-secondary text-xs">
                      <div className="flex justify-between">
                        <span className="text-content-muted">Morphology:</span>
                        <span className="text-telemetry-cyan font-semibold">{hs.primary_morphology}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-content-muted">High-Risk Dies in Cluster:</span>
                        <span className="text-telemetry-red font-bold">{hs.high_risk_die_count} of {hs.die_count} dies</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-content-muted">Mean Risk Probability:</span>
                        <span className="text-content-primary font-bold">{(hs.mean_risk_prob * 100).toFixed(1)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-content-muted">Centroid Coordinates:</span>
                        <span className="text-telemetry-purple font-bold">Row {hs.center_row}, Col {hs.center_col}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-content-muted">Observed Defects:</span>
                        <span className="text-telemetry-amber font-semibold">{hs.observed_defect_count} ground-truth defects</span>
                      </div>
                    </div>

                    <div className="p-2 rounded bg-silicon-canvas/80 border border-silicon-border/60 text-[11px] text-content-secondary leading-relaxed">
                      <span className="text-telemetry-cyan font-semibold mr-1">SIGNATURE:</span>
                      {hs.suspicious_signature}
                    </div>

                    {/* Quick drill-down to centroid / highest-risk die */}
                    <div className="pt-1 flex items-center justify-end">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleSelectHotspot(hs);
                        }}
                        className="px-2.5 py-1 rounded bg-telemetry-amber/15 hover:bg-telemetry-amber/25 border border-telemetry-amber/30 text-telemetry-amber text-[11px] font-semibold transition-colors flex items-center gap-1"
                      >
                        <span>Inspect Hotspot Center</span>
                        <ArrowRight className="w-3 h-3" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </Panel>
          )}

          {/* Spatial Pattern Analysis Summary */}
          {spatialMetrics && (
            <Panel
              title="Spatial Defect Distribution & Coordinate Symmetry"
              subtitle="Mathematically computed coordinate layout without heuristic clustering assumptions"
            >
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
                <div className="p-3 bg-silicon-elevated rounded border border-silicon-border">
                  <div className="text-content-muted text-[11px]">Edge vs Center</div>
                  <div className="text-base font-bold text-content-primary mt-1">
                    {spatialMetrics.edgeDefects} Edge / {spatialMetrics.centerDefects} Center
                  </div>
                  <div className="text-[11px] text-content-secondary mt-0.5">
                    {spatialMetrics.edgeDefects + spatialMetrics.centerDefects > 0
                      ? `${(
                          (spatialMetrics.edgeDefects /
                            (spatialMetrics.edgeDefects + spatialMetrics.centerDefects)) *
                          100
                        ).toFixed(1)}% Perimeter Ring Concentration`
                      : 'Zero Defectives'}
                  </div>
                </div>

                <div className="p-3 bg-silicon-elevated rounded border border-silicon-border">
                  <div className="text-content-muted text-[11px]">North-East (Q1)</div>
                  <div className="text-base font-bold text-telemetry-cyan mt-1">
                    {spatialMetrics.q1} Defective Dies
                  </div>
                  <div className="text-[11px] text-content-secondary mt-0.5">Top-Right Reticle Zone</div>
                </div>

                <div className="p-3 bg-silicon-elevated rounded border border-silicon-border">
                  <div className="text-content-muted text-[11px]">North-West (Q2)</div>
                  <div className="text-base font-bold text-telemetry-purple mt-1">
                    {spatialMetrics.q2} Defective Dies
                  </div>
                  <div className="text-[11px] text-content-secondary mt-0.5">Top-Left Reticle Zone</div>
                </div>

                <div className="p-3 bg-silicon-elevated rounded border border-silicon-border">
                  <div className="text-content-muted text-[11px]">South Hemispheres (Q3 + Q4)</div>
                  <div className="text-base font-bold text-telemetry-amber mt-1">
                    {spatialMetrics.q3 + spatialMetrics.q4} Defective Dies
                  </div>
                  <div className="text-[11px] text-content-secondary mt-0.5">
                    Q3: {spatialMetrics.q3} • Q4: {spatialMetrics.q4} (Notch zone)
                  </div>
                </div>
              </div>
            </Panel>
          )}
        </div>
      ) : null}

      {/* Cross-Wafer Comparison Modal */}
      {showCompareModal && currentSummary && wafersList?.wafers && (
        <CrossWaferCompare
          currentWafer={currentSummary}
          allWafers={wafersList.wafers}
          onClose={() => setShowCompareModal(false)}
          onSwitchWafer={handleSelectWafer}
        />
      )}
    </div>
  );
};
