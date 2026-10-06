import { useQuery } from '@tanstack/react-query';
import { modelService } from '../services/models';

export function useModelMetrics() {
  return useQuery({
    queryKey: ['models', 'metrics'],
    queryFn: modelService.getMetrics,
    staleTime: Infinity, // Authoritative metrics are constant
  });
}

export function useModelRelease() {
  return useQuery({
    queryKey: ['models', 'release'],
    queryFn: modelService.getRelease,
    staleTime: Infinity,
  });
}

export function useModelBenchmark() {
  return useQuery({
    queryKey: ['models', 'benchmark'],
    queryFn: modelService.getBenchmark,
    staleTime: Infinity,
  });
}

export function useErrorDrilldown(group: string = 'recovered_fns', limit: number = 50) {
  return useQuery({
    queryKey: ['models', 'error-drilldown', group, limit],
    queryFn: () => modelService.getErrorDrilldown(group, limit),
  });
}

export function useProbabilityDistribution() {
  return useQuery({
    queryKey: ['models', 'probability-distribution'],
    queryFn: modelService.getProbabilityDistribution,
    staleTime: Infinity,
  });
}

export function useRocPrCurves() {
  return useQuery({
    queryKey: ['models', 'roc-pr-curves'],
    queryFn: modelService.getRocPrCurves,
    staleTime: Infinity,
  });
}

export function useHighRiskDies(limit: number = 50, waferId?: string) {
  return useQuery({
    queryKey: ['models', 'high-risk-dies', limit, waferId || 'all'],
    queryFn: () => modelService.getHighRiskDies(limit, waferId),
    staleTime: 60_000,
  });
}

