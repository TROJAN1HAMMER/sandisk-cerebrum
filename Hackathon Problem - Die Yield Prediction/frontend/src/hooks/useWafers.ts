import { useQuery } from '@tanstack/react-query';
import { waferService } from '../services/wafers';

export function useWafers() {
  return useQuery({
    queryKey: ['wafers', 'list'],
    queryFn: waferService.getWafers,
  });
}

export function useWaferSummary(waferId: string) {
  return useQuery({
    queryKey: ['wafers', 'summary', waferId],
    queryFn: () => waferService.getWaferSummary(waferId),
    enabled: !!waferId,
  });
}

export function useWaferDies(waferId: string, filterMode: string = 'all', limit: number = 3000) {
  return useQuery({
    queryKey: ['wafers', 'dies', waferId, filterMode, limit],
    queryFn: () => waferService.getWaferDies(waferId, filterMode, limit),
    enabled: !!waferId,
  });
}

export function useSingleWaferDie(waferId: string, dieRow: number, dieCol: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['wafers', 'single-die', waferId, dieRow, dieCol],
    queryFn: () => waferService.getSingleDie(waferId, dieRow, dieCol),
    enabled: enabled && !!waferId,
  });
}

export function useWaferHotspots(waferId: string, enabled: boolean = true) {
  return useQuery({
    queryKey: ['wafers', 'hotspots', waferId],
    queryFn: () => waferService.getWaferHotspots(waferId),
    enabled: enabled && !!waferId,
  });
}
