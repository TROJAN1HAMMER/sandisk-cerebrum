import { useQuery } from '@tanstack/react-query';
import { systemService } from '../services/system';

export function useSystemHealth() {
  return useQuery({
    queryKey: ['system', 'health'],
    queryFn: systemService.getHealth,
    refetchInterval: 30000, // Check health every 30s
  });
}

export function useSystemStatus() {
  return useQuery({
    queryKey: ['system', 'status'],
    queryFn: systemService.getStatus,
    refetchInterval: 60000,
  });
}
