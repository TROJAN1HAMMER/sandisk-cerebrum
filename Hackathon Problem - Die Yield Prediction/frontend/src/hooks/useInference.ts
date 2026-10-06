import { useQuery, useMutation } from '@tanstack/react-query';
import { inferenceService } from '../services/inference';
import { SingleDieRequest } from '../types/inference';

export function useDemoSamples() {
  return useQuery({
    queryKey: ['inference', 'demo-samples'],
    queryFn: inferenceService.getDemoSamples,
    staleTime: Infinity,
  });
}

export function useSingleDieInference() {
  return useMutation({
    mutationFn: (req: SingleDieRequest) => inferenceService.predictSingleDie(req),
  });
}

export function useBatchInference() {
  return useMutation({
    mutationFn: ({ file, gateLower, gateUpper }: { file: File; gateLower?: number; gateUpper?: number }) =>
      inferenceService.predictBatch(file, gateLower, gateUpper),
  });
}

export function useBlockAttribution(waferId: string, dieRow: number, dieCol: number, enabled: boolean = true) {
  return useQuery({
    queryKey: ['inference', 'block-attribution', waferId, dieRow, dieCol],
    queryFn: () => inferenceService.getBlockAttribution(waferId, dieRow, dieCol),
    enabled,
  });
}
