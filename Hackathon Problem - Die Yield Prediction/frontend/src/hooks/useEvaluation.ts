import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { evaluationService } from '../services/evaluation';

export function useLatestEvaluation() {
  return useQuery({
    queryKey: ['evaluation', 'latest'],
    queryFn: evaluationService.getLatestEvaluation,
  });
}

export function useRunEvaluation() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: evaluationService.runEvaluation,
    onSuccess: (data) => {
      queryClient.setQueryData(['evaluation', 'latest'], data);
    },
  });
}
