import { request } from './api';
import { EvaluationRunResponse } from '../types/models';

export const evaluationService = {
  runEvaluation: () =>
    request<EvaluationRunResponse>('/api/v1/evaluation/run', {
      method: 'POST',
    }),

  getLatestEvaluation: () =>
    request<EvaluationRunResponse>('/api/v1/evaluation/latest'),
};
