import { request } from './api';
import { EvaluationRunResponse } from '../types/models';

export const evaluationService = {
  runEvaluation: () =>
    request<EvaluationRunResponse>('/api/v1/evaluation/run', {
      method: 'POST',
    }),

  getLatestEvaluation: () =>
    request<EvaluationRunResponse>('/api/v1/evaluation/latest'),

  simulatePolicy: (params: import('../types/models').PolicySimulationRequest) =>
    request<import('../types/models').PolicySimulationResponse>('/api/v1/evaluation/simulate-policy', {
      method: 'POST',
      body: JSON.stringify(params),
    }),
};
