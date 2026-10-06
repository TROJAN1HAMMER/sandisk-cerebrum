import { request } from './api';
import {
  ModelMetricsResponse,
  ModelReleaseResponse,
  BenchmarkTableResponse,
  ErrorDrilldownResponse,
  ProbabilityDistributionResponse,
  RocPrCurvesResponse,
  HighRiskDiesResponse,
} from '../types/models';

export const modelService = {
  getMetrics: () => request<ModelMetricsResponse>('/api/v1/models/metrics'),

  getRelease: () => request<ModelReleaseResponse>('/api/v1/models/release'),

  getBenchmark: () => request<BenchmarkTableResponse>('/api/v1/models/benchmark'),

  getErrorDrilldown: (group: string = 'recovered_fns', limit: number = 50) => {
    const params = new URLSearchParams({
      group,
      limit: limit.toString(),
    });
    return request<ErrorDrilldownResponse>(`/api/v1/models/error-drilldown?${params.toString()}`);
  },

  getProbabilityDistribution: () =>
    request<ProbabilityDistributionResponse>('/api/v1/models/probability-distribution'),

  getRocPrCurves: () =>
    request<RocPrCurvesResponse>('/api/v1/models/roc-pr-curves'),

  getHighRiskDies: (limit: number = 50, waferId?: string) => {
    const params = new URLSearchParams({ limit: limit.toString() });
    if (waferId) {
      params.append('wafer_id', waferId);
    }
    return request<HighRiskDiesResponse>(`/api/v1/models/high-risk-dies?${params.toString()}`);
  },
};

