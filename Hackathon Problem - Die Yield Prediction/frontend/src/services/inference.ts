import { request } from './api';
import {
  SingleDieRequest,
  DieInferenceResponse,
  BatchInferenceResponse,
  DemoSampleItem,
  BlockAttributionResponse,
} from '../types/inference';

export const inferenceService = {
  predictSingleDie: (req: SingleDieRequest) =>
    request<DieInferenceResponse>('/api/v1/inference/die', {
      method: 'POST',
      body: JSON.stringify(req),
    }),

  predictBatch: (file: File, gateLower?: number, gateUpper?: number) => {
    const formData = new FormData();
    formData.append('file', file);
    let query = '';
    const params = new URLSearchParams();
    if (gateLower !== undefined) params.append('gate_lower', gateLower.toString());
    if (gateUpper !== undefined) params.append('gate_upper', gateUpper.toString());
    if (params.toString()) query = `?${params.toString()}`;

    return request<BatchInferenceResponse>(`/api/v1/inference/batch${query}`, {
      method: 'POST',
      body: formData,
    });
  },

  getDemoSamples: () =>
    request<DemoSampleItem[]>('/api/v1/inference/demo-samples'),

  getBlockAttribution: (waferId: string, dieRow: number, dieCol: number) => {
    const params = new URLSearchParams({
      wafer_id: waferId,
      die_row: dieRow.toString(),
      die_col: dieCol.toString(),
    });
    return request<BlockAttributionResponse>(`/api/v1/inference/block-attribution?${params.toString()}`);
  },
};
