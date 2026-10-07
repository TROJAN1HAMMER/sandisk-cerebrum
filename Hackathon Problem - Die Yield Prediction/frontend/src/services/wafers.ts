import { request } from './api';
import { WaferListResponse, WaferSummaryItem, WaferDetailResponse, DiePoint } from '../types/wafers';

export const waferService = {
  getWafers: () => request<WaferListResponse>('/api/v1/wafers'),

  getWaferSummary: (waferId: string) =>
    request<WaferSummaryItem>(`/api/v1/wafers/${waferId}`),

  getWaferDies: (waferId: string, filterMode: string = 'all', limit: number = 3000) => {
    const params = new URLSearchParams({
      filter_mode: filterMode,
      limit: limit.toString(),
    });
    return request<WaferDetailResponse>(`/api/v1/wafers/${waferId}/dies?${params.toString()}`);
  },

  getSingleDie: (waferId: string, dieRow: number, dieCol: number) =>
    request<DiePoint>(`/api/v1/wafers/${waferId}/dies/${dieRow}/${dieCol}`),

  getWaferHotspots: (waferId: string) =>
    request<import('../types/wafers').WaferHotspotsResponse>(`/api/v1/wafers/${waferId}/hotspots`),
};
