import { request } from './api';
import { HealthResponse, StatusResponse } from '../types/system';

export const systemService = {
  getHealth: () => request<HealthResponse>('/api/v1/system/health'),
  getStatus: () => request<StatusResponse>('/api/v1/system/status'),
};
