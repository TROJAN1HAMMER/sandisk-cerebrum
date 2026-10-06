import { ApiError } from '../types/api';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || '';

export async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${BASE_URL}${endpoint}`;
  
  const headers = new Headers(options.headers || {});
  if (!headers.has('Content-Type') && !(options.body instanceof FormData)) {
    headers.set('Content-Type', 'application/json');
  }

  const response = await fetch(url, {
    ...options,
    headers,
  });

  if (!response.ok) {
    let errorData: ApiError;
    try {
      const data = await response.json();
      errorData = data.error || data.detail || {
        code: `HTTP_${response.status}`,
        message: response.statusText,
      };
    } catch {
      errorData = {
        code: `HTTP_${response.status}`,
        message: response.statusText || 'Unknown API Error',
      };
    }
    throw errorData;
  }

  return response.json() as Promise<T>;
}
