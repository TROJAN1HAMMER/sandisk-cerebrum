export interface ApiError {
  code: string;
  message: string;
  details?: unknown;
}

export interface ApiResponseError {
  error: ApiError;
}
