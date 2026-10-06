import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 2, // 2 minutes stale time for static holdout data
      gcTime: 1000 * 60 * 10,   // 10 minutes cache garbage collection
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});
