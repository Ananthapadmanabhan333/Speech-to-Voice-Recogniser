import axios, {
  AxiosError,
  AxiosInstance,
  InternalAxiosRequestConfig,
  AxiosResponse,
} from 'axios';
import type {
  ApiResponse,
  PaginatedResponse,
  DashboardMetrics,
  AnalyticsMetrics,
  CommunicationSession,
  UserProfile,
  Message,
  EmotionDistribution,
  CommunicationEfficiency,
  LearningCurve,
  AdaptationMetric,
} from '@/types';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';
const MAX_RETRIES = 3;
const RETRY_DELAY = 1000;

class ApiClient {
  private client: AxiosInstance;
  private refreshPromise: Promise<string | null> | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        const token = this.getAccessToken();
        if (token && config.headers) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error: AxiosError) => Promise.reject(error)
    );

    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & {
          _retry?: boolean;
          _retryCount?: number;
        };

        if (!originalRequest) {
          return Promise.reject(error);
        }

        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;
          try {
            const newToken = await this.refreshToken();
            if (newToken && originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${newToken}`;
              return this.client(originalRequest);
            }
          } catch {
            this.clearTokens();
            if (typeof window !== 'undefined') {
              window.location.href = '/login';
            }
          }
        }

        if (
          error.response?.status !== 401 &&
          (originalRequest._retryCount ?? 0) < MAX_RETRIES
        ) {
          originalRequest._retryCount = (originalRequest._retryCount ?? 0) + 1;
          await this.delay(RETRY_DELAY * (originalRequest._retryCount ?? 1));
          return this.client(originalRequest);
        }

        return Promise.reject(this.normalizeError(error));
      }
    );
  }

  private getAccessToken(): string | null {
    if (typeof window === 'undefined') return null;
    try {
      const stored = localStorage.getItem('neurolink_auth');
      if (stored) {
        const parsed = JSON.parse(stored);
        return parsed.accessToken || null;
      }
    } catch {
      return null;
    }
    return null;
  }

  private getRefreshToken(): string | null {
    if (typeof window === 'undefined') return null;
    try {
      const stored = localStorage.getItem('neurolink_auth');
      if (stored) {
        const parsed = JSON.parse(stored);
        return parsed.refreshToken || null;
      }
    } catch {
      return null;
    }
    return null;
  }

  private clearTokens(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('neurolink_auth');
    }
  }

  private async refreshToken(): Promise<string | null> {
    if (this.refreshPromise) {
      return this.refreshPromise;
    }

    this.refreshPromise = (async () => {
      const refreshToken = this.getRefreshToken();
      if (!refreshToken) return null;

      try {
        const response = await axios.post(`${API_BASE_URL}/auth/refresh`, {
          refreshToken,
        });
        const { accessToken, refreshToken: newRefreshToken } = response.data.data;
        this.storeTokens(accessToken, newRefreshToken);
        return accessToken;
      } catch {
        this.clearTokens();
        return null;
      } finally {
        this.refreshPromise = null;
      }
    })();

    return this.refreshPromise;
  }

  private storeTokens(accessToken: string, refreshToken: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem(
        'neurolink_auth',
        JSON.stringify({ accessToken, refreshToken })
      );
    }
  }

  private normalizeError(error: AxiosError): Error {
    if (error.response) {
      const data = error.response.data as { message?: string };
      return new Error(data?.message || `Request failed with status ${error.response.status}`);
    }
    if (error.request) {
      return new Error('Network error. Please check your connection.');
    }
    return new Error(error.message || 'An unexpected error occurred');
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  async getDashboardMetrics(): Promise<ApiResponse<DashboardMetrics>> {
    const { data } = await this.client.get<ApiResponse<DashboardMetrics>>('/dashboard/metrics');
    return data;
  }

  async getAnalyticsMetrics(
    startDate?: string,
    endDate?: string
  ): Promise<ApiResponse<AnalyticsMetrics>> {
    const params = new URLSearchParams();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const query = params.toString();
    const { data } = await this.client.get<ApiResponse<AnalyticsMetrics>>(
      `/analytics/metrics${query ? `?${query}` : ''}`
    );
    return data;
  }

  async getEmotionDistribution(
    startDate?: string,
    endDate?: string
  ): Promise<ApiResponse<EmotionDistribution[]>> {
    const params = new URLSearchParams();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const query = params.toString();
    const { data } = await this.client.get<ApiResponse<EmotionDistribution[]>>(
      `/analytics/emotions${query ? `?${query}` : ''}`
    );
    return data;
  }

  async getCommunicationEfficiency(
    startDate?: string,
    endDate?: string
  ): Promise<ApiResponse<CommunicationEfficiency[]>> {
    const params = new URLSearchParams();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const query = params.toString();
    const { data } = await this.client.get<ApiResponse<CommunicationEfficiency[]>>(
      `/analytics/efficiency${query ? `?${query}` : ''}`
    );
    return data;
  }

  async getLearningCurve(
    startDate?: string,
    endDate?: string
  ): Promise<ApiResponse<LearningCurve[]>> {
    const params = new URLSearchParams();
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const query = params.toString();
    const { data } = await this.client.get<ApiResponse<LearningCurve[]>>(
      `/analytics/learning-curve${query ? `?${query}` : ''}`
    );
    return data;
  }

  async getAdaptationMetrics(): Promise<ApiResponse<AdaptationMetric[]>> {
    const { data } = await this.client.get<ApiResponse<AdaptationMetric[]>>(
      '/analytics/adaptation'
    );
    return data;
  }

  async getSessions(
    page = 1,
    limit = 20
  ): Promise<PaginatedResponse<CommunicationSession>> {
    const { data } = await this.client.get<PaginatedResponse<CommunicationSession>>(
      `/sessions?page=${page}&limit=${limit}`
    );
    return data;
  }

  async getSession(id: string): Promise<ApiResponse<CommunicationSession>> {
    const { data } = await this.client.get<ApiResponse<CommunicationSession>>(
      `/sessions/${id}`
    );
    return data;
  }

  async getSessionMessages(
    sessionId: string,
    page = 1,
    limit = 50
  ): Promise<PaginatedResponse<Message>> {
    const { data } = await this.client.get<PaginatedResponse<Message>>(
      `/sessions/${sessionId}/messages?page=${page}&limit=${limit}`
    );
    return data;
  }

  async getUserProfile(): Promise<ApiResponse<UserProfile>> {
    const { data } = await this.client.get<ApiResponse<UserProfile>>('/user/profile');
    return data;
  }

  async updateUserProfile(
    profile: Partial<UserProfile>
  ): Promise<ApiResponse<UserProfile>> {
    const { data } = await this.client.put<ApiResponse<UserProfile>>(
      '/user/profile',
      profile
    );
    return data;
  }

  async login(
    email: string,
    password: string
  ): Promise<ApiResponse<{ accessToken: string; refreshToken: string; user: UserProfile }>> {
    const { data } = await this.client.post<
      ApiResponse<{ accessToken: string; refreshToken: string; user: UserProfile }>
    >('/auth/login', { email, password });
    if (data.success && data.data) {
      this.storeTokens(data.data.accessToken, data.data.refreshToken);
    }
    return data;
  }

  async register(
    name: string,
    email: string,
    password: string
  ): Promise<ApiResponse<{ accessToken: string; refreshToken: string; user: UserProfile }>> {
    const { data } = await this.client.post<
      ApiResponse<{ accessToken: string; refreshToken: string; user: UserProfile }>
    >('/auth/register', { name, email, password });
    if (data.success && data.data) {
      this.storeTokens(data.data.accessToken, data.data.refreshToken);
    }
    return data;
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
    } finally {
      this.clearTokens();
    }
  }

  async exportAnalytics(
    format: 'csv' | 'json' = 'csv',
    startDate?: string,
    endDate?: string
  ): Promise<Blob> {
    const params = new URLSearchParams({ format });
    if (startDate) params.set('start_date', startDate);
    if (endDate) params.set('end_date', endDate);
    const { data } = await this.client.get<Blob>(
      `/analytics/export?${params.toString()}`,
      { responseType: 'blob' }
    );
    return data;
  }

  async healthCheck(): Promise<ApiResponse<{ status: string }>> {
    const { data } = await this.client.get<ApiResponse<{ status: string }>>('/health');
    return data;
  }
}

export const api = new ApiClient();
export default api;
