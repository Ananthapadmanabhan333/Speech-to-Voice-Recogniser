import { create } from 'zustand';
import type {
  AnalyticsMetrics,
  EmotionDistribution,
  CommunicationEfficiency,
  LearningCurve,
  AdaptationMetric,
  TimeSeriesPoint,
} from '@/types';
import api from '@/lib/api';

interface AnalyticsState {
  metrics: AnalyticsMetrics | null;
  emotionDistribution: EmotionDistribution[];
  communicationEfficiency: CommunicationEfficiency[];
  learningCurve: LearningCurve[];
  adaptationMetrics: AdaptationMetric[];
  dateRange: { start: string | null; end: string | null };
  isLoading: boolean;
  error: string | null;

  fetchAllMetrics: () => Promise<void>;
  fetchAnalyticsMetrics: () => Promise<void>;
  fetchEmotionDistribution: () => Promise<void>;
  fetchCommunicationEfficiency: () => Promise<void>;
  fetchLearningCurve: () => Promise<void>;
  fetchAdaptationMetrics: () => Promise<void>;
  setDateRange: (start: string | null, end: string | null) => void;
  exportData: (format: 'csv' | 'json') => Promise<Blob>;
  updateMetricsRealtime: (metrics: Partial<AnalyticsMetrics>) => void;
  clearError: () => void;
  reset: () => void;

  getGestureAccuracyPoints: () => TimeSeriesPoint[];
  getSpeechConfidencePoints: () => TimeSeriesPoint[];
  getEmotionDetectionPoints: () => TimeSeriesPoint[];
  getLatencyPoints: () => TimeSeriesPoint[];
  getOverallProgress: () => number;
}

const initialState = {
  metrics: null,
  emotionDistribution: [],
  communicationEfficiency: [],
  learningCurve: [],
  adaptationMetrics: [],
  dateRange: { start: null, end: null },
  isLoading: false,
  error: null,
};

export const useAnalyticsStore = create<AnalyticsState>()((set, get) => ({
  ...initialState,

  fetchAllMetrics: async () => {
    set({ isLoading: true, error: null });
    try {
      const { dateRange } = get();
      const [
        analyticsRes,
        emotionRes,
        efficiencyRes,
        learningRes,
        adaptationRes,
      ] = await Promise.all([
        api.getAnalyticsMetrics(dateRange.start ?? undefined, dateRange.end ?? undefined),
        api.getEmotionDistribution(dateRange.start ?? undefined, dateRange.end ?? undefined),
        api.getCommunicationEfficiency(dateRange.start ?? undefined, dateRange.end ?? undefined),
        api.getLearningCurve(dateRange.start ?? undefined, dateRange.end ?? undefined),
        api.getAdaptationMetrics(),
      ]);

      set({
        metrics: analyticsRes.success ? analyticsRes.data : null,
        emotionDistribution: emotionRes.success ? emotionRes.data : [],
        communicationEfficiency: efficiencyRes.success ? efficiencyRes.data : [],
        learningCurve: learningRes.success ? learningRes.data : [],
        adaptationMetrics: adaptationRes.success ? adaptationRes.data : [],
        isLoading: false,
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch analytics';
      set({ isLoading: false, error: message });
    }
  },

  fetchAnalyticsMetrics: async () => {
    try {
      const { dateRange } = get();
      const response = await api.getAnalyticsMetrics(
        dateRange.start ?? undefined,
        dateRange.end ?? undefined
      );
      if (response.success) {
        set({ metrics: response.data });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch metrics';
      set({ error: message });
    }
  },

  fetchEmotionDistribution: async () => {
    try {
      const { dateRange } = get();
      const response = await api.getEmotionDistribution(
        dateRange.start ?? undefined,
        dateRange.end ?? undefined
      );
      if (response.success) {
        set({ emotionDistribution: response.data });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch emotion data';
      set({ error: message });
    }
  },

  fetchCommunicationEfficiency: async () => {
    try {
      const { dateRange } = get();
      const response = await api.getCommunicationEfficiency(
        dateRange.start ?? undefined,
        dateRange.end ?? undefined
      );
      if (response.success) {
        set({ communicationEfficiency: response.data });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch efficiency data';
      set({ error: message });
    }
  },

  fetchLearningCurve: async () => {
    try {
      const { dateRange } = get();
      const response = await api.getLearningCurve(
        dateRange.start ?? undefined,
        dateRange.end ?? undefined
      );
      if (response.success) {
        set({ learningCurve: response.data });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch learning curve';
      set({ error: message });
    }
  },

  fetchAdaptationMetrics: async () => {
    try {
      const response = await api.getAdaptationMetrics();
      if (response.success) {
        set({ adaptationMetrics: response.data });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Failed to fetch adaptation metrics';
      set({ error: message });
    }
  },

  setDateRange: (start, end) => {
    set({ dateRange: { start, end } });
    get().fetchAllMetrics();
  },

  exportData: async (format) => {
    const { dateRange } = get();
    return api.exportAnalytics(
      format,
      dateRange.start ?? undefined,
      dateRange.end ?? undefined
    );
  },

  updateMetricsRealtime: (partial) => {
    set((state) => ({
      metrics: state.metrics ? { ...state.metrics, ...partial } : null,
    }));
  },

  clearError: () => set({ error: null }),

  reset: () => set(initialState),

  getGestureAccuracyPoints: () => {
    const { learningCurve } = get();
    return learningCurve.map((point) => ({
      timestamp: new Date(point.date).getTime(),
      value: point.gestureAccuracy,
      label: point.date,
    }));
  },

  getSpeechConfidencePoints: () => {
    const { learningCurve } = get();
    return learningCurve.map((point) => ({
      timestamp: new Date(point.date).getTime(),
      value: point.speechAccuracy,
      label: point.date,
    }));
  },

  getEmotionDetectionPoints: () => {
    const { learningCurve } = get();
    return learningCurve.map((point) => ({
      timestamp: new Date(point.date).getTime(),
      value: point.emotionAccuracy,
      label: point.date,
    }));
  },

  getLatencyPoints: () => {
    const { metrics } = get();
    return metrics?.latency ?? [];
  },

  getOverallProgress: () => {
    const { adaptationMetrics } = get();
    if (adaptationMetrics.length === 0) return 0;
    const total = adaptationMetrics.reduce((sum, m) => sum + m.current, 0);
    const maxTotal = adaptationMetrics.reduce((sum, m) => sum + m.target, 0);
    return maxTotal > 0 ? (total / maxTotal) * 100 : 0;
  },
}));
