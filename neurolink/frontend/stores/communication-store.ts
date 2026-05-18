import { create } from 'zustand';
import type {
  Gesture,
  SpeechResult,
  EmotionResult,
  MultimodalResult,
  Message,
  CommunicationSession,
  ModalType,
} from '@/types';
import { wsClient } from '@/lib/websocket';

interface CommunicationState {
  messages: Message[];
  currentInput: string;
  session: CommunicationSession | null;
  isRecording: boolean;
  isProcessing: boolean;
  gesture: {
    current: Gesture | null;
    history: Gesture[];
    isTracking: boolean;
  };
  speech: {
    transcript: string;
    isListening: boolean;
    audioLevel: number;
    interimResults: SpeechResult[];
  };
  emotion: {
    current: EmotionResult | null;
    history: EmotionResult[];
  };
  multimodal: {
    current: MultimodalResult | null;
    history: MultimodalResult[];
  };
  suggestedPhrases: string[];
  connectionStatus: string;
  error: string | null;

  // Gesture actions
  setCurrentGesture: (gesture: Gesture | null) => void;
  addGestureToHistory: (gesture: Gesture) => void;
  setGestureTracking: (tracking: boolean) => void;

  // Speech actions
  setTranscript: (transcript: string) => void;
  setListening: (listening: boolean) => void;
  setAudioLevel: (level: number) => void;
  addInterimResult: (result: SpeechResult) => void;

  // Emotion actions
  setCurrentEmotion: (emotion: EmotionResult | null) => void;
  addEmotionToHistory: (emotion: EmotionResult) => void;

  // Multimodal actions
  setMultimodalResult: (result: MultimodalResult | null) => void;
  addMultimodalToHistory: (result: MultimodalResult) => void;

  // Message actions
  addMessage: (message: Message) => void;
  updateLastMessage: (content: string) => void;
  setMessages: (messages: Message[]) => void;

  // Session actions
  setSession: (session: CommunicationSession | null) => void;
  updateSession: (partial: Partial<CommunicationSession>) => void;

  // Recording actions
  startRecording: () => void;
  stopRecording: () => void;

  // Input actions
  setCurrentInput: (input: string) => void;
  sendMessage: (content: string, modality?: ModalType) => void;

  // Suggestions
  setSuggestedPhrases: (phrases: string[]) => void;

  // Connection
  setConnectionStatus: (status: string) => void;

  // Error
  setError: (error: string | null) => void;
  clearError: () => void;

  // Reset
  reset: () => void;

  // Computed
  getLastGestures: (count: number) => Gesture[];
  getRecentEmotions: (count: number) => EmotionResult[];
  getMessagesByModality: (modality: ModalType) => Message[];
}

const initialState = {
  messages: [],
  currentInput: '',
  session: null,
  isRecording: false,
  isProcessing: false,
  gesture: {
    current: null,
    history: [],
    isTracking: false,
  },
  speech: {
    transcript: '',
    isListening: false,
    audioLevel: 0,
    interimResults: [],
  },
  emotion: {
    current: null,
    history: [],
  },
  multimodal: {
    current: null,
    history: [],
  },
  suggestedPhrases: [],
  connectionStatus: 'disconnected',
  error: null,
};

export const useCommunicationStore = create<CommunicationState>()((set, get) => ({
  ...initialState,

  setCurrentGesture: (gesture) =>
    set((state) => ({
      gesture: { ...state.gesture, current: gesture },
    })),

  addGestureToHistory: (gesture) =>
    set((state) => ({
      gesture: {
        ...state.gesture,
        history: [...state.gesture.history.slice(-50), gesture],
      },
    })),

  setGestureTracking: (tracking) =>
    set((state) => ({
      gesture: { ...state.gesture, isTracking: tracking },
    })),

  setTranscript: (transcript) =>
    set((state) => ({
      speech: { ...state.speech, transcript },
    })),

  setListening: (listening) =>
    set((state) => ({
      speech: { ...state.speech, isListening: listening },
    })),

  setAudioLevel: (level) =>
    set((state) => ({
      speech: { ...state.speech, audioLevel: level },
    })),

  addInterimResult: (result) =>
    set((state) => ({
      speech: {
        ...state.speech,
        interimResults: [...state.speech.interimResults.slice(-10), result],
        transcript: result.text,
      },
    })),

  setCurrentEmotion: (emotion) =>
    set((state) => ({
      emotion: { ...state.emotion, current: emotion },
    })),

  addEmotionToHistory: (emotion) =>
    set((state) => ({
      emotion: {
        ...state.emotion,
        history: [...state.emotion.history.slice(-100), emotion],
      },
    })),

  setMultimodalResult: (result) =>
    set((state) => ({
      multimodal: { ...state.multimodal, current: result },
    })),

  addMultimodalToHistory: (result) =>
    set((state) => ({
      multimodal: {
        ...state.multimodal,
        history: [...state.multimodal.history.slice(-50), result],
      },
    })),

  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),

  updateLastMessage: (content) =>
    set((state) => {
      const messages = [...state.messages];
      if (messages.length > 0) {
        messages[messages.length - 1] = {
          ...messages[messages.length - 1],
          content,
        };
      }
      return { messages };
    }),

  setMessages: (messages) => set({ messages }),

  setSession: (session) => set({ session }),

  updateSession: (partial) =>
    set((state) => ({
      session: state.session ? { ...state.session, ...partial } : null,
    })),

  startRecording: () => {
    set({ isRecording: true, isProcessing: false });
    wsClient.emit('session:start', { type: 'multimodal' });
  },

  stopRecording: () => {
    set({ isRecording: false, isProcessing: false });
    wsClient.emit('session:stop', {});
  },

  setCurrentInput: (input) => set({ currentInput: input }),

  sendMessage: (content, modality = 'text') => {
    if (!content.trim()) return;

    const message: Message = {
      id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      role: 'user',
      content,
      timestamp: Date.now(),
      confidence: 1,
      modality,
      emotion: get().emotion.current ?? undefined,
      gesture: get().gesture.current ?? undefined,
    };

    set((state) => ({
      messages: [...state.messages, message],
      currentInput: '',
      isProcessing: true,
    }));

    wsClient.emit('message:new', message);
  },

  setSuggestedPhrases: (phrases) => set({ suggestedPhrases: phrases }),

  setConnectionStatus: (status) => set({ connectionStatus: status }),

  setError: (error) => set({ error }),

  clearError: () => set({ error: null }),

  reset: () => set(initialState),

  getLastGestures: (count) => {
    const { gesture } = get();
    return gesture.history.slice(-count);
  },

  getRecentEmotions: (count) => {
    const { emotion } = get();
    return emotion.history.slice(-count);
  },

  getMessagesByModality: (modality) => {
    const { messages } = get();
    return messages.filter((m) => m.modality === modality);
  },
}));
