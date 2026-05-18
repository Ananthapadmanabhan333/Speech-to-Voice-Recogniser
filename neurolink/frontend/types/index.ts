export type GestureType =
  | 'thumbs_up'
  | 'thumbs_down'
  | 'pointing'
  | 'wave'
  | 'ok'
  | 'peace'
  | 'fist'
  | 'open_palm'
  | 'pinch'
  | 'custom';

export type EmotionType =
  | 'happy'
  | 'sad'
  | 'angry'
  | 'surprise'
  | 'neutral'
  | 'fear'
  | 'disgust'
  | 'contempt';

export type MessageRole = 'user' | 'assistant' | 'system';

export type SessionStatus = 'active' | 'paused' | 'completed' | 'error';

export type ModalType = 'gesture' | 'speech' | 'text' | 'emotion' | 'multimodal';

export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export interface Gesture {
  id: string;
  type: GestureType;
  landmarks: Point3D[];
  confidence: number;
  timestamp: number;
  handedness: 'left' | 'right' | 'both';
  boundingBox?: { x: number; y: number; width: number; height: number };
}

export interface GestureSegment {
  start: number;
  end: number;
  gesture: Gesture;
  duration: number;
}

export interface SpeechResult {
  id: string;
  text: string;
  language: string;
  confidence: number;
  segments: SpeechSegment[];
  isFinal: boolean;
  timestamp: number;
  duration: number;
}

export interface SpeechSegment {
  text: string;
  start: number;
  end: number;
  confidence: number;
  speaker?: string;
}

export interface EmotionResult {
  id: string;
  emotion: EmotionType;
  confidence: number;
  arousal: number;
  valence: number;
  facial: EmotionComponent;
  vocal: EmotionComponent;
  timestamp: number;
}

export interface EmotionComponent {
  emotion: EmotionType;
  confidence: number;
  features: Record<string, number>;
}

export interface MultimodalResult {
  id: string;
  intent: string;
  emotion: EmotionResult;
  urgency: number;
  confidence: number;
  suggestions: string[];
  fusedAt: number;
  modalities: ModalType[];
  interpretation: string;
  alternativeInterpretations?: string[];
}

export interface UserProfile {
  id: string;
  name: string;
  email: string;
  avatar?: string;
  preferences: UserPreferences;
  accessibility: AccessibilitySettings;
  adaptationLevel: number;
  communicationHistory: number;
  joinedAt: number;
  lastActive: number;
}

export interface UserPreferences {
  theme: 'dark' | 'light' | 'system';
  language: string;
  speechSpeed: number;
  gestureSensitivity: number;
  preferredModality: ModalType;
  reducedMotion: boolean;
  highContrast: boolean;
}

export interface AccessibilitySettings {
  captionEnabled: boolean;
  signLanguageEnabled: boolean;
  hapticFeedback: boolean;
  visualCues: boolean;
  fontScale: number;
  colorBlindMode: boolean;
}

export interface CommunicationSession {
  id: string;
  type: ModalType;
  messages: Message[];
  metrics: SessionMetrics;
  status: SessionStatus;
  startedAt: number;
  endedAt?: number;
  participants: string[];
}

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  translation?: string;
  timestamp: number;
  confidence: number;
  modality: ModalType;
  gesture?: Gesture;
  emotion?: EmotionResult;
  metadata?: Record<string, unknown>;
}

export interface SessionMetrics {
  totalMessages: number;
  averageConfidence: number;
  duration: number;
  modalityDistribution: Record<ModalType, number>;
  emotionProgression: EmotionResult[];
  gestureAccuracy: number;
  speechAccuracy: number;
}

export interface AnalyticsMetrics {
  accuracy: TimeSeriesPoint[];
  latency: TimeSeriesPoint[];
  userSatisfaction: number;
  adaptationProgress: number;
  sessionsCompleted: number;
  averageSessionDuration: number;
  gestureRecognitionRate: number;
  speechRecognitionRate: number;
  emotionDetectionRate: number;
}

export interface TimeSeriesPoint {
  timestamp: number;
  value: number;
  label?: string;
}

export interface EmotionDistribution {
  emotion: EmotionType;
  count: number;
  percentage: number;
}

export interface CommunicationEfficiency {
  date: string;
  accuracy: number;
  speed: number;
  understanding: number;
}

export interface LearningCurve {
  date: string;
  gestureAccuracy: number;
  speechAccuracy: number;
  emotionAccuracy: number;
  multimodalAccuracy: number;
}

export interface AdaptationMetric {
  category: string;
  current: number;
  previous: number;
  target: number;
  unit: string;
  trend: 'up' | 'down' | 'stable';
}

export interface WebSocketEventMap {
  'gesture:recognized': Gesture;
  'gesture:tracking': Gesture;
  'speech:interim': SpeechResult;
  'speech:final': SpeechResult;
  'emotion:detected': EmotionResult;
  'multimodal:fused': MultimodalResult;
  'session:created': CommunicationSession;
  'session:updated': Partial<CommunicationSession>;
  'session:ended': CommunicationSession;
  'message:new': Message;
  'message:stream': { content: string; sessionId: string };
  'analytics:metrics': AnalyticsMetrics;
  'system:status': { status: string; message: string };
  'system:error': { code: string; message: string };
  'system:heartbeat': { timestamp: number };
  'user:presence': { userId: string; status: 'online' | 'offline' | 'away' };
  'adaptation:updated': { metric: AdaptationMetric };
  'suggestion:phrases': string[];
}

export type WebSocketEventCallback<T extends keyof WebSocketEventMap> = (
  data: WebSocketEventMap[T]
) => void;

export interface ApiResponse<T> {
  data: T;
  success: boolean;
  message?: string;
  errors?: string[];
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  total: number;
  page: number;
  limit: number;
  hasMore: boolean;
}

export interface DashboardMetrics {
  gestureAccuracy: number;
  speechConfidence: number;
  emotionDetectionRate: number;
  sessionsToday: number;
  activeUsers: number;
  systemUptime: number;
  adaptationScore: number;
}
