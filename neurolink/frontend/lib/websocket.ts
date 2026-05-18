import { io, Socket } from 'socket.io-client';
import type { WebSocketEventMap, WebSocketEventCallback } from '@/types';

const WS_BASE_URL = process.env.NEXT_PUBLIC_WS_URL || 'http://localhost:8000';
const RECONNECTION_ATTEMPTS = 10;
const RECONNECTION_DELAY = 2000;
const HEARTBEAT_INTERVAL = 30000;

type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

class WebSocketClient {
  private socket: Socket | null = null;
  private listeners: Map<string, Set<(...args: unknown[]) => void>> = new Map();
  private connectionState: ConnectionState = 'disconnected';
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private connectionListeners: Set<(state: ConnectionState) => void> = new Set();
  private errorListeners: Set<(error: Error) => void> = new Set();

  connect(token?: string): void {
    if (this.socket?.connected) return;

    this.connectionState = 'connecting';
    this.notifyConnectionListeners();

    this.socket = io(WS_BASE_URL, {
      auth: token ? { token } : undefined,
      transports: ['websocket', 'polling'],
      reconnection: true,
      reconnectionAttempts: RECONNECTION_ATTEMPTS,
      reconnectionDelay: RECONNECTION_DELAY,
      reconnectionDelayMax: 30000,
      randomizationFactor: 0.5,
      timeout: 20000,
      forceNew: true,
    });

    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      this.connectionState = 'connected';
      this.notifyConnectionListeners();
      this.startHeartbeat();
    });

    this.socket.on('disconnect', (reason: string) => {
      this.connectionState = 'disconnected';
      this.notifyConnectionListeners();
      this.stopHeartbeat();

      if (reason === 'io server disconnect' || reason === 'transport close') {
        this.socket?.connect();
      }
    });

    this.socket.on('connect_error', (error: Error) => {
      this.connectionState = 'reconnecting';
      this.notifyConnectionListeners();
      this.errorListeners.forEach((listener) => listener(error));
    });

    this.socket.on('reconnect_attempt', (attempt: number) => {
      this.connectionState = 'reconnecting';
      this.notifyConnectionListeners();
    });

    this.socket.on('reconnect', () => {
      this.connectionState = 'connected';
      this.notifyConnectionListeners();
      this.startHeartbeat();
    });

    this.socket.on('reconnect_failed', () => {
      this.connectionState = 'disconnected';
      this.notifyConnectionListeners();
      this.stopHeartbeat();
    });

    this.socket.on('error', (error: Error) => {
      this.errorListeners.forEach((listener) => listener(error));
    });

    this.socket.io.on('reconnect', () => {
      this.resubscribeAll();
    });
  }

  private startHeartbeat(): void {
    this.stopHeartbeat();
    this.heartbeatTimer = setInterval(() => {
      if (this.socket?.connected) {
        this.socket.emit('heartbeat', { timestamp: Date.now() });
      }
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private resubscribeAll(): void {
    this.listeners.forEach((callbacks, event) => {
      callbacks.forEach((callback) => {
        this.socket?.on(event, callback);
      });
    });
  }

  on<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<K>
  ): () => void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback as (...args: unknown[]) => void);
    this.socket?.on(event, callback as (...args: unknown[]) => void);

    return () => {
      this.listeners.get(event)?.delete(callback as (...args: unknown[]) => void);
      this.socket?.off(event, callback as (...args: unknown[]) => void);
      if (this.listeners.get(event)?.size === 0) {
        this.listeners.delete(event);
      }
    };
  }

  once<K extends keyof WebSocketEventMap>(
    event: K,
    callback: WebSocketEventCallback<K>
  ): void {
    this.socket?.once(event, callback as (...args: unknown[]) => void);
  }

  emit<K extends keyof WebSocketEventMap>(
    event: K,
    data: WebSocketEventMap[K]
  ): void {
    if (this.socket?.connected) {
      this.socket.emit(event, data);
    } else {
      console.warn(`Cannot emit ${event}: socket not connected`);
    }
  }

  off<K extends keyof WebSocketEventMap>(
    event: K,
    callback?: WebSocketEventCallback<K>
  ): void {
    if (callback) {
      this.listeners.get(event)?.delete(callback as (...args: unknown[]) => void);
      this.socket?.off(event, callback as (...args: unknown[]) => void);
    } else {
      this.listeners.delete(event);
      this.socket?.off(event);
    }
  }

  onConnectionChange(listener: (state: ConnectionState) => void): () => void {
    this.connectionListeners.add(listener);
    listener(this.connectionState);
    return () => {
      this.connectionListeners.delete(listener);
    };
  }

  onError(listener: (error: Error) => void): () => void {
    this.errorListeners.add(listener);
    return () => {
      this.errorListeners.delete(listener);
    };
  }

  private notifyConnectionListeners(): void {
    this.connectionListeners.forEach((listener) => {
      listener(this.connectionState);
    });
  }

  get isConnected(): boolean {
    return this.socket?.connected ?? false;
  }

  get state(): ConnectionState {
    return this.connectionState;
  }

  disconnect(): void {
    this.stopHeartbeat();
    this.listeners.clear();
    if (this.socket) {
      this.socket.removeAllListeners();
      this.socket.disconnect();
      this.socket = null;
    }
    this.connectionState = 'disconnected';
    this.notifyConnectionListeners();
  }
}

export const wsClient = new WebSocketClient();
export default wsClient;
