# System Architecture

## Overview

Neurolink is an Adaptive Multimodal Communication Intelligence System that processes gesture, speech, facial expression, and text inputs through a unified AI pipeline. It enables real-time communication for individuals with speech and motor impairments by translating multimodal signals into synthesized speech, text, and sign language.

The system employs a microservices architecture with event-driven communication, supporting both cloud and edge deployments.

## Architecture Diagram

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                                Client Layer                                 │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ │
  │  │ Web App  │  │ Mobile   │  │ Edge     │  │ 3rd Party│  │ Accessibility│ │
  │  │ (Next.js)│  │ (PWA)    │  │ Device   │  │ API      │  │ Tools        │ │
  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘ │
  └───────┼──────────────┼────────────┼──────────────┼───────────────┼─────────┘
          │              │            │              │               │
  ┌───────┼──────────────┼────────────┼──────────────┼───────────────┼─────────┐
  │       │              │       ┌────┴────┐         │               │         │
  │       │              │       │  NGINX  │         │               │         │
  │       │              │       │ Gateway │         │               │         │
  │       │              │       └────┬────┘         │               │         │
  │       └──────────────┼────────────┼──────────────┼───────────────┼─────────┤
  │                      │      ┌─────┴──────┐       │               │         │
  │              ┌───────┴──────┴─┐           │       │               │         │
  │              │   FastAPI     │           │       │               │         │
  │              │   Backend     │   WebSocket│       │               │         │
  │              │   :8000       │   :8000/ws │       │               │         │
  │              └───────┬───────┬─┘           │       │               │         │
  │                      │       │             │       │               │         │
  │              ┌───────┴───────┴────────────┴───────┴───────────────┴─────────┤
  │              │                       AI Worker Layer                        │
  │              │  ┌──────────────────────────────────────────────────────┐    │
  │              │  │              AI Orchestrator                        │    │
  │              │  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌─────────────┐  │    │
  │              │  │  │Gesture │ │ Speech │ │Emotion │ │Multimodal   │  │    │
  │              │  │  │Engine  │ │ Engine │ │ Engine │ │Fusion Engine│  │    │
  │              │  │  └────────┘ └────────┘ └────────┘ └─────────────┘  │    │
  │              │  │  ┌────────┐ ┌────────┐ ┌──────────────────────┐   │    │
  │              │  │  │Intent  │ │Context │ │ Adaptation Learner   │   │    │
  │              │  │  │Engine  │ │Manager │ │ (RL-based)           │   │    │
  │              │  │  └────────┘ └────────┘ └──────────────────────┘   │    │
  │              │  └──────────────────────────────────────────────────────┘    │
  │              └───────────────────────────────────────────────────────────────┤
  │                                                                              │
  │  ┌───────────────────────────────────────────────────────────────────────────┤
  │  │                         Data Layer                                        │
  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐     │
  │  │  │PostgreSQL│  │  Redis   │  │ ChromaDB │  │    Vector Memory     │     │
  │  │  │(Primary) │  │ (Cache)  │  │(Vectors) │  │  (User Embeddings)   │     │
  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘     │
  │  └───────────────────────────────────────────────────────────────────────────┤
  │                                                                              │
  │  ┌───────────────────────────────────────────────────────────────────────────┤
  │  │                       Monitoring Layer                                    │
  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐     │
  │  │  │Prometheus│  │  Grafana │  │  Sentry  │  │   OpenTelemetry      │     │
  │  │  │(Metrics) │  │(Dashboards│ │(Errors)  │  │  (Distributed Tracing)│     │
  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────────┘     │
  │  └───────────────────────────────────────────────────────────────────────────┘
  └─────────────────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

### 1. Frontend (Next.js + React)

- **Web Application**: Server-rendered React application with Tailwind CSS
- **Real-time UI**: WebSocket-based live updates for gesture visualization, emotion display, and communication panel
- **Accessibility**: WCAG 2.1 AA compliant, screen reader optimized, high-contrast mode
- **State Management**: Zustand for client state, React Query for server state
- **WebSocket Client**: Socket.IO client for bidirectional real-time communication

### 2. API Gateway (NGINX)

- **Reverse Proxy**: Routes requests to appropriate services
- **SSL Termination**: HTTPS termination with automatic certificate renewal
- **Rate Limiting**: Per-IP and per-route rate limiting
- **Load Balancing**: Round-robin distribution across backend instances
- **WebSocket Proxy**: Upgrades and maintains WebSocket connections

### 3. Backend (FastAPI)

- **REST API**: Versioned (v1) REST endpoints for CRUD operations
- **WebSocket Server**: Real-time bidirectional communication for streaming data
- **Authentication**: JWT-based auth with refresh tokens and API key support
- **Rate Limiting**: Redis-backed distributed rate limiting
- **Metrics**: Prometheus metrics for request volume, latency, error rates
- **Health Checks**: Comprehensive health endpoints for all services
- **File Uploads**: Audio file upload for STT processing

### 4. AI Worker

The AI Worker is the computational core, composed of specialized engines:

#### Gesture Engine
- **Hand Detector**: MediaPipe-based hand landmark detection (21 landmarks per hand)
- **Hand Tracker**: Multi-hand tracking with temporal consistency
- **Gesture Classifier**: CNN + LSTM hybrid for gesture classification from landmark sequences
- **Sequence Model**: Transformer-based temporal sequence modeling for continuous gesture recognition

#### Speech Engine
- **Transcriber**: Whisper-based speech-to-text with multi-language support (15+ languages)
- **Synthesizer**: Neural TTS with emotion-aware prosody control
- **Vocal Emotion Analyzer**: Audio emotion classification (happy, sad, angry, neutral, etc.)

#### Emotion Engine
- **Facial Emotion Analyzer**: CNN-based facial expression recognition (7 basic emotions)
- **Vocal Emotion Analyzer**: Audio prosody analysis
- **Emotion Fusion**: Weighted fusion of facial and vocal emotion signals with context awareness

#### Multimodal Fusion
- **Cross-Modal Attention**: Transformer attention between gesture, speech, and emotion modalities
- **Embedding Fusion**: Multi-layer fusion of modality-specific embeddings
- **Inference Pipeline**: Coordinated inference across all modalities with temporal alignment

#### Intent Engine
- **Intent Classifier**: Multi-label intent classification with context history
- **Context Manager**: Short-term and long-term conversation context tracking
- **Phrase Predictor**: Next-phrase prediction using language model with personalization

#### Adaptation Engine
- **User Profiler**: Builds and maintains user communication profiles
- **RL Adaptation**: Reinforcement learning from user feedback for continuous improvement
- **Personalization Memory**: User-specific gesture patterns, vocabulary, and preferences

### 5. Data Layer

- **PostgreSQL**: Primary database for users, sessions, gesture history, translations, analytics
- **Redis**: Caching, rate limiting, WebSocket pub/sub, session store
- **ChromaDB**: Vector database for semantic memory and embedding storage
- **Vector Memory**: Sentence-transformer based user memory with semantic search

### 6. Monitoring Layer

- **Prometheus**: Time-series metrics collection and alerting
- **Grafana**: Dashboards for system health, ML performance, business metrics
- **Sentry**: Error tracking and performance monitoring
- **OpenTelemetry**: Distributed tracing across services

## Data Flow

### Real-Time Communication Flow

```
User Gesture/Audio → WebSocket → Backend → AI Worker → Orchestrator
  → Gesture Pipeline || Speech Pipeline || Emotion Pipeline
  → Multimodal Fusion → Intent Classification
  → Context Update → Response Generation
  → WebSocket → Client UI
```

### REST API Flow

```
Client Request → NGINX → FastAPI → Auth Middleware
  → Rate Limiter → Route Handler → DB/AI Operations
  → Response → Client
```

### Training Pipeline Flow

```
Data Collection → Validation → Preprocessing
  → Model Training → Evaluation → Comparison with Baseline
  → Model Registry → Deployment (if accuracy improves)
```

## Deployment Architecture

### Cloud Deployment (Production)

```
                    ┌──────────────┐
                    │  Route 53    │
                    │  / CloudFront│
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   NGINX      │
                    │  (ALB/NLB)   │
                    └──────┬───────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────┴──────┐ ┌──────┴──────┐ ┌──────┴──────┐
    │  Backend    │ │ AI Worker   │ │  Frontend   │
    │  (ECS/Farg) │ │ (GPU ECS)   │ │ (ECS/Farg)  │
    └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
           │               │               │
    ┌──────┴───────────────────────────────┴──────┐
    │           Services (RDS/ElastiCache)         │
    │  PostgreSQL  Redis  ChromaDB  S3 (Models)   │
    └─────────────────────────────────────────────┘
```

### Edge Deployment

```
┌─────────────────────────────────────────┐
│           Edge Device                    │
│  ┌─────────┐  ┌──────────┐              │
│  │ Camera  │  │ Microphone│             │
│  └────┬────┘  └────┬─────┘              │
│       │            │                    │
│  ┌────┴────────────┴─────┐              │
│  │    Edge Agent          │             │
│  │  (NVIDIA Jetson / RPi) │             │
│  │  ┌───────────────────┐ │             │
│  │  │  ONNX Runtime     │ │             │
│  │  │  TensorRT (Jetson)│ │             │
│  │  └───────────────────┘ │             │
│  └───────────┬────────────┘              │
│              │                          │
│       ┌──────┴──────┐                   │
│       │   Speaker   │                   │
│       └─────────────┘                   │
└─────────────────────────────────────────┘
```

## Scalability Design

- **Horizontal Scaling**: Backend and AI worker services scale horizontally behind load balancers
- **Connection Pooling**: Database connection pooling with configurable pool sizes
- **Caching Layer**: Redis for session cache, rate limit counters, and WebSocket pub/sub
- **Async Processing**: All I/O operations use asyncio for non-blocking execution
- **GPU Resource Management**: AI worker uses semaphore-based concurrent request limiting
- **Model Caching**: Models loaded on first use with lazy loading pattern
- **Database Indexing**: Strategic indexes on user_id, timestamp, and session_type columns

## Security Architecture

- **Authentication**: JWT access/refresh token pair with configurable expiration
- **API Security**: API key authentication for machine-to-machine communication
- **Password Security**: bcrypt hashing with configurable rounds
- **Rate Limiting**: Redis-backed distributed rate limiting per user/IP
- **WebSocket Auth**: JWT-based WebSocket authentication at connection time
- **CORS**: Configurable CORS policies for frontend origins
- **Input Validation**: Pydantic models with strict validation on all endpoints
- **SQL Injection Prevention**: SQLAlchemy ORM with parameterized queries
- **Audit Logging**: Structured logging of all authentication events
- **Secrets Management**: Environment-based configuration with .env support
