# API Documentation

## Overview

Neurolink exposes a REST API (v1) and WebSocket endpoint for real-time communication. All API responses follow a consistent JSON format. Authentication is via JWT tokens or API keys.

**Base URL**: `/api/v1`

**Content-Type**: `application/json`

## Authentication

### Endpoints

#### POST `/api/v1/auth/register`

Create a new user account.

```json
{
  "email": "user@example.com",
  "name": "Jane Doe",
  "password": "securePassword123"
}
```

**Response** `201 Created`:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### POST `/api/v1/auth/login`

Authenticate and receive tokens. Rate limited.

```json
{
  "email": "user@example.com",
  "password": "securePassword123"
}
```

**Response** `200 OK`:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### POST `/api/v1/auth/refresh`

Obtain a new access token using a refresh token.

```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIs..."
}
```

**Response** `200 OK`:
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
  "token_type": "bearer"
}
```

#### GET `/api/v1/auth/me`

Get current authenticated user profile.

**Headers**: `Authorization: Bearer <access_token>`

**Response** `200 OK`:
```json
{
  "id": "uuid",
  "email": "user@example.com",
  "name": "Jane Doe",
  "is_active": true,
  "is_verified": true,
  "preferences": {},
  "accessibility_settings": {},
  "created_at": "2025-01-01T00:00:00+00:00",
  "updated_at": "2025-01-01T00:00:00+00:00"
}
```

## Gesture API

### Endpoints

#### POST `/api/v1/gestures/recognize`

Recognize a gesture from a video frame.

```json
{
  "frame_data": "base64_encoded_image_data",
  "model_version": "v2.1"
}
```

**Response** `200 OK`:
```json
{
  "gesture_type": "thumbs_up",
  "confidence": 0.94,
  "landmarks": [{"x": 0.45, "y": 0.32, "z": -0.01}],
  "processing_time_ms": 45.2
}
```

#### POST `/api/v1/gestures/train`

Add a training sample for a custom gesture.

```json
{
  "gesture_type": "custom_wave",
  "landmarks": [{"x": 0.45, "y": 0.32, "z": -0.01}],
  "label": "hello_wave"
}
```

**Response** `200 OK`:
```json
{
  "gesture_type": "custom_wave",
  "samples_collected": 15,
  "status": "success"
}
```

#### GET `/api/v1/gestures/history`

Get gesture recognition history (paginated).

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |
| `gesture_type` | string | null | Filter by gesture type |

**Response** `200 OK`:
```json
{
  "items": [
    {
      "id": "uuid",
      "gesture_type": "thumbs_up",
      "confidence": 0.94,
      "timestamp": "2025-01-01T00:00:00+00:00"
    }
  ],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```

#### DELETE `/api/v1/gestures/gesture/{gesture_id}`

Delete a specific gesture history entry.

**Response** `204 No Content`

## Speech API

### Endpoints

#### POST `/api/v1/speech/stt`

Speech-to-text from base64-encoded audio.

```json
{
  "audio_data": "base64_encoded_audio",
  "language": "en",
  "sample_rate": 16000
}
```

**Response** `200 OK`:
```json
{
  "text": "Hello, how are you?",
  "confidence": 0.92,
  "language": "en",
  "duration_seconds": 2.3,
  "processing_time_ms": 350.0
}
```

#### POST `/api/v1/speech/stt/upload`

Speech-to-text from uploaded audio file (multipart/form-data).

**Form Data**:
| Field | Type | Description |
|-------|------|-------------|
| `file` | File | Audio file (wav, mp3, ogg, webm) |
| `language` | string | Language code (default: "en") |

**Response** `200 OK`: Same as STT response.

#### POST `/api/v1/speech/tts`

Text-to-speech synthesis.

```json
{
  "text": "Hello, how are you?",
  "language": "en",
  "voice": "en_female_1",
  "speed": 1.0
}
```

**Response** `200 OK`:
```json
{
  "audio_data": "base64_encoded_audio",
  "format": "wav",
  "sample_rate": 24000,
  "duration_seconds": 1.5
}
```

#### POST `/api/v1/speech/analyze-emotion`

Analyze emotion from speech audio.

```json
{
  "audio_data": "base64_encoded_audio",
  "language": "en",
  "sample_rate": 16000
}
```

**Response** `200 OK`:
```json
{
  "emotion": "happy",
  "confidence": 0.85,
  "emotions": {"happy": 0.85, "neutral": 0.10, "sad": 0.03, "angry": 0.02},
  "processing_time_ms": 120.0
}
```

#### GET `/api/v1/speech/languages`

Get supported languages for STT and TTS.

**Response** `200 OK`:
```json
{
  "languages": [
    {"code": "en", "name": "English", "stt_supported": true, "tts_supported": true},
    {"code": "es", "name": "Spanish", "stt_supported": true, "tts_supported": true}
  ]
}
```

## Communication API

### Endpoints

#### POST `/api/v1/communication/session`

Create a new communication session.

```json
{
  "session_type": "multimodal",
  "metadata": {"device": "web", "app_version": "1.0.0"}
}
```

**Session Types**: `gesture`, `speech`, `multimodal`, `text`

**Response** `201 Created`:
```json
{
  "id": "uuid",
  "session_type": "multimodal",
  "start_time": "2025-01-01T00:00:00+00:00",
  "is_active": true,
  "metadata": {"device": "web"}
}
```

#### POST `/api/v1/communication/translate`

Translate text between languages.

```json
{
  "source_text": "Hello, how are you?",
  "source_lang": "en",
  "target_lang": "es"
}
```

**Response** `200 OK`:
```json
{
  "target_text": "Hola, como estas?",
  "source_lang": "en",
  "target_lang": "es",
  "confidence": 0.96,
  "processing_time_ms": 150.0
}
```

#### GET `/api/v1/communication/suggest`

Get communication suggestions based on context.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `context` | string | "" | Current conversation context |
| `max_suggestions` | integer | 5 | Max suggestions (1-20) |

**Response** `200 OK`:
```json
{
  "suggestions": ["I need help", "Yes, please", "Thank you", "I am hungry", "Where is the bathroom?"],
  "context": {"last_intent": "request", "emotion": "neutral"},
  "processing_time_ms": 30.0
}
```

#### POST `/api/v1/communication/feedback`

Submit feedback for a session.

```json
{
  "session_id": "uuid",
  "rating": 4,
  "feedback_type": "general",
  "comment": "Great experience!"
}
```

**Response** `200 OK`:
```json
{
  "status": "success",
  "message": "Feedback recorded"
}
```

## Analytics API

### Endpoints

#### GET `/api/v1/analytics/metrics`

Get system-wide metrics (admin).

**Response** `200 OK`:
```json
{
  "total_users": 1250,
  "total_sessions": 45000,
  "total_gestures": 120000,
  "total_translations": 35000,
  "active_sessions": 45,
  "uptime_hours": 720.5
}
```

#### GET `/api/v1/analytics/user/{user_id}/progress`

Get user progress data.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `days` | integer | 30 | Lookback period (1-365) |

**Response** `200 OK`:
```json
{
  "user_id": "uuid",
  "total_sessions": 45,
  "total_gestures": 1200,
  "average_confidence": 0.87,
  "progress_over_time": [
    {"date": "2025-01-01", "value": 40, "metric": "gestures"}
  ]
}
```

#### GET `/api/v1/analytics/user/{user_id}/accuracy`

Get gesture accuracy metrics per user.

**Response** `200 OK`:
```json
{
  "user_id": "uuid",
  "overall_accuracy": 0.89,
  "by_gesture": [
    {"gesture_type": "thumbs_up", "total": 200, "correct": 190, "accuracy": 0.95, "average_confidence": 0.94}
  ]
}
```

#### GET `/api/v1/analytics/user/{user_id}/adaptation`

Get user adaptation metrics.

**Query Parameters**:
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric_type` | string | null | Filter by metric type |
| `limit` | integer | 100 | Max records (1-1000) |

**Response** `200 OK`:
```json
{
  "user_id": "uuid",
  "metrics": [
    {"metric_type": "gesture_accuracy", "value": 0.92, "recorded_at": "2025-01-01T00:00:00+00:00", "metadata": {}}
  ]
}
```

#### GET `/api/v1/analytics/realtime`

Get real-time analytics snapshot.

**Response** `200 OK`:
```json
{
  "timestamp": "2025-01-01T00:00:00+00:00",
  "active_connections": 42,
  "requests_per_second": 15.3,
  "average_latency_ms": 85.0,
  "gesture_throughput": 8.2,
  "speech_throughput": 3.1
}
```

## System Endpoints

### Endpoints

#### GET `/health`

Comprehensive health check.

**Response** `200 OK` (healthy) / `503 Service Unavailable` (degraded):
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "production",
  "uptime_hours": 720.5,
  "database": {"status": "healthy", "latency_ms": 2.3},
  "websocket_connections": 42,
  "active_users": 38
}
```

#### GET `/metrics`

Prometheus-formatted metrics.

**Response** `200 OK`: Prometheus text format.
```
# HELP http_requests_total Total HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="GET",path="/health",status="200"} 15000
```

#### GET `/`

Root endpoint with service information.

**Response** `200 OK`:
```json
{
  "name": "Neurolink",
  "version": "1.0.0",
  "environment": "production",
  "docs": "/docs",
  "health": "/health",
  "metrics": "/metrics"
}
```

## WebSocket Events

### Connection

**URL**: `ws://host:8000/ws`

**Query Parameters**:
| Parameter | Required | Description |
|-----------|----------|-------------|
| `token` | Yes | JWT access token |
| `session_id` | No | Existing session ID to resume |

### Client-to-Server Events

#### `gesture:recognize`

```json
{
  "event": "gesture:recognize",
  "data": {
    "frame_data": "base64_image",
    "model_version": "v2.1"
  }
}
```

**Response**: `gesture:result`
```json
{
  "event": "gesture:result",
  "data": {
    "gesture_type": "thumbs_up",
    "confidence": 0.94,
    "landmarks": [],
    "processing_time_ms": 45.2
  }
}
```

#### `speech:transcribe`

```json
{
  "event": "speech:transcribe",
  "data": {
    "audio_data": "base64_audio",
    "language": "en",
    "sample_rate": 16000
  }
}
```

**Response**: `speech:result`
```json
{
  "event": "speech:result",
  "data": {
    "text": "Hello world",
    "confidence": 0.92,
    "processing_time_ms": 350.0
  }
}
```

#### `multimodal:process`

```json
{
  "event": "multimodal:process",
  "data": {
    "frame_data": "base64_image",
    "audio_data": "base64_audio",
    "text_input": null,
    "session_id": "uuid"
  }
}
```

**Response**: `multimodal:result`
```json
{
  "event": "multimodal:result",
  "data": {
    "intent": "request_help",
    "gesture": "help",
    "text": "I need assistance",
    "emotion": "distressed",
    "urgency": 0.85,
    "suggestions": ["I need help", "Please call my caregiver"],
    "processing_time_ms": 280.0
  }
}
```

#### `heartbeat`

```json
{
  "event": "heartbeat",
  "data": {}
}
```

**Response**: `heartbeat:ack`
```json
{
  "event": "heartbeat:ack",
  "data": {"timestamp": "2025-01-01T00:00:00+00:00"}
}
```

### Server-to-Client Events

| Event | Description |
|-------|-------------|
| `connection:established` | Connection confirmed with session ID |
| `connection:error` | Authentication or connection error |
| `gesture:result` | Gesture recognition result |
| `speech:result` | Speech transcription result |
| `speech:interim` | Interim transcription (partial) |
| `multimodal:result` | Complete multimodal processing result |
| `emotion:update` | Real-time emotion analysis update |
| `suggestion:update` | Updated communication suggestions |
| `error` | Error with message and code |
| `heartbeat:ack` | Heartbeat acknowledgment |

## Authentication Headers

```
Authorization: Bearer <jwt_access_token>
X-API-Key: <api_key_for_service_to_service>
```

## Rate Limits

| Endpoint | Rate | Window |
|----------|------|--------|
| `/auth/login` | 10 requests | 1 minute |
| `/auth/register` | 3 requests | 1 hour |
| `/gestures/recognize` | 60 requests | 1 minute |
| `/speech/stt` | 20 requests | 1 minute |
| `/speech/tts` | 30 requests | 1 minute |
| `/communication/translate` | 60 requests | 1 minute |
| All other endpoints | 100 requests | 1 minute |

Rate limit headers are returned in all responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `AUTHENTICATION_ERROR` | 401 | Invalid or expired credentials |
| `AUTHORIZATION_ERROR` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 422 | Request validation failed |
| `RATE_LIMIT_ERROR` | 429 | Too many requests |
| `GESTURE_PROCESSING_ERROR` | 422 | Gesture processing failed |
| `SPEECH_PROCESSING_ERROR` | 422 | Speech processing failed |
| `EMOTION_DETECTION_ERROR` | 422 | Emotion detection failed |
| `MULTIMODAL_FUSION_ERROR` | 500 | Multimodal fusion failed |
| `PERSONALIZATION_ERROR` | 422 | Personalization operation failed |
| `EDGE_DEPLOYMENT_ERROR` | 500 | Edge deployment failed |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `INTERNAL_ERROR` | 500 | Unexpected internal error |

### Error Response Format

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "context": {
      "details": [
        {
          "loc": ["body", "email"],
          "msg": "value is not a valid email address",
          "type": "value_error.email"
        }
      ]
    }
  }
}
```

## Pagination

List endpoints support cursor-based or offset-based pagination:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | integer | 1 | Page number |
| `page_size` | integer | 20 | Items per page (max 100) |

Response includes pagination metadata:
```json
{
  "items": [],
  "total": 150,
  "page": 1,
  "page_size": 20
}
```
