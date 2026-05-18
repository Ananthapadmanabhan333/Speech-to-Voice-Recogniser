# Multimodal Pipeline Architecture

This document describes the complete multimodal processing pipeline, from raw sensor input to synthesized communication output.

## Pipeline Overview

```
                        ┌─────────────────────────────────────┐
                        │       Multimodal Input              │
                        │  ┌──────┐ ┌──────┐ ┌──────┐ ┌────┐ │
                        │  │Video │ │Audio │ │Text  │ │Context│
                        │  │Frame │ │Stream│ │Input │ │      │
                        │  └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ │
                        └─────┼────────┼────────┼─────────┼─────┘
                              │        │        │         │
        ┌─────────────────────┼────────┼────────┼─────────┼──────────────┐
        │                     │        │        │         │              │
        │  ┌──────────────────┴──┐  ┌──┴────────┴──┐  ┌──┴───────────┐  │
        │  │  Gesture Pipeline   │  │Speech Pipeline│  │Emotion Pipeline│ │
        │  └──────────┬──────────┘  └───────┬───────┘  └───────┬───────┘  │
        │             │                     │                   │          │
        │  ┌──────────┴─────────────────────┴───────────────────┴──────┐   │
        │  │                  Multimodal Fusion                         │   │
        │  │     Temporal Alignment -> Cross-Modal Attention ->        │   │
        │  │     Feature Fusion -> Context Integration -> Inference    │   │
        │  └──────────────────────────┬────────────────────────────────┘   │
        │                             │                                    │
        │  ┌──────────────────────────┴────────────────────────────────┐   │
        │  │                    Output Generation                       │   │
        │  │     Intent -> Phrase Prediction -> TTS -> Sign Language   │   │
        │  └───────────────────────────────────────────────────────────┘   │
        └──────────────────────────────────────────────────────────────────┘
```

## Gesture Pipeline

### Pipeline Stages

```
Camera Frame ─► Hand Detection ─► Landmark Extraction ─► Tracking
    │                 │                    │                   │
    │            MediaPipe          21 landmarks          Temporal
    │           palm detection       per hand             consistency
    │                 │                    │                   │
    └─────────────────┴────────────────────┴───────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
            Gesture Classification    Sequence Modeling
                    │                       │
              CNN classifier          Transformer/LSTM
              (static gestures)       (dynamic gestures)
                    │                       │
                    └───────────┬───────────┘
                                │
                      ┌─────────┴─────────┐
                      │   Gesture Result   │
                      │  type+confidence   │
                      │  +landmark seq     │
                      └───────────────────┘
```

### Implementation Details

| Stage | Library | Model | Input | Output | Latency Budget |
|-------|---------|-------|-------|--------|---------------|
| Hand Detection | MediaPipe | BlazePalm | 640x480 frame | Bounding boxes | <30ms |
| Landmark Extraction | MediaPipe | Hand Landmark | Cropped hand ROI | 21 landmarks (x,y,z) | <15ms |
| Tracking | Custom | Kalman Filter | Sequential landmarks | Tracked hand IDs | <5ms |
| Classification | PyTorch | CNN (3 conv layers) | Single frame landmarks | Gesture class + confidence | <20ms |
| Sequence Modeling | PyTorch | Bi-LSTM + Attention | 30-frame landmark sequence | Dynamic gesture class | <25ms |

### Supported Gesture Types

- Static: Point, Thumbs Up, OK, Peace, Fist, Open Palm, Pinch
- Dynamic: Swipe Left/Right/Up/Down, Circle, Wave, Clap, Zoom In/Out
- Sign Language: A-Z fingerspelling (static), 50+ common ASL signs (dynamic)

### Error Handling

- **No hand detected**: Return empty result with `no_hand_detected` flag
- **Low confidence (<0.5)**: Return with `low_confidence` warning, use fallback gesture
- **Tracking loss**: Reinitialize tracker, use last known good state
- **Occlusion**: Temporal interpolation of missing landmarks
- **Edge case**: Two hands crossing - assign hands based on trajectory consistency

## Speech Pipeline

### Pipeline Stages

```
Audio Stream ─► Preprocessing ─► VAD ─► STT (Whisper) ─► NLU
    │               │            │           │              │
    │         16kHz resample   Voice      OpenAI        Intent +
    │         noise reduction  Activity   Whisper       Entity
    │         normalization   Detection   base/large   Extraction
    │               │            │           │              │
    └───────────────┴────────────┴───────────┴──────────────┴───┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
              TTS Synthesis         Translation
                    │                     │
            Tacotron2/VITS       NLLB/M2M-100
            + HiFi-GAN Vocoder    (100+ languages)
                    │                     │
                    └──────────┬──────────┘
                               │
                      ┌────────┴────────┐
                      │  Speech Result   │
                      │  text + audio    │
                      └─────────────────┘
```

### Implementation Details

| Stage | Model | Input | Output | Latency Budget |
|-------|-------|-------|--------|---------------|
| VAD | Silero VAD | 16kHz audio chunks | Voice activity segments | <10ms |
| STT | Whisper base/large | 16kHz audio (30s max) | Transcribed text + confidence | <500ms (base) / <2s (large) |
| NLU | Custom BERT | Transcribed text | Intent + entities + sentiment | <50ms |
| TTS | Tacotron2 + HiFi-GAN | Text with prosody markers | 24kHz audio waveform | <300ms (per sentence) |
| Translation | NLLB-200/M2M-100 | Source text + languages | Translated text | <200ms |

### Supported Languages

STT: 15 languages (en, es, fr, de, zh, ja, ko, ar, hi, pt, ru, it, nl, tr, pl)
TTS: 8 languages (en, es, fr, de, zh, ja, hi, pt)
Translation: 100+ language pairs via NLLB-200

### Error Handling

- **Silence/no speech**: Return empty result, trigger gesture fallback
- **Low confidence transcription (<0.6)**: Flag for user confirmation
- **Unsupported language**: Auto-detect closest supported language, inform user
- **Audio corruption**: Request re-recording, use previous good transcription
- **TTS failure**: Fallback to text display with audio unavailable notification

## Emotion Pipeline

### Pipeline Stages

```
┌───────────────── Facial Pipeline ─────────────────┐
│  Face Detection ─► Landmark Detection ─► CNN ─►   │
│     MTCNN/Retina   68 landmarks      ResNet-18    │
│        Face           Face ROI         Emotion     │
└────────────────────────────────────────────────────┘
                         │
┌────────────────── Vocal Pipeline ──────────────────┐
│  Audio Features ─► Prosody Analysis ─► DNN ─►     │
│  MFCC + pitch +   Energy, pitch,     Speech        │
│  spectral         speaking rate     Emotion Net    │
└────────────────────────────────────────────────────┘
                         │
              ┌──────────┴──────────┐
              │   Emotion Fusion     │
              │  Weighted average    │
              │  Context-aware       │
              │  Temporal smoothing  │
              └──────────┬──────────┘
                         │
                ┌────────┴────────┐
                │ Emotion Output   │
                │ type + intensity │
                │ modality weights │
                └─────────────────┘
```

### Implementation Details

| Stage | Model | Input | Output | Latency Budget |
|-------|-------|-------|--------|---------------|
| Face Detection | MTCNN/RetinaFace | Video frame | Face bounding box | <20ms |
| Facial Emotion | ResNet-18 | 224x224 face crop | 7 emotion probabilities | <15ms |
| Vocal Emotion | CNN-LSTM | 3s audio MFCC | 7 emotion probabilities | <50ms |
| Emotion Fusion | Weighted fusion | Face + voice probs | Fused emotion + intensity | <5ms |

### Emotion Categories

- Universal: happy, sad, angry, fearful, surprised, disgusted, neutral
- Extended (facial): contempt, confused, frustrated, interested
- Extended (vocal): calm, excited, bored, anxious, confident

### Fusion Strategy

```
fused_emotion = α * facial_emotion + β * vocal_emotion + γ * context_bias

Weights:
  α = 0.6, β = 0.3, γ = 0.1  (video available)
  α = 0.0, β = 0.9, γ = 0.1  (audio only)
  α = 0.0, β = 0.0, γ = 1.0  (no sensor input, use context)
```

### Error Handling

- **No face detected**: Fall back to vocal-only emotion analysis
- **Low quality audio**: Reduce vocal weight, increase facial weight
- **Ambient noise**: Apply noise gate before vocal feature extraction
- **Conflicting modalities**: Use context-based disambiguation
- **Rapid mood swings**: Temporal smoothing with EMA (α=0.3)

## Multimodal Fusion

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────┐
│               Temporal Alignment                          │
│  Gesture frames ──┐                                       │
│  Speech chunks  ──┼──► Dynamic Time Warping              │
│  Emotion frames ──┘    Align to common timeline           │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────┐
│              Cross-Modal Attention                         │
│                                                           │
│     ┌──────────┐      ┌──────────┐      ┌──────────┐    │
│     │ Gesture  │      │  Speech  │      │ Emotion  │    │
│     │ Embedding│      │ Embedding│      │ Embedding│    │
│     └─────┬────┘      └────┬─────┘      └────┬─────┘    │
│           │                │                  │          │
│           └────────────────┼──────────────────┘          │
│                            │                             │
│                ┌───────────┴───────────┐                │
│                │   Cross-Modal          │                │
│                │   Transformer         │                │
│                │   (self + cross-attn)  │                │
│                └───────────┬───────────┘                │
│                            │                             │
│                ┌───────────┴───────────┐                │
│                │   Fused Embedding     │                │
│                │   (512-dim)           │                │
│                └───────────────────────┘                │
└──────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────┐
│                   Context Integration                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐  │
│  │ Conversation│  │  User       │  │  Environmental  │  │
│  │ History     │  │  Profile    │  │  Context        │  │
│  └─────────────┘  └─────────────┘  └─────────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │
┌──────────────────────────┴───────────────────────────────┐
│                   Inference                                │
│  ┌────────────┐  ┌────────────┐  ┌────────────────────┐ │
│  │  Intent    │  │  Urgency   │  │  Communication    │ │
│  │  Prediction│  │  Detection │  │  Confidence       │ │
│  └────────────┘  └────────────┘  └────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

### Fusion Architecture

The multimodal fusion uses a Transformer encoder with cross-modal attention:

```
Input Modalities:
  - Gesture: landmark sequence → MLP → 128-dim embedding
  - Speech: audio features → Whisper encoder → 512-dim embedding
  - Emotion: facial+vocal → ResNet → 128-dim embedding
  - Text: input text → BERT → 768-dim embedding

Fusion Layers:
  - Projection: 4x linear layers projecting to 256-dim common space
  - Cross-Attention: Each modality attends to all others
  - Self-Attention: Modality-specific temporal context
  - Feed-Forward: 2-layer MLP with GELU activation
  - Layer Norm: Pre-norm architecture for training stability

Output:
  - Fused embedding (512-dim) used for intent, urgency, confidence
```

## Edge Pipeline

### Optimization Pipeline

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ PyTorch  │───►│ ONNX     │───►│ Quantize │───►│ Optimize │
│ Model    │    │ Export   │    │ FP16/INT8│    │ TensorRT │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                    │
                                          ┌─────────┴─────────┐
                                          │                   │
                                    ┌─────┴─────┐     ┌──────┴──────┐
                                    │  Jetson   │     │ Raspberry Pi│
                                    │  TensorRT │     │  ONNX CPU   │
                                    │  FP16     │     │  INT8       │
                                    └───────────┘     └─────────────┘
```

### Edge Deployment Targets

| Device | Architecture | GPU | RAM | Storage | Models |
|--------|-------------|-----|-----|---------|--------|
| NVIDIA Jetson Orin | ARM Cortex-A78AE | Ampere 2048 CUDA | 16GB | 64GB | Gesture + Emotion (FP16 TensorRT) |
| NVIDIA Jetson Nano | ARM Cortex-A57 | Maxwell 128 CUDA | 4GB | 16GB | Gesture only (FP16) |
| Raspberry Pi 5 | ARM Cortex-A76 | VideoCore VII | 8GB | 32GB | Gesture only (INT8 ONNX) |

### Latency Budgets

| Pipeline | Cloud Target | Jetson Target | RPi Target | Maximum Acceptable |
|----------|-------------|---------------|------------|-------------------|
| Gesture Recognition | <75ms | <100ms | <200ms | 300ms |
| Speech-to-Text | <500ms | N/A | N/A | 1.5s |
| Text-to-Speech | <300ms | N/A | N/A | 1s |
| Emotion Detection | <100ms | <200ms | N/A | 500ms |
| Multimodal Fusion | <150ms | <250ms | N/A | 500ms |
| Total Pipeline | <1s | <1.5s | <2s | 3s |

### Error Handling Strategy

| Error Type | Detection | Recovery |
|------------|-----------|----------|
| Model load failure | Exception on init | Fall back to CPU, log warning |
| Inference timeout | Timer > threshold | Return cached result, queue retry |
| Out of memory | torch.cuda.OutOfMemoryError | Unload unused models, clear cache |
| Invalid input | Shape/type validation | Return error with details |
| Service unavailable | Connection refused | Retry with exponential backoff |
| Cascade failure | Downstream dependency check | Degrade gracefully (e.g., gesture-only mode) |

## End-to-End Flow Example

### Scenario: User gestures "help" while speaking "I need assistance" with distressed expression

```
1. Camera captures 60fps video, microphone captures 16kHz audio
2. Gesture Pipeline:
   - Frame 1-5: Hand detected at ROI (0.2, 0.3, 0.5, 0.7)
   - Frame 6-30: Landmarks extracted, tracking initialized
   - Frames 1-60: Gesture classified as "help" (confidence 0.92)
   - Sequence model confirms temporal consistency

3. Speech Pipeline:
   - VAD detects speech segment (2.3s)
   - Whisper transcribes: "I need assistance" (confidence 0.88)
   - NLU extracts intent: "request_help" + entity: "assistance"

4. Emotion Pipeline:
   - Facial: "distressed" (0.72), "fearful" (0.15), "sad" (0.10)
   - Vocal: "anxious" (0.65), "distressed" (0.20), "neutral" (0.10)
   - Fused: "distressed" (0.68) with urgency 0.85

5. Multimodal Fusion:
   - Temporal alignment: gesture (0-1s), speech (0.3-2.6s), emotion (0-3s)
   - Cross-modal attention: gesture reinforces "help/request_help"
   - Fused intent: "emergency_request" (confidence 0.94)
   - Combined output: {"intent": "emergency_request", "urgency": 0.85}

6. Output Generation:
   - Phrase prediction: "I need help urgently"
   - TTS synthesis: audio with urgent prosody
   - Display: text + gesture visualization + emotion indicator
   - Alert: high-urgency notification to caregiver
```
