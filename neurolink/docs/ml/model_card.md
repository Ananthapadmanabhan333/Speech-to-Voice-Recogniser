# Model Card: Neurolink ML Models

## Overview

Neurolink uses a suite of specialized neural network models for gesture recognition, sign language translation, multimodal fusion, emotion detection, and personalization. This document provides detailed information about each model's architecture, training data, performance, limitations, and ethical considerations.

---

## 1. Gesture Classification Model

### Model Details

| Attribute | Description |
|-----------|-------------|
| **Model Name** | `neurolink-gesture-classifier-v2` |
| **Model Type** | CNN + Bi-LSTM hybrid |
| **Task** | Static and dynamic gesture classification |
| **Framework** | PyTorch 2.5.0 |
| **Architecture** | 3-layer CNN for spatial features → Bi-LSTM with attention for temporal sequence |
| **Input** | 21 hand landmarks (x, y, z) × 30 frames |
| **Output** | 62 gesture classes (12 static + 50 dynamic) |
| **Parameters** | 2.1M |
| **Model Size** | 8.4 MB (FP32), 4.2 MB (FP16), 2.1 MB (INT8) |
| **Inference Hardware** | CPU, NVIDIA GPU, Jetson TensorRT, Raspberry Pi ONNX |

### Training Data

| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| HaGRD (Hand Gesture Recognition Dataset) | 552,992 | 18 | Open source |
| NVIDIA Dynamic Gesture Dataset | 50,000 | 25 | Open source |
| Custom in-house dataset | 120,000 | 62 | Collected internally |
| ASL Fingerspelling (A-Z) | 780,000 | 26 | Open source |
| Augmented data (rotation, scaling, noise) | 2,000,000 | 62 | Generated |

**Total training samples**: ~3.5M

**Data split**: Train 80% / Validation 10% / Test 10%

### Performance Metrics

| Metric | FP32 GPU | FP16 TensorRT | INT8 ONNX CPU |
|--------|----------|---------------|---------------|
| Top-1 Accuracy | 94.2% | 93.8% | 91.5% |
| Top-5 Accuracy | 98.7% | 98.5% | 97.2% |
| F1 Score (macro) | 0.93 | 0.92 | 0.90 |
| Inference Latency | 15ms | 8ms | 45ms |
| Throughput | 200/s | 400/s | 50/s |

### Class-wise Performance

| Gesture | Precision | Recall | F1 Score |
|---------|-----------|--------|----------|
| Thumbs Up | 0.97 | 0.96 | 0.96 |
| Point | 0.95 | 0.94 | 0.94 |
| Fist | 0.96 | 0.95 | 0.95 |
| Open Palm | 0.94 | 0.93 | 0.93 |
| Swipe Left | 0.91 | 0.89 | 0.90 |
| Swipe Right | 0.90 | 0.88 | 0.89 |
| Wave | 0.92 | 0.90 | 0.91 |
| ASL A-Z (avg) | 0.88 | 0.86 | 0.87 |
| Custom Gestures | 0.85 | 0.82 | 0.83 |

### Limitations

- Performance degrades under poor lighting conditions (< 50 lux)
- Two-hand gestures have 15% lower accuracy than single-hand
- Skin tone bias: 3% lower accuracy for darker skin tones (Mitigation: training augmentation with diverse skin tones)
- Speed sensitivity: gestures performed too quickly (< 200ms) or too slowly (> 3s) reduce accuracy
- Camera angle > 45 degrees off-frontal reduces landmark accuracy

---

## 2. Sign Language Model

### Model Details

| Attribute | Description |
|-----------|-------------|
| **Model Name** | `neurolink-sign-language-v1` |
| **Model Type** | Transformer Encoder + CTC Decoder |
| **Task** | Continuous sign language recognition and translation |
| **Framework** | PyTorch 2.5.0 |
| **Architecture** | MediaPipe landmarks → Transformer encoder (6 layers, 8 heads) → CTC decoder |
| **Input** | 21 hand landmarks × 2 hands + body pose (33 landmarks) × 150 frames |
| **Output** | Text translation in target language |
| **Parameters** | 12.5M |
| **Model Size** | 50 MB (FP32), 25 MB (FP16) |
| **Supported Languages** | ASL (American), BSL (British) - output in English text |

### Training Data

| Dataset | Samples | Language | Source |
|---------|---------|----------|--------|
| WLASL (Word-Level ASL) | 200,000 | ASL | Open source |
| RWTH-PHOENIX-Weather | 100,000 | DGS (German) | Open source |
| MS-ASL | 100,000 | ASL | Open source |
| How2Sign | 80,000 | ASL | Open source |
| Custom recorded data | 50,000 | ASL, BSL | Collected internally |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Word Error Rate (WER) | 24.3% |
| Sign Error Rate (SER) | 18.7% |
| BLEU-4 Score | 0.42 |
| ROUGE-L Score | 0.58 |
| Fingerspelling Accuracy | 89.2% |
| Continuous Sign Recognition Accuracy | 72.5% |

### Limitations

- Currently supports only ASL and BSL (limited BSL coverage)
- Requires clear upper body visibility (not just hands)
- Sentence-level translation accuracy drops for complex grammar (WER increases to 32%)
- Limited vocabulary: ~2,000 signs per supported language
- Regional sign variations not fully captured
- Non-manual markers (facial expressions in sign language) not yet integrated

---

## 3. Multimodal Fusion Model

### Model Details

| Attribute | Description |
|-----------|-------------|
| **Model Name** | `neurolink-multimodal-fusion-v1` |
| **Model Type** | Cross-Modal Transformer |
| **Task** | Fuse gesture, speech, emotion, and text modalities |
| **Framework** | PyTorch 2.5.0 |
| **Architecture** | 4 modality encoders → 4-layer cross-modal transformer → MLP decoder |
| **Input** | Gesture embedding (128), Speech embedding (512), Emotion embedding (128), Text embedding (768) |
| **Output** | Fused intent (20 classes), urgency (0-1), confidence |
| **Parameters** | 8.3M |
| **Model Size** | 33 MB (FP32) |

### Training Data

| Dataset | Samples | Modalities | Source |
|---------|---------|------------|--------|
| IEMOCAP | 12,000 | Speech + Emotion | Academic |
| CMU-MOSEI | 23,000 | Video + Speech + Text | Academic |
| MELD | 13,000 | Video + Speech + Emotion | Academic |
| Custom multimodal sessions | 50,000 | Gesture + Speech + Emotion | Collected |
| Synthetic multimodal data | 200,000 | All 4 modalities | Generated |

### Performance Metrics

| Metric | Value |
|--------|-------|
| Intent Classification Accuracy | 91.2% |
| F1 Score (intent) | 0.89 |
| Urgency Detection Accuracy | 87.5% |
| Ablation: Gesture only | 72.1% |
| Ablation: Speech only | 81.3% |
| Ablation: Text only | 85.7% |
| Ablation: Emotion only | 65.2% |
| **Full multimodal** | **91.2%** |

### Modality Contribution Analysis

| Modality | Average Attention Weight | Performance Drop When Removed |
|----------|-------------------------|-------------------------------|
| Speech | 0.35 | -12.3% |
| Text | 0.28 | -9.8% |
| Gesture | 0.22 | -7.5% |
| Emotion | 0.15 | -4.1% |

### Limitations

- Degraded performance when 2+ modalities are missing simultaneously
- Temporal misalignment between modalities reduces fusion quality
- Speech-text redundancy is not always optimally utilized
- Training data limited to English-language emotional speech
- Cross-modal attention may over-weight dominant modalities

---

## 4. Emotion Detection Model

### Model Details

| Attribute | Description |
|-----------|-------------|
| **Model Name** | `neurolink-emotion-detection-v2` |
| **Model Type** | Dual-pathway CNN + Fusion |
| **Task** | Emotion recognition from facial and vocal signals |
| **Framework** | PyTorch 2.5.0 |
| **Architecture** | Facial: ResNet-18 (224x224 face crop) → 512-d embedding; Vocal: CNN-LSTM (MFCC features) → 256-d embedding; Fusion: weighted combination |
| **Input** | Facial: 224x224 RGB face image; Vocal: 3 seconds 16kHz audio → 40 MFCCs |
| **Output** | 7 basic emotions + intensity score |
| **Parameters** | 12.8M (facial) + 3.2M (vocal) + 0.5M (fusion) = 16.5M total |

### Training Data

#### Facial
| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| FER2013 | 35,887 | 7 | Kaggle |
| AffectNet | 450,000 | 8 | Academic |
| RAF-DB | 30,000 | 7 | Academic |
| EMOTIC | 35,000 | 26 | Academic |

#### Vocal
| Dataset | Samples | Classes | Source |
|---------|---------|---------|--------|
| RAVDESS | 7,356 | 8 | Academic |
| CREMA-D | 7,442 | 6 | Academic |
| TESS | 2,800 | 7 | Academic |
| IEMOCAP | 12,000 | 4 | Academic |

### Performance Metrics

| Metric | Facial | Vocal | Fused |
|--------|--------|-------|-------|
| Accuracy | 85.3% | 72.1% | 88.7% |
| F1 Score | 0.84 | 0.70 | 0.87 |
| AUC-ROC | 0.93 | 0.85 | 0.95 |

### Per-Emotion Performance (Fused)

| Emotion | Precision | Recall | F1 |
|---------|-----------|--------|-----|
| Happy | 0.94 | 0.92 | 0.93 |
| Sad | 0.90 | 0.88 | 0.89 |
| Angry | 0.91 | 0.89 | 0.90 |
| Surprised | 0.87 | 0.84 | 0.85 |
| Fearful | 0.82 | 0.79 | 0.80 |
| Disgusted | 0.79 | 0.75 | 0.77 |
| Neutral | 0.92 | 0.95 | 0.93 |

### Limitations

- Cultural differences in emotion expression not fully captured (trained primarily on Western datasets)
- Subtle emotions often confused (e.g., fear and surprise)
- Performance degrades with face masks, occlusions, or low-resolution input
- Vocal emotion accuracy drops significantly with background noise (SNR < 10dB)
- Emotional intensity calibration not yet validated clinically

---

## 5. Personalization Model

### Model Details

| Attribute | Description |
|-----------|-------------|
| **Model Name** | `neurolink-personalization-v1` |
| **Model Type** | Reinforcement Learning + Memory Network |
| **Task** | Adaptive personalization of gesture recognition, phrase prediction, and communication suggestions |
| **Framework** | PyTorch 2.5.0 |
| **Architecture** | Memory-augmented neural network with DQN-based RL agent |
| **Input** | User feedback signals (gesture corrections, phrase selections, ratings) |
| **Output** | Updated model weights, personalized suggestions, adapted gesture templates |
| **Parameters** | 1.2M (memory) + 0.8M (RL policy) = 2.0M total |

### Training Data

- Synthetically generated user interaction sequences: 1M episodes
- Real user interaction logs: 50,000 sessions (anonymized)
- Simulated user profiles: 10,000 synthetic users with varying communication patterns

### Performance Metrics

| Metric | Before Personalization | After 100 Sessions | Improvement |
|--------|----------------------|-------------------|-------------|
| Gesture Recognition Accuracy | 91.2% | 94.8% | +3.6% |
| Phrase Prediction Accuracy | 72.3% | 81.5% | +9.2% |
| Suggestion Acceptance Rate | 45.1% | 62.3% | +17.2% |
| User Satisfaction (1-5) | 3.8 | 4.3 | +0.5 |
| Adaptation Convergence | - | ~50 sessions | - |

### Learning Curves

| Sessions | Gesture Accuracy | Suggestion Acceptance |
|----------|-----------------|---------------------|
| 0 | 91.2% | 45.1% |
| 10 | 92.5% | 50.2% |
| 25 | 93.8% | 56.8% |
| 50 | 94.5% | 61.0% |
| 100 | 94.8% | 62.3% |
| 200+ | 94.9% | 62.5% |

### Limitations

- Requires minimum ~25 sessions for noticeable personalization
- Cold-start problem for new users (uses population-level defaults)
- RL exploration rate must balance adaptation speed vs. stability
- Privacy-preserving personalization limits data retention
- Cross-user knowledge transfer not yet implemented

---

## Ethical Considerations

### Bias and Fairness

- **Skin tone bias**: Gesture models show 3% lower accuracy for darker skin tones. We mitigate through data augmentation and are actively collecting diverse training data.
- **Gender bias**: Emotion models show slight bias (2%) in recognizing female-expressed anger. Investigated through fairness audits.
- **Language bias**: Speech models are optimized for English. Support for other languages varies by availability of training data.
- **Accessibility bias**: Models are designed for speech/motor impairments but may not cover all types of disabilities equally.

### Privacy

- All personalization data is end-to-end encrypted
- Users can request data deletion at any time (right to be forgotten)
- Interaction logs are anonymized after 30 days
- On-device processing is prioritized to minimize data transmission
- Voice data is processed in real-time and not stored unless explicitly saved by user

### Safety

- The system is not a medical device and should not replace professional medical advice
- Emergency detection (high urgency flag) should be validated by a human caregiver
- TTS should not be used to impersonate individuals without explicit consent
- Gesture recognition should not be used in high-stakes control systems without fail-safes

### Environmental Impact

- Model training: estimated 500 kg CO2 per full training run (all models)
- Inference: ~5W (edge device) to ~150W (GPU server) per inference node
- Model quantization reduces energy consumption by 60% on edge devices

### Intended Use

Neurolink models are designed to assist individuals with communication disabilities, including:
- Amyotrophic Lateral Sclerosis (ALS)
- Cerebral Palsy
- Stroke-related aphasia
- Locked-in syndrome
- Non-verbal autism spectrum
- Traumatic brain injury

### Out-of-Scope Use

- Autonomous decision-making in critical systems
- Surveillance or behavioral monitoring without consent
- Lie detection or credibility assessment
- Emotion-based manipulation or targeted advertising
- Replacement of human caregivers or medical professionals
