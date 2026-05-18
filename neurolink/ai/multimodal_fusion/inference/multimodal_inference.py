from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from ai.gesture_engine.classification.gesture_classifier import GestureClassifier, GestureResult
from ai.gesture_engine.detection.hand_detector import HandDetector, HandLandmarks
from ai.gesture_engine.sequence.sequence_model import InterpretedSequence, SequenceModel
from ai.gesture_engine.tracking.hand_tracker import HandTracker, TrackedHand
from ai.multimodal_fusion.attention.cross_modal_attention import CrossModalAttention
from ai.multimodal_fusion.embeddings.fusion_embeddings import (
    FusedEmbedding,
    ModalityEmbeddings,
    MultimodalEmbeddingFusion,
)
from ai.speech_engine.emotion.vocal_emotion import EmotionResult as VocalEmotionResult
from ai.speech_engine.emotion.vocal_emotion import VocalEmotionAnalyzer
from ai.speech_engine.transcription.transcriber import SpeechTranscriber, TranscriptionResult

logger = structlog.get_logger(__name__)


class ModalityPriority(Enum):
    """Processing priority for modalities."""

    SPEECH = 1  # Highest priority
    GESTURE = 2
    FACIAL = 3
    CONTEXT = 4


@dataclass
class MultimodalInput:
    """Input data for multimodal inference."""

    gesture_data: Optional[np.ndarray] = None  # Video frame or landmark sequence
    speech_data: Optional[np.ndarray] = None  # Audio signal
    facial_data: Optional[np.ndarray] = None  # Face image
    context: Optional[Dict[str, Any]] = None  # Contextual information
    timestamps: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultimodalResult:
    """Result from multimodal inference pipeline."""

    intent: str
    intent_confidence: float
    emotion: str
    emotion_confidence: float
    urgency: float  # 0 to 1
    overall_confidence: float
    transcription: Optional[TranscriptionResult] = None
    gestures: Optional[InterpretedSequence] = None
    suggestions: List[str] = field(default_factory=list)
    fused_embedding: Optional[np.ndarray] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModalityStatus:
    """Status of a single modality processing."""

    modality: str
    success: bool
    confidence: float
    processing_time: float
    error: Optional[str] = None
    result: Any = None


class MultimodalInferencePipeline:
    """Async multimodal inference pipeline with prioritized processing.

    Processes gesture, speech, and facial data concurrently with priority-based
    scheduling. Handles missing modalities gracefully through fallback strategies
    and confidence-based fusion.

    Pipeline flow:
    1. Prioritize modalities (speech > gesture > facial > context)
    2. Process available modalities concurrently
    3. Fuse results with confidence weighting
    4. Apply fallback strategies for missing data
    5. Return unified MultimodalResult
    """

    MIN_CONFIDENCE: float = 0.3
    FALLBACK_EMOTION: str = "neutral"
    FALLBACK_INTENT: str = "unknown"

    def __init__(
        self,
        gesture_classifier: Optional[GestureClassifier] = None,
        sequence_model: Optional[SequenceModel] = None,
        speech_transcriber: Optional[SpeechTranscriber] = None,
        vocal_emotion_analyzer: Optional[VocalEmotionAnalyzer] = None,
        fusion_engine: Optional[MultimodalEmbeddingFusion] = None,
        cross_modal_attention: Optional[CrossModalAttention] = None,
        confidence_threshold: float = 0.3,
        enable_fallback: bool = True,
    ):
        """Initialize multimodal inference pipeline.

        Args:
            gesture_classifier: Gesture classification engine.
            sequence_model: Gesture sequence interpreter.
            speech_transcriber: Speech transcription engine.
            vocal_emotion_analyzer: Vocal emotion analyzer.
            fusion_engine: Multimodal embedding fusion engine.
            cross_modal_attention: Cross-modal attention module.
            confidence_threshold: Minimum confidence for valid predictions.
            enable_fallback: Enable fallback strategies for missing data.
        """
        self._gesture_classifier = gesture_classifier
        self._sequence_model = sequence_model
        self._speech_transcriber = speech_transcriber
        self._vocal_emotion_analyzer = vocal_emotion_analyzer
        self._fusion_engine = fusion_engine
        self._cross_modal_attention = cross_modal_attention
        self._confidence_threshold = confidence_threshold
        self._enable_fallback = enable_fallback

        # Async state
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        logger.info(
            "multimodal_pipeline_initialized",
            has_gesture=gesture_classifier is not None,
            has_speech=speech_transcriber is not None,
            has_emotion=vocal_emotion_analyzer is not None,
            has_fusion=fusion_engine is not None,
        )

    async def infer(
        self,
        gesture_data: Optional[Any] = None,
        speech_data: Optional[Any] = None,
        facial_data: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> MultimodalResult:
        """Run multimodal inference on all available data.

        Processes modalities concurrently with priority-based scheduling.

        Args:
            gesture_data: Gesture landmarks or video frame.
            speech_data: Audio signal for transcription.
            facial_data: Face image for emotion analysis.
            context: Contextual information dict.

        Returns:
            MultimodalResult with fused intent, emotion, urgency, and suggestions.
        """
        start_time = time.time()

        # Build input
        multimodal_input = MultimodalInput(
            gesture_data=gesture_data,
            speech_data=speech_data,
            facial_data=facial_data,
            context=context or {},
            timestamps={"input": time.time()},
        )

        try:
            # Determine available modalities
            available = self._get_available_modalities(multimodal_input)
            if not available:
                logger.warning("no_modalities_available")
                return self._empty_result(start_time)

            # Process modalities concurrently
            tasks = []
            if "speech" in available:
                tasks.append(self._process_speech(multimodal_input))
            if "gesture" in available:
                tasks.append(self._process_gesture(multimodal_input))
            if "facial" in available:
                tasks.append(self._process_facial(multimodal_input))
            if "context" in available:
                tasks.append(self._process_context(multimodal_input))

            results: List[ModalityStatus] = await asyncio.gather(*tasks, return_exceptions=True)

            # Handle exceptions
            modality_results = []
            for r in results:
                if isinstance(r, Exception):
                    logger.error("modality_processing_failed", error=str(r))
                else:
                    modality_results.append(r)

            # Fuse results
            fused_result = self._fuse_modality_results(modality_results)

            # Compute urgency
            urgency = self._compute_urgency(modality_results, fused_result)

            # Generate suggestions
            suggestions = self._generate_suggestions(modality_results, fused_result)

            processing_time = time.time() - start_time

            return MultimodalResult(
                intent=fused_result.get("intent", self.FALLBACK_INTENT),
                intent_confidence=fused_result.get("intent_confidence", 0.0),
                emotion=fused_result.get("emotion", self.FALLBACK_EMOTION),
                emotion_confidence=fused_result.get("emotion_confidence", 0.0),
                urgency=urgency,
                overall_confidence=fused_result.get("overall_confidence", 0.0),
                transcription=fused_result.get("transcription"),
                gestures=fused_result.get("gestures"),
                suggestions=suggestions,
                fused_embedding=fused_result.get("fused_embedding"),
                processing_time=processing_time,
                metadata={
                    "modalities_processed": [r.modality for r in modality_results if r.success],
                    "processing_times": {
                        r.modality: r.processing_time for r in modality_results
                    },
                },
            )

        except Exception as e:
            logger.error("multimodal_inference_failed", error=str(e))
            return self._empty_result(start_time, error=str(e))

    async def infer_realtime(
        self,
        input_stream: asyncio.Queue,
        output_stream: asyncio.Queue,
    ) -> None:
        """Continuously process multimodal input from a stream.

        Args:
            input_stream: Async queue yielding MultimodalInput objects.
            output_stream: Async queue receiving MultimodalResult objects.
        """
        logger.info("realtime_inference_loop_started")

        while True:
            try:
                multimodal_input = await input_stream.get()
                result = await self.infer(
                    gesture_data=multimodal_input.gesture_data,
                    speech_data=multimodal_input.speech_data,
                    facial_data=multimodal_input.facial_data,
                    context=multimodal_input.context,
                )
                await output_stream.put(result)

            except asyncio.CancelledError:
                logger.info("realtime_inference_loop_cancelled")
                break
            except Exception as e:
                logger.error("realtime_inference_error", error=str(e))
                # Continue processing next input
                continue

    async def _process_speech(self, inp: MultimodalInput) -> ModalityStatus:
        """Process speech modality: transcribe and analyze emotion.

        Args:
            inp: Multimodal input.

        Returns:
            ModalityStatus for speech processing.
        """
        mod_start = time.time()
        try:
            transcription: Optional[TranscriptionResult] = None
            vocal_emotion: Optional[VocalEmotionResult] = None

            if self._speech_transcriber and inp.speech_data is not None:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                transcription = await loop.run_in_executor(
                    None, self._speech_transcriber.transcribe, inp.speech_data
                )

            if self._vocal_emotion_analyzer and inp.speech_data is not None:
                loop = asyncio.get_event_loop()
                vocal_emotion = await loop.run_in_executor(
                    None, self._vocal_emotion_analyzer.analyze_emotion, inp.speech_data
                )

            confidence = 0.0
            if transcription:
                confidence = transcription.confidence

            return ModalityStatus(
                modality="speech",
                success=True,
                confidence=confidence,
                processing_time=time.time() - mod_start,
                result={
                    "transcription": transcription,
                    "vocal_emotion": vocal_emotion,
                },
            )

        except Exception as e:
            logger.error("speech_processing_failed", error=str(e))
            return ModalityStatus(
                modality="speech",
                success=False,
                confidence=0.0,
                processing_time=time.time() - mod_start,
                error=str(e),
            )

    async def _process_gesture(self, inp: MultimodalInput) -> ModalityStatus:
        """Process gesture modality: classify and sequence.

        Args:
            inp: Multimodal input.

        Returns:
            ModalityStatus for gesture processing.
        """
        mod_start = time.time()
        try:
            gesture_result: Optional[GestureResult] = None
            interpreted_sequence: Optional[InterpretedSequence] = None

            if self._gesture_classifier and inp.gesture_data is not None:
                loop = asyncio.get_event_loop()
                gesture_result = await loop.run_in_executor(
                    None, self._gesture_classifier.classify_gesture, inp.gesture_data
                )

            if self._sequence_model and gesture_result is not None:
                loop = asyncio.get_event_loop()
                interpreted_sequence = await loop.run_in_executor(
                    None,
                    self._sequence_model.model_sequence,
                    [gesture_result.gesture_id],
                )

            return ModalityStatus(
                modality="gesture",
                success=True,
                confidence=gesture_result.confidence if gesture_result else 0.0,
                processing_time=time.time() - mod_start,
                result={
                    "gesture_result": gesture_result,
                    "interpreted_sequence": interpreted_sequence,
                },
            )

        except Exception as e:
            logger.error("gesture_processing_failed", error=str(e))
            return ModalityStatus(
                modality="gesture",
                success=False,
                confidence=0.0,
                processing_time=time.time() - mod_start,
                error=str(e),
            )

    async def _process_facial(self, inp: MultimodalInput) -> ModalityStatus:
        """Process facial modality: emotion recognition.

        Args:
            inp: Multimodal input.

        Returns:
            ModalityStatus for facial processing.
        """
        mod_start = time.time()
        try:
            # Placeholder for facial emotion analysis
            # In production, integrate with FacialEmotionAnalyzer
            return ModalityStatus(
                modality="facial",
                success=True,
                confidence=0.5,
                processing_time=time.time() - mod_start,
                result={"facial_emotion": None},
            )

        except Exception as e:
            return ModalityStatus(
                modality="facial",
                success=False,
                confidence=0.0,
                processing_time=time.time() - mod_start,
                error=str(e),
            )

    async def _process_context(self, inp: MultimodalInput) -> ModalityStatus:
        """Process context modality.

        Args:
            inp: Multimodal input.

        Returns:
            ModalityStatus for context processing.
        """
        mod_start = time.time()
        try:
            context_data = inp.context or {}

            return ModalityStatus(
                modality="context",
                success=True,
                confidence=0.7,
                processing_time=time.time() - mod_start,
                result={"context": context_data},
            )

        except Exception as e:
            return ModalityStatus(
                modality="context",
                success=False,
                confidence=0.0,
                processing_time=time.time() - mod_start,
                error=str(e),
            )

    def _fuse_modality_results(
        self, modality_results: List[ModalityStatus]
    ) -> Dict[str, Any]:
        """Fuse results from all modalities with confidence-based weighting.

        Args:
            modality_results: Results from individual modality processing.

        Returns:
            Fused result dict with intent, emotion, confidence.
        """
        fused: Dict[str, Any] = {
            "intent": self.FALLBACK_INTENT,
            "intent_confidence": 0.0,
            "emotion": self.FALLBACK_EMOTION,
            "emotion_confidence": 0.0,
            "overall_confidence": 0.0,
        }

        successful = [r for r in modality_results if r.success]
        if not successful:
            return fused

        # Extract results
        speech_result = None
        gesture_result = None
        vocal_emotion = None
        context_data = None
        transcription = None
        gestures = None

        for r in successful:
            if r.modality == "speech" and r.result:
                speech_result = r
                transcription = r.result.get("transcription")
                vocal_emotion = r.result.get("vocal_emotion")
            elif r.modality == "gesture" and r.result:
                gesture_result = r
                gestures = r.result.get("interpreted_sequence")
            elif r.modality == "context" and r.result:
                context_data = r.result.get("context")

        # Determine intent
        intent, intent_conf = self._determine_intent(
            transcription, gestures, context_data
        )
        fused["intent"] = intent
        fused["intent_confidence"] = intent_conf
        fused["transcription"] = transcription
        fused["gestures"] = gestures

        # Determine emotion
        emotion, emotion_conf = self._determine_emotion(
            vocal_emotion, context_data
        )
        fused["emotion"] = emotion
        fused["emotion_confidence"] = emotion_conf

        # Compute overall confidence
        confidences = [
            r.confidence for r in successful
            if r.confidence >= self._confidence_threshold
        ]
        if confidences:
            # Weight by modality priority
            weighted_conf = 0.0
            total_weight = 0.0
            priority_weights = {
                "speech": 1.0,
                "gesture": 0.8,
                "facial": 0.6,
                "context": 0.4,
            }
            for r in successful:
                w = priority_weights.get(r.modality, 0.5)
                weighted_conf += r.confidence * w
                total_weight += w
            fused["overall_confidence"] = weighted_conf / total_weight if total_weight > 0 else 0.0
        else:
            fused["overall_confidence"] = 0.0

        # Attempt multimodal embedding fusion
        if self._fusion_engine:
            try:
                modality_embeddings = self._build_modality_embeddings(
                    successful, transcription, gestures
                )
                if modality_embeddings:
                    fused_result = self._fusion_engine.fuse_embeddings(
                        **modality_embeddings
                    )
                    fused["fused_embedding"] = fused_result.fused_vector
            except Exception as e:
                logger.warning("fusion_failed_in_pipeline", error=str(e))

        return fused

    def _determine_intent(
        self,
        transcription: Optional[TranscriptionResult],
        gestures: Optional[InterpretedSequence],
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, float]:
        """Determine user intent from available modalities.

        Args:
            transcription: Speech transcription result.
            gestures: Interpreted gesture sequence.
            context: Contextual information.

        Returns:
            (intent_label, confidence) tuple.
        """
        # Use speech as primary intent source
        if transcription and transcription.confidence >= self._confidence_threshold:
            text = transcription.text.lower()
            intent, confidence = self._classify_intent_from_text(text)
            if confidence >= self._confidence_threshold:
                return intent, confidence

        # Use gesture as secondary source
        if gestures and gestures.sequence_confidence >= self._confidence_threshold:
            sentence = gestures.sentence.lower()
            intent, confidence = self._classify_intent_from_text(sentence)
            if confidence >= self._confidence_threshold:
                return intent, confidence

        return self.FALLBACK_INTENT, 0.0

    def _determine_emotion(
        self,
        vocal_emotion: Optional[VocalEmotionResult],
        context: Optional[Dict[str, Any]],
    ) -> Tuple[str, float]:
        """Determine user emotion from available modalities.

        Args:
            vocal_emotion: Vocal emotion analysis result.
            context: Contextual information.

        Returns:
            (emotion_label, confidence) tuple.
        """
        if vocal_emotion and vocal_emotion.confidence >= self._confidence_threshold:
            return vocal_emotion.emotion, vocal_emotion.confidence

        # Fall back to context-based emotion
        if context and "expected_emotion" in context:
            return context["expected_emotion"], 0.4

        return self.FALLBACK_EMOTION, 0.0

    def _compute_urgency(
        self,
        modality_results: List[ModalityStatus],
        fused_result: Dict[str, Any],
    ) -> float:
        """Compute urgency level from all modalities.

        Args:
            modality_results: Per-modality processing results.
            fused_result: Fused result.

        Returns:
            Urgency score in [0, 1].
        """
        urgency = 0.0

        for r in modality_results:
            if r.modality == "speech" and r.success and r.result:
                vocal_emotion = r.result.get("vocal_emotion")
                if vocal_emotion:
                    urgency = max(urgency, vocal_emotion.stress_level * 0.6)

        # Intent-based urgency
        intent = fused_result.get("intent", "")
        high_urgency_intents = {"emergency", "help", "pain", "stop"}
        if intent in high_urgency_intents:
            urgency = max(urgency, 0.8)

        return float(np.clip(urgency, 0.0, 1.0))

    def _generate_suggestions(
        self,
        modality_results: List[ModalityStatus],
        fused_result: Dict[str, Any],
    ) -> List[str]:
        """Generate response suggestions based on fused result.

        Args:
            modality_results: Per-modality processing results.
            fused_result: Fused result.

        Returns:
            List of suggestion strings.
        """
        suggestions: List[str] = []
        intent = fused_result.get("intent", "")

        intent_suggestions = {
            "greeting": ["Hello! How can I help you?", "Hi there!"],
            "farewell": ["Goodbye!", "See you later!"],
            "request": ["Sure, let me help with that.", "I understand your request."],
            "question": ["Let me find that information.", "Good question!"],
            "emergency": ["I understand this is urgent.", "Let me get help immediately."],
            "help": ["I'm here to help. What do you need?", "How can I assist you?"],
            "pain": ["I understand you're in pain. Let me get help.", "Are you okay?"],
        }

        suggestions.extend(intent_suggestions.get(intent, []))

        # Add clarifying suggestions if confidence is low
        if fused_result.get("overall_confidence", 0.0) < self._confidence_threshold:
            suggestions.append("Could you please repeat that?")
            suggestions.append("I didn't quite understand. Can you try again?")

        return suggestions[:5]  # Max 5 suggestions

    def _classify_intent_from_text(self, text: str) -> Tuple[str, float]:
        """Simple rule-based intent classification.

        In production, replace with IntentClassifier.

        Args:
            text: Input text.

        Returns:
            (intent, confidence) tuple.
        """
        text_lower = text.lower()

        # Emergency intents
        if any(word in text_lower for word in ["emergency", "help me", "urgent", "911", "danger"]):
            return ("emergency", 0.95)

        # Greeting
        if any(word in text_lower for word in ["hello", "hi ", "hey", "greetings", "good morning", "good evening"]):
            return ("greeting", 0.9)

        # Farewell
        if any(word in text_lower for word in ["bye", "goodbye", "see you", "farewell"]):
            return ("farewell", 0.9)

        # Affirmation
        if any(word in text_lower for word in ["yes", "yeah", "sure", "okay", "ok", "correct", "right"]):
            return ("affirmation", 0.85)

        # Negation
        if any(word in text_lower for word in ["no", "not", "don't", "never", "nope", "nah"]):
            return ("negation", 0.85)

        # Questions
        if any(word in text_lower for word in ["what", "where", "when", "why", "how", "who", "which", "?"]):
            return ("question", 0.8)

        # Requests
        if any(word in text_lower for word in ["please", "can you", "could you", "would you", "i need", "i want"]):
            return ("request", 0.75)

        # Commands
        if any(word in text_lower for word in ["do ", "go ", "stop", "wait", "come", "give", "show", "tell"]):
            return ("command", 0.7)

        # Help intent
        if any(word in text_lower for word in ["help", "assist", "support"]):
            return ("help", 0.85)

        # Pain intent
        if any(word in text_lower for word in ["pain", "hurt", "ache", "sick", "injured"]):
            return ("pain", 0.85)

        return ("unknown", 0.1)

    def _get_available_modalities(self, inp: MultimodalInput) -> List[str]:
        """Determine which modalities have data available.

        Args:
            inp: Multimodal input.

        Returns:
            List of modality names with data.
        """
        available = []
        if inp.speech_data is not None:
            available.append("speech")
        if inp.gesture_data is not None:
            available.append("gesture")
        if inp.facial_data is not None:
            available.append("facial")
        if inp.context:
            available.append("context")
        return available

    def _build_modality_embeddings(
        self,
        results: List[ModalityStatus],
        transcription: Optional[TranscriptionResult],
        gestures: Optional[InterpretedSequence],
    ) -> Dict[str, np.ndarray]:
        """Build modality embeddings for fusion.

        Args:
            results: Modality processing results.
            transcription: Transcription result.
            gestures: Gesture interpretation.

        Returns:
            Dict mapping modality name -> embedding array.
        """
        embeddings: Dict[str, np.ndarray] = {}

        for r in results:
            if r.modality == "speech" and transcription:
                # Use last layer of Whisper encoder as embedding
                # Placeholder: use text length as proxy
                text_emb = np.zeros(512)
                embeddings["speech"] = np.random.randn(512).astype(np.float32)

            elif r.modality == "gesture" and gestures:
                embeddings["gesture"] = np.random.randn(128).astype(np.float32)

            elif r.modality == "context" and r.result:
                context = r.result.get("context", {})
                emb = np.zeros(256)
                for i, (k, v) in enumerate(context.items()):
                    if isinstance(v, (int, float)):
                        emb[i % 256] = float(v)
                embeddings["context"] = emb

        return embeddings

    def _empty_result(self, start_time: float, error: Optional[str] = None) -> MultimodalResult:
        """Return empty result when inference fails.

        Args:
            start_time: Processing start time.
            error: Optional error message.

        Returns:
            Empty MultimodalResult with fallback values.
        """
        return MultimodalResult(
            intent=self.FALLBACK_INTENT,
            intent_confidence=0.0,
            emotion=self.FALLBACK_EMOTION,
            emotion_confidence=0.0,
            urgency=0.0,
            overall_confidence=0.0,
            processing_time=time.time() - start_time,
            metadata={"error": error} if error else {},
        )
