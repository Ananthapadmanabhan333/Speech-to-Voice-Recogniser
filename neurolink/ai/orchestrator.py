from __future__ import annotations

import asyncio
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import structlog
import torch

from ai.adaptation_engine.personalization.user_profiler import UserProfile, UserProfiler
from ai.adaptation_engine.rl.reinforcement_learning import AdaptationLearner
from ai.emotion_engine.facial.facial_emotion import FacialEmotionAnalyzer
from ai.emotion_engine.fusion.emotion_fusion import EmotionFusion, FinalEmotionResult
from ai.gesture_engine.classification.gesture_classifier import GestureClassifier
from ai.gesture_engine.detection.hand_detector import HandDetector
from ai.gesture_engine.sequence.sequence_model import SequenceModel
from ai.gesture_engine.tracking.hand_tracker import HandTracker
from ai.intent_engine.classifier.intent_classifier import IntentClassifier, IntentResult
from ai.intent_engine.context.context_manager import Context, ContextManager
from ai.intent_engine.predictor.phrase_predictor import PhrasePredictor, PredictedPhrase
from ai.multimodal_fusion.attention.cross_modal_attention import CrossModalAttention
from ai.multimodal_fusion.embeddings.fusion_embeddings import MultimodalEmbeddingFusion
from ai.multimodal_fusion.inference.multimodal_inference import (
    MultimodalInput,
    MultimodalInferencePipeline,
    MultimodalResult,
)
from ai.recommendation_engine.phrases.phrase_recommender import PhraseRecommender, RecommendedPhrase
from ai.speech_engine.emotion.vocal_emotion import VocalEmotionAnalyzer
from ai.speech_engine.synthesis.synthesizer import SpeechSynthesizer
from ai.speech_engine.transcription.transcriber import SpeechTranscriber

logger = structlog.get_logger(__name__)


class OrchestratorError(Exception):
    """Raised when orchestration fails."""


@dataclass
class OrchestrationMetrics:
    """Metrics for orchestration performance monitoring."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    peak_memory_mb: float = 0.0
    model_load_times: Dict[str, float] = field(default_factory=dict)
    modality_counts: Dict[str, int] = field(default_factory=dict)


@dataclass
class OrchestrationResult:
    """Complete result from the AI orchestrator."""

    multimodal_result: MultimodalResult
    intent_result: IntentResult
    emotion_result: FinalEmotionResult
    context: Context
    predictions: List[PredictedPhrase]
    recommendations: List[RecommendedPhrase]
    processing_time: float
    metrics: OrchestrationMetrics


class AIOrchestrator:
    """Main AI orchestrator that initializes all engines and coordinates
    multimodal processing.

    Manages the complete pipeline:
    1. Gesture processing (detect -> track -> classify -> sequence)
    2. Speech processing (transcribe -> emotion analysis)
    3. Facial emotion recognition
    4. Multimodal fusion with cross-modal attention
    5. Intent classification with context
    6. Phrase prediction and recommendation
    7. Emotion fusion
    8. RL-based adaptation from feedback

    Features:
    - Lazy model loading for memory efficiency
    - Concurrent request handling with asyncio
    - Graceful error handling with fallbacks
    - Metrics collection for monitoring
    - Model lifecycle management (load/unload)
    """

    def __init__(
        self,
        models_dir: Optional[Path] = None,
        enable_gpu: bool = True,
        lazy_loading: bool = True,
        max_concurrent: int = 4,
        model_preference: str = "balanced",
    ):
        """Initialize the AI orchestrator.

        Args:
            models_dir: Directory containing model checkpoints.
            enable_gpu: Enable GPU acceleration.
            lazy_loading: Load models on first use (memory efficient).
            max_concurrent: Maximum concurrent inference requests.
            model_preference: 'speed', 'balanced', or 'accuracy'.
        """
        self._models_dir = models_dir or Path("models")
        self._enable_gpu = enable_gpu
        self._lazy_loading = lazy_loading
        self._model_preference = model_preference

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and enable_gpu else "cpu"
        )

        # Metrics
        self._metrics = OrchestrationMetrics()
        self._request_semaphore = asyncio.Semaphore(max_concurrent)

        # Engine instances (lazy loaded)
        self._hand_detector: Optional[HandDetector] = None
        self._hand_tracker: Optional[HandTracker] = None
        self._gesture_classifier: Optional[GestureClassifier] = None
        self._sequence_model: Optional[SequenceModel] = None
        self._speech_transcriber: Optional[SpeechTranscriber] = None
        self._speech_synthesizer: Optional[SpeechSynthesizer] = None
        self._vocal_emotion_analyzer: Optional[VocalEmotionAnalyzer] = None
        self._facial_emotion_analyzer: Optional[FacialEmotionAnalyzer] = None
        self._fusion_engine: Optional[MultimodalEmbeddingFusion] = None
        self._cross_modal_attention: Optional[CrossModalAttention] = None
        self._inference_pipeline: Optional[MultimodalInferencePipeline] = None
        self._intent_classifier: Optional[IntentClassifier] = None
        self._context_manager: Optional[ContextManager] = None
        self._phrase_predictor: Optional[PhrasePredictor] = None
        self._phrase_recommender: Optional[PhraseRecommender] = None
        self._emotion_fusion: Optional[EmotionFusion] = None
        self._adaptation_learner: Optional[AdaptationLearner] = None
        self._user_profiler: Optional[UserProfiler] = None

        # Engine ready flags
        self._engine_ready: Dict[str, bool] = {}

        logger.info(
            "ai_orchestrator_initialized",
            device=str(self._device),
            lazy_loading=lazy_loading,
            model_preference=model_preference,
            max_concurrent=max_concurrent,
        )

    async def process_multimodal_input(
        self,
        video_frame: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        text_input: Optional[str] = None,
        session_id: str = "default",
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> OrchestrationResult:
        """Main entry point for multimodal processing.

        Processes all available input modalities and returns a comprehensive
        result with intent, emotion, predictions, and recommendations.

        Args:
            video_frame: Video frame for gesture/facial analysis.
            audio: Audio signal for speech processing.
            text_input: Direct text input (bypasses ASR).
            session_id: Conversation session identifier.
            user_id: User identifier for personalization.
            context: Additional context data.

        Returns:
            OrchestrationResult with all analysis results.

        Raises:
            OrchestratorError: If orchestration fails.
        """
        start_time = time.time()
        self._metrics.total_requests += 1

        async with self._request_semaphore:
            try:
                # 1. Ensure engines are loaded
                await self._ensure_engines_ready(video_frame, audio, text_input)

                # 2. Process gestures
                gesture_result = await self._process_gestures(video_frame)

                # 3. Process speech
                transcription_result = None
                vocal_emotion_result = None
                if audio is not None:
                    speech_results = await self._process_speech(audio)
                    transcription_result = speech_results.get("transcription")
                    vocal_emotion_result = speech_results.get("vocal_emotion")

                # 4. Process facial emotion
                facial_emotion_result = None
                if video_frame is not None and self._facial_emotion_analyzer:
                    facial_emotion_result = await asyncio.get_event_loop().run_in_executor(
                        None, self._facial_emotion_analyzer.analyze_emotion, video_frame
                    )

                # 5. Multimodal inference
                multimodal_input = MultimodalInput(
                    gesture_data=gesture_result.get("landmark_sequence") if gesture_result else None,
                    speech_data=audio,
                    facial_data=video_frame,
                    context=context,
                )

                multimodal_result = await self._inference_pipeline.infer(
                    gesture_data=multimodal_input.gesture_data,
                    speech_data=multimodal_input.speech_data,
                    facial_data=multimodal_input.facial_data,
                    context=multimodal_input.context,
                )

                # 6. Intent classification
                intent_result = await self._classify_intent(
                    text=text_input or (transcription_result.text if transcription_result else ""),
                    multimodal_result=multimodal_result,
                    session_id=session_id,
                    context=context,
                )

                # 7. Context management
                ctx = self._context_manager.maintain_conversation_context(
                    session_id=session_id,
                    utterance_text=text_input or multimodal_result.transcription.text if multimodal_result.transcription else "",
                    intent=intent_result.intent,
                    emotion=multimodal_result.emotion,
                    metadata={
                        "modalities_used": multimodal_result.metadata.get("modalities_processed", []),
                        "confidence": multimodal_result.overall_confidence,
                    },
                )

                # 8. Emotion fusion
                emotion_result = self._emotion_fusion.fuse_emotions(
                    facial_result=facial_emotion_result,
                    vocal_result=vocal_emotion_result,
                    context=context,
                    session_id=session_id,
                )

                # 9. Phrase prediction
                predictions = self._phrase_predictor.predict_next_phrase(
                    context=self._get_context_text(ctx),
                    user_id=user_id,
                )

                # 10. Phrase recommendation
                user_profile = None
                if user_id and self._user_profiler:
                    user_profile = self._user_profiler.build_user_profile(user_id)

                recommendations = self._phrase_recommender.recommend_phrases(
                    context=self._get_context_text(ctx),
                    user_profile=user_profile,
                    current_intent=intent_result.intent,
                )

                # 11. RL adaptation feedback (async, non-blocking)
                if user_id and self._adaptation_learner:
                    asyncio.ensure_future(
                        self._adaptation_learner.learn_from_feedback(
                            user_feedback={"intent_accuracy": intent_result.confidence},
                            context={
                                "session_id": session_id,
                                "user_id": user_id,
                                "modalities": multimodal_result.metadata.get("modalities_processed", []),
                            },
                        )
                    )

                # Update metrics
                processing_time = time.time() - start_time
                self._successful_request(processing_time)

                return OrchestrationResult(
                    multimodal_result=multimodal_result,
                    intent_result=intent_result,
                    emotion_result=emotion_result,
                    context=ctx,
                    predictions=predictions,
                    recommendations=recommendations,
                    processing_time=processing_time,
                    metrics=self._metrics,
                )

            except Exception as e:
                self._metrics.failed_requests += 1
                logger.error(
                    "orchestration_failed",
                    error=str(e),
                    traceback=traceback.format_exc(),
                    session_id=session_id,
                )
                raise OrchestratorError(f"Orchestration failed: {e}") from e

    async def get_speech_synthesis(
        self, text: str, voice_id: str = "default_female", emotion: Optional[str] = None
    ) -> np.ndarray:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize.
            voice_id: Voice profile ID.
            emotion: Emotion for prosody.

        Returns:
            Audio array.
        """
        await self._ensure_engine("speech_synthesizer")
        result = self._speech_synthesizer.synthesize(text, voice_id=voice_id, emotion=emotion)
        return result.audio

    async def provide_feedback(
        self,
        session_id: str,
        user_id: str,
        feedback: Dict[str, Any],
    ) -> None:
        """Provide feedback for RL adaptation.

        Args:
            session_id: Session identifier.
            user_id: User identifier.
            feedback: Feedback data for adaptation.
        """
        if self._adaptation_learner:
            await self._adaptation_learner.learn_from_feedback(
                user_feedback=feedback,
                context={"session_id": session_id, "user_id": user_id},
            )

    def get_metrics(self) -> OrchestrationMetrics:
        """Get current orchestration metrics.

        Returns:
            Current metrics snapshot.
        """
        return self._metrics

    async def shutdown(self) -> None:
        """Gracefully shut down all engines and release resources."""
        logger.info("orchestrator_shutting_down")

        # Persist context
        if self._context_manager:
            self._context_manager.persist()

        # Release model resources
        for engine_name in ["hand_detector", "hand_tracker", "gesture_classifier",
                            "speech_transcriber", "facial_emotion_analyzer"]:
            engine = getattr(self, f"_{engine_name}", None)
            if engine and hasattr(engine, "close"):
                try:
                    engine.close()
                except Exception as e:
                    logger.warning("engine_close_failed", engine=engine_name, error=str(e))
                setattr(self, f"_{engine_name}", None)
                self._engine_ready[engine_name] = False

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("orchestrator_shutdown_complete")

    async def _ensure_engines_ready(
        self,
        video_frame: Optional[np.ndarray] = None,
        audio: Optional[np.ndarray] = None,
        text_input: Optional[str] = None,
    ) -> None:
        """Ensure required engines are loaded based on input.

        Lazy-loads engines on first use.

        Args:
            video_frame: Indicates need for vision engines.
            audio: Indicates need for audio engines.
            text_input: Indicates need for text engines.
        """
        tasks = []

        if video_frame is not None:
            if not self._engine_ready.get("hand_detector"):
                tasks.append(self._ensure_engine("hand_detector"))
            if not self._engine_ready.get("hand_tracker"):
                tasks.append(self._ensure_engine("hand_tracker"))
            if not self._engine_ready.get("gesture_classifier"):
                tasks.append(self._ensure_engine("gesture_classifier"))
            if not self._engine_ready.get("facial_emotion_analyzer"):
                tasks.append(self._ensure_engine("facial_emotion_analyzer"))

        if audio is not None:
            if not self._engine_ready.get("speech_transcriber"):
                tasks.append(self._ensure_engine("speech_transcriber"))
            if not self._engine_ready.get("vocal_emotion_analyzer"):
                tasks.append(self._ensure_engine("vocal_emotion_analyzer"))

        if text_input is not None or audio is not None:
            if not self._engine_ready.get("intent_classifier"):
                tasks.append(self._ensure_engine("intent_classifier"))

        # Always ensure these
        for engine_name in ["context_manager", "emotion_fusion", "inference_pipeline",
                            "phrase_predictor", "phrase_recommender"]:
            if not self._engine_ready.get(engine_name):
                tasks.append(self._ensure_engine(engine_name))

        if tasks:
            await asyncio.gather(*tasks)

    async def _ensure_engine(self, engine_name: str) -> None:
        """Lazy-load a single engine.

        Args:
            engine_name: Engine attribute name (without leading _).
        """
        if self._engine_ready.get(engine_name):
            return

        load_start = time.time()
        logger.info("loading_engine", engine=engine_name)

        try:
            if engine_name == "hand_detector":
                self._hand_detector = HandDetector(
                    max_hands=2, min_detection_confidence=0.7, skip_rate=0
                )
            elif engine_name == "hand_tracker":
                self._hand_tracker = HandTracker(max_tracks=4)
            elif engine_name == "gesture_classifier":
                self._gesture_classifier = GestureClassifier(
                    device=str(self._device)
                )
            elif engine_name == "sequence_model":
                self._sequence_model = SequenceModel()
            elif engine_name == "speech_transcriber":
                self._speech_transcriber = SpeechTranscriber(
                    model_name="base",
                    device=str(self._device),
                )
            elif engine_name == "speech_synthesizer":
                self._speech_synthesizer = SpeechSynthesizer(
                    device=str(self._device),
                )
            elif engine_name == "vocal_emotion_analyzer":
                self._vocal_emotion_analyzer = VocalEmotionAnalyzer(
                    device=str(self._device),
                )
            elif engine_name == "facial_emotion_analyzer":
                self._facial_emotion_analyzer = FacialEmotionAnalyzer(
                    device=str(self._device),
                )
            elif engine_name == "fusion_engine":
                self._fusion_engine = MultimodalEmbeddingFusion(
                    device=str(self._device),
                )
            elif engine_name == "cross_modal_attention":
                self._cross_modal_attention = CrossModalAttention(
                    modality_dims={"gesture": 128, "speech": 512, "emotion": 128},
                )
            elif engine_name == "inference_pipeline":
                self._inference_pipeline = MultimodalInferencePipeline(
                    gesture_classifier=self._gesture_classifier,
                    sequence_model=self._sequence_model,
                    speech_transcriber=self._speech_transcriber,
                    vocal_emotion_analyzer=self._vocal_emotion_analyzer,
                    fusion_engine=self._fusion_engine,
                    cross_modal_attention=self._cross_modal_attention,
                )
            elif engine_name == "intent_classifier":
                self._intent_classifier = IntentClassifier(
                    device=str(self._device),
                )
            elif engine_name == "context_manager":
                self._context_manager = ContextManager()
            elif engine_name == "phrase_predictor":
                self._phrase_predictor = PhrasePredictor(
                    device=str(self._device),
                )
            elif engine_name == "phrase_recommender":
                self._phrase_recommender = PhraseRecommender()
            elif engine_name == "emotion_fusion":
                self._emotion_fusion = EmotionFusion()
            elif engine_name == "adaptation_learner":
                self._adaptation_learner = AdaptationLearner()
            elif engine_name == "user_profiler":
                self._user_profiler = UserProfiler()
            else:
                logger.warning("unknown_engine", engine=engine_name)
                return

            self._engine_ready[engine_name] = True
            load_time = time.time() - load_start
            self._metrics.model_load_times[engine_name] = load_time
            logger.info("engine_loaded", engine=engine_name, load_time=load_time)

        except Exception as e:
            logger.error("engine_load_failed", engine=engine_name, error=str(e))
            self._engine_ready[engine_name] = False

    async def _process_gestures(
        self, video_frame: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Process gesture pipeline: detect -> track -> classify -> sequence.

        Args:
            video_frame: Input video frame.

        Returns:
            Dict with landmark_sequence, gesture_result, and tracked_hands.
        """
        if video_frame is None:
            return {}

        result: Dict[str, Any] = {}

        # Detect
        if self._hand_detector:
            detections = self._hand_detector.detect_hands(video_frame)
            result["detections"] = detections

            # Track
            if self._hand_tracker and detections:
                tracked = self._hand_tracker.update(detections)
                result["tracked_hands"] = tracked

                # Classify
                if self._gesture_classifier and tracked:
                    # Use landmark sequence from most confident track
                    best_track = max(tracked, key=lambda t: t.confidence)
                    if best_track.history and len(best_track.history) >= 5:
                        landmark_seq = np.array(list(best_track.history))
                        gesture_result = self._gesture_classifier.classify_gesture(landmark_seq)
                        result["gesture_result"] = gesture_result
                        result["landmark_sequence"] = landmark_seq

        return result

    async def _process_speech(
        self, audio: np.ndarray
    ) -> Dict[str, Any]:
        """Process speech pipeline: transcribe -> emotion analysis.

        Args:
            audio: Audio signal.

        Returns:
            Dict with transcription and vocal_emotion.
        """
        result: Dict[str, Any] = {}

        if self._speech_transcriber:
            loop = asyncio.get_event_loop()
            transcription = await loop.run_in_executor(
                None, self._speech_transcriber.transcribe, audio
            )
            result["transcription"] = transcription

        if self._vocal_emotion_analyzer:
            loop = asyncio.get_event_loop()
            vocal_emotion = await loop.run_in_executor(
                None, self._vocal_emotion_analyzer.analyze_emotion, audio
            )
            result["vocal_emotion"] = vocal_emotion

        return result

    async def _classify_intent(
        self,
        text: str,
        multimodal_result: MultimodalResult,
        session_id: str,
        context: Optional[Dict[str, Any]],
    ) -> IntentResult:
        """Classify intent from text and multimodal context.

        Args:
            text: Input text.
            multimodal_result: Multimodal inference result.
            session_id: Session identifier.
            context: Additional context.

        Returns:
            IntentResult.
        """
        if not self._intent_classifier:
            return IntentResult(
                intent="unknown",
                confidence=0.0,
                intent_probs={},
            )

        # Build context for intent classification
        intent_context = {
            "session_id": session_id,
            "previous_intent": None,
            "emotion": multimodal_result.emotion,
        }

        if context:
            intent_context.update(context)

        # Get previous intent from context manager
        session_context = self._context_manager.get_context(session_id)
        if session_context and session_context.active_intents:
            intent_context["previous_intent"] = session_context.active_intents[-1]

        return self._intent_classifier.classify_intent(
            text=text,
            context=intent_context,
            multimodal_input={
                "emotion": multimodal_result.emotion,
                "urgency": multimodal_result.urgency,
            },
        )

    def _get_context_text(self, ctx: Context) -> str:
        """Get recent context as text for prediction/recommendation.

        Args:
            ctx: Conversation context.

        Returns:
            Recent utterance text.
        """
        if ctx.short_term:
            return " ".join(u.text for u in ctx.short_term[-3:])
        return ""

    def _successful_request(self, processing_time: float) -> None:
        """Update metrics after successful request.

        Args:
            processing_time: Request processing time.
        """
        self._metrics.successful_requests += 1
        self._metrics.total_processing_time += processing_time
        self._metrics.avg_processing_time = (
            self._metrics.total_processing_time / self._metrics.successful_requests
        )
