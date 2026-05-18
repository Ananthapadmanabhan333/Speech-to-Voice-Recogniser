from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class IntentClassificationError(Exception):
    """Raised when intent classification fails."""


@dataclass
class IntentResult:
    """Result of intent classification."""

    intent: str
    confidence: float
    intent_probs: Dict[str, float]
    context_used: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    timestamp: float = field(default_factory=time.time)


# Standard intent categories
INTENT_CATEGORIES: List[str] = [
    "request",
    "question",
    "command",
    "affirmation",
    "negation",
    "greeting",
    "farewell",
    "emergency",
    "help",
    "pain",
    "thanks",
    "apology",
    "clarification",
    "confirmation",
    "information",
    "suggestion",
    "complaint",
    "opinion",
    "small_talk",
    "unknown",
]


class IntentEncoder(nn.Module):
    """Transformer-based encoder for intent classification."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        max_seq_length: int = 128,
        num_intents: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = self._create_pe(max_seq_length, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_intents),
        )

    def _create_pe(self, max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return nn.Parameter(pe, requires_grad=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = min(x.size(1), self.max_seq_length)
        x = x[:, :seq_len]

        emb = self.embedding(x) * np.sqrt(self.d_model)
        emb = emb + self.pos_encoding[:, :seq_len, :]

        # Transformer encoder
        if mask is not None:
            mask = mask[:, :seq_len]
            # Create padding mask for transformer (True = masked)
            padding_mask = ~mask.bool() if mask is not None else None
        else:
            padding_mask = None

        encoded = self.transformer(emb, src_key_padding_mask=padding_mask)

        # Mean pooling
        if padding_mask is not None:
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            pooled = encoded.sum(dim=1) / (~padding_mask).float().sum(dim=1, keepdim=True).clamp(min=1)
        else:
            pooled = encoded.mean(dim=1)

        logits = self.classifier(pooled)
        return logits


class IntentClassifier:
    """Intent classification with few-shot learning support.

    Classifies multimodal input into predefined intent categories.
    Supports few-shot learning for new intents via prototype-based
    classification. Context-aware classification integrates conversation
    history and user patterns.

    Intent categories:
    - request, question, command, affirmation, negation
    - greeting, farewell, emergency, help, pain
    - thanks, apology, clarification, confirmation
    - information, suggestion, complaint, opinion, small_talk, unknown
    """

    def __init__(
        self,
        intent_categories: Optional[List[str]] = None,
        d_model: int = 256,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.4,
    ):
        """Initialize intent classifier.

        Args:
            intent_categories: List of intent labels. Defaults to INTENT_CATEGORIES.
            d_model: Encoder dimension.
            device: Device to run on.
            model_path: Path to pretrained model.
            confidence_threshold: Minimum confidence for prediction.
        """
        self._intent_categories = intent_categories or INTENT_CATEGORIES.copy()
        self._num_intents = len(self._intent_categories)
        self._confidence_threshold = confidence_threshold

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Intent label mappings
        self._intent_to_id: Dict[str, int] = {
            intent: i for i, intent in enumerate(self._intent_categories)
        }
        self._id_to_intent: Dict[int, str] = {
            i: intent for i, intent in enumerate(self._intent_categories)
        }

        # Few-shot prototypes (intent -> list of embedding vectors)
        self._few_shot_prototypes: Dict[str, List[np.ndarray]] = {}

        # Tokenizer placeholder (in production, use a proper tokenizer)
        self._vocab: Dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self._vocab_size = 2

        # Model
        self._encoder = IntentEncoder(
            vocab_size=self._vocab_size,
            d_model=d_model,
            num_intents=self._num_intents,
        ).to(self._device)

        self._eval_mode = False

        if model_path:
            self.load(model_path)

        logger.info(
            "intent_classifier_initialized",
            num_intents=self._num_intents,
            device=str(self._device),
            intents=self._intent_categories,
        )

    def classify_intent(
        self,
        text: str,
        context: Optional[Dict[str, Any]] = None,
        multimodal_input: Optional[Dict[str, Any]] = None,
    ) -> IntentResult:
        """Classify intent from text and optional multimodal input.

        Args:
            text: Input text to classify.
            context: Conversation context dict.
            multimodal_input: Dict with additional modality data.

        Returns:
            IntentResult with predicted intent and confidence.

        Raises:
            IntentClassificationError: If classification fails.
        """
        if not text or not text.strip():
            raise ValueError("Empty text for intent classification")

        start_time = time.time()

        try:
            # Tokenize (simple placeholder - use proper tokenizer in production)
            input_ids = self._tokenize(text)

            # Check few-shot prototypes first
            few_shot_intent, few_shot_conf = self._classify_few_shot(text)
            if few_shot_intent and few_shot_conf >= self._confidence_threshold:
                return IntentResult(
                    intent=few_shot_intent,
                    confidence=few_shot_conf,
                    intent_probs={few_shot_intent: few_shot_conf},
                    context_used=context or {},
                    processing_time=time.time() - start_time,
                )

            # Run through model
            self._ensure_eval_mode()
            with torch.no_grad():
                input_tensor = torch.tensor([input_ids], device=self._device)
                logits = self._encoder(input_tensor)
                probs = F.softmax(logits, dim=-1).cpu().numpy()[0]

            # Get top prediction
            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            intent = self._id_to_intent.get(pred_idx, "unknown")

            # Build probability dict
            intent_probs = {
                self._id_to_intent[i]: float(probs[i])
                for i in range(self._num_intents)
            }

            # Apply context-aware adjustment
            if context:
                intent, confidence = self._adjust_with_context(intent, confidence, context)

            return IntentResult(
                intent=intent,
                confidence=confidence,
                intent_probs=intent_probs,
                context_used=context or {},
                processing_time=time.time() - start_time,
            )

        except Exception as e:
            logger.error("intent_classification_failed", error=str(e))
            raise IntentClassificationError(f"Intent classification failed: {e}") from e

    def add_few_shot_example(self, intent: str, text: str) -> None:
        """Add a few-shot example for a new or existing intent.

        Args:
            intent: Intent label.
            text: Example text for this intent.

        Raises:
            ValueError: If intent is invalid or text is empty.
        """
        if not text or not text.strip():
            raise ValueError("Empty example text")

        # Add to vocabulary
        for token in text.lower().split():
            if token not in self._vocab:
                self._vocab[token] = self._vocab_size
                self._vocab_size += 1

        # Compute embedding
        input_ids = self._tokenize(text)
        self._ensure_eval_mode()
        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], device=self._device)
            # Get embedding from encoder (before classifier)
            mask = torch.ones(1, len(input_ids), dtype=torch.bool, device=self._device)
            emb = self._encoder.embedding(input_tensor) * np.sqrt(self._encoder.d_model)
            emb = emb + self._encoder.pos_encoding[:, :len(input_ids), :]
            padding_mask = ~mask.bool()
            encoded = self._encoder.transformer(emb, src_key_padding_mask=padding_mask)
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            embedding = encoded.sum(dim=1).cpu().numpy()[0]

        if intent not in self._few_shot_prototypes:
            self._few_shot_prototypes[intent] = []
        self._few_shot_prototypes[intent].append(embedding)

        logger.info("few_shot_example_added", intent=intent, text=text)

    def add_new_intent(self, intent: str, examples: List[str]) -> None:
        """Add a new intent category with few-shot examples.

        Args:
            intent: New intent label.
            examples: List of example texts.

        Raises:
            ValueError: If intent already exists or examples are empty.
        """
        if intent in self._intent_to_id:
            raise ValueError(f"Intent '{intent}' already exists")

        if not examples:
            raise ValueError("Must provide at least one example")

        self._intent_categories.append(intent)
        intent_id = len(self._intent_categories) - 1
        self._intent_to_id[intent] = intent_id
        self._id_to_intent[intent_id] = intent
        self._num_intents = len(self._intent_categories)

        # Update classifier output dimension
        old_classifier = self._encoder.classifier
        in_features = old_classifier[0].in_features
        new_classifier = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, self._num_intents),
        ).to(self._device)
        # Copy weights for existing classes
        with torch.no_grad():
            new_classifier[0].weight[:in_features // 2, :] = old_classifier[0].weight
            new_classifier[0].bias[:in_features // 2] = old_classifier[0].bias
            new_classifier[3].weight[:self._num_intents - 1, :] = old_classifier[3].weight
            new_classifier[3].bias[:self._num_intents - 1] = old_classifier[3].bias
        self._encoder.classifier = new_classifier

        # Add few-shot examples
        for example in examples:
            self.add_few_shot_example(intent, example)

        logger.info("new_intent_added", intent=intent, num_examples=len(examples))

    def load(self, model_path: str) -> None:
        """Load model weights.

        Args:
            model_path: Path to checkpoint.
        """
        checkpoint = torch.load(model_path, map_location=self._device)
        self._encoder.load_state_dict(checkpoint["encoder"])
        if "intent_categories" in checkpoint:
            self._intent_categories = checkpoint["intent_categories"]
            self._intent_to_id = {intent: i for i, intent in enumerate(self._intent_categories)}
            self._id_to_intent = {i: intent for i, intent in enumerate(self._intent_categories)}
        logger.info("intent_classifier_loaded", path=model_path)

    def save(self, model_path: str) -> None:
        """Save model weights.

        Args:
            model_path: Path to save checkpoint.
        """
        checkpoint = {
            "encoder": self._encoder.state_dict(),
            "intent_categories": self._intent_categories,
        }
        torch.save(checkpoint, model_path)
        logger.info("intent_classifier_saved", path=model_path)

    def _tokenize(self, text: str) -> List[int]:
        """Tokenize text to token IDs.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        tokens = text.lower().split()
        ids = [self._vocab.get(t, self._vocab.get("<UNK>", 1)) for t in tokens]

        # Add tokens to vocab if not seen
        for t in tokens:
            if t not in self._vocab:
                self._vocab[t] = self._vocab_size
                self._vocab_size += 1

        # Truncate to max sequence length
        max_len = self._encoder.max_seq_length
        if len(ids) > max_len:
            ids = ids[-max_len:]

        return ids

    def _classify_few_shot(self, text: str) -> Tuple[Optional[str], float]:
        """Classify using few-shot prototypes.

        Args:
            text: Input text.

        Returns:
            (intent, confidence) or (None, 0.0) if no prototypes.
        """
        if not self._few_shot_prototypes:
            return (None, 0.0)

        input_ids = self._tokenize(text)
        self._ensure_eval_mode()

        with torch.no_grad():
            input_tensor = torch.tensor([input_ids], device=self._device)
            mask = torch.ones(1, len(input_ids), dtype=torch.bool, device=self._device)
            emb = self._encoder.embedding(input_tensor) * np.sqrt(self._encoder.d_model)
            emb = emb + self._encoder.pos_encoding[:, :len(input_ids), :]
            padding_mask = ~mask.bool()
            encoded = self._encoder.transformer(emb, src_key_padding_mask=padding_mask)
            encoded = encoded.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            query_emb = encoded.sum(dim=1).cpu().numpy()[0]

        best_intent = None
        best_score = 0.0

        for intent, prototypes in self._few_shot_prototypes.items():
            for proto in prototypes:
                sim = np.dot(query_emb, proto) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(proto) + 1e-8
                )
                score = max(0.0, float(sim))
                if score > best_score:
                    best_score = score
                    best_intent = intent

        return (best_intent, best_score)

    def _adjust_with_context(
        self, intent: str, confidence: float, context: Dict[str, Any]
    ) -> Tuple[str, float]:
        """Adjust intent prediction based on conversation context.

        Args:
            intent: Currently predicted intent.
            confidence: Prediction confidence.
            context: Conversation context.

        Returns:
            (adjusted_intent, adjusted_confidence).
        """
        # If previous intent was a question and current is "unknown",
        # it's likely an answer
        prev_intent = context.get("previous_intent")
        if prev_intent == "question" and intent in ("unknown", "affirmation", "negation"):
            return ("answer", max(confidence, 0.6))

        # Emergency escalation
        if intent == "emergency":
            return (intent, max(confidence, 0.9))

        # Greeting follow-up: if greeting was recent, current is likely request
        if prev_intent == "greeting" and intent == "unknown":
            return ("request", 0.5)

        return (intent, confidence)

    def _ensure_eval_mode(self) -> None:
        if not self._eval_mode:
            self._encoder.eval()
            self._eval_mode = True
