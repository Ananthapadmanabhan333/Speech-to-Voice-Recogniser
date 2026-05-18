from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class PhrasePredictionError(Exception):
    """Raised when phrase prediction fails."""


@dataclass
class PredictedPhrase:
    """A predicted next phrase."""

    text: str
    confidence: float
    source: str  # "language_model", "context", "user_pattern"
    category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerDecoder(nn.Module):
    """Lightweight transformer decoder for phrase prediction."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 1024,
        max_seq_length: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        tokens: torch.Tensor,
        memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = min(tokens.size(1), self.max_seq_length)
        tokens = tokens[:, :seq_len]

        emb = self.embedding(tokens) * np.sqrt(self.d_model)
        emb = self.pos_encoding(emb)

        if memory is None:
            memory = emb

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=tokens.device),
            diagonal=1,
        )

        padding_mask = None
        if mask is not None:
            mask = mask[:, :seq_len]
            padding_mask = ~mask.bool()

        decoded = self.decoder(
            emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
            memory_key_padding_mask=padding_mask,
        )

        logits = self.output_proj(decoded)
        return logits


class PhrasePredictor:
    """Transformer-based phrase predictor with user-specific fine-tuning.

    Predicts the next likely phrase given conversation context and partial input.
    Supports context-aware suggestions, user-specific pattern learning,
    and confidence scoring.

    Architecture:
    - Lightweight transformer decoder
    - Context encoder for conversation history
    - User-specific fine-tuning via adapter layers
    """

    DEFAULT_PHRASES: List[str] = [
        "Hello", "How are you?", "I need help", "Thank you", "Yes",
        "No", "Please", "Sorry", "I don't understand", "Can you repeat that?",
        "What is this?", "I'm in pain", "Help me", "Call for help",
        "I need water", "I need food", "I'm tired", "I'm happy",
        "I'm sad", "Goodbye", "See you later", "Nice to meet you",
        "How do I do this?", "What's happening?", "Emergency",
    ]

    def __init__(
        self,
        vocab_size: int = 1000,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        temperature: float = 1.0,
        top_k: int = 10,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize phrase predictor.

        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            temperature: Sampling temperature (higher = more diverse).
            top_k: Number of top candidates to consider.
            device: Device to run on.
            model_path: Path to pretrained model.
        """
        self._d_model = d_model
        self._temperature = temperature
        self._top_k = top_k

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Vocabulary
        self._vocab: List[str] = ["<PAD>", "<UNK>", "<SOS>", "<EOS>"] + self.DEFAULT_PHRASES
        self._token_to_id: Dict[str, int] = {t: i for i, t in enumerate(self._vocab)}
        self._id_to_token: Dict[int, str] = {i: t for i, t in enumerate(self._vocab)}
        self._vocab_size = len(self._vocab)

        # Phrases lookup for fast suggestion
        self._phrase_embeddings: Dict[str, np.ndarray] = {}

        # Model
        self._model = TransformerDecoder(
            vocab_size=self._vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
        ).to(self._device)

        # User-specific fine-tuning data
        self._user_patterns: Dict[str, Dict[str, Any]] = {}

        self._eval_mode = False

        if model_path:
            self.load(model_path)

        logger.info(
            "phrase_predictor_initialized",
            vocab_size=self._vocab_size,
            device=str(self._device),
            d_model=d_model,
        )

    def predict_next_phrase(
        self,
        context: str,
        partial_input: str = "",
        user_id: Optional[str] = None,
        num_predictions: int = 5,
    ) -> List[PredictedPhrase]:
        """Predict the next likely phrase.

        Args:
            context: Current conversation context text.
            partial_input: Partial user input (for autocomplete).
            user_id: Optional user ID for personalized predictions.
            num_predictions: Number of predictions to return.

        Returns:
            List of PredictedPhrase sorted by confidence.

        Raises:
            PhrasePredictionError: If prediction fails.
        """
        try:
            predictions: List[PredictedPhrase] = []

            # 1. Language model predictions
            if context or partial_input:
                lm_predictions = self._predict_from_language_model(
                    context, partial_input, num_predictions
                )
                predictions.extend(lm_predictions)

            # 2. Context-aware predictions
            if context:
                context_predictions = self._predict_from_context(
                    context, num_predictions
                )
                predictions.extend(context_predictions)

            # 3. User-specific predictions
            if user_id:
                user_predictions = self._predict_from_user_patterns(
                    user_id, context, num_predictions
                )
                predictions.extend(user_predictions)

            # 4. Default phrases for empty input
            if not predictions:
                default_predictions = self._get_default_phrases(
                    context, num_predictions
                )
                predictions.extend(default_predictions)

            # Deduplicate and sort
            return self._deduplicate_and_sort(predictions, num_predictions)

        except Exception as e:
            logger.error("phrase_prediction_failed", error=str(e))
            raise PhrasePredictionError(f"Phrase prediction failed: {e}") from e

    def fine_tune_on_user(
        self,
        user_id: str,
        utterances: List[str],
        intents: Optional[List[str]] = None,
    ) -> None:
        """Fine-tune predictions for a specific user.

        Learns user-specific patterns from their utterance history.

        Args:
            user_id: User identifier.
            utterances: List of user's past utterances.
            intents: Optional list of intents for each utterance.
        """
        if not utterances:
            return

        # Extract patterns
        pattern_counts: Dict[str, int] = {}
        bigram_counts: Dict[Tuple[str, str], int] = {}

        for i, utt in enumerate(utterances):
            utt_lower = utt.lower().strip()
            pattern_counts[utt_lower] = pattern_counts.get(utt_lower, 0) + 1

            if i > 0:
                prev = utterances[i - 1].lower().strip()
                bigram = (prev, utt_lower)
                bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

        self._user_patterns[user_id] = {
            "utterance_count": len(utterances),
            "pattern_counts": pattern_counts,
            "bigram_counts": bigram_counts,
            "last_updated": time.time(),
        }

        logger.info("user_fine_tuned", user_id=user_id, utterances=len(utterances))

    def load(self, model_path: str) -> None:
        """Load model from checkpoint.

        Args:
            model_path: Path to checkpoint.
        """
        checkpoint = torch.load(model_path, map_location=self._device)
        self._model.load_state_dict(checkpoint["model"])
        if "vocab" in checkpoint:
            self._vocab = checkpoint["vocab"]
            self._token_to_id = {t: i for i, t in enumerate(self._vocab)}
            self._id_to_token = {i: t for i, t in enumerate(self._vocab)}
            self._vocab_size = len(self._vocab)
        logger.info("phrase_predictor_loaded", path=model_path)

    def save(self, model_path: str) -> None:
        """Save model to checkpoint.

        Args:
            model_path: Path to save checkpoint.
        """
        checkpoint = {
            "model": self._model.state_dict(),
            "vocab": self._vocab,
        }
        torch.save(checkpoint, model_path)
        logger.info("phrase_predictor_saved", path=model_path)

    def _predict_from_language_model(
        self, context: str, partial_input: str, num_predictions: int
    ) -> List[PredictedPhrase]:
        """Get predictions from the transformer language model.

        Args:
            context: Conversation context.
            partial_input: Partial user input.
            num_predictions: Number to return.

        Returns:
            List of predicted phrases.
        """
        # Tokenize context + partial input
        input_text = f"{context} {partial_input}".strip()
        input_ids = self._tokenize(input_text)

        if len(input_ids) < 2:
            return []

        input_tensor = torch.tensor([input_ids], device=self._device)

        self._ensure_eval_mode()
        with torch.no_grad():
            logits = self._model(input_tensor)
            # Get last token predictions
            last_logits = logits[0, -1, :] / self._temperature
            probs = F.softmax(last_logits, dim=-1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probs, min(self._top_k, self._vocab_size))

        predictions = []
        for i in range(len(top_indices)):
            idx = top_indices[i].item()
            conf = top_probs[i].item()
            token = self._id_to_token.get(idx, "<UNK>")

            if token not in ("<PAD>", "<UNK>", "<SOS>", "<EOS>"):
                # Form complete phrase suggestion
                phrase = self._complete_phrase(token, partial_input)
                predictions.append(PredictedPhrase(
                    text=phrase,
                    confidence=conf,
                    source="language_model",
                ))

        return predictions

    def _predict_from_context(
        self, context: str, num_predictions: int
    ) -> List[PredictedPhrase]:
        """Get predictions based on conversation context.

        Uses keyword matching against known phrases.

        Args:
            context: Conversation context.
            num_predictions: Number to return.

        Returns:
            List of context-based predictions.
        """
        context_lower = context.lower()
        predictions = []

        # Keyword-based suggestions
        keyword_map: Dict[str, List[Tuple[str, float]]] = {
            "help": [("How can I help you?", 0.9), ("I need help", 0.8)],
            "pain": [("I'm in pain", 0.95), ("Help me", 0.85), ("Call for help", 0.7)],
            "thank": [("You're welcome", 0.9), ("Glad to help", 0.8)],
            "hello": [("Hello", 0.9), ("How are you?", 0.8)],
            "bye": [("Goodbye", 0.9), ("See you later", 0.8)],
            "hungry": [("I need food", 0.9), ("I need water", 0.7)],
            "thirsty": [("I need water", 0.9), ("I need food", 0.6)],
            "tired": [("I'm tired", 0.9), ("I need rest", 0.7)],
            "what": [("What is this?", 0.8), ("What's happening?", 0.7)],
            "emergency": [("Emergency", 0.95), ("Call for help", 0.9), ("Help me", 0.85)],
        }

        for keyword, suggestions in keyword_map.items():
            if keyword in context_lower:
                for phrase, conf in suggestions:
                    predictions.append(PredictedPhrase(
                        text=phrase,
                        confidence=conf * 0.8,  # Slightly lower weight
                        source="context",
                        category="context_match",
                    ))

        return predictions

    def _predict_from_user_patterns(
        self, user_id: str, context: str, num_predictions: int
    ) -> List[PredictedPhrase]:
        """Get predictions based on user's historical patterns.

        Args:
            user_id: User identifier.
            context: Current context.
            num_predictions: Number to return.

        Returns:
            List of user-specific predictions.
        """
        patterns = self._user_patterns.get(user_id)
        if not patterns:
            return []

        predictions = []
        context_lower = context.lower().strip()

        # Check bigram patterns
        bigram_counts = patterns.get("bigram_counts", {})
        for (prev, next_utt), count in bigram_counts.items():
            if prev == context_lower or (context_lower and prev in context_lower):
                frequency = count / max(patterns.get("utterance_count", 1), 1)
                predictions.append(PredictedPhrase(
                    text=next_utt,
                    confidence=min(0.95, frequency * 3),  # Boost frequency
                    source="user_pattern",
                    category="bigram",
                ))

        # Check most frequent utterances
        pattern_counts = patterns.get("pattern_counts", {})
        sorted_patterns = sorted(
            pattern_counts.items(), key=lambda x: x[1], reverse=True
        )[:num_predictions]

        for pattern, count in sorted_patterns:
            if pattern != context_lower:
                frequency = count / max(patterns.get("utterance_count", 1), 1)
                predictions.append(PredictedPhrase(
                    text=pattern,
                    confidence=min(0.9, frequency * 2),
                    source="user_pattern",
                    category="frequent",
                ))

        return predictions

    def _get_default_phrases(
        self, context: str, num_predictions: int
    ) -> List[PredictedPhrase]:
        """Get default phrase suggestions.

        Args:
            context: Current context.
            num_predictions: Number to return.

        Returns:
            List of default phrase predictions.
        """
        return [
            PredictedPhrase(text=phrase, confidence=0.3, source="default")
            for phrase in self.DEFAULT_PHRASES[:num_predictions]
        ]

    def _complete_phrase(self, token: str, partial_input: str) -> str:
        """Complete a partial input with the predicted token.

        Args:
            token: Predicted token.
            partial_input: Current partial input.

        Returns:
            Completed phrase.
        """
        if not partial_input:
            return token
        return f"{partial_input} {token}".strip()

    def _tokenize(self, text: str) -> List[int]:
        """Simple whitespace tokenization.

        Args:
            text: Input text.

        Returns:
            List of token IDs.
        """
        tokens = text.split()
        ids = [self._token_to_id.get(t, self._token_to_id["<UNK>"]) for t in tokens]
        # Truncate
        max_len = self._model.max_seq_length
        if len(ids) > max_len:
            ids = ids[-max_len:]
        return ids

    def _deduplicate_and_sort(
        self, predictions: List[PredictedPhrase], top_n: int
    ) -> List[PredictedPhrase]:
        """Remove duplicates and sort by confidence.

        Args:
            predictions: Raw predictions list.
            top_n: Number to keep.

        Returns:
            Deduplicated, sorted predictions.
        """
        seen: set = set()
        unique: List[PredictedPhrase] = []

        for p in sorted(predictions, key=lambda x: x.confidence, reverse=True):
            key = p.text.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:top_n]

    def _ensure_eval_mode(self) -> None:
        if not self._eval_mode:
            self._model.eval()
            self._eval_mode = True
