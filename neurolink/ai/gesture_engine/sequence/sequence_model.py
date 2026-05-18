from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = structlog.get_logger(__name__)


class SequenceModelingError(Exception):
    """Raised when sequence modeling fails."""


@dataclass
class InterpretedSequence:
    """Result of gesture sequence interpretation."""

    tokens: List[str]
    sentence: str
    token_confidences: List[float]
    sequence_confidence: float
    alignment: List[Tuple[int, int]]  # start, end frame indices
    is_complete: bool
    metadata: Dict = field(default_factory=dict)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block with pre-norm."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=mask)
        x = x + self.dropout(attn_out)
        # Feedforward with pre-norm
        x_norm = self.norm2(x)
        ff_out = self.feedforward(x_norm)
        x = x + self.dropout(ff_out)
        return x


class GestureSequenceTransformer(nn.Module):
    """Transformer encoder for gesture sequence modeling.

    Encodes variable-length gesture sequences with temporal attention
    and produces interpreted token outputs.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 100,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embedding = nn.Linear(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (batch, seq_len, vocab_size)
        seq_len = x.size(1)
        if seq_len > self.max_seq_length:
            x = x[:, -self.max_seq_length:, :]
            if mask is not None:
                mask = mask[:, -self.max_seq_length:]

        x = self.embedding(x) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x, mask)

        logits = self.output_proj(x)
        return logits


class CRFLayer(nn.Module):
    """Conditional Random Field for sequence smoothing.

    Implements linear-chain CRF for optimal sequence decoding.
    """

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # Transition scores (num_tags, num_tags) where transitions[i, j] = score of i->j
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # Constrain transitions from start to first tag
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        # Constrain transitions from last tag to end
        self.end_transitions = nn.Parameter(torch.randn(num_tags))

    def forward(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute CRF log-likelihood.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len) - 1 for valid, 0 for padded

        Returns:
            Log-likelihood score.
        """
        return self._score(emissions, mask)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Find optimal tag sequence using Viterbi decoding.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)

        Returns:
            (batch, seq_len) optimal tag indices.
        """
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # Initialize
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0, :]  # (batch, num_tags)
        backpointers = torch.zeros(batch_size, seq_len, num_tags, dtype=torch.long, device=device)

        for t in range(1, seq_len):
            # score: (batch, num_tags, 1) + transitions: (1, num_tags, num_tags) + emissions: (batch, 1, num_tags)
            broadcast_score = score.unsqueeze(2)  # (batch, num_tags, 1)
            broadcast_trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            broadcast_emissions = emissions[:, t, :].unsqueeze(1)  # (batch, 1, num_tags)
            next_score = broadcast_score + broadcast_trans + broadcast_emissions
            # next_score: (batch, num_tags, num_tags)
            best_scores, best_tags = next_score.max(dim=1)
            score = best_scores
            backpointers[:, t, :] = best_tags

            # Apply mask: where mask[:, t] == 0, keep previous score
            mask_t = mask[:, t].unsqueeze(1)  # (batch, 1)
            score = torch.where(mask_t.bool(), score, score)  # handled by mask

        # Add end transitions
        score = score + self.end_transitions.unsqueeze(0)

        # Backtrack
        best_tags_list = []
        _, best_last_tag = score.max(dim=1)
        best_tags_list.append(best_last_tag.unsqueeze(1))

        for t in range(seq_len - 1, 0, -1):
            prev_best = backpointers[:, t, best_last_tag]
            best_tags_list.append(prev_best.unsqueeze(1))
            best_last_tag = prev_best

        best_tags = torch.cat(best_tags_list[::-1], dim=1)
        return best_tags

    def _score(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute log partition function."""
        batch_size, seq_len, num_tags = emissions.shape
        device = emissions.device

        # Start transitions
        score = self.start_transitions + emissions[:, 0]

        for t in range(1, seq_len):
            # Expand score for broadcasting
            broadcast_score = score.unsqueeze(2)  # (batch, num_tags, 1)
            broadcast_trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            broadcast_emissions = emissions[:, t, :].unsqueeze(1)  # (batch, 1, num_tags)
            next_score = broadcast_score + broadcast_trans + broadcast_emissions
            # Log-sum-exp over previous tags
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[:, t].unsqueeze(1).bool(), next_score, score)

        # End transitions
        score = score + self.end_transitions
        # Log-sum-exp
        final_score = torch.logsumexp(score, dim=1)
        return final_score


class SequenceModel:
    """Temporal gesture sequence modeling using Transformer encoder with CRF smoothing.

    Takes a sequence of gesture classifications and produces an interpreted
    sequence (e.g., words or sentences). Handles varying-length sequences,
    applies temporal attention, and performs sequence-level smoothing.

    The model learns contextual dependencies between consecutive gestures
    and can correct isolated misclassifications using the CRF layer.
    """

    PAD_TOKEN_ID: int = 0
    SOS_TOKEN_ID: int = 1
    EOS_TOKEN_ID: int = 2
    UNK_TOKEN_ID: int = 3
    SPECIAL_TOKENS: List[str] = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]

    def __init__(
        self,
        vocab: Optional[List[str]] = None,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 100,
        device: Optional[str] = None,
        model_path: Optional[str] = None,
    ):
        """Initialize sequence model.

        Args:
            vocab: Full vocabulary list. If None, initialized with special tokens.
            d_model: Transformer embedding dimension.
            nhead: Number of attention heads.
            num_layers: Number of transformer layers.
            dim_feedforward: Feedforward dimension.
            dropout: Dropout rate.
            max_seq_length: Maximum sequence length.
            device: Device to run on.
            model_path: Path to pretrained model.
        """
        if vocab is None:
            vocab = list(self.SPECIAL_TOKENS)

        self._vocab: List[str] = vocab
        self._token_to_id: Dict[str, int] = {t: i for i, t in enumerate(vocab)}
        self._id_to_token: Dict[int, str] = {i: t for i, t in enumerate(vocab)}
        self._vocab_size = len(vocab)
        self._d_model = d_model
        self._max_seq_length = max_seq_length

        if device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self._transformer = GestureSequenceTransformer(
            vocab_size=self._vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_length=max_seq_length,
        ).to(self._device)

        self._crf = CRFLayer(num_tags=self._vocab_size).to(self._device)

        self._eval_mode = False

        if model_path:
            self.load(model_path)

        logger.info(
            "sequence_model_initialized",
            vocab_size=self._vocab_size,
            device=str(self._device),
            d_model=d_model,
            num_layers=num_layers,
        )

    def model_sequence(
        self, gesture_sequence: List[str]
    ) -> InterpretedSequence:
        """Model and interpret a gesture sequence.

        Args:
            gesture_sequence: List of gesture label strings.

        Returns:
            InterpretedSequence with token-level interpretations.

        Raises:
            SequenceModelingError: If modeling fails.
        """
        if not gesture_sequence:
            raise ValueError("Empty gesture sequence")

        try:
            # Convert to token IDs
            token_ids = self._encode_sequence(gesture_sequence)
            seq_len = len(token_ids)

            # Create input tensor (batch=1, seq_len, vocab_size)
            input_tensor = torch.zeros(1, seq_len, self._vocab_size, device=self._device)
            for t, tid in enumerate(token_ids):
                input_tensor[0, t, tid] = 1.0

            # Create mask (all valid)
            mask = torch.ones(1, seq_len, dtype=torch.bool, device=self._device)

            self._ensure_eval_mode()

            with torch.no_grad():
                # Transformer encoding
                logits = self._transformer(input_tensor, mask=~mask)  # attention mask: True = masked

                # CRF decoding
                crf_mask = mask
                best_tags = self._crf.decode(logits, crf_mask)
                smoothed_ids = best_tags[0].cpu().tolist()

                # Get confidences
                probs = F.softmax(logits, dim=-1)
                confidences = probs[0].max(dim=-1)[0].cpu().tolist()

            # Decode tokens
            tokens = [self._id_to_token.get(tid, "<UNK>") for tid in smoothed_ids]

            # Filter out special tokens for sentence
            sentence_tokens = [
                t for t in tokens
                if t not in self.SPECIAL_TOKENS and not t.startswith("_")
            ]
            sentence = " ".join(sentence_tokens)

            # Compute alignment (dummy alignment by position)
            alignment = [(i, i + 1) for i in range(len(tokens))]

            # Overall confidence
            seq_conf = float(np.mean(confidences)) if confidences else 0.0

            # Determine if sequence is complete
            is_complete = self._is_complete(gesture_sequence, tokens)

            return InterpretedSequence(
                tokens=tokens,
                sentence=sentence,
                token_confidences=confidences,
                sequence_confidence=seq_conf,
                alignment=alignment,
                is_complete=is_complete,
                metadata={
                    "raw_gestures": gesture_sequence,
                    "smoothed_ids": smoothed_ids,
                    "seq_len": seq_len,
                },
            )

        except Exception as e:
            logger.error("sequence_modeling_failed", error=str(e))
            raise SequenceModelingError(f"Sequence modeling failed: {e}") from e

    def predict_next_tokens(
        self, gesture_sequence: List[str], num_predictions: int = 5
    ) -> List[Tuple[str, float]]:
        """Predict likely next tokens given current sequence.

        Args:
            gesture_sequence: Current gesture sequence.
            num_predictions: Number of top predictions to return.

        Returns:
            List of (token, confidence) tuples.
        """
        if not gesture_sequence:
            return []

        try:
            token_ids = self._encode_sequence(gesture_sequence)
            seq_len = len(token_ids)

            input_tensor = torch.zeros(1, seq_len, self._vocab_size, device=self._device)
            for t, tid in enumerate(token_ids):
                input_tensor[0, t, tid] = 1.0

            self._ensure_eval_mode()

            with torch.no_grad():
                logits = self._transformer(input_tensor)
                # Take last timestep logits
                last_logits = logits[0, -1, :]
                probs = F.softmax(last_logits, dim=-1)

            top_probs, top_indices = torch.topk(probs, num_predictions)

            predictions = []
            for i in range(num_predictions):
                tid = top_indices[i].item()
                conf = top_probs[i].item()
                token = self._id_to_token.get(tid, "<UNK>")
                if token not in self.SPECIAL_TOKENS:
                    predictions.append((token, conf))

            return predictions

        except Exception as e:
            logger.error("next_token_prediction_failed", error=str(e))
            return []

    def load(self, model_path: str) -> None:
        """Load model and CRF weights from checkpoint.

        Args:
            model_path: Path to checkpoint file.
        """
        checkpoint = torch.load(model_path, map_location=self._device)
        self._transformer.load_state_dict(checkpoint["transformer"])
        self._crf.load_state_dict(checkpoint["crf"])
        if "vocab" in checkpoint:
            self._vocab = checkpoint["vocab"]
            self._token_to_id = {t: i for i, t in enumerate(self._vocab)}
            self._id_to_token = {i: t for i, t in enumerate(self._vocab)}
        logger.info("sequence_model_loaded", path=model_path)

    def save(self, model_path: str) -> None:
        """Save model and CRF weights to checkpoint.

        Args:
            model_path: Path to save checkpoint.
        """
        checkpoint = {
            "transformer": self._transformer.state_dict(),
            "crf": self._crf.state_dict(),
            "vocab": self._vocab,
        }
        torch.save(checkpoint, model_path)
        logger.info("sequence_model_saved", path=model_path)

    def _encode_sequence(self, gesture_sequence: List[str]) -> List[int]:
        """Encode a gesture sequence to token IDs.

        Args:
            gesture_sequence: List of gesture labels.

        Returns:
            List of token IDs.
        """
        token_ids = []
        for gesture in gesture_sequence:
            tid = self._token_to_id.get(gesture, self.UNK_TOKEN_ID)
            token_ids.append(tid)

        # Truncate if too long
        if len(token_ids) > self._max_seq_length:
            token_ids = token_ids[-self._max_seq_length:]

        return token_ids

    def _is_complete(self, raw_gestures: List[str], tokens: List[str]) -> bool:
        """Determine if the interpreted sequence forms a complete utterance.

        Heuristic: sequence is complete if it ends with a sentence-ending
        gesture or has high enough confidence.

        Args:
            raw_gestures: Original gesture labels.
            tokens: Interpreted tokens.

        Returns:
            True if the sequence appears complete.
        """
        if not tokens:
            return False

        end_markers = {"stop", "done", "end", "period", "<EOS>"}
        last_token = tokens[-1].lower() if tokens else ""

        if last_token in end_markers:
            return True

        # Complete if sequence is long enough and has decent confidence
        return len(tokens) >= 5

    def _ensure_eval_mode(self) -> None:
        """Ensure model is in evaluation mode."""
        if not self._eval_mode:
            self._transformer.eval()
            self._crf.eval()
            self._eval_mode = True
