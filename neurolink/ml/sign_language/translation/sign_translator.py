"""
Sign-to-text translation service for Neurolink.

Provides the SignTranslator class that wraps the SignLanguageTransformer
for end-to-end sign language translation. Includes contextual correction
using a language model, confidence scoring, and OOV handling.
"""

import json
import logging
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from ..models.sign_language_transformer import (
    SignLanguageTransformer,
    TransformerConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of a sign-to-text translation."""

    text: str
    tokens: List[str]
    token_ids: List[int]
    confidence: float
    token_confidences: List[float]
    alternative_hypotheses: List[Tuple[str, float]]
    has_oov: bool
    oov_tokens: List[str]
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "confidence": self.confidence,
            "token_confidences": self.token_confidences,
            "has_oov": self.has_oov,
            "oov_tokens": self.oov_tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class SignTranslatorConfig:
    """Configuration for the SignTranslator."""

    model_path: Optional[Path] = None
    vocab_path: Optional[Path] = None
    lm_model_path: Optional[Path] = None
    beam_size: int = 5
    max_length: int = 100
    length_penalty: float = 0.6
    confidence_threshold: float = 0.5
    use_lm_correction: bool = True
    lm_weight: float = 0.3
    oov_threshold: int = 3  # min occurrences to add to vocab
    max_alternatives: int = 3
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class NGramLanguageModel:
    """Simple n-gram language model for contextual correction."""

    def __init__(self, n: int = 3, smoothing: float = 0.01):
        self.n = n
        self.smoothing = smoothing
        self.ngram_counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.context_counts: Counter = Counter()
        self.vocab: Set[str] = set()

    def fit(self, texts: List[str]):
        """Fit the language model on a corpus of texts."""
        for text in texts:
            tokens = text.lower().split()
            self.vocab.update(tokens)
            for i in range(len(tokens)):
                for j in range(1, min(self.n, len(tokens) - i) + 1):
                    ngram = tuple(tokens[i : i + j])
                    if len(ngram) == 1:
                        self.ngram_counts[tuple()][ngram[0]] += 1
                        self.context_counts[tuple()] += 1
                    elif len(ngram) > 1:
                        context = ngram[:-1]
                        word = ngram[-1]
                        self.ngram_counts[context][word] += 1
                        self.context_counts[context] += 1

        logger.info(
            f"LM fitted on {len(texts)} texts, vocab={len(self.vocab)}, "
            f"contexts={len(self.context_counts)}"
        )

    def score(self, word: str, context: Tuple[str, ...]) -> float:
        """Compute log probability of word given context."""
        context = tuple(w.lower() for w in context)
        word = word.lower()
        ngram_key = context

        count = self.ngram_counts[ngram_key].get(word, 0)
        context_total = self.context_counts[ngram_key]
        vocab_size = len(self.vocab)

        if context_total == 0:
            return -np.log(vocab_size)

        # Add-k smoothing
        prob = (count + self.smoothing) / (
            context_total + self.smoothing * (vocab_size + 1)
        )
        return np.log(prob)

    def correct_candidates(
        self, word: str, context: Tuple[str, ...], top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get top-k word candidates for correction."""
        context = tuple(w.lower() for w in context)
        if context not in self.context_counts:
            return [(word, self.score(word, context))]

        candidates = []
        for w, c in self.ngram_counts[context].items():
            score = self.score(w, context)
            candidates.append((w, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:top_k]


class SignTranslator:
    """End-to-end sign language to text translator.

    Wraps the SignLanguageTransformer and provides:
    - Translation with beam search
    - Contextual correction via n-gram LM
    - Per-token confidence scoring
    - OOV detection and handling
    """

    def __init__(
        self,
        model: Optional[SignLanguageTransformer] = None,
        config: Optional[SignTranslatorConfig] = None,
        tokenizer: Optional[Any] = None,
    ):
        self.config = config or SignTranslatorConfig()
        self.device = torch.device(self.config.device)

        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        self.tokenizer = tokenizer
        self.lm: Optional[NGramLanguageModel] = None
        if self.config.use_lm_correction:
            self.lm = NGramLanguageModel(n=3)

        self._vocab: Set[str] = set()
        logger.info(f"SignTranslator initialized (device={self.device})")

    def _load_vocab(self, path: Path):
        """Load vocabulary from file."""
        with open(path, "r") as f:
            self._vocab = set(line.strip() for line in f if line.strip())
        logger.info(f"Loaded vocabulary with {len(self._vocab)} tokens")

    def translate(
        self,
        gesture_sequence: np.ndarray,
        return_alternatives: bool = True,
    ) -> TranslationResult:
        """Translate a gesture sequence to text.

        Args:
            gesture_sequence: (seq_len, gesture_token_dim) numpy array of gesture tokens.
            return_alternatives: Whether to return alternative hypotheses.

        Returns:
            TranslationResult with translated text and metadata.
        """
        if self.model is None:
            raise RuntimeError("No model loaded for translation")

        start_time = torch.cuda.Event(enable_timing=True)  # placeholder
        import time as _time
        t0 = _time.time()

        # Prepare input
        if gesture_sequence.ndim == 2:
            gesture_sequence = np.expand_dims(gesture_sequence, axis=0)  # (1, seq_len, dim)

        gesture_tensor = torch.tensor(
            gesture_sequence, dtype=torch.float32, device=self.device
        )

        # Generate translation via beam search
        with torch.no_grad():
            sequences = self.model.beam_search(
                gesture_tensor,
                beam_size=self.config.beam_size,
                max_length=self.config.max_length,
                length_penalty=self.config.length_penalty,
            )

        latency_ms = (_time.time() - t0) * 1000.0

        # Decode beam results
        hypotheses: List[Tuple[List[int], float]] = []
        for seq_ids in sequences:
            if self.config.eos_token_id in seq_ids:
                end_idx = seq_ids.index(self.config.eos_token_id)
                seq_ids = seq_ids[:end_idx]
            score = self._score_sequence(seq_ids)
            hypotheses.append((seq_ids, score))

        hypotheses.sort(key=lambda x: x[1], reverse=True)

        # Get token probabilities
        best_seq_ids = hypotheses[0][0] if hypotheses else []
        token_confidences = self._get_token_confidences(
            gesture_tensor, best_seq_ids
        )

        # Detokenize
        tokens = self._decode_ids(best_seq_ids)
        text = " ".join(tokens)

        # Contextual correction with LM
        oov_tokens: List[str] = []
        if self.config.use_lm_correction and self.lm is not None:
            corrected_tokens = list(tokens)
            for i, token in enumerate(tokens):
                if token not in self._vocab:
                    oov_tokens.append(token)
                    context = tuple(tokens[max(0, i - 2) : i])
                    candidates = self.lm.correct_candidates(token, context)
                    if candidates and candidates[0][0] != token:
                        corrected_tokens[i] = candidates[0][0]
            text = " ".join(corrected_tokens)

        # Overall confidence
        confidence = float(np.mean(token_confidences)) if token_confidences else 0.0

        # Alternative hypotheses
        alternatives: List[Tuple[str, float]] = []
        if return_alternatives:
            for alt_ids, alt_score in hypotheses[1 : 1 + self.config.max_alternatives]:
                alt_tokens = self._decode_ids(alt_ids)
                alt_text = " ".join(alt_tokens)
                alternatives.append((alt_text, alt_score))

        result = TranslationResult(
            text=text,
            tokens=tokens,
            token_ids=best_seq_ids,
            confidence=confidence,
            token_confidences=token_confidences,
            alternative_hypotheses=alternatives,
            has_oov=len(oov_tokens) > 0,
            oov_tokens=oov_tokens,
            latency_ms=latency_ms,
        )

        logger.debug(
            f"Translated: '{text}' (confidence={confidence:.3f}, "
            f"latency={latency_ms:.1f}ms, oov={len(oov_tokens)})"
        )

        return result

    def translate_batch(
        self, gesture_sequences: List[np.ndarray]
    ) -> List[TranslationResult]:
        """Translate a batch of gesture sequences."""
        return [self.translate(seq) for seq in gesture_sequences]

    def _score_sequence(self, token_ids: List[int]) -> float:
        """Score a token sequence using model probabilities and optional LM."""
        if not token_ids:
            return -float("inf")
        # Length-normalized score from beam search is already included;
        # this is a simple length penalty
        return len(token_ids) ** self.config.length_penalty

    @torch.no_grad()
    def _get_token_confidences(
        self, gesture_tensor: torch.Tensor, token_ids: List[int]
    ) -> List[float]:
        """Compute per-token confidence scores."""
        if not token_ids:
            return []

        confidences: List[float] = []
        memory = self.model.encode(gesture_tensor)
        generated = torch.tensor(
            [[self.config.sos_token_id]], dtype=torch.long, device=self.device
        )

        for next_id in token_ids:
            logits = self.model.decode(generated, memory)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            conf = float(probs[0, next_id].item())
            confidences.append(conf)

            next_tensor = torch.tensor(
                [[next_id]], dtype=torch.long, device=self.device
            )
            generated = torch.cat([generated, next_tensor], dim=1)

        return confidences

    def _decode_ids(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to text tokens using the tokenizer or id mapping."""
        if self.tokenizer is not None:
            return self.tokenizer.decode(token_ids)
        return [f"tok_{tid}" for tid in token_ids if tid >= 3]  # skip special tokens

    def add_to_vocabulary(self, tokens: List[str]):
        """Add new tokens to the vocabulary (for OOV handling)."""
        for token in tokens:
            self._vocab.add(token)
        logger.info(f"Added {len(tokens)} tokens to vocabulary (total={len(self._vocab)})")

    def fit_language_model(self, texts: List[str]):
        """Fit the internal n-gram language model on a text corpus."""
        if self.lm is None:
            self.lm = NGramLanguageModel(n=3)
        self.lm.fit(texts)
        logger.info(f"Language model fitted on {len(texts)} sentences")

    def save(self, path: Union[str, Path]):
        """Save translator state to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        if self.model:
            self.model.save_pretrained(path / "model")

        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)

        with open(path / "vocab.txt", "w") as f:
            for token in sorted(self._vocab):
                f.write(f"{token}\n")

        logger.info(f"Translator saved to {path}")

    @classmethod
    def load(
        cls,
        path: Union[str, Path],
        tokenizer: Optional[Any] = None,
    ) -> "SignTranslator":
        """Load translator from disk."""
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = SignTranslatorConfig(**config_dict)

        if (path / "model").exists():
            model = SignLanguageTransformer.from_pretrained(path / "model")
        else:
            model = None

        translator = cls(model=model, config=config, tokenizer=tokenizer)

        vocab_path = path / "vocab.txt"
        if vocab_path.exists():
            translator._load_vocab(vocab_path)

        logger.info(f"Translator loaded from {path}")
        return translator
