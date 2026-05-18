"""
Sign language transformer model for Neurolink.

Defines the SignLanguageTransformer implementing a Transformer encoder-decoder
architecture for sequence-to-sequence sign language to text translation.
Supports teacher forcing during training and beam search during inference.
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for the SignLanguageTransformer."""

    vocab_size: int = 10000
    max_seq_length: int = 200
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    layer_norm_eps: float = 1e-5
    positional_dropout: float = 0.1
    gesture_token_dim: int = 128
    gesture_embed_dim: int = 512
    num_gesture_tokens: int = 1000
    pad_token_id: int = 0
    sos_token_id: int = 1
    eos_token_id: int = 2
    max_beam_size: int = 5
    beam_length_penalty: float = 0.6
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer sequences."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class GestureEmbedding(nn.Module):
    """Learned embedding layer for gesture tokens."""

    def __init__(
        self,
        num_gesture_tokens: int,
        gesture_token_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed = nn.Linear(gesture_token_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, gesture_tokens: torch.Tensor) -> torch.Tensor:
        # gesture_tokens: (batch, seq_len, gesture_token_dim)
        x = self.embed(gesture_tokens)  # (batch, seq_len, d_model)
        x = self.pos_encoding(x)
        return self.dropout(x)


class SignLanguageTransformer(nn.Module):
    """Transformer encoder-decoder for sign language to text translation.

    The encoder processes gesture token sequences, and the decoder
    autoregressively generates text tokens. Supports teacher forcing
    during training and beam search during inference.
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Gesture encoder
        self.gesture_embedding = GestureEmbedding(
            num_gesture_tokens=config.num_gesture_tokens,
            gesture_token_dim=config.gesture_token_dim,
            d_model=config.d_model,
            dropout=config.positional_dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )

        # Text decoder
        self.text_embedding = nn.Embedding(
            config.vocab_size, config.d_model, padding_idx=config.pad_token_id
        )
        self.text_pos_encoding = PositionalEncoding(
            config.d_model, max_len=config.max_seq_length,
            dropout=config.positional_dropout,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )

        self.output_projection = nn.Linear(config.d_model, config.vocab_size)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_square_subsequent_mask(
        self, sz: int, device: torch.device
    ) -> torch.Tensor:
        """Generate causal mask for autoregressive decoding."""
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
        return mask

    def _create_padding_mask(
        self, tokens: torch.Tensor, pad_token_id: int
    ) -> torch.Tensor:
        """Create padding mask (True = masked position)."""
        return tokens == pad_token_id  # (batch, seq_len)

    def encode(self, gesture_tokens: torch.Tensor) -> torch.Tensor:
        """Encode gesture token sequences into memory.

        Args:
            gesture_tokens: (batch, src_seq_len, gesture_token_dim)

        Returns:
            memory: (batch, src_seq_len, d_model)
        """
        x = self.gesture_embedding(gesture_tokens)
        memory = self.encoder(x)
        return memory

    def decode(
        self,
        target_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode target tokens conditioned on encoder memory.

        Args:
            target_tokens: (batch, tgt_seq_len)
            memory: (batch, src_seq_len, d_model)
            tgt_mask: Optional causal mask for target sequence.
            tgt_key_padding_mask: Padding mask for target tokens.
            memory_key_padding_mask: Padding mask for source memory.

        Returns:
            logits: (batch, tgt_seq_len, vocab_size)
        """
        tgt = self.text_embedding(target_tokens)
        tgt = self.text_pos_encoding(tgt)

        if tgt_mask is None:
            tgt_len = target_tokens.size(1)
            tgt_mask = self._generate_square_subsequent_mask(
                tgt_len, target_tokens.device
            )

        output = self.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.output_projection(output)
        return logits

    def forward(
        self,
        gesture_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        use_teacher_forcing: bool = True,
        teacher_forcing_ratio: float = 1.0,
    ) -> torch.Tensor:
        """Forward pass with optional teacher forcing.

        Args:
            gesture_tokens: (batch, src_seq_len, gesture_token_dim)
            target_tokens: (batch, tgt_seq_len) including SOS at position 0
            use_teacher_forcing: Whether to use teacher forcing.
            teacher_forcing_ratio: Probability of using teacher forcing.

        Returns:
            logits: (batch, tgt_seq_len - 1, vocab_size)
        """
        memory = self.encode(gesture_tokens)

        if use_teacher_forcing and torch.rand(1).item() < teacher_forcing_ratio:
            # Teacher forcing: feed ground truth tokens
            tgt_input = target_tokens[:, :-1]  # remove EOS
            logits = self.decode(tgt_input, memory)
            return logits
        else:
            # Autoregressive decoding (used during scheduled sampling)
            batch_size = gesture_tokens.size(0)
            max_len = target_tokens.size(1) - 1
            device = gesture_tokens.device

            logits_list: List[torch.Tensor] = []
            current_token = torch.full(
                (batch_size, 1), self.config.sos_token_id, dtype=torch.long, device=device
            )
            generated = current_token

            for _ in range(max_len):
                tgt_mask = self._generate_square_subsequent_mask(
                    generated.size(1), device
                )
                step_logits = self.decode(generated, memory, tgt_mask=tgt_mask)
                next_logit = step_logits[:, -1:, :]  # (batch, 1, vocab_size)
                logits_list.append(next_logit)

                # Sample or greedy
                if use_teacher_forcing:
                    next_token = target_tokens[:, _ : _ + 1]
                else:
                    next_token = next_logit.argmax(dim=-1)

                generated = torch.cat([generated, next_token], dim=1)

            return torch.cat(logits_list, dim=1)

    @torch.no_grad()
    def beam_search(
        self,
        gesture_tokens: torch.Tensor,
        beam_size: int = 5,
        max_length: int = 50,
        length_penalty: float = 0.6,
    ) -> List[List[int]]:
        """Beam search decoding for inference.

        Args:
            gesture_tokens: (batch, src_seq_len, gesture_token_dim)
            beam_size: Beam width for search.
            max_length: Maximum generation length.
            length_penalty: Length penalty exponent (shorter = higher penalty).

        Returns:
            List of token ID sequences for each batch item.
        """
        self.eval()
        batch_size = gesture_tokens.size(0)
        device = gesture_tokens.device
        memory = self.encode(gesture_tokens)

        results: List[List[int]] = []
        for batch_idx in range(batch_size):
            mem = memory[batch_idx : batch_idx + 1]  # (1, src_len, d_model)

            # Initial beam: (SOS, log_prob=0)
            beam = [(self.config.sos_token_id, 0.0)]
            completed: List[Tuple[List[int], float]] = []

            for step in range(max_length):
                candidates: List[Tuple[List[int], float]] = []

                for seq, score in beam:
                    if seq == self.config.eos_token_id and len(str(seq)) > 1:
                        pass

                    seq_tensor = torch.tensor([seq], dtype=torch.long, device=device)

                    # Skip if EOS already generated (but check the last token)
                    if isinstance(seq, int):
                        seq_list = [seq]
                    elif isinstance(seq, list):
                        seq_list = seq
                    else:
                        seq_list = [seq]

                    if len(seq_list) > 0 and seq_list[-1] == self.config.eos_token_id:
                        completed.append((seq_list, score))
                        continue
                    if len(seq_list) >= max_length:
                        completed.append((seq_list, score))
                        continue

                    seq_tensor = torch.tensor(
                        [seq_list], dtype=torch.long, device=device
                    )
                    tgt_mask = self._generate_square_subsequent_mask(
                        seq_tensor.size(1), device
                    )

                    logits = self.decode(seq_tensor, mem, tgt_mask=tgt_mask)
                    next_logits = logits[:, -1, :]  # (1, vocab_size)
                    log_probs = F.log_softmax(next_logits, dim=-1)

                    # Top-k candidates
                    top_k_log_probs, top_k_tokens = log_probs.topk(beam_size, dim=-1)

                    for i in range(beam_size):
                        token = top_k_tokens[0, i].item()
                        new_score = score + top_k_log_probs[0, i].item()
                        candidates.append((seq_list + [token], new_score))

                # Prune to beam_size
                beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                # Check if all beams have completed
                all_completed = all(
                    len(seq) > 0 and seq[-1] == self.config.eos_token_id
                    for seq, _ in beam
                )
                if all_completed and len(beam) > 0:
                    break

            # Add remaining beams to completed
            for seq, score in beam:
                if not (len(seq) > 0 and seq[-1] == self.config.eos_token_id):
                    completed.append((seq if isinstance(seq, list) else [seq], score))

            # Score with length penalty
            scored = []
            for seq, score in completed:
                lp = ((5 + len(seq)) ** length_penalty) / ((5 + 1) ** length_penalty)
                adjusted_score = score / lp
                scored.append((seq, adjusted_score))

            scored.sort(key=lambda x: x[1], reverse=True)
            best_seq = scored[0][0] if scored else [self.config.eos_token_id]
            results.append(best_seq)

        return results

    @torch.no_grad()
    def greedy_decode(
        self,
        gesture_tokens: torch.Tensor,
        max_length: int = 50,
    ) -> torch.Tensor:
        """Greedy decoding for fast inference.

        Args:
            gesture_tokens: (batch, src_seq_len, gesture_token_dim)
            max_length: Maximum generation length.

        Returns:
            Token sequences: (batch, seq_len)
        """
        self.eval()
        batch_size = gesture_tokens.size(0)
        device = gesture_tokens.device
        memory = self.encode(gesture_tokens)

        generated = torch.full(
            (batch_size, 1), self.config.sos_token_id, dtype=torch.long, device=device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_length):
            if finished.all():
                break

            tgt_mask = self._generate_square_subsequent_mask(
                generated.size(1), device
            )
            logits = self.decode(generated, memory, tgt_mask=tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (batch, 1)

            # Mark finished sequences
            just_finished = next_token.squeeze(-1) == self.config.eos_token_id
            finished = finished | just_finished
            next_token[finished] = self.config.pad_token_id

            generated = torch.cat([generated, next_token], dim=1)

            if finished.all():
                break

        return generated

    def save_pretrained(self, path: Union[str, Path]):
        """Save model weights and config."""
        import json as _json

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        config_dict = {
            k: v for k, v in self.config.__dict__.items()
            if not k.startswith("_")
        }
        with open(path / "config.json", "w") as f:
            _json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"Model saved to {path}")

    @classmethod
    def from_pretrained(cls, path: Union[str, Path]) -> "SignLanguageTransformer":
        """Load model from saved weights and config."""
        import json as _json

        path = Path(path)
        with open(path / "config.json", "r") as f:
            config_dict = _json.load(f)
        config = TransformerConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        logger.info(f"Model loaded from {path}")
        return model
