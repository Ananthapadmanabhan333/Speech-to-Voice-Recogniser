"""
Vocal emotion detection model for Neurolink.

Provides the VocalEmotionModel combining a Wav2Vec2/BERT-style audio encoder
with a prosodic feature CNN for emotion classification and arousal/valence
regression.
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class VocalEmotionConfig:
    """Configuration for the VocalEmotionModel."""

    # Audio encoder
    audio_encoder: str = "wav2vec2"  # "wav2vec2" or "hubert"
    audio_feat_dim: int = 768  # wav2vec2 base hidden size
    audio_num_layers: int = 12
    freeze_encoder: bool = True

    # Prosodic features
    prosodic_dim: int = 40  # MFCC + pitch + energy + zero-crossing etc.
    prosodic_cnn_channels: List[int] = field(
        default_factory=lambda: [64, 128, 256]
    )
    prosodic_kernel_size: int = 5

    # Fusion
    d_model: int = 512

    # Emotion classes
    num_emotions: int = 8  # angry, disgust, fear, happy, neutral, sad, surprise, others

    # Arousal/valence
    arousal_std: float = 1.0
    valence_std: float = 1.0

    # Dropout
    dropout: float = 0.3

    # Pretrained checkpoint
    pretrained_ckpt: Optional[str] = None

    # Input
    sample_rate: int = 16000
    max_audio_length: float = 10.0  # seconds

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class Wav2Vec2Encoder(nn.Module):
    """Wav2Vec2/HuBERT-style transformer encoder for raw audio.

    Implements a simplified version of the Wav2Vec2 architecture with
    convolutional feature encoder + transformer context network.
    """

    def __init__(
        self,
        feat_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feat_dim = feat_dim

        # Feature encoder (CNN feature extractor)
        self.feature_encoder = nn.Sequential(
            nn.Conv1d(1, 512, kernel_size=10, stride=5, padding=3),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(512, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

        # Feature projection
        self.feature_proj = nn.Linear(feat_dim, feat_dim)

        # Positional embeddings
        self.pos_conv = nn.Conv1d(feat_dim, feat_dim, kernel_size=128, groups=16, padding=64)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=num_heads,
            dim_feedforward=feat_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.layer_norm = nn.LayerNorm(feat_dim)

    def forward(
        self, waveform: torch.Tensor
    ) -> torch.Tensor:
        """Encode raw waveform.

        Args:
            waveform: (batch, audio_samples) or (batch, 1, audio_samples).

        Returns:
            encoded: (batch, seq_len, feat_dim) frame-level features.
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)

        # CNN feature encoder
        features = self.feature_encoder(waveform)  # (batch, feat_dim, t)
        features = features.permute(0, 2, 1)  # (batch, t, feat_dim)
        features = self.feature_proj(features)

        # Positional encoding via convolution
        pos = self.pos_conv(features.permute(0, 2, 1)).permute(0, 2, 1)
        features = features + pos

        # Transformer context network
        features = self.transformer(features)
        features = self.layer_norm(features)

        return features


class ProsodicFeatureCNN(nn.Module):
    """CNN for processing prosodic features extracted from audio."""

    def __init__(
        self,
        input_dim: int = 40,
        channels: List[int] = None,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()
        channels = channels or [64, 128, 256]

        layers = []
        prev_c = input_dim
        for c in channels:
            layers.extend([
                nn.Conv1d(prev_c, c, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.MaxPool1d(2),
            ])
            prev_c = c
        self.cnn = nn.Sequential(*layers)
        self.output_dim = prev_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, prosodic_dim, seq_len)
        return self.cnn(x)


class VocalEmotionModel(nn.Module):
    """Vocal emotion recognition model combining Wav2Vec2 audio encoding
    with prosodic feature analysis.

    Architecture:
        Audio encoder (Wav2Vec2) -> Frame-level features
        Prosodic features -> CNN -> Segment-level features
        Fusion -> Emotion classification + Arousal/Valence regression
    """

    def __init__(self, config: VocalEmotionConfig):
        super().__init__()
        self.config = config

        # 1. Audio encoder
        if config.audio_encoder == "wav2vec2":
            self.audio_encoder = Wav2Vec2Encoder(
                feat_dim=config.audio_feat_dim,
                num_layers=config.audio_num_layers,
                dropout=config.dropout,
            )
        else:
            raise ValueError(f"Unsupported audio encoder: {config.audio_encoder}")

        if config.freeze_encoder:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False

        # 2. Prosodic feature CNN
        self.prosodic_cnn = ProsodicFeatureCNN(
            input_dim=config.prosodic_dim,
            channels=config.prosodic_cnn_channels,
            kernel_size=config.prosodic_kernel_size,
            dropout=config.dropout,
        )

        # 3. Fusion layer
        audio_feat_dim = config.audio_feat_dim
        prosodic_out = config.prosodic_cnn_channels[-1]
        fusion_input = audio_feat_dim + prosodic_out
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # 4. Emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_emotions),
        )

        # 5. Arousal/Valence regressor
        self.arousal_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout // 2),
            nn.Linear(config.d_model // 2, 1),
        )
        self.valence_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout // 2),
            nn.Linear(config.d_model // 2, 1),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        waveform: torch.Tensor,
        prosodic_features: torch.Tensor,
        return_all: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            waveform: (batch, audio_samples) raw audio waveform.
            prosodic_features: (batch, prosodic_dim, seq_len) prosodic features.
            return_all: Whether to return intermediate features.

        Returns:
            Dict with 'emotion_logits', 'arousal', 'valence', and optionally
            'audio_features', 'prosodic_features', 'fused_features'.
        """
        # Audio encoding
        audio_feats = self.audio_encoder(waveform)  # (batch, t, audio_feat_dim)
        audio_pooled = audio_feats.mean(dim=1)  # (batch, audio_feat_dim)

        # Prosodic CNN
        prosodic_out = self.prosodic_cnn(prosodic_features)  # (batch, C', t')
        prosodic_pooled = prosodic_out.mean(dim=-1)  # (batch, C')

        # Fusion
        concat = torch.cat([audio_pooled, prosodic_pooled], dim=-1)
        fused = self.fusion(concat)

        # Outputs
        emotion_logits = self.emotion_classifier(fused)
        arousal = self.arousal_head(fused)
        valence = self.valence_head(fused)

        outputs = {
            "emotion_logits": emotion_logits,
            "arousal": arousal,
            "valence": valence,
        }

        if return_all:
            outputs["audio_features"] = audio_feats
            outputs["audio_pooled"] = audio_pooled
            outputs["prosodic_features"] = prosodic_out
            outputs["prosodic_pooled"] = prosodic_pooled
            outputs["fused_features"] = fused

        return outputs

    def compute_emotion_scores(
        self,
        waveform: torch.Tensor,
        prosodic_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute emotion probabilities and arousal/valence values."""
        with torch.no_grad():
            outputs = self.forward(waveform, prosodic_features, return_all=False)
            probs = F.softmax(outputs["emotion_logits"], dim=-1)
            return {
                "emotion_probs": probs,
                "arousal": outputs["arousal"],
                "valence": outputs["valence"],
            }

    def load_pretrained_checkpoint(
        self, ckpt_path: Union[str, Path], strict: bool = True
    ):
        """Load weights from a pretrained checkpoint."""
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu")
        self.load_state_dict(state, strict=strict)
        logger.info(f"Loaded pretrained checkpoint from {ckpt_path}")

    def save_pretrained(self, path: Union[str, Path]):
        """Save model and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        config_dict = {
            k: v for k, v in self.config.__dict__.items() if not k.startswith("_")
        }
        with open(path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2, default=str)
        logger.info(f"VocalEmotionModel saved to {path}")

    @classmethod
    def from_pretrained(
        cls, path: Union[str, Path]
    ) -> "VocalEmotionModel":
        """Load model from saved weights."""
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = VocalEmotionConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        logger.info(f"VocalEmotionModel loaded from {path}")
        return model
