"""
Facial emotion detection model for Neurolink.

Provides the FacialEmotionModel using a CNN backbone (ResNet-18/EfficientNet)
with facial landmark conditioned attention for recognizing 7 basic emotions
plus compound emotions, along with Action Unit regression.
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
import torchvision.models as models

logger = logging.getLogger(__name__)


@dataclass
class FacialEmotionConfig:
    """Configuration for the FacialEmotionModel."""

    backbone: str = "resnet18"  # "resnet18", "resnet50", "efficientnet_b0"
    pretrained: bool = True
    freeze_backbone: bool = False

    # Landmarks
    num_landmarks: int = 68
    landmark_dim: int = 2  # (x, y)
    landmark_hidden_dim: int = 128

    # Emotion classes
    basic_emotions: int = 7  # anger, disgust, fear, happy, sad, surprise, neutral
    compound_emotions: int = 15  # happily surprised, sadly angry, etc.
    num_emotions: int = field(init=False)

    # Action Units
    num_action_units: int = 45

    # Attention
    attention_heads: int = 8
    attention_features: int = 512

    # Image size
    image_size: int = 224

    # Dropout
    dropout: float = 0.3

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self):
        self.num_emotions = self.basic_emotions + self.compound_emotions


class LandmarkConditionedAttention(nn.Module):
    """Attention mechanism conditioned on facial landmark positions.

    Uses landmark coordinates to generate spatial attention maps over
    CNN feature maps, focusing on emotion-relevant facial regions.
    """

    def __init__(
        self,
        feature_dim: int,
        num_landmarks: int = 68,
        landmark_dim: int = 2,
        landmark_hidden: int = 128,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_landmarks = num_landmarks
        self.num_heads = num_heads

        # Landmark embedding
        self.landmark_encoder = nn.Sequential(
            nn.Linear(num_landmarks * landmark_dim, landmark_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(landmark_hidden, landmark_hidden),
            nn.ReLU(inplace=True),
        )

        # Spatial attention generation
        self.attention_proj = nn.Linear(landmark_hidden, num_heads * 2)

        # Feature modulation
        head_dim = feature_dim // num_heads
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        features: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> torch.Tensor:
        """Apply landmark-conditioned attention.

        Args:
            features: (batch, channels, h, w) CNN feature maps.
            landmarks: (batch, num_landmarks, 2) landmark coordinates.

        Returns:
            attended_features: (batch, channels, h, w).
        """
        batch, channels, h, w = features.shape

        # Flatten landmarks and encode
        lm_flat = landmarks.reshape(batch, -1)  # (batch, num_landmarks * 2)
        lm_encoded = self.landmark_encoder(lm_flat)  # (batch, landmark_hidden)

        # Generate attention parameters (mean and std for Gaussian attention)
        attn_params = self.attention_proj(lm_encoded)  # (batch, num_heads * 2)
        attn_mean = attn_params[:, :self.num_heads]  # (batch, num_heads)
        attn_std = torch.softplus(attn_params[:, self.num_heads:]) + 1e-4

        # Generate spatial coordinates grid
        y_coords = torch.arange(h, device=features.device).float() / h
        x_coords = torch.arange(w, device=features.device).float() / w
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (h, w, 2)

        # Compute attention weights per head
        attn_maps = []
        for head in range(self.num_heads):
            mean = attn_mean[:, head].view(batch, 1, 1, 1)  # (batch, 1, 1, 1)
            std = attn_std[:, head].view(batch, 1, 1, 1)
            # Gaussian attention centered at predicted location
            dist = ((grid.unsqueeze(0) - mean.unsqueeze(-1).unsqueeze(-1)) ** 2).sum(
                dim=-1
            )
            attn_map = torch.exp(-dist / (2 * std ** 2))
            attn_maps.append(attn_map)

        # Combine multi-head attention
        attn_weights = torch.stack(attn_maps, dim=1)  # (batch, heads, h, w)
        attn_weights = attn_weights / (attn_weights.sum(dim=(2, 3), keepdim=True) + 1e-8)

        # Apply attention to features
        value = self.value_proj(
            features.permute(0, 2, 3, 1)
        )  # (batch, h, w, channels)

        # Weighted sum
        attended = torch.zeros_like(features)
        for head in range(self.num_heads):
            w = attn_weights[:, head:head+1, :, :]  # (batch, 1, h, w)
            attended = attended + w * features

        attended = self.output_proj(
            attended.permute(0, 2, 3, 1)
        ).permute(0, 3, 1, 2)
        attended = self.dropout(attended)

        return attended + features  # residual connection


class ActionUnitRegressor(nn.Module):
    """Action Unit intensity regression head."""

    def __init__(self, feature_dim: int, num_units: int = 45):
        super().__init__()
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, feature_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 4, num_units),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.regressor(x)


class FacialEmotionModel(nn.Module):
    """Facial emotion recognition model combining CNN backbone with
    landmark-conditioned attention.

    Architecture:
        CNN backbone (ResNet/EfficientNet) -> Landmark-conditioned attention ->
        Global pooling -> Emotion classification + AU regression heads
    """

    def __init__(self, config: FacialEmotionConfig):
        super().__init__()
        self.config = config

        # 1. CNN backbone
        self.backbone, feat_dim = self._build_backbone(config)

        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 2. Landmark-conditioned attention
        self.landmark_attention = LandmarkConditionedAttention(
            feature_dim=feat_dim,
            num_landmarks=config.num_landmarks,
            landmark_dim=config.landmark_dim,
            landmark_hidden=config.landmark_hidden_dim,
            num_heads=config.attention_heads,
        )

        # 3. Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 4. Feature projector
        self.feature_proj = nn.Sequential(
            nn.Linear(feat_dim, config.attention_features),
            nn.LayerNorm(config.attention_features),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
        )

        # 5. Classification heads
        self.basic_emotion_head = nn.Linear(
            config.attention_features, config.basic_emotions
        )
        self.compound_emotion_head = nn.Linear(
            config.attention_features, config.compound_emotions
        )

        # 6. Action Unit regressor
        self.au_regressor = ActionUnitRegressor(
            config.attention_features, config.num_action_units
        )

        self._reset_parameters()

    def _build_backbone(
        self, config: FacialEmotionConfig
    ) -> Tuple[nn.Module, int]:
        """Build CNN backbone and return (backbone, feature_dim)."""
        if config.backbone == "resnet18":
            backbone = models.resnet18(weights="DEFAULT" if config.pretrained else None)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif config.backbone == "resnet50":
            backbone = models.resnet50(weights="DEFAULT" if config.pretrained else None)
            feat_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif config.backbone == "efficientnet_b0":
            backbone = models.efficientnet_b0(
                weights="DEFAULT" if config.pretrained else None
            )
            feat_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {config.backbone}")

        logger.info(
            f"Built {config.backbone} backbone (pretrained={config.pretrained}, "
            f"feat_dim={feat_dim})"
        )
        return backbone, feat_dim

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(
        self,
        images: torch.Tensor,
        landmarks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            images: (batch, 3, H, W) face images.
            landmarks: (batch, num_landmarks, 2) facial landmark coordinates
                      normalized to [0, 1].

        Returns:
            Dict with 'basic_emotion_logits', 'compound_emotion_logits',
            'au_intensities', 'features', 'attention_weights'.
        """
        # CNN features
        if hasattr(self.backbone, "features"):
            # EfficientNet
            features = self.backbone.features(images)
        else:
            # ResNet
            x = self.backbone.conv1(images)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)
            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            features = self.backbone.layer4(x)

        # Landmark-conditioned attention
        attended = self.landmark_attention(features, landmarks)

        # Global pooling and projection
        pooled = self.global_pool(attended).flatten(1)  # (batch, feat_dim)
        feat = self.feature_proj(pooled)  # (batch, attention_features)

        # Outputs
        basic_logits = self.basic_emotion_head(feat)
        compound_logits = self.compound_emotion_head(feat)
        au_intensities = self.au_regressor(feat)

        return {
            "basic_emotion_logits": basic_logits,
            "compound_emotion_logits": compound_logits,
            "emotion_logits": torch.cat(
                [basic_logits, compound_logits], dim=-1
            ),
            "au_intensities": au_intensities,
            "features": feat,
        }

    def freeze_backbone(self):
        """Freeze backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone frozen")

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen")

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
        logger.info(f"FacialEmotionModel saved to {path}")

    @classmethod
    def from_pretrained(
        cls, path: Union[str, Path]
    ) -> "FacialEmotionModel":
        """Load model from saved weights."""
        path = Path(path)
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        config = FacialEmotionConfig(**config_dict)
        model = cls(config)
        model.load_state_dict(torch.load(path / "model.pt", map_location="cpu"))
        logger.info(f"FacialEmotionModel loaded from {path}")
        return model
