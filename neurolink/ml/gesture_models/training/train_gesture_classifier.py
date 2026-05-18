"""
Gesture classifier training pipeline for Neurolink.

Provides the GestureTrainer class that handles the complete training lifecycle
for the Temporal CNN + LSTM gesture classification model including data loading,
augmentation, scheduling, checkpointing, and ONNX export.
"""

import json
import logging
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class GestureTrainerConfig:
    """Configuration for the GestureTrainer pipeline."""

    # Data paths
    data_dir: Path = Path("data/gestures")
    checkpoint_dir: Path = Path("checkpoints/gesture")
    log_dir: Path = Path("logs/gesture")

    # Model hyperparameters
    num_classes: int = 30
    landmark_dim: int = 63  # 21 landmarks x 3 (x, y, z)
    hidden_dim: int = 128
    lstm_layers: int = 2
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    kernel_size: int = 3
    dropout: float = 0.3

    # Training hyperparameters
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    min_delta: float = 1e-4

    # Scheduler
    t_max: int = 50
    eta_min: float = 1e-6

    # Augmentation
    rotation_range: float = 0.1
    scaling_range: float = 0.1
    noise_std: float = 0.01
    augmentation_prob: float = 0.5

    # Sequence
    max_seq_length: int = 150
    min_seq_length: int = 10

    # ONNX export
    onnx_export: bool = True
    onnx_opset: int = 17

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True

    # Logging
    log_interval: int = 10
    eval_interval: int = 1


class TemporalCNN(nn.Module):
    """1D temporal CNN for extracting spatial-temporal features from landmark sequences."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_c = in_channels
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
        self._output_dim = prev_c

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        return self.cnn(x)


class GestureClassifierModel(nn.Module):
    """Temporal CNN + LSTM for gesture sequence classification."""

    def __init__(self, config: GestureTrainerConfig):
        super().__init__()
        self.config = config

        self.cnn = TemporalCNN(
            in_channels=config.landmark_dim,
            channels=config.cnn_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )

        cnn_out = self.cnn.output_dim
        self.lstm = nn.LSTM(
            input_size=cnn_out,
            hidden_size=config.hidden_dim,
            num_layers=config.lstm_layers,
            batch_first=True,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            bidirectional=True,
        )
        lstm_out = config.hidden_dim * 2  # bidirectional

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_classes),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for name, param in self.named_parameters():
            if "weight" in name:
                if param.dim() >= 2:
                    nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(
        self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # x: (batch, seq_len, landmark_dim)
        batch_size, seq_len, feat_dim = x.shape

        # CNN expects (batch, channels, seq_len)
        x_cnn = x.permute(0, 2, 1)  # (batch, landmark_dim, seq_len)
        cnn_out = self.cnn(x_cnn)  # (batch, cnn_channels, reduced_seq_len)
        cnn_out = cnn_out.permute(0, 2, 1)  # (batch, reduced_seq_len, cnn_channels)

        if lengths is not None:
            # Adjust lengths after CNN pooling layers
            pooled_lengths = lengths
            for _ in self.config.cnn_channels:
                pooled_lengths = torch.div(
                    pooled_lengths + 1, 2, rounding_mode="floor"
                )
            pooled_lengths = pooled_lengths.clamp(min=1)

            packed = pack_padded_sequence(
                cnn_out,
                pooled_lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            lstm_out, (h_n, _) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        else:
            lstm_out, (h_n, _) = self.lstm(cnn_out)

        # Use final hidden states from both directions
        h_forward = h_n[-2, :, :]  # (batch, hidden_dim)
        h_backward = h_n[-1, :, :]  # (batch, hidden_dim)
        h_concat = torch.cat([h_forward, h_backward], dim=-1)  # (batch, 2*hidden_dim)

        logits = self.classifier(h_concat)
        return logits


class GestureDataset(Dataset):
    """Dataset for loading gesture landmark sequences from CSV/JSON files."""

    def __init__(
        self,
        data: List[Tuple[np.ndarray, int]],
        max_seq_length: int = 150,
        transform: Optional[Any] = None,
    ):
        self.data = data
        self.max_seq_length = max_seq_length
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        landmarks, label = self.data[idx]

        if self.transform:
            landmarks = self.transform(landmarks)

        seq_len = landmarks.shape[0]
        # Pad or truncate
        if seq_len > self.max_seq_length:
            landmarks = landmarks[: self.max_seq_length]
            seq_len = self.max_seq_length
        else:
            pad_len = self.max_seq_length - seq_len
            if pad_len > 0:
                landmarks = np.pad(
                    landmarks,
                    ((0, pad_len), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

        return (
            torch.tensor(landmarks, dtype=torch.float32),
            torch.tensor(min(seq_len, self.max_seq_length), dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def load_landmark_data(
    data_dir: Union[str, Path],
    format: str = "csv",
) -> List[Tuple[np.ndarray, int]]:
    """Load landmark sequences from CSV or JSON files.

    CSV format: each row has landmark_x0, landmark_y0, landmark_z0, ..., label
    JSON format: list of dicts with keys 'landmarks' and 'label'
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    samples: List[Tuple[np.ndarray, int]] = []

    if format == "csv":
        for fpath in sorted(data_path.glob("*.csv")):
            with open(fpath, "r") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                for row in reader:
                    if len(row) < 2:
                        continue
                    values = np.array(row[:-1], dtype=np.float32)
                    label = int(row[-1])
                    samples.append((values.reshape(-1, 63), label))
    elif format == "json":
        for fpath in sorted(data_path.glob("*.json")):
            with open(fpath, "r") as f:
                records = json.load(f)
                for record in records:
                    landmarks = np.array(record["landmarks"], dtype=np.float32)
                    label = int(record["label"])
                    samples.append((landmarks, label))
    else:
        raise ValueError(f"Unsupported format: {format}")

    if not samples:
        raise RuntimeError(f"No samples loaded from {data_dir}")

    logger.info(f"Loaded {len(samples)} samples from {data_dir}")
    return samples


class AugmentationPipeline:
    """Online data augmentation for landmark sequences."""

    def __init__(self, config: GestureTrainerConfig):
        self.config = config

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        if np.random.random() > self.config.augmentation_prob:
            return landmarks

        landmarks = landmarks.copy()
        # Rotation (around z-axis)
        if np.random.random() < 0.5:
            angle = np.random.uniform(-self.config.rotation_range, self.config.rotation_range)
            rot_mat = np.array([
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ], dtype=np.float32)
            landmarks = landmarks @ rot_mat.T

        # Scaling
        if np.random.random() < 0.5:
            scale = np.random.uniform(
                1.0 - self.config.scaling_range, 1.0 + self.config.scaling_range
            )
            landmarks = landmarks * scale

        # Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(0, self.config.noise_std, size=landmarks.shape).astype(
                np.float32
            )
            landmarks = landmarks + noise

        # Random temporal scaling (speed variation)
        if np.random.random() < 0.3:
            orig_len = landmarks.shape[0]
            scale_factor = np.random.uniform(0.8, 1.2)
            new_len = max(3, int(orig_len * scale_factor))
            indices = np.linspace(0, orig_len - 1, new_len).astype(int)
            landmarks = landmarks[indices]

        return landmarks


class GestureTrainer:
    """Complete training pipeline for the gesture classifier model."""

    def __init__(self, config: Optional[GestureTrainerConfig] = None):
        self.config = config or GestureTrainerConfig()
        self.device = torch.device(self.config.device)

        self._setup_directories()
        self.model: Optional[GestureClassifierModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[CosineAnnealingLR] = None
        self.criterion: Optional[nn.Module] = None
        self.writer: Optional[SummaryWriter] = None
        self.best_val_acc: float = 0.0
        self.epoch: int = 0

        logger.info(
            f"Initialized GestureTrainer (device={self.device}, "
            f"num_classes={config.num_classes if config else 30})"
        )

    def _setup_directories(self):
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.config.log_dir.mkdir(parents=True, exist_ok=True)

    def _build_model(self) -> GestureClassifierModel:
        model = GestureClassifierModel(self.config)
        return model.to(self.device)

    def _build_criterion(self) -> nn.Module:
        return nn.CrossEntropyLoss(label_smoothing=0.1)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

    def _build_scheduler(self) -> CosineAnnealingLR:
        return CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.t_max,
            eta_min=self.config.eta_min,
        )

    def _create_data_loaders(
        self, data_dir: Union[str, Path]
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        samples = load_landmark_data(data_dir)

        # Stratified split
        labels = [s[1] for s in samples]
        indices = np.arange(len(samples))
        unique_labels = np.unique(labels)

        train_idx, val_idx, test_idx = [], [], []
        for lbl in unique_labels:
            lbl_indices = indices[np.array(labels) == lbl]
            np.random.shuffle(lbl_indices)
            n = len(lbl_indices)
            n_train = int(0.7 * n)
            n_val = int(0.15 * n)
            train_idx.extend(lbl_indices[:n_train].tolist())
            val_idx.extend(lbl_indices[n_train : n_train + n_val].tolist())
            test_idx.extend(lbl_indices[n_train + n_val :].tolist())

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]

        augment = AugmentationPipeline(self.config)

        train_dataset = GestureDataset(
            train_samples,
            max_seq_length=self.config.max_seq_length,
            transform=augment,
        )
        val_dataset = GestureDataset(
            val_samples,
            max_seq_length=self.config.max_seq_length,
        )
        test_dataset = GestureDataset(
            test_samples,
            max_seq_length=self.config.max_seq_length,
        )

        # Class-balanced sampler for training
        train_labels = [train_samples[i][1] for i in range(len(train_samples))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / (class_counts + 1e-8)
        sample_weights = [class_weights[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )

        logger.info(
            f"Data loaders created: train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}"
        )
        return train_loader, val_loader, test_loader

    def train_epoch(
        self, train_loader: DataLoader
    ) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()

        for batch_idx, (landmarks, lengths, labels) in enumerate(train_loader):
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(landmarks, lengths)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if (batch_idx + 1) % self.config.log_interval == 0:
                logger.debug(
                    f"Train batch {batch_idx + 1}/{len(train_loader)}: "
                    f"loss={loss.item():.4f}"
                )

        elapsed = time.time() - start_time
        metrics = {
            "loss": total_loss / len(train_loader),
            "accuracy": correct / total if total > 0 else 0.0,
            "throughput": total / elapsed,
        }
        return metrics

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds: List[int] = []
        all_labels: List[int] = []

        for landmarks, lengths, labels in val_loader:
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(landmarks, lengths)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        metrics = {
            "loss": total_loss / len(val_loader),
            "accuracy": correct / total if total > 0 else 0.0,
        }
        self._last_val_preds = all_preds
        self._last_val_labels = all_labels
        return metrics

    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch: int,
    ):
        if self.writer is None:
            return
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"train/{key}", value, epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"val/{key}", value, epoch)

        # Log learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.writer.add_scalar("train/lr", current_lr, epoch)

    def _check_early_stopping(self, val_acc: float) -> bool:
        if val_acc > self.best_val_acc + self.config.min_delta:
            self.best_val_acc = val_acc
            self._patience_counter = 0
            return False
        self._patience_counter += 1
        if self._patience_counter >= self.config.patience:
            logger.info(
                f"Early stopping triggered after {self.config.patience} epochs "
                f"without improvement (best={self.best_val_acc:.4f})"
            )
            return True
        return False

    def _save_checkpoint(self, epoch: int, val_acc: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_acc": self.best_val_acc,
            "config": self.config,
        }
        suffix = "_best" if is_best else f"_epoch{epoch}"
        path = self.config.checkpoint_dir / f"gesture_classifier{suffix}.pt"
        torch.save(state, path)
        logger.info(f"Checkpoint saved: {path}")

    def _log_confusion_matrix(
        self, preds: List[int], labels: List[int], epoch: int
    ):
        if self.writer is None:
            return
        from torchmetrics import ConfusionMatrix

        cm = ConfusionMatrix(task="multiclass", num_classes=self.config.num_classes)
        cm_tensor = cm(
            torch.tensor(preds), torch.tensor(labels)
        )
        num_classes = min(self.config.num_classes, 20)  # cap for visualization
        fig, ax = plt.subplots(figsize=(num_classes, num_classes))
        im = ax.imshow(cm_tensor[:num_classes, :num_classes].cpu().numpy(), cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.colorbar(im, ax=ax)
        self.writer.add_figure("confusion_matrix", fig, epoch)
        plt.close(fig)

    def fit(
        self,
        data_dir: Union[str, Path],
        resume_from: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Run the full training pipeline."""
        logger.info("Starting training pipeline...")

        self.model = self._build_model()
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.writer = SummaryWriter(log_dir=str(self.config.log_dir))
        self.best_val_acc = 0.0
        self._patience_counter = 0
        self._last_val_preds = []
        self._last_val_labels = []

        epoch_start = 0
        if resume_from:
            state = torch.load(resume_from, map_location=self.device)
            self.model.load_state_dict(state["model_state_dict"])
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
            self.scheduler.load_state_dict(state["scheduler_state_dict"])
            epoch_start = state["epoch"] + 1
            self.best_val_acc = state.get("best_val_acc", 0.0)
            logger.info(f"Resumed from checkpoint: {resume_from} (epoch {epoch_start})")

        train_loader, val_loader, test_loader = self._create_data_loaders(data_dir)

        for epoch in range(epoch_start, self.config.max_epochs):
            self.epoch = epoch
            train_metrics = self.train_epoch(train_loader)
            self.scheduler.step()

            val_metrics = self.validate(val_loader)

            self._log_metrics(train_metrics, val_metrics, epoch)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.max_epochs}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.4f}, "
                f"val_loss={val_metrics['loss']:.4f}, "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            # Save checkpoint periodically
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch, val_metrics["accuracy"])

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self._save_checkpoint(epoch, val_metrics["accuracy"], is_best=True)
                self._log_confusion_matrix(
                    self._last_val_preds, self._last_val_labels, epoch
                )

            if self._check_early_stopping(val_metrics["accuracy"]):
                logger.info(f"Stopping at epoch {epoch + 1}")
                break

        # Final test evaluation
        test_metrics = self.validate(test_loader)
        logger.info(
            f"Final test metrics: loss={test_metrics['loss']:.4f}, "
            f"acc={test_metrics['accuracy']:.4f}"
        )
        self.writer.add_hparams(
            {
                "lr": self.config.learning_rate,
                "batch_size": self.config.batch_size,
                "hidden_dim": self.config.hidden_dim,
            },
            {"test_accuracy": test_metrics["accuracy"]},
        )

        if self.config.onnx_export:
            self.export_onnx()

        self.writer.close()

        return {
            "best_val_acc": self.best_val_acc,
            "test_acc": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "epochs_trained": self.epoch + 1,
        }

    @torch.no_grad()
    def export_onnx(self, output_path: Optional[Union[str, Path]] = None):
        """Export the trained model to ONNX format."""
        if self.model is None:
            raise RuntimeError("No trained model to export")

        self.model.eval()
        path = Path(output_path or self.config.checkpoint_dir / "gesture_classifier.onnx")

        dummy_input = torch.randn(
            1, self.config.max_seq_length, self.config.landmark_dim,
            device=self.device,
        )
        dummy_lengths = torch.tensor([self.config.max_seq_length], device=self.device)

        torch.onnx.export(
            self.model,
            (dummy_input, dummy_lengths),
            str(path),
            input_names=["landmarks", "lengths"],
            output_names=["logits"],
            dynamic_axes={
                "landmarks": {0: "batch_size", 1: "sequence_length"},
                "lengths": {0: "batch_size"},
                "logits": {0: "batch_size"},
            },
            opset_version=self.config.onnx_opset,
            export_params=True,
            do_constant_folding=True,
        )
        logger.info(f"Model exported to ONNX: {path}")

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        map_location: Optional[str] = None,
    ) -> "GestureTrainer":
        """Load a trainer from a checkpoint file."""
        state = torch.load(checkpoint_path, map_location=map_location)
        config = state.get("config", GestureTrainerConfig())
        trainer = cls(config)
        trainer.model = trainer._build_model()
        trainer.model.load_state_dict(state["model_state_dict"])
        trainer.best_val_acc = state.get("best_val_acc", 0.0)
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {state['epoch']})")
        return trainer


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
