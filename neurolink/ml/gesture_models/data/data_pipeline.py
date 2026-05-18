"""
Gesture data preprocessing pipeline for Neurolink.

Provides the GestureDataPipeline class that handles landmark normalization,
sequence padding/truncation, dataset splitting, class balancing, and
augmentation for gesture recognition training.
"""

import json
import logging
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torch.utils.data import ConcatDataset, Subset, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class DataPipelineConfig:
    """Configuration for the gesture data pipeline."""

    landmark_dim: int = 63
    max_seq_length: int = 150
    min_seq_length: int = 10
    batch_size: int = 64
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    num_workers: int = 4
    pin_memory: bool = True
    normalize_per_frame: bool = True
    use_rotation_invariance: bool = True
    use_scale_invariance: bool = True
    use_translation_invariance: bool = True

    # Augmentation parameters
    rotation_std: float = 0.05
    scaling_std: float = 0.05
    noise_std: float = 0.01
    time_warp_std: float = 0.05
    dropout_landmark_prob: float = 0.05


def normalize_landmarks(
    landmarks: np.ndarray,
    use_translation: bool = True,
    use_scale: bool = True,
    use_rotation: bool = True,
) -> np.ndarray:
    """Normalize landmark sequence to be translation, scale, and rotation invariant.

    Args:
        landmarks: Array of shape (seq_len, landmark_dim) where landmarks are
                   grouped as (x0, y0, z0, x1, y1, z1, ...)
        use_translation: Center landmarks around wrist (landmark 0)
        use_scale: Normalize by hand bounding box diagonal
        use_rotation: Align to wrist-middle_finger_mcp vector

    Returns:
        Normalized landmarks with same shape.
    """
    result = landmarks.copy().astype(np.float64)
    seq_len, dim = result.shape
    num_landmarks = dim // 3

    for t in range(seq_len):
        frame = result[t].reshape(-1, 3)  # (num_landmarks, 3)

        # Translation invariance: center at wrist (landmark 0)
        if use_translation:
            wrist = frame[0].copy()
            frame = frame - wrist

        # Rotation invariance: align wrist->middle_finger_mcp to y-axis
        if use_rotation and num_landmarks > 9:
            mcp_idx = 9  # middle_finger_mcp
            vec = frame[mcp_idx] - frame[0]
            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
                target = np.array([0.0, 1.0, 0.0])
                # Compute rotation matrix to align vec to target
                v = np.cross(vec, target)
                c = np.dot(vec, target)
                if abs(c) < 1.0 - 1e-9:
                    skew = np.array([
                        [0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0],
                    ])
                    rot = np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + c))
                else:
                    rot = np.eye(3) if c > 0 else -np.eye(3)
                frame = (rot @ frame.T).T

        # Scale invariance: normalize by bounding box diagonal
        if use_scale:
            min_pt = frame.min(axis=0)
            max_pt = frame.max(axis=0)
            diag = np.linalg.norm(max_pt - min_pt)
            if diag > 1e-6:
                frame = frame / diag

        result[t] = frame.reshape(-1)

    return result.astype(np.float32)


def pad_or_truncate_sequence(
    landmarks: np.ndarray, max_length: int
) -> np.ndarray:
    """Pad or truncate a landmark sequence to max_length.

    If the sequence is shorter than max_length, pad with zeros.
    If it is longer, truncate evenly from both ends or take the middle section.
    """
    seq_len = landmarks.shape[0]
    if seq_len == max_length:
        return landmarks
    if seq_len > max_length:
        # Take middle section
        start = (seq_len - max_length) // 2
        return landmarks[start : start + max_length]
    # Pad
    pad_len = max_length - seq_len
    padding = np.zeros((pad_len, landmarks.shape[1]), dtype=landmarks.dtype)
    return np.vstack([landmarks, padding])


def create_sequence_segments(
    landmarks: np.ndarray,
    segment_length: int,
    stride: int,
    min_length: int = 10,
) -> List[np.ndarray]:
    """Create overlapping segments from a landmark sequence."""
    segments = []
    seq_len = landmarks.shape[0]
    for start in range(0, seq_len - segment_length + 1, stride):
        seg = landmarks[start : start + segment_length]
        segments.append(seg)
    # Also add the last segment if we didn't cover the end
    if seq_len > segment_length and (seq_len - segment_length) % stride != 0:
        seg = landmarks[-segment_length:]
        segments.append(seg)
    return segments


class LandmarkAugmentation:
    """Online landmark sequence augmentation pipeline."""

    def __init__(self, config: DataPipelineConfig):
        self.config = config

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        result = landmarks.copy()

        # Random rotation perturbation
        if np.random.random() < 0.5:
            angle = np.random.normal(0, self.config.rotation_std)
            c, s = np.cos(angle), np.sin(angle)
            rot_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
            for t in range(result.shape[0]):
                frame = result[t].reshape(-1, 3)
                frame = (rot_z @ frame.T).T
                result[t] = frame.reshape(-1)

        # Random scaling
        if np.random.random() < 0.5:
            scale = 1.0 + np.random.normal(0, self.config.scaling_std)
            result = result * scale

        # Gaussian noise
        if np.random.random() < 0.5:
            noise = np.random.normal(
                0, self.config.noise_std, size=result.shape
            ).astype(np.float32)
            result = result + noise

        # Random landmark dropout
        if np.random.random() < self.config.dropout_landmark_prob:
            num_landmarks = result.shape[1] // 3
            drop_idx = np.random.randint(0, num_landmarks)
            result[:, drop_idx * 3 : (drop_idx + 1) * 3] = 0.0

        # Time warping (random frame duplication/removal)
        if np.random.random() < 0.3:
            seq_len = result.shape[0]
            if seq_len > 5:
                warp = np.random.normal(0, self.config.time_warp_std, seq_len)
                new_indices = np.clip(
                    np.arange(seq_len) + warp, 0, seq_len - 1
                ).astype(int)
                result = result[new_indices]

        return result


class GestureDataset(Dataset):
    """PyTorch Dataset for gesture landmark sequences."""

    def __init__(
        self,
        sequences: List[np.ndarray],
        labels: List[int],
        max_seq_length: int = 150,
        transform: Optional[Any] = None,
        normalize: bool = True,
        normalize_kwargs: Optional[Dict[str, bool]] = None,
    ):
        if len(sequences) != len(labels):
            raise ValueError(
                f"Sequences ({len(sequences)}) and labels ({len(labels)}) must match"
            )
        self.sequences = sequences
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.transform = transform
        self.normalize = normalize
        self.normalize_kwargs = normalize_kwargs or {}

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        landmarks = self.sequences[idx].copy()
        label = self.labels[idx]

        if self.normalize:
            landmarks = normalize_landmarks(landmarks, **self.normalize_kwargs)

        if self.transform:
            landmarks = self.transform(landmarks)

        original_length = landmarks.shape[0]
        landmarks = pad_or_truncate_sequence(landmarks, self.max_seq_length)
        seq_len = min(original_length, landmarks.shape[0])

        return (
            torch.tensor(landmarks, dtype=torch.float32),
            torch.tensor(seq_len, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


class GestureDataPipeline:
    """Complete gesture data preprocessing and loading pipeline."""

    def __init__(self, config: Optional[DataPipelineConfig] = None):
        self.config = config or DataPipelineConfig()
        self._rng = np.random.RandomState(self.config.random_seed)
        self.augmentation = LandmarkAugmentation(self.config)
        logger.info(
            "GestureDataPipeline initialized "
            f"(max_seq_len={self.config.max_seq_length}, "
            f"batch_size={self.config.batch_size})"
        )

    @staticmethod
    def load_csv(data_path: Union[str, Path]) -> List[Tuple[np.ndarray, int]]:
        """Load landmark sequences from CSV files.

        Expected format: each row is a flattened landmark sequence
        (landmark_x0, y0, z0, ..., landmark_x20, y20, z20) followed by label.
        """
        data_path = Path(data_path)
        samples: List[Tuple[np.ndarray, int]] = []

        if data_path.is_file():
            paths = [data_path]
        else:
            paths = sorted(data_path.glob("*.csv"))

        for fpath in paths:
            with open(fpath, "r", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                seq_buffer: List[np.ndarray] = []
                seq_labels: List[int] = []
                for row in reader:
                    if len(row) < 2:
                        continue
                    label = int(row[-1])
                    values = np.array(row[:-1], dtype=np.float32)
                    seq_buffer.append(values)
                    seq_labels.append(label)

                if seq_buffer:
                    samples.append((np.stack(seq_buffer, axis=0), seq_labels[0]))

        logger.info(f"Loaded {len(samples)} sequences from {data_path}")
        return samples

    @staticmethod
    def load_json(data_path: Union[str, Path]) -> List[Tuple[np.ndarray, int]]:
        """Load landmark sequences from JSON files.

        Expected format:
        [
            {
                "landmarks": [[x0,y0,z0, x1,y1,z1, ...], ...],
                "label": int
            },
            ...
        ]
        """
        data_path = Path(data_path)
        samples: List[Tuple[np.ndarray, int]] = []

        if data_path.is_file():
            paths = [data_path]
        else:
            paths = sorted(data_path.glob("*.json"))

        for fpath in paths:
            with open(fpath, "r") as f:
                records = json.load(f)
                for record in records:
                    landmarks = np.array(record["landmarks"], dtype=np.float32)
                    label = int(record["label"])
                    samples.append((landmarks, label))

        logger.info(f"Loaded {len(samples)} sequences from {data_path}")
        return samples

    def filter_sequences(
        self, samples: List[Tuple[np.ndarray, int]]
    ) -> List[Tuple[np.ndarray, int]]:
        """Remove sequences that are too short or too long."""
        filtered = []
        for landmarks, label in samples:
            seq_len = landmarks.shape[0]
            if self.config.min_seq_length <= seq_len <= self.config.max_seq_length * 2:
                filtered.append((landmarks, label))
            else:
                logger.debug(
                    f"Filtered sequence of length {seq_len} (label={label})"
                )
        removed = len(samples) - len(filtered)
        if removed > 0:
            logger.info(f"Filtered {removed} sequences by length")
        return filtered

    def balance_classes(
        self, samples: List[Tuple[np.ndarray, int]]
    ) -> List[Tuple[np.ndarray, int]]:
        """Oversample minority classes to balance the dataset."""
        labels = [s[1] for s in samples]
        counter = Counter(labels)
        max_count = max(counter.values())
        if max_count == min(counter.values()):
            logger.info("Classes are already balanced")
            return samples

        balanced = list(samples)
        for label_id, count in counter.items():
            if count < max_count:
                label_samples = [s for s in samples if s[1] == label_id]
                oversample_count = max_count - count
                extra = self._rng.choice(
                    label_samples, size=oversample_count, replace=True
                ).tolist()
                balanced.extend(extra)

        logger.info(
            f"Balanced dataset: {len(samples)} -> {len(balanced)} samples"
        )
        return balanced

    def split_data(
        self, samples: List[Tuple[np.ndarray, int]]
    ) -> Tuple[
        List[Tuple[np.ndarray, int]],
        List[Tuple[np.ndarray, int]],
        List[Tuple[np.ndarray, int]],
    ]:
        """Stratified train/val/test split."""
        labels = [s[1] for s in samples]
        unique_labels = np.unique(labels)
        indices_by_class: Dict[int, List[int]] = defaultdict(list)
        for i, lbl in enumerate(labels):
            indices_by_class[lbl].append(i)

        train_idx, val_idx, test_idx = [], [], []
        for lbl in unique_labels:
            lbl_indices = indices_by_class[lbl]
            self._rng.shuffle(lbl_indices)
            n = len(lbl_indices)
            n_val = max(1, int(self.config.val_split * n))
            n_test = max(1, int(self.config.test_split * n))
            n_train = n - n_val - n_test
            if n_train < 1:
                n_train = 1
                n_val = max(1, (n - n_train) // 2)
                n_test = n - n_train - n_val

            train_idx.extend(lbl_indices[:n_train])
            val_idx.extend(lbl_indices[n_train : n_train + n_val])
            test_idx.extend(lbl_indices[n_train + n_val :])

        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        test_samples = [samples[i] for i in test_idx]

        logger.info(
            f"Split: train={len(train_samples)}, "
            f"val={len(val_samples)}, test={len(test_samples)}"
        )
        return train_samples, val_samples, test_samples

    def create_dataloaders(
        self,
        train_samples: List[Tuple[np.ndarray, int]],
        val_samples: List[Tuple[np.ndarray, int]],
        test_samples: List[Tuple[np.ndarray, int]],
        augment: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create DataLoaders from sample lists."""
        transform = self.augmentation if augment else None

        train_dataset = GestureDataset(
            [s[0] for s in train_samples],
            [s[1] for s in train_samples],
            max_seq_length=self.config.max_seq_length,
            transform=transform,
        )
        val_dataset = GestureDataset(
            [s[0] for s in val_samples],
            [s[1] for s in val_samples],
            max_seq_length=self.config.max_seq_length,
        )
        test_dataset = GestureDataset(
            [s[0] for s in test_samples],
            [s[1] for s in test_samples],
            max_seq_length=self.config.max_seq_length,
        )

        # Class-balanced sampler
        train_labels = [s[1] for s in train_samples]
        label_counts = Counter(train_labels)
        weights = [1.0 / label_counts[lbl] for lbl in train_labels]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

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

        return train_loader, val_loader, test_loader

    def run(
        self,
        data_path: Union[str, Path],
        format: str = "csv",
        balance: bool = True,
        augment: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Run the full pipeline end-to-end.

        Args:
            data_path: Path to CSV file, JSON file, or directory containing files.
            format: Data format - "csv" or "json".
            balance: Whether to balance classes via oversampling.
            augment: Whether to apply online augmentation to training data.

        Returns:
            Tuple of (train_loader, val_loader, test_loader).
        """
        logger.info(f"Running data pipeline on {data_path} (format={format})")

        if format == "csv":
            samples = self.load_csv(data_path)
        elif format == "json":
            samples = self.load_json(data_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        samples = self.filter_sequences(samples)

        if balance:
            samples = self.balance_classes(samples)

        train_samples, val_samples, test_samples = self.split_data(samples)

        return self.create_dataloaders(
            train_samples, val_samples, test_samples, augment=augment
        )

    def get_class_weights(
        self, samples: List[Tuple[np.ndarray, int]]
    ) -> torch.Tensor:
        """Compute class weights for weighted loss functions."""
        labels = [s[1] for s in samples]
        counter = Counter(labels)
        num_classes = max(labels) + 1
        counts = [counter.get(i, 0) for i in range(num_classes)]
        total = sum(counts)
        weights = [total / (num_classes * c) if c > 0 else 1.0 for c in counts]
        return torch.tensor(weights, dtype=torch.float32)
