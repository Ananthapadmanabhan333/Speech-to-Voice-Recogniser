"""
Gesture model evaluation module for Neurolink.

Provides the GestureEvaluator class for computing classification metrics,
confusion matrices, per-class performance, sequence accuracy, and
real-time FPS/latency benchmarking.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


@dataclass
class EvaluationMetrics:
    """Container for gesture classification evaluation metrics."""

    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    per_class_precision: Optional[np.ndarray] = None
    per_class_recall: Optional[np.ndarray] = None
    per_class_f1: Optional[np.ndarray] = None
    per_class_support: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    sequence_accuracy: float = 0.0
    avg_inference_time_ms: float = 0.0
    fps: float = 0.0
    total_samples: int = 0
    correct_predictions: int = 0


class GestureEvaluator:
    """Evaluator for gesture classification models providing comprehensive metrics."""

    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.model.eval()
        self.class_names = class_names
        logger.info(
            f"GestureEvaluator initialized (device={self.device})"
        )

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        benchmark: bool = True,
    ) -> EvaluationMetrics:
        """Run full evaluation on a DataLoader.

        Args:
            dataloader: DataLoader providing (landmarks, lengths, labels) tuples.
            benchmark: Whether to measure latency and FPS.

        Returns:
            EvaluationMetrics object with all computed metrics.
        """
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss = 0.0
        criterion = nn.CrossEntropyLoss()

        inference_times: List[float] = []
        total_frames = 0

        for landmarks, lengths, labels in dataloader:
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            labels = labels.to(self.device)

            if benchmark:
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                start_time = time.perf_counter()

            logits = self.model(landmarks, lengths)

            if benchmark:
                torch.cuda.synchronize() if self.device.type == "cuda" else None
                elapsed = time.perf_counter() - start_time
                inference_times.append(elapsed)
                total_frames += landmarks.size(0)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        return self._compute_metrics(
            all_preds, all_labels, total_loss / len(dataloader),
            inference_times, total_frames, len(dataloader.dataset),
        )

    def _compute_metrics(
        self,
        all_preds: List[int],
        all_labels: List[int],
        avg_loss: float,
        inference_times: List[float],
        total_frames: int,
        total_samples: int,
    ) -> EvaluationMetrics:
        """Compute all metrics from predictions and ground truth."""
        preds = np.array(all_preds)
        labels = np.array(all_labels)
        num_classes = max(max(preds), max(labels)) + 1

        # Confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for p, l in zip(preds, labels):
            cm[l, p] += 1

        # Per-class metrics
        support = cm.sum(axis=1)
        correct_per_class = np.diag(cm)

        with np.errstate(divide="ignore", invalid="ignore"):
            precision = np.diag(cm) / (cm.sum(axis=0) + 1e-15)
            recall = np.diag(cm) / (support + 1e-15)
            f1 = 2 * precision * recall / (precision + recall + 1e-15)

        # Macro-averages
        macro_precision = float(np.nanmean(precision[support > 0]))
        macro_recall = float(np.nanmean(recall[support > 0]))
        macro_f1 = float(np.nanmean(f1[support > 0]))

        accuracy = float((preds == labels).sum()) / len(labels)

        # Timing metrics
        avg_time_ms = (
            (np.mean(inference_times) * 1000.0) if inference_times else 0.0
        )
        fps = total_frames / sum(inference_times) if inference_times else 0.0

        class_names = self.class_names or [
            f"class_{i}" for i in range(num_classes)
        ]

        metrics = EvaluationMetrics(
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1_score=macro_f1,
            confusion_matrix=cm,
            per_class_precision=precision,
            per_class_recall=recall,
            per_class_f1=f1,
            per_class_support=support,
            class_names=class_names[:num_classes],
            avg_inference_time_ms=avg_time_ms,
            fps=fps,
            total_samples=total_samples,
            correct_predictions=int((preds == labels).sum()),
        )

        logger.info(
            f"Evaluation: accuracy={accuracy:.4f}, precision={macro_precision:.4f}, "
            f"recall={macro_recall:.4f}, f1={macro_f1:.4f}, "
            f"fps={fps:.1f}, latency={avg_time_ms:.2f}ms"
        )

        return metrics

    @torch.no_grad()
    def evaluate_sequence_accuracy(
        self,
        dataloader: DataLoader,
        sequence_length: int = 10,
        majority_vote: bool = True,
    ) -> float:
        """Evaluate accuracy on grouped sequences using majority voting."""
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []

        for landmarks, lengths, labels in dataloader:
            landmarks = landmarks.to(self.device)
            lengths = lengths.to(self.device)
            logits = self.model(landmarks, lengths)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # Group into sequences and apply majority vote
        seq_correct = 0
        seq_total = 0
        for start in range(0, len(all_preds), sequence_length):
            chunk_preds = all_preds[start : start + sequence_length]
            chunk_labels = all_labels[start : start + sequence_length]
            if len(chunk_preds) < sequence_length // 2:
                continue
            if majority_vote:
                final_pred = max(set(chunk_preds), key=chunk_preds.count)
                final_label = max(set(chunk_labels), key=chunk_labels.count)
            else:
                final_pred = chunk_preds[-1]
                final_label = chunk_labels[-1]
            if final_pred == final_label:
                seq_correct += 1
            seq_total += 1

        seq_acc = seq_correct / seq_total if seq_total > 0 else 0.0
        logger.info(f"Sequence accuracy (len={sequence_length}): {seq_acc:.4f}")
        return seq_acc

    @torch.no_grad()
    def benchmark_fps(
        self,
        input_shape: Tuple[int, int, int] = (1, 150, 63),
        num_runs: int = 500,
        warmup_runs: int = 50,
    ) -> Dict[str, float]:
        """Benchmark model inference FPS and latency."""
        self.model.eval()

        dummy_input = torch.randn(*input_shape, device=self.device)
        dummy_lengths = torch.tensor(
            [input_shape[1]], device=self.device
        )

        # Warmup
        for _ in range(warmup_runs):
            _ = self.model(dummy_input, dummy_lengths)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        latencies: List[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.model(dummy_input, dummy_lengths)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)

        latencies = np.array(latencies)
        avg_latency = float(np.mean(latencies) * 1000.0)  # ms
        p50_latency = float(np.median(latencies) * 1000.0)
        p95_latency = float(np.percentile(latencies, 95) * 1000.0)
        p99_latency = float(np.percentile(latencies, 99) * 1000.0)
        fps = float(input_shape[0] / np.mean(latencies))

        results = {
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "fps": fps,
            "batch_size": input_shape[0],
            "sequence_length": input_shape[1],
        }

        logger.info(
            f"FPS benchmark: {fps:.1f} fps, "
            f"avg latency={avg_latency:.2f}ms, "
            f"p95={p95_latency:.2f}ms"
        )

        return results

    def confusion_matrix_report(
        self, metrics: EvaluationMetrics, output_path: Optional[Path] = None
    ) -> str:
        """Generate a text-based confusion matrix report."""
        if metrics.confusion_matrix is None:
            return "No confusion matrix available."

        cm = metrics.confusion_matrix
        num_classes = cm.shape[0]
        names = metrics.class_names or [f"C{i}" for i in range(num_classes)]
        max_name_len = max(len(n) for n in names)

        lines = ["Confusion Matrix:", "=" * 80]
        header = " " * (max_name_len + 2)
        for i in range(num_classes):
            header += f"{names[i][:6]:>7}"
        lines.append(header)
        lines.append("-" * 80)

        for i in range(num_classes):
            row = f"{names[i]:>{max_name_len}}  "
            for j in range(num_classes):
                row += f"{cm[i, j]:>7d}"
            lines.append(row)

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Confusion matrix report saved to {output_path}")

        return report

    def per_class_report(
        self, metrics: EvaluationMetrics, output_path: Optional[Path] = None
    ) -> str:
        """Generate a per-class metrics report."""
        if metrics.per_class_precision is None:
            return "No per-class metrics available."

        num_classes = len(metrics.per_class_precision)
        names = metrics.class_names or [f"C{i}" for i in range(num_classes)]

        lines = [
            "Per-Class Metrics:",
            "=" * 80,
            f"{'Class':>20}  {'Precision':>10}  {'Recall':>10}  "
            f"{'F1-Score':>10}  {'Support':>8}",
            "-" * 80,
        ]

        for i in range(num_classes):
            lines.append(
                f"{names[i]:>20}  "
                f"{metrics.per_class_precision[i]:>10.4f}  "
                f"{metrics.per_class_recall[i]:>10.4f}  "
                f"{metrics.per_class_f1[i]:>10.4f}  "
                f"{metrics.per_class_support[i]:>8d}"
            )

        lines.append("-" * 80)
        lines.append(
            f"{'Macro Avg':>20}  "
            f"{metrics.precision:>10.4f}  "
            f"{metrics.recall:>10.4f}  "
            f"{metrics.f1_score:>10.4f}  "
            f"{metrics.total_samples:>8d}"
        )
        lines.append(
            f"{'Accuracy':>20}  "
            f"{metrics.accuracy:>10.4f}"
        )

        report = "\n".join(lines)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Per-class report saved to {output_path}")

        return report

    def full_report(
        self, dataloader: DataLoader, output_dir: Optional[Path] = None
    ) -> EvaluationMetrics:
        """Generate a complete evaluation report with all metrics."""
        metrics = self.evaluate(dataloader, benchmark=True)
        seq_acc = self.evaluate_sequence_accuracy(dataloader)
        metrics.sequence_accuracy = seq_acc
        benchmark = self.benchmark_fps()

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            summary = {
                "accuracy": metrics.accuracy,
                "precision": metrics.precision,
                "recall": metrics.recall,
                "f1_score": metrics.f1_score,
                "sequence_accuracy": metrics.sequence_accuracy,
                "fps": metrics.fps,
                "avg_latency_ms": metrics.avg_inference_time_ms,
                "benchmark": benchmark,
                "total_samples": metrics.total_samples,
                "correct_predictions": metrics.correct_predictions,
            }
            with open(output_dir / "evaluation_summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"Evaluation summary saved to {output_dir / 'evaluation_summary.json'}")

            self.confusion_matrix_report(metrics, output_dir / "confusion_matrix.txt")
            self.per_class_report(metrics, output_dir / "per_class_metrics.txt")

        return metrics
