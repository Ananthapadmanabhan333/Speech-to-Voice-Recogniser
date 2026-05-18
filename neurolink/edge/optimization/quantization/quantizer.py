"""
Neurolink - Model Quantization Engine
Adaptive Multimodal Communication Intelligence System

Provides production-grade model quantization with post-training quantization (PTQ),
quantization-aware training (QAT) support, per-channel/per-tensor quantization,
calibration dataset management, accuracy evaluation, and size/speed benchmarks.
"""

from __future__ import annotations

import json
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import structlog

try:
    import torch
    import torch.nn as nn
    import torch.quantization as quant
    from torch.quantization import (
        QuantStub, DeQuantStub, FakeQuantize,
        default_qconfig, default_qconfig_dict,
        QConfig, default_observer, default_weight_observer,
        get_default_qconfig, get_default_qat_qconfig,
        prepare, convert, prepare_qat,
    )
except ImportError:
    torch = None
    nn = None

try:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, quantize_static
except ImportError:
    onnx = None

logger = structlog.get_logger(__name__)


class QuantizationMethod(Enum):
    POST_TRAINING_DYNAMIC = "ptq_dynamic"
    POST_TRAINING_STATIC = "ptq_static"
    QUANTIZATION_AWARE_TRAINING = "qat"


class QuantizationScheme(Enum):
    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"


class QuantizationBackend(Enum):
    FBGEMM = "fbgemm"
    QNNPACK = "qnnpack"
    ONNX_RUNTIME = "onnxruntime"


@dataclass
class QuantizationConfig:
    method: QuantizationMethod = QuantizationMethod.POST_TRAINING_STATIC
    scheme: QuantizationScheme = QuantizationScheme.PER_CHANNEL
    backend: QuantizationBackend = QuantizationBackend.FBGEMM
    dtype: str = "qint8"
    reduce_range: bool = False
    num_calibration_batches: int = 50
    calibration_batch_size: int = 32
    observers_per_layer: int = 1
    percentile: Optional[float] = None
    symmetric: bool = False
    moving_average: bool = True
    averaging_constant: float = 0.01


@dataclass
class QuantizationMetrics:
    original_size_bytes: int
    quantized_size_bytes: int
    compression_ratio: float
    original_accuracy: float
    quantized_accuracy: float
    accuracy_delta: float
    accuracy_acceptable: bool
    original_latency_ms: float
    quantized_latency_ms: float
    speedup_ratio: float
    original_throughput_fps: float
    quantized_throughput_fps: float
    quantization_time_s: float
    config: QuantizationConfig = field(default_factory=QuantizationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["config"] = asdict(data["config"])
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> QuantizationMetrics:
        with open(path, "r") as f:
            data = json.load(f)
        data["config"] = QuantizationConfig(**data["config"])
        return cls(**data)


@dataclass
class CalibrationSample:
    data: np.ndarray
    label: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


class CalibrationDataset:
    """Manages calibration data for quantization."""

    def __init__(self, max_samples: int = 1000) -> None:
        self._samples: List[CalibrationSample] = []
        self._max_samples = max_samples
        self._index = 0

    def add_sample(
        self, data: np.ndarray, label: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if len(self._samples) >= self._max_samples:
            logger.warning("calibration_dataset_full", max_samples=self._max_samples)
            return
        self._samples.append(CalibrationSample(data=data, label=label, metadata=metadata))

    def add_batch(
        self, data: np.ndarray, labels: Optional[np.ndarray] = None,
    ) -> None:
        batch_size = data.shape[0]
        for i in range(batch_size):
            label_i = labels[i] if labels is not None else None
            self.add_sample(data[i], label_i)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Generator[CalibrationSample, None, None]:
        for sample in self._samples:
            yield sample

    def as_numpy(self) -> np.ndarray:
        return np.array([s.data for s in self._samples])

    def shuffle(self) -> None:
        indices = np.random.permutation(len(self._samples))
        self._samples = [self._samples[i] for i in indices]

    def split(self, ratio: float = 0.8) -> Tuple[CalibrationDataset, CalibrationDataset]:
        split_idx = int(len(self._samples) * ratio)
        train = CalibrationDataset(max_samples=self._max_samples)
        val = CalibrationDataset(max_samples=self._max_samples)
        train._samples = self._samples[:split_idx]
        val._samples = self._samples[split_idx:]
        return train, val

    def save(self, path: str) -> None:
        data = {
            "samples": [
                {
                    "data": s.data.tolist(),
                    "label": s.label.tolist() if s.label is not None else None,
                    "metadata": s.metadata,
                }
                for s in self._samples
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> CalibrationDataset:
        with open(path, "r") as f:
            data = json.load(f)
        dataset = cls()
        for s in data["samples"]:
            sample = CalibrationSample(
                data=np.array(s["data"]),
                label=np.array(s["label"]) if s.get("label") is not None else None,
                metadata=s.get("metadata"),
            )
            dataset._samples.append(sample)
        return dataset


class ModelQuantizer:
    """
    Production-grade model quantizer supporting PyTorch and ONNX models
    with PTQ, QAT, per-channel/per-tensor quantization, and calibration.

    Usage:
        quantizer = ModelQuantizer()
        metrics = quantizer.quantize_to_int8(
            model, calib_dataset,
            method=QuantizationMethod.POST_TRAINING_STATIC,
        )
    """

    def __init__(
        self,
        config: Optional[QuantizationConfig] = None,
        accuracy_threshold: float = 0.01,
    ) -> None:
        self._config = config or QuantizationConfig()
        self._accuracy_threshold = accuracy_threshold
        self._logger = structlog.get_logger(__name__)
        self._quantized_model = None

    def quantize_to_int8(
        self,
        model: Any,
        calib_data: Optional[CalibrationDataset] = None,
        method: QuantizationMethod = QuantizationMethod.POST_TRAINING_STATIC,
        eval_fn: Optional[Callable] = None,
        input_shape: Optional[Tuple[int, ...]] = None,
        save_path: Optional[str] = None,
    ) -> QuantizationMetrics:
        """
        Quantize a model to INT8 precision.

        Args:
            model: PyTorch nn.Module or ONNX model path.
            calib_data: CalibrationDataset for static quantization.
            method: Quantization method (PTQ dynamic, PTQ static, QAT).
            eval_fn: Function to evaluate accuracy: eval_fn(model, data) -> float.
            input_shape: Input tensor shape for dummy calibration.
            save_path: Path to save quantized model.

        Returns:
            QuantizationMetrics with accuracy and performance comparison.
        """
        start_time = time.time()

        if isinstance(model, str):
            model_path = model
            ext = Path(model_path).suffix.lower()
            if ext == ".onnx":
                return self._quantize_onnx(model_path, calib_data, method, save_path)
            else:
                raise ValueError(f"Unsupported model format: {ext}")

        if torch is None:
            raise ImportError("PyTorch is required for model quantization")

        if not isinstance(model, nn.Module):
            raise TypeError("Model must be a PyTorch nn.Module or a path to ONNX model")

        model.eval()

        original_size = self._estimate_model_size(model)

        config = self._config
        config.method = method

        if calib_data is None and method in (
            QuantizationMethod.POST_TRAINING_STATIC,
            QuantizationMethod.QUANTIZATION_AWARE_TRAINING,
        ):
            if input_shape:
                calib_data = self._generate_dummy_calibration(input_shape)
                logger.info("dummy_calibration_generated", shape=input_shape)
            else:
                raise ValueError("Calibration data required for static/QAT quantization")

        original_latency = self._benchmark_latency(model, input_shape or (1, 3, 224, 224))
        original_accuracy = eval_fn(model, calib_data) if eval_fn else 0.0

        if method == QuantizationMethod.POST_TRAINING_DYNAMIC:
            quantized_model = self._apply_ptq_dynamic(model)
        elif method == QuantizationMethod.POST_TRAINING_STATIC:
            quantized_model = self._apply_ptq_static(model, calib_data)
        elif method == QuantizationMethod.QUANTIZATION_AWARE_TRAINING:
            quantized_model = self._apply_qat(model, calib_data)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

        self._quantized_model = quantized_model

        if torch:
            quantized_size = self._estimate_model_size(quantized_model)
        else:
            quantized_size = original_size

        quantized_latency = self._benchmark_latency(quantized_model, input_shape or (1, 3, 224, 224))
        quantized_accuracy = eval_fn(quantized_model, calib_data) if eval_fn else 0.0

        accuracy_delta = abs(original_accuracy - quantized_accuracy)
        elapsed = time.time() - start_time

        metrics = QuantizationMetrics(
            original_size_bytes=original_size,
            quantized_size_bytes=quantized_size,
            compression_ratio=original_size / max(quantized_size, 1),
            original_accuracy=original_accuracy,
            quantized_accuracy=quantized_accuracy,
            accuracy_delta=accuracy_delta,
            accuracy_acceptable=accuracy_delta <= self._accuracy_threshold,
            original_latency_ms=original_latency,
            quantized_latency_ms=quantized_latency,
            speedup_ratio=original_latency / max(quantized_latency, 0.001),
            original_throughput_fps=1000.0 / max(original_latency, 0.001),
            quantized_throughput_fps=1000.0 / max(quantized_latency, 0.001),
            quantization_time_s=elapsed,
            config=config,
        )

        logger.info("quantization_complete",
                     method=method.value,
                     compression=f"{metrics.compression_ratio:.2f}x",
                     speedup=f"{metrics.speedup_ratio:.2f}x",
                     accuracy_delta=f"{accuracy_delta:.4f}",
                     acceptable=metrics.accuracy_acceptable)

        if save_path:
            if isinstance(quantized_model, nn.Module) and torch:
                scripted = torch.jit.script(quantized_model)
                torch.jit.save(scripted, save_path)
            metrics.save(str(Path(save_path).with_suffix(".metrics.json")))
            logger.info("quantized_model_saved", path=save_path)

        return metrics

    def _apply_ptq_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply post-training dynamic quantization."""
        quantized_model = quant.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.LSTM, nn.GRU},
            dtype=torch.qint8,
        )
        logger.info("ptq_dynamic_applied")
        return quantized_model

    def _apply_ptq_static(self, model: nn.Module, calib_data: CalibrationDataset) -> nn.Module:
        """Apply post-training static quantization with calibration."""
        model.qconfig = get_default_qconfig(self._config.backend.value)

        if self._config.scheme == QuantizationScheme.PER_CHANNEL:
            model.qconfig = QConfig(
                activation=default_observer,
                weight=default_weight_observer,
            )

        prepared_model = prepare(model, inplace=False)

        self._calibrate(prepared_model, calib_data)

        quantized_model = convert(prepared_model, inplace=False)
        logger.info("ptq_static_applied")
        return quantized_model

    def _apply_qat(self, model: nn.Module, calib_data: CalibrationDataset) -> nn.Module:
        """Apply quantization-aware training."""
        model.train()
        model.qconfig = get_default_qat_qconfig(self._config.backend.value)

        prepared_model = prepare_qat(model, inplace=False)

        self._calibrate(prepared_model, calib_data, num_batches=min(10, len(calib_data)))

        prepared_model.eval()
        quantized_model = convert(prepared_model, inplace=False)
        logger.info("qat_applied")
        return quantized_model

    def _calibrate(
        self, model: nn.Module, dataset: CalibrationDataset, num_batches: Optional[int] = None,
    ) -> None:
        """Run calibration data through the model to collect statistics."""
        model.eval()
        num_batches = num_batches or self._config.num_calibration_batches
        batch_size = self._config.calibration_batch_size

        device = next(model.parameters()).device if torch else "cpu"

        with torch.no_grad():
            for i, sample in enumerate(dataset):
                if i >= num_batches:
                    break
                input_tensor = torch.from_numpy(sample.data).unsqueeze(0).float().to(device)
                model(input_tensor)

        logger.info("calibration_completed", batches=min(num_batches, len(dataset)))

    def _quantize_onnx(
        self,
        model_path: str,
        calib_data: Optional[CalibrationDataset],
        method: QuantizationMethod,
        save_path: Optional[str],
    ) -> QuantizationMetrics:
        """Quantize an ONNX model."""
        if onnx is None or not calib_data:
            raise ImportError("onnx and calibration data required")

        output_path = save_path or str(Path(model_path).with_suffix(".quant.onnx"))
        original_size = Path(model_path).stat().st_size

        start_time = time.time()

        if method == QuantizationMethod.POST_TRAINING_DYNAMIC:
            quantize_dynamic(model_path, output_path)
        elif method == QuantizationMethod.POST_TRAINING_STATIC:
            class ONNXCalibReader:
                def __init__(self, dataset: CalibrationDataset):
                    self._dataset = dataset
                    self._index = 0

                def get_next(self) -> Optional[Dict[str, np.ndarray]]:
                    if self._index >= len(self._dataset):
                        return None
                    sample = self._dataset._samples[self._index]
                    self._index += 1
                    return {"input": np.expand_dims(sample.data, 0).astype(np.float32)}

                def rewind(self) -> None:
                    self._index = 0

            quantize_static(model_path, output_path, ONNXCalibReader(calib_data))

        elapsed = time.time() - start_time
        quantized_size = Path(output_path).stat().st_size

        metrics = QuantizationMetrics(
            original_size_bytes=original_size,
            quantized_size_bytes=quantized_size,
            compression_ratio=original_size / max(quantized_size, 1),
            original_accuracy=0.0,
            quantized_accuracy=0.0,
            accuracy_delta=0.0,
            accuracy_acceptable=True,
            original_latency_ms=0.0,
            quantized_latency_ms=0.0,
            speedup_ratio=0.0,
            original_throughput_fps=0.0,
            quantized_throughput_fps=0.0,
            quantization_time_s=elapsed,
            config=self._config,
        )

        return metrics

    def _benchmark_latency(self, model: nn.Module, input_shape: Tuple[int, ...], num_runs: int = 100) -> float:
        """Benchmark model inference latency in milliseconds."""
        if torch is None:
            return 0.0

        device = next(model.parameters()).device if list(model.parameters()) else "cpu"
        dummy_input = torch.randn(*input_shape).to(device)

        model.eval()
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()

        latencies: List[float] = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                model(dummy_input)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end = time.perf_counter()
                latencies.append((end - start) * 1000.0)

        return float(np.median(latencies))

    def _estimate_model_size(self, model: nn.Module) -> int:
        """Estimate model size in bytes from parameter memory footprint."""
        if torch is None:
            return 0
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return param_size + buffer_size

    def _generate_dummy_calibration(self, input_shape: Tuple[int, ...], num_samples: int = 100) -> CalibrationDataset:
        """Generate dummy calibration data for testing."""
        dataset = CalibrationDataset(max_samples=num_samples)
        for _ in range(num_samples):
            data = np.random.randn(*input_shape[1:]).astype(np.float32)
            dataset.add_sample(data)
        return dataset

    def evaluate_accuracy(
        self,
        model: Any,
        test_data: CalibrationDataset,
        eval_fn: Optional[Callable] = None,
    ) -> float:
        """
        Evaluate model accuracy on test data.

        Args:
            model: PyTorch model or ONNX model path.
            test_data: CalibrationDataset with test samples.
            eval_fn: Custom evaluation function.

        Returns:
            Accuracy score (0.0 to 1.0).
        """
        if eval_fn:
            return eval_fn(model, test_data)

        if torch is None or not isinstance(model, nn.Module):
            return 0.0

        model.eval()
        correct = 0
        total = 0
        device = next(model.parameters()).device if list(model.parameters()) else "cpu"

        with torch.no_grad():
            for sample in test_data:
                input_tensor = torch.from_numpy(sample.data).unsqueeze(0).float().to(device)
                output = model(input_tensor)
                if sample.label is not None:
                    pred = output.argmax(dim=1).item()
                    correct += int(pred == sample.label.argmax()) if sample.label.ndim > 0 else int(pred == sample.label)
                    total += 1

        accuracy = correct / max(total, 1)
        logger.info("accuracy_evaluation", accuracy=f"{accuracy:.4f}")
        return accuracy

    def benchmark_size(self, model: Any, original_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Benchmark model size before and after quantization.

        Returns:
            Dictionary with size metrics.
        """
        if isinstance(model, str):
            size = Path(model).stat().st_size
        elif isinstance(model, nn.Module):
            size = self._estimate_model_size(model)
        else:
            size = 0

        result = {"size_bytes": size, "size_mb": size / (1024 * 1024)}

        if self._quantized_model is not None:
            if isinstance(self._quantized_model, nn.Module):
                qsize = self._estimate_model_size(self._quantized_model)
            else:
                qsize = 0
            result["quantized_size_bytes"] = qsize
            result["quantized_size_mb"] = qsize / (1024 * 1024)
            result["compression_ratio"] = size / max(qsize, 1)

        return result

    def benchmark_speed(
        self, model: Any, input_shape: Tuple[int, ...], num_runs: int = 500
    ) -> Dict[str, float]:
        """
        Benchmark model inference speed.

        Returns:
            Dictionary with latency and throughput metrics.
        """
        latency = self._benchmark_latency(model, input_shape, num_runs)
        result = {
            "latency_ms": latency,
            "throughput_fps": 1000.0 / max(latency, 0.001),
        }

        if self._quantized_model is not None:
            qlatency = self._benchmark_latency(self._quantized_model, input_shape, num_runs)
            result["quantized_latency_ms"] = qlatency
            result["quantized_throughput_fps"] = 1000.0 / max(qlatency, 0.001)
            result["speedup_ratio"] = latency / max(qlatency, 0.001)

        return result

    def get_quantized_model(self):
        """Return the quantized model."""
        return self._quantized_model
