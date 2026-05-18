#!/usr/bin/env python3
"""
Neurolink - Edge Benchmarking Suite
Adaptive Multimodal Communication Intelligence System

Comprehensive benchmarking framework for edge devices covering inference
latency, throughput, memory usage, power consumption, and accuracy across
multiple devices and precision levels.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = structlog.get_logger(__name__)


class DeviceCategory(Enum):
    JETSON = "jetson"
    RASPBERRY_PI = "raspberry_pi"
    X86_CPU = "x86_cpu"
    X86_GPU = "x86_gpu"
    CORAL_TPU = "coral_tpu"


class BenchmarkType(Enum):
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    POWER = "power"
    ACCURACY = "accuracy"
    COMPREHENSIVE = "comprehensive"


@dataclass
class BenchmarkSample:
    iteration: int
    latency_ms: float
    memory_mb: float
    power_mw: float
    temperature_c: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class BenchmarkReport:
    benchmark_type: BenchmarkType
    device_category: DeviceCategory
    device_name: str
    model_name: str
    precision: str
    input_shape: Tuple[int, ...]
    batch_size: int
    num_samples: int
    num_warmup: int
    samples: List[BenchmarkSample] = field(default_factory=list)

    mean_latency_ms: float = 0.0
    median_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    std_latency_ms: float = 0.0

    throughput_fps: float = 0.0
    throughput_items_per_sec: float = 0.0

    mean_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0

    mean_power_mw: float = 0.0
    peak_power_mw: float = 0.0
    energy_per_inference_mj: float = 0.0

    mean_temperature_c: float = 0.0
    peak_temperature_c: float = 0.0

    accuracy: float = 0.0
    baseline_accuracy: float = 0.0
    accuracy_delta: float = 0.0
    accuracy_acceptable: bool = True

    model_size_mb: float = 0.0
    compression_ratio: float = 1.0

    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    duration_seconds: float = 0.0

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["benchmark_type"] = self.benchmark_type.value
        d["device_category"] = self.device_category.value
        return d

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("benchmark_report_saved", path=path)

    @classmethod
    def load(cls, path: str) -> BenchmarkReport:
        with open(path, "r") as f:
            data = json.load(f)
        data["benchmark_type"] = BenchmarkType(data["benchmark_type"])
        data["device_category"] = DeviceCategory(data["device_category"])
        data["input_shape"] = tuple(data["input_shape"])
        data["samples"] = [BenchmarkSample(**s) for s in data.get("samples", [])]
        return cls(**data)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  Benchmark Report: {self.model_name}",
            f"  Device: {self.device_name} ({self.device_category.value})",
            f"  Precision: {self.precision} | Batch: {self.batch_size}",
            "-" * 60,
            f"  Latency:  mean={self.mean_latency_ms:.3f}ms  p50={self.median_latency_ms:.3f}ms  "
            f"p95={self.p95_latency_ms:.3f}ms  p99={self.p99_latency_ms:.3f}ms",
            f"  Throughput: {self.throughput_fps:.1f} FPS  ({self.throughput_items_per_sec:.0f} items/s)",
        ]
        if self.peak_memory_mb > 0:
            lines.append(f"  Memory:   mean={self.mean_memory_mb:.1f}MB  peak={self.peak_memory_mb:.1f}MB")
        if self.mean_power_mw > 0:
            lines.append(f"  Power:    mean={self.mean_power_mw:.0f}mW  "
                         f"energy={self.energy_per_inference_mj:.2f}mJ/inference")
        if self.accuracy > 0:
            lines.append(f"  Accuracy: {self.accuracy:.4f} (delta={self.accuracy_delta:.4f})")
        lines.append(f"  Model Size: {self.model_size_mb:.2f}MB (compression: {self.compression_ratio:.2f}x)")
        lines.append(f"  Duration: {self.duration_seconds:.1f}s")
        lines.append("=" * 60)
        return "\n".join(lines)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for edge device inference.

    Supports latency, throughput, memory, power, and accuracy benchmarks
    across Jetson, Raspberry Pi, and other edge devices with comparison
    and visualization capabilities.

    Usage:
        suite = BenchmarkSuite()
        report = suite.benchmark_inference_latency(inference_fn, input_shape)
        comparison = suite.compare_reports([report1, report2])
    """

    def __init__(self, results_dir: Optional[str] = None) -> None:
        self._results_dir = Path(results_dir or "benchmark_results")
        self._results_dir.mkdir(parents=True, exist_ok=True)
        self._logger = structlog.get_logger(__name__)

    def benchmark_inference_latency(
        self,
        inference_fn: Callable[[np.ndarray], Any],
        input_shape: Tuple[int, ...],
        num_warmup: int = 50,
        num_samples: int = 500,
        batch_size: int = 1,
        model_name: str = "unknown",
        device_category: DeviceCategory = DeviceCategory.X86_CPU,
        device_name: str = "unknown",
        precision: str = "fp32",
        measure_power: bool = False,
    ) -> BenchmarkReport:
        """
        Benchmark inference latency with statistical analysis.

        Args:
            inference_fn: Callable that runs inference on a single input.
            input_shape: Shape of the input tensor.
            num_warmup: Number of warmup iterations.
            num_samples: Number of timed iterations.
            batch_size: Batch dimension.
            model_name: Name of the model.
            device_category: Category of the device.
            device_name: Name of the device.
            precision: Model precision.
            measure_power: Measure power consumption during benchmark.

        Returns:
            BenchmarkReport with detailed latency statistics.
        """
        report = BenchmarkReport(
            benchmark_type=BenchmarkType.LATENCY,
            device_category=device_category,
            device_name=device_name,
            model_name=model_name,
            precision=precision,
            input_shape=input_shape,
            batch_size=batch_size,
            num_samples=num_samples,
            num_warmup=num_warmup,
            model_size_mb=self._get_model_size(inference_fn),
        )

        shape = (batch_size, *input_shape[1:])
        dummy_input = np.random.randn(*shape).astype(np.float32)

        logger.info("benchmark_started", type="latency", samples=num_samples)

        for _ in range(num_warmup):
            inference_fn(dummy_input)

        gc_start_mem = self._get_memory_usage()

        power_reader = PowerReader() if measure_power else None

        samples: List[BenchmarkSample] = []
        peak_mem = 0.0

        for i in range(num_samples):
            start = time.perf_counter()
            inference_fn(dummy_input)
            end = time.perf_counter()

            latency_ms = (end - start) * 1000.0
            mem_mb = self._get_memory_usage()
            peak_mem = max(peak_mem, mem_mb)

            power_mw = 0.0
            temp_c = 0.0
            if power_reader:
                pw, tmp = power_reader.read()
                power_mw = pw
                temp_c = tmp

            samples.append(BenchmarkSample(
                iteration=i,
                latency_ms=latency_ms,
                memory_mb=mem_mb,
                power_mw=power_mw,
                temperature_c=temp_c,
            ))

        if power_reader:
            power_reader.close()

        report.samples = samples
        report = self._compute_statistics(report, peak_mem, gc_start_mem)
        report.completed_at = datetime.utcnow().isoformat()
        report.duration_seconds = (datetime.fromisoformat(report.completed_at) -
                                   datetime.fromisoformat(report.started_at)).total_seconds()

        logger.info("benchmark_completed", type="latency",
                     mean_latency_ms=f"{report.mean_latency_ms:.3f}")
        return report

    def benchmark_throughput(
        self,
        inference_fn: Callable[[np.ndarray], Any],
        input_shape: Tuple[int, ...],
        batch_sizes: List[int] = None,
        duration_s: float = 30.0,
        model_name: str = "unknown",
        device_category: DeviceCategory = DeviceCategory.X86_CPU,
        device_name: str = "unknown",
        precision: str = "fp32",
    ) -> Dict[int, BenchmarkReport]:
        """
        Benchmark throughput at various batch sizes.

        Args:
            inference_fn: Callable that runs inference.
            input_shape: Shape of the input tensor.
            batch_sizes: List of batch sizes to test.
            duration_s: Duration of each throughput test.
            model_name: Name of the model.
            device_category: Category of the device.
            device_name: Name of the device.
            precision: Model precision.

        Returns:
            Dict mapping batch size to BenchmarkReport.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8, 16, 32]

        results: Dict[int, BenchmarkReport] = {}

        for batch_size in batch_sizes:
            shape = (batch_size, *input_shape[1:])
            dummy_input = np.random.randn(*shape).astype(np.float32)

            for _ in range(10):
                inference_fn(dummy_input)

            count = 0
            start = time.perf_counter()
            while time.perf_counter() - start < duration_s:
                inference_fn(dummy_input)
                count += 1

            elapsed = time.perf_counter() - start
            throughput = count / elapsed
            latency = (elapsed / count) * 1000.0

            report = BenchmarkReport(
                benchmark_type=BenchmarkType.THROUGHPUT,
                device_category=device_category,
                device_name=device_name,
                model_name=model_name,
                precision=precision,
                input_shape=input_shape,
                batch_size=batch_size,
                num_samples=count,
                num_warmup=10,
                mean_latency_ms=latency,
                median_latency_ms=latency,
                p95_latency_ms=latency,
                p99_latency_ms=latency,
                throughput_fps=throughput,
                throughput_items_per_sec=throughput * batch_size,
                model_size_mb=self._get_model_size(inference_fn),
            )
            report.completed_at = datetime.utcnow().isoformat()
            report.duration_seconds = elapsed

            results[batch_size] = report
            logger.info("throughput_benchmark", batch=batch_size,
                         throughput_fps=f"{throughput:.1f}",
                         latency_ms=f"{latency:.3f}")

        return results

    def benchmark_memory_usage(
        self,
        inference_fn: Callable[[np.ndarray], Any],
        input_shape: Tuple[int, ...],
        model_name: str = "unknown",
        device_category: DeviceCategory = DeviceCategory.X86_CPU,
        device_name: str = "unknown",
        precision: str = "fp32",
    ) -> BenchmarkReport:
        """Benchmark peak and average memory usage during inference."""
        report = BenchmarkReport(
            benchmark_type=BenchmarkType.MEMORY,
            device_category=device_category,
            device_name=device_name,
            model_name=model_name,
            precision=precision,
            input_shape=input_shape,
            batch_size=1,
            num_samples=50,
            num_warmup=10,
            model_size_mb=self._get_model_size(inference_fn),
        )

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(10):
            inference_fn(dummy_input)

        mem_readings: List[float] = []
        peak_mem = 0.0

        for _ in range(50):
            inference_fn(dummy_input)
            mem = self._get_memory_usage()
            mem_readings.append(mem)
            peak_mem = max(peak_mem, mem)

        report.mean_memory_mb = float(np.mean(mem_readings))
        report.peak_memory_mb = peak_mem

        logger.info("memory_benchmark_complete",
                     mean_mb=f"{report.mean_memory_mb:.1f}",
                     peak_mb=f"{report.peak_memory_mb:.1f}")
        return report

    def benchmark_power_consumption(
        self,
        inference_fn: Callable[[np.ndarray], Any],
        input_shape: Tuple[int, ...],
        duration_s: float = 60.0,
        model_name: str = "unknown",
        device_category: DeviceCategory = DeviceCategory.JETSON,
        device_name: str = "unknown",
        precision: str = "fp16",
    ) -> BenchmarkReport:
        """Benchmark power consumption during sustained inference."""
        report = BenchmarkReport(
            benchmark_type=BenchmarkType.POWER,
            device_category=device_category,
            device_name=device_name,
            model_name=model_name,
            precision=precision,
            input_shape=input_shape,
            batch_size=1,
            num_samples=0,
            num_warmup=10,
        )

        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        power_reader = PowerReader()

        for _ in range(10):
            inference_fn(dummy_input)

        power_readings: List[float] = []
        temp_readings: List[float] = []

        start = time.perf_counter()
        count = 0
        report_start = datetime.utcnow().isoformat()

        while time.perf_counter() - start < duration_s:
            inference_fn(dummy_input)
            count += 1
            if count % 10 == 0:
                pw, tmp = power_reader.read()
                power_readings.append(pw)
                temp_readings.append(tmp)

        elapsed = time.perf_counter() - start

        power_reader.close()

        report.mean_power_mw = float(np.mean(power_readings)) if power_readings else 0.0
        report.peak_power_mw = float(np.max(power_readings)) if power_readings else 0.0
        report.mean_temperature_c = float(np.mean(temp_readings)) if temp_readings else 0.0
        report.peak_temperature_c = float(np.max(temp_readings)) if temp_readings else 0.0
        report.mean_latency_ms = (elapsed / count) * 1000.0
        report.throughput_fps = count / elapsed
        report.num_samples = count
        report.started_at = report_start
        report.completed_at = datetime.utcnow().isoformat()
        report.duration_seconds = elapsed

        if report.mean_power_mw > 0 and count > 0:
            report.energy_per_inference_mj = (report.mean_power_mw * (elapsed * 1000.0 / count)) / 1000.0

        logger.info("power_benchmark_complete",
                     mean_power_mw=f"{report.mean_power_mw:.0f}",
                     mean_temp_c=f"{report.mean_temperature_c:.1f}")
        return report

    def benchmark_accuracy(
        self,
        inference_fn: Callable[[np.ndarray], np.ndarray],
        test_data: np.ndarray,
        test_labels: np.ndarray,
        baseline_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        model_name: str = "unknown",
        device_category: DeviceCategory = DeviceCategory.X86_CPU,
        device_name: str = "unknown",
        precision: str = "fp32",
        accuracy_fn: Optional[Callable] = None,
    ) -> BenchmarkReport:
        """
        Benchmark model accuracy on test data.

        Args:
            inference_fn: Quantized/optimized model inference function.
            test_data: Test data array (N, C, H, W).
            test_labels: Ground truth labels.
            baseline_fn: Baseline (full precision) model for comparison.
            model_name: Name of the model.
            device_category: Category of the device.
            device_name: Name of the device.
            precision: Model precision.
            accuracy_fn: Custom accuracy function (default: top-1).

        Returns:
            BenchmarkReport with accuracy metrics.
        """
        report = BenchmarkReport(
            benchmark_type=BenchmarkType.ACCURACY,
            device_category=device_category,
            device_name=device_name,
            model_name=model_name,
            precision=precision,
            input_shape=tuple(test_data.shape[1:]),
            batch_size=1,
            num_samples=len(test_data),
            num_warmup=0,
        )

        if accuracy_fn is None:
            accuracy_fn = self._top1_accuracy

        predictions = []
        for i in range(len(test_data)):
            input_tensor = np.expand_dims(test_data[i], 0).astype(np.float32)
            output = inference_fn(input_tensor)
            predictions.append(output)

        predictions = np.array(predictions)
        report.accuracy = accuracy_fn(predictions, test_labels)

        if baseline_fn is not None:
            baseline_predictions = []
            for i in range(len(test_data)):
                input_tensor = np.expand_dims(test_data[i], 0).astype(np.float32)
                output = baseline_fn(input_tensor)
                baseline_predictions.append(output)
            baseline_predictions = np.array(baseline_predictions)
            report.baseline_accuracy = accuracy_fn(baseline_predictions, test_labels)
            report.accuracy_delta = report.baseline_accuracy - report.accuracy
            report.accuracy_acceptable = report.accuracy_delta <= 0.01

        logger.info("accuracy_benchmark_complete",
                     accuracy=f"{report.accuracy:.4f}",
                     baseline=f"{report.baseline_accuracy:.4f}",
                     delta=f"{report.accuracy_delta:.4f}")
        return report

    def _top1_accuracy(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute top-1 accuracy."""
        pred_classes = np.argmax(predictions, axis=1) if predictions.ndim > 1 else predictions
        true_classes = np.argmax(labels, axis=1) if labels.ndim > 1 else labels
        return float(np.mean(pred_classes == true_classes))

    def _compute_statistics(
        self, report: BenchmarkReport, peak_mem: float, base_mem: float,
    ) -> BenchmarkReport:
        """Compute statistical measures from benchmark samples."""
        latencies = np.array([s.latency_ms for s in report.samples])
        mem_values = np.array([s.memory_mb for s in report.samples])
        power_values = np.array([s.power_mw for s in report.samples])
        temp_values = np.array([s.temperature_c for s in report.samples])

        report.mean_latency_ms = float(np.mean(latencies))
        report.median_latency_ms = float(np.median(latencies))
        report.p95_latency_ms = float(np.percentile(latencies, 95))
        report.p99_latency_ms = float(np.percentile(latencies, 99))
        report.min_latency_ms = float(np.min(latencies))
        report.max_latency_ms = float(np.max(latencies))
        report.std_latency_ms = float(np.std(latencies))

        report.throughput_fps = 1000.0 / max(report.mean_latency_ms, 0.001)
        report.throughput_items_per_sec = report.throughput_fps * report.batch_size

        report.mean_memory_mb = float(np.mean(mem_values))
        report.peak_memory_mb = peak_mem

        if len(power_values) > 0 and np.any(power_values > 0):
            report.mean_power_mw = float(np.mean(power_values[power_values > 0]))
            report.peak_power_mw = float(np.max(power_values))
        if len(temp_values) > 0 and np.any(temp_values > 0):
            report.mean_temperature_c = float(np.mean(temp_values[temp_values > 0]))
            report.peak_temperature_c = float(np.max(temp_values))

        if report.mean_power_mw > 0:
            report.energy_per_inference_mj = (
                report.mean_power_mw * report.mean_latency_ms / 1000.0
            )

        return report

    def _get_model_size(self, inference_fn: Callable) -> float:
        """Estimate model size in MB."""
        model_obj = getattr(inference_fn, "__self__", None)
        if model_obj:
            model_path = getattr(model_obj, "_model_path", None) or \
                         getattr(model_obj, "_model_path", None)
            if model_path and Path(model_path).exists():
                return Path(model_path).stat().st_size / (1024 * 1024)
        return 0.0

    def _get_memory_usage(self) -> float:
        """Get current process memory usage in MB."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except (ImportError, Exception):
            return 0.0

    def compare_reports(
        self,
        reports: List[BenchmarkReport],
        metric: str = "mean_latency_ms",
    ) -> Dict[str, Any]:
        """Compare multiple benchmark reports."""
        comparison: Dict[str, Any] = {
            "metric": metric,
            "best": None,
            "worst": None,
            "results": [],
            "speedups": {},
        }

        values = [getattr(r, metric, 0) for r in reports]
        names = [f"{r.model_name}_{r.precision}_{r.device_name}" for r in reports]

        if metric == "mean_latency_ms":
            best_idx = int(np.argmin(values))
            is_lower_better = True
        elif metric == "throughput_fps":
            best_idx = int(np.argmax(values))
            is_lower_better = False
        else:
            best_idx = int(np.argmin(values))
            is_lower_better = True

        for i, (report, name) in enumerate(zip(reports, names)):
            entry = {
                "name": name,
                "value": values[i],
                "model": report.model_name,
                "device": report.device_name,
                "precision": report.precision,
                "batch_size": report.batch_size,
            }

            if i != best_idx and values[best_idx] > 0:
                if is_lower_better:
                    entry["speedup_vs_best"] = values[i] / values[best_idx]
                else:
                    entry["speedup_vs_best"] = values[i] / values[best_idx] if values[i] > 0 else 0

            comparison["results"].append(entry)

        comparison["best"] = comparison["results"][best_idx]
        comparison["worst"] = comparison["results"][int(np.argmax(values))] if is_lower_better else \
                              comparison["results"][int(np.argmin(values))]

        return comparison

    def generate_report(
        self,
        reports: Union[BenchmarkReport, List[BenchmarkReport]],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a formatted benchmark report."""
        if isinstance(reports, BenchmarkReport):
            reports = [reports]

        lines: List[str] = [
            "=" * 80,
            "  Neurolink - Edge Benchmark Report",
            f"  Generated: {datetime.utcnow().isoformat()}",
            "=" * 80,
        ]

        for report in reports:
            lines.extend(["", report.summary()])

        if len(reports) > 1:
            lines.extend(["", "=" * 60, "  Cross-Device Comparison", "=" * 60])

            comparison = self.compare_reports(reports, "mean_latency_ms")
            lines.append(f"\n  Latency Comparison ({comparison['metric']}):")
            for result in comparison["results"]:
                speedup = result.get("speedup_vs_best", 1.0)
                lines.append(
                    f"    {result['name']:40s} {result['value']:10.3f} ms  "
                    f"(speedup: {speedup:.2f}x)"
                )

            comparison = self.compare_reports(reports, "throughput_fps")
            lines.append(f"\n  Throughput Comparison:")
            for result in comparison["results"]:
                lines.append(
                    f"    {result['name']:40s} {result['value']:10.1f} FPS"
                )

        report_text = "\n".join(lines)
        print(report_text)

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(report_text)
            logger.info("report_generated", path=output_path)

            json_path = Path(output_path).with_suffix(".json")
            data = {
                "generated_at": datetime.utcnow().isoformat(),
                "reports": [r.to_dict() for r in reports],
                "text_report": report_text,
            }
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return report_text

    def plot_results(
        self,
        reports: List[BenchmarkReport],
        output_dir: Optional[str] = None,
    ) -> None:
        """
        Generate visualization of benchmark results.

        Creates latency distribution, throughput comparison, and
        device comparison plots.
        """
        output_dir_path = Path(output_dir or self._results_dir / "plots")
        output_dir_path.mkdir(parents=True, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            if reports:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                ax = axes[0, 0]
                names = [f"{r.model_name}\n{r.precision}" for r in reports]
                latencies = [r.mean_latency_ms for r in reports]
                err_bars = [r.std_latency_ms for r in reports]
                ax.bar(names, latencies, yerr=err_bars, capsize=5)
                ax.set_ylabel("Latency (ms)")
                ax.set_title("Mean Inference Latency")
                ax.tick_params(axis="x", rotation=45)

                ax = axes[0, 1]
                fps_values = [r.throughput_fps for r in reports]
                ax.bar(names, fps_values)
                ax.set_ylabel("Throughput (FPS)")
                ax.set_title("Inference Throughput")
                ax.tick_params(axis="x", rotation=45)

                ax = axes[1, 0]
                for report in reports:
                    if report.samples:
                        lat = [s.latency_ms for s in report.samples[:200]]
                        ax.plot(lat, label=f"{report.model_name} ({report.precision})", alpha=0.7)
                ax.set_xlabel("Sample")
                ax.set_ylabel("Latency (ms)")
                ax.set_title("Latency Distribution over Samples")
                ax.legend()

                ax = axes[1, 1]
                mem_values = [r.peak_memory_mb for r in reports]
                power_values = [r.mean_power_mw / 1000.0 for r in reports]
                x = np.arange(len(reports))
                width = 0.35
                ax.bar(x - width / 2, mem_values, width, label="Peak Memory (MB)")
                ax.bar(x + width / 2, power_values, width, label="Avg Power (W)")
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=45)
                ax.set_title("Resource Usage")
                ax.legend()

                plt.tight_layout()
                plot_path = output_dir_path / "benchmark_comparison.png"
                plt.savefig(plot_path, dpi=150)
                plt.close()
                logger.info("benchmark_plot_saved", path=str(plot_path))

        except ImportError:
            logger.warning("matplotlib_not_available_skipping_plots")


class PowerReader:
    """Reads power consumption and temperature from the system."""

    def __init__(self) -> None:
        self._has_jetson_power = False
        self._has_nvml = False
        self._has_rpi_power = False

        try:
            if Path("/sys/bus/i2c/drivers/ina3221").exists():
                self._has_jetson_power = True
        except Exception:
            pass

        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            self._nvml_handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            self._has_nvml = True
        except ImportError:
            self._has_nvml = False

        try:
            if Path("/sys/class/thermal/thermal_zone0/temp").exists():
                self._has_rpi_power = True
        except Exception:
            pass

    def read(self) -> Tuple[float, float]:
        """Read power (mW) and temperature (C)."""
        power_mw = 0.0
        temp_c = 0.0

        if self._has_nvml and hasattr(self, "_nvml_handle"):
            try:
                import nvidia_smi
                power = nvidia_smi.nvmlDeviceGetPowerUsage(self._nvml_handle)
                power_mw = float(power)
                temp = nvidia_smi.nvmlDeviceGetTemperature(
                    self._nvml_handle, nvidia_smi.NVML_TEMPERATURE_GPU
                )
                temp_c = float(temp)
            except Exception:
                pass

        if self._has_jetson_power:
            try:
                rail_paths = [
                    "/sys/bus/i2c/drivers/ina3221/0-0040/iio:device0/in_power0_input",
                    "/sys/bus/i2c/drivers/ina3221/0-0041/iio:device1/in_power0_input",
                ]
                total_power = 0.0
                for path in rail_paths:
                    if Path(path).exists():
                        total_power += float(Path(path).read_text().strip())
                if total_power > 0:
                    power_mw = total_power
            except Exception:
                pass

        if self._has_rpi_power:
            try:
                temp_raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
                temp_c = float(temp_raw) / 1000.0
            except Exception:
                pass

        return power_mw, temp_c

    def close(self) -> None:
        if self._has_nvml:
            try:
                import nvidia_smi
                nvidia_smi.nvmlShutdown()
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neurolink Edge Benchmarking Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--model", type=str, help="Path to model file")
    parser.add_argument("--device", type=str, default="local", help="Device name")
    parser.add_argument("--device-category", type=str,
                        choices=["jetson", "raspberry_pi", "x86_cpu", "x86_gpu"],
                        default="x86_cpu")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "int8"])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--num-warmup", type=int, default=50)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    parser.add_argument("--test-data", type=str, help="Path to test data (npy)")
    parser.add_argument("--test-labels", type=str, help="Path to test labels (npy)")

    args = parser.parse_args()

    suite = BenchmarkSuite(results_dir=args.output_dir)

    if args.test_data and Path(args.test_data).exists():
        test_data = np.load(args.test_data)
        test_labels = np.load(args.test_labels) if args.test_labels else None
    else:
        test_data = np.random.randn(100, 3, 224, 224).astype(np.float32)
        test_labels = np.random.randint(0, 1000, size=100)

    def dummy_inference(x: np.ndarray) -> np.ndarray:
        time.sleep(0.01)
        return np.random.randn(x.shape[0], 1000).astype(np.float32)

    reports: List[BenchmarkReport] = []

    print(f"\n  Running benchmark on {args.device}...")

    latency_report = suite.benchmark_inference_latency(
        dummy_inference, (1, 3, 224, 224),
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        model_name=Path(args.model).stem if args.model else "dummy",
        device_category=DeviceCategory(args.device_category),
        device_name=args.device,
        precision=args.precision,
    )
    reports.append(latency_report)

    if len(args.batch_sizes) > 1 or args.batch_sizes[0] != 1:
        throughput_results = suite.benchmark_throughput(
            dummy_inference, (1, 3, 224, 224),
            batch_sizes=args.batch_sizes,
            duration_s=args.duration,
            model_name=Path(args.model).stem if args.model else "dummy",
            device_category=DeviceCategory(args.device_category),
            device_name=args.device,
            precision=args.precision,
        )
        for batch_size, report in throughput_results.items():
            reports.append(report)

    accuracy_report = suite.benchmark_accuracy(
        dummy_inference, test_data, test_labels,
        baseline_fn=dummy_inference,
        model_name=Path(args.model).stem if args.model else "dummy",
        device_category=DeviceCategory(args.device_category),
        device_name=args.device,
        precision=args.precision,
    )
    reports.append(accuracy_report)

    output_path = str(Path(args.output_dir) / f"{args.device}_benchmark_report.txt")
    suite.generate_report(reports, output_path=output_path)

    try:
        suite.plot_results(reports, output_dir=args.output_dir)
    except Exception as e:
        logger.warning("plotting_failed", error=str(e))

    print(f"\n  Benchmark complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
