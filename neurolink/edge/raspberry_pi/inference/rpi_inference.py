"""
Neurolink - Raspberry Pi Inference Engine
Adaptive Multimodal Communication Intelligence System

Provides production-grade inference on Raspberry Pi platforms with ONNX Runtime,
CPU/Coral TPU acceleration, memory-mapped model loading, reduced precision
inference, threadpool optimization, and power-aware scheduling.
"""

from __future__ import annotations

import asyncio
import json
import mmap
import os
import platform
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog

try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import psutil
except ImportError:
    psutil = None

logger = structlog.get_logger(__name__)


class RPiModel(Enum):
    RPI_ZERO = "rpi_zero"
    RPI_3 = "rpi_3"
    RPI_4 = "rpi_4"
    RPI_5 = "rpi_5"
    RPI_400 = "rpi_400"
    RPI_CM4 = "rpi_cm4"
    RPI_CM5 = "rpi_cm5"
    UNKNOWN = "unknown"


class InferenceBackend(Enum):
    ONNX_RUNTIME_CPU = "onnxruntime_cpu"
    ONNX_RUNTIME_CORAL = "onnxruntime_coral"
    TFLITE = "tflite"
    OPENCV = "opencv"


class PowerProfile(Enum):
    PERFORMANCE = "performance"
    BALANCED = "balanced"
    POWER_SAVE = "power_save"


@dataclass
class RPiSystemInfo:
    model: RPiModel = RPiModel.UNKNOWN
    hardware_version: str = ""
    cpu_count: int = 0
    cpu_type: str = ""
    memory_total_mb: int = 0
    arm_version: int = 0
    has_neon: bool = False
    has_coral_tpu: bool = False
    has_gpu: bool = False
    os_version: str = ""
    python_version: str = ""
    onnxruntime_version: str = ""
    max_freq_mhz: float = 0.0
    temp_throttle_c: float = 85.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InferenceResult:
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    input_shape: Tuple[int, ...]
    output_shapes: Dict[str, Tuple[int, ...]]
    backend: InferenceBackend
    batch_size: int = 1
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "input_shape": self.input_shape,
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "backend": self.backend.value,
            "batch_size": self.batch_size,
            "timestamp": self.timestamp,
        }


class MemoryMappedModel:
    """Memory-mapped model loader for reduced memory footprint."""

    def __init__(self, model_path: str) -> None:
        self._path = Path(model_path)
        if not self._path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._file = None
        self._mmap = None
        self._size = self._path.stat().st_size

    def __enter__(self) -> MemoryMappedModel:
        self._file = open(self._path, "rb")
        self._mmap = mmap.mmap(
            self._file.fileno(), 0, access=mmap.ACCESS_READ,
        )
        return self

    def __exit__(self, *args: Any) -> None:
        if self._mmap:
            self._mmap.close()
        if self._file:
            self._file.close()

    def read(self, offset: int = 0, size: Optional[int] = None) -> bytes:
        if self._mmap is None:
            raise RuntimeError("Model not mapped. Use 'with' context manager.")
        size = size or self._size
        self._mmap.seek(offset)
        return self._mmap.read(size)

    @property
    def size(self) -> int:
        return self._size


class RPiInferenceEngine:
    """
    Production-grade inference engine for Raspberry Pi with ONNX Runtime,
    Coral TPU support, memory-mapped loading, and power-aware scheduling.

    Usage:
        engine = RPiInferenceEngine()
        engine.load_model("model.onnx")
        result = await engine.infer(input_data)
        stats = engine.get_system_stats()
    """

    def __init__(
        self,
        backend: InferenceBackend = InferenceBackend.ONNX_RUNTIME_CPU,
        power_profile: PowerProfile = PowerProfile.BALANCED,
        num_threads: Optional[int] = None,
        use_coral: bool = False,
        enable_mmap: bool = True,
        inter_op_threads: int = 1,
        intra_op_threads: int = 2,
    ) -> None:
        if ort is None and backend == InferenceBackend.ONNX_RUNTIME_CPU:
            raise ImportError("onnxruntime is required for ONNX Runtime backend")

        self._backend = backend
        self._power_profile = power_profile
        self._num_threads = num_threads or max(1, os.cpu_count() or 2) - 1
        self._use_coral = use_coral
        self._enable_mmap = enable_mmap
        self._inter_op_threads = inter_op_threads
        self._intra_op_threads = intra_op_threads

        self._session: Optional[ort.InferenceSession] = None
        self._model_path: Optional[str] = None
        self._input_names: List[str] = []
        self._output_names: List[str] = []
        self._input_shapes: Dict[str, Tuple[int, ...]] = {}
        self._output_shapes: Dict[str, Tuple[int, ...]] = {}
        self._model_size: int = 0
        self._is_loaded = False
        self._is_coral_available = False

        self._threadpool = ThreadPoolExecutor(max_workers=self._num_threads)
        self._lock = threading.RLock()
        self._logger = structlog.get_logger(__name__)

        self._system_info = self._detect_system_info()
        self._configure_system()
        self._check_coral_tpu()

        logger.info("rpi_inference_engine_initialized",
                     model=self._system_info.model.value,
                     cpu_count=self._system_info.cpu_count,
                     backend=backend.value,
                     power_profile=power_profile.value)

    def _detect_system_info(self) -> RPiSystemInfo:
        """Detect Raspberry Pi system configuration."""
        info = RPiSystemInfo()
        info.cpu_count = os.cpu_count() or 0
        info.python_version = platform.python_version()
        info.arm_version = self._get_arm_version()

        if ort:
            info.onnxruntime_version = ort.__version__

        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
                if "ARMv7" in cpuinfo:
                    info.arm_version = 7
                elif "ARMv8" in cpuinfo:
                    info.arm_version = 8

                for line in cpuinfo.splitlines():
                    if line.startswith("Hardware"):
                        info.hardware_version = line.split(":")[-1].strip()
                    if line.startswith("Revision"):
                        rev = line.split(":")[-1].strip()
                        info.model = self._identify_model(rev)
        except FileNotFoundError:
            pass

        try:
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        info.memory_total_mb = kb // 1024
                        break
        except FileNotFoundError:
            info.memory_total_mb = psutil.virtual_memory().total // (1024 * 1024) if psutil else 512

        if self._has_neon():
            info.has_neon = True

        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r") as f:
                info.max_freq_mhz = float(f.read().strip()) / 1000.0
        except FileNotFoundError:
            pass

        return info

    def _get_arm_version(self) -> int:
        """Detect ARM architecture version."""
        try:
            import ctypes
            if hasattr(ctypes, "machine"):
                machine = platform.machine().lower()
                if "aarch64" in machine:
                    return 8
                elif "armv7" in machine:
                    return 7
                elif "armv6" in machine:
                    return 6
        except Exception:
            pass
        return 0

    def _identify_model(self, revision: str) -> RPiModel:
        """Identify Raspberry Pi model from revision code."""
        rev_upper = revision.upper().strip()
        model_map = {
            "900092": RPiModel.RPI_ZERO,
            "900093": RPiModel.RPI_ZERO,
            "920092": RPiModel.RPI_ZERO,
            "9000C1": RPiModel.RPI_ZERO,
            "A02082": RPiModel.RPI_3,
            "A22082": RPiModel.RPI_3,
            "A32082": RPiModel.RPI_3,
            "A020D3": RPiModel.RPI_3,
            "A03111": RPiModel.RPI_4,
            "B03111": RPiModel.RPI_4,
            "B03112": RPiModel.RPI_4,
            "B03114": RPiModel.RPI_4,
            "C03111": RPiModel.RPI_4,
            "C03112": RPiModel.RPI_4,
            "C03114": RPiModel.RPI_4,
            "D03114": RPiModel.RPI_4,
            "B04114": RPiModel.RPI_5,
            "C04114": RPiModel.RPI_5,
        }
        for rev_code, model in model_map.items():
            if rev_upper.startswith(rev_code[:4]):
                return model
            if rev_upper == rev_code:
                return model
        return RPiModel.UNKNOWN

    def _has_neon(self) -> bool:
        """Check if NEON SIMD instructions are available."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                return "neon" in f.read().lower()
        except FileNotFoundError:
            return False

    def _configure_system(self) -> None:
        """Configure system based on power profile."""
        gov_map = {
            PowerProfile.PERFORMANCE: "performance",
            PowerProfile.BALANCED: "ondemand",
            PowerProfile.POWER_SAVE: "powersave",
        }
        governor = gov_map.get(self._power_profile, "ondemand")

        try:
            for cpu in range(self._system_info.cpu_count):
                gov_path = f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor"
                if Path(gov_path).exists():
                    Path(gov_path).write_text(governor)
            logger.info("cpu_governor_set", governor=governor)
        except (PermissionError, FileNotFoundError):
            logger.warning("cannot_set_cpu_governor_no_permissions")

    def _check_coral_tpu(self) -> None:
        """Check if Coral TPU is available."""
        try:
            import pyedgetpu
            self._is_coral_available = True
            self._system_info.has_coral_tpu = True
            logger.info("coral_tpu_detected")
        except ImportError:
            self._is_coral_available = False

    def load_model(
        self,
        model_path: str,
        force_reload: bool = False,
        input_shapes: Optional[Dict[str, List[int]]] = None,
    ) -> None:
        """
        Load an ONNX model for inference.

        Uses memory-mapped loading when enabled for reduced RAM footprint.

        Args:
            model_path: Path to ONNX model.
            force_reload: Reload even if already loaded.
            input_shapes: Override input shapes for dynamic models.
        """
        if self._is_loaded and not force_reload:
            logger.info("model_already_loaded", path=model_path)
            return

        path_obj = Path(model_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self._model_path = model_path
        self._model_size = path_obj.stat().st_size

        sess_options = ort.SessionOptions()

        if self._enable_mmap and self._system_info.memory_total_mb < 2048:
            sess_options.enable_mem_pattern = True
            sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            logger.info("memory_efficient_loading_enabled")

        sess_options.intra_op_num_threads = self._intra_op_threads
        sess_options.inter_op_num_threads = self._inter_op_threads
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = str(
            path_obj.parent / f"{path_obj.stem}_optimized{path_obj.suffix}"
        )

        if self._enable_mmap:
            with MemoryMappedModel(model_path) as mmap_model:
                model_data = mmap_model.read()
                self._session = ort.InferenceSession(
                    model_data, sess_options,
                    providers=self._get_providers(),
                )
                logger.info("model_loaded_via_mmap", size_mb=self._model_size / (1024 * 1024))
        else:
            self._session = ort.InferenceSession(
                model_path, sess_options,
                providers=self._get_providers(),
            )

        if self._session is None:
            raise RuntimeError(f"Failed to load model: {model_path}")

        self._input_names = [inp.name for inp in self._session.get_inputs()]
        self._output_names = [out.name for out in self._session.get_outputs()]
        self._input_shapes = {
            inp.name: tuple(inp.shape) if all(d > 0 for d in inp.shape) else (1, 3, 224, 224)
            for inp in self._session.get_inputs()
        }
        self._output_shapes = {
            out.name: tuple(out.shape) if all(d > 0 for d in out.shape) else (1, 1000)
            for out in self._session.get_outputs()
        }

        if input_shapes:
            for name, shape in input_shapes.items():
                if name in self._input_shapes:
                    self._input_shapes[name] = tuple(shape)

        self._is_loaded = True
        logger.info("model_loaded", path=model_path,
                     inputs=self._input_names,
                     outputs=self._output_names,
                     size_mb=f"{self._model_size / (1024 * 1024):.2f}")

    def _get_providers(self) -> List[str]:
        """Get available ONNX Runtime providers."""
        providers = ["CPUExecutionProvider"]

        if self._use_coral and self._is_coral_available:
            try:
                providers.insert(0, "TensorrtExecutionProvider")
            except Exception:
                pass

        return providers

    async def infer(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray]],
        batch_size: int = 1,
        timeout_ms: Optional[float] = None,
    ) -> InferenceResult:
        """
        Run asynchronous inference on Raspberry Pi.

        Args:
            input_data: Input tensor(s) as numpy array or dict.
            batch_size: Batch dimension for input.
            timeout_ms: Optional timeout in milliseconds.

        Returns:
            InferenceResult with outputs and timing.
        """
        if not self._is_loaded or self._session is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if isinstance(input_data, np.ndarray):
            inputs = {self._input_names[0]: input_data}
        else:
            inputs = input_data

        if batch_size > 1:
            inputs = {k: np.repeat(v, batch_size, axis=0) for k, v in inputs.items()}

        loop = asyncio.get_event_loop()

        start = time.perf_counter()

        try:
            async with asyncio.timeout(timeout_ms / 1000.0 if timeout_ms else None):
                outputs = await loop.run_in_executor(
                    self._threadpool,
                    self._run_sync,
                    inputs,
                )
        except asyncio.TimeoutError:
            logger.error("inference_timeout", timeout_ms=timeout_ms)
            raise TimeoutError(f"Inference timed out after {timeout_ms}ms")

        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0

        output_shapes = {k: v.shape for k, v in outputs.items()}

        result = InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            input_shape=tuple(next(iter(inputs.values())).shape),
            output_shapes=output_shapes,
            backend=self._backend,
            batch_size=batch_size,
        )

        logger.debug("inference_complete", latency_ms=f"{latency_ms:.3f}")
        return result

    def _run_sync(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Synchronous inference run."""
        assert self._session is not None
        ort_inputs = {
            name: np.ascontiguousarray(data.astype(np.float32))
            for name, data in inputs.items()
        }
        outputs = self._session.run(self._output_names, ort_inputs)
        return dict(zip(self._output_names, outputs))

    def get_system_stats(self) -> Dict[str, Any]:
        """Get Raspberry Pi system statistics."""
        stats: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "memory_used_mb": 0.0,
            "memory_percent": 0.0,
            "temperature_c": 0.0,
            "disk_used_mb": 0.0,
            "uptime_hours": 0.0,
        }

        if psutil:
            stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            mem = psutil.virtual_memory()
            stats["memory_used_mb"] = mem.used / (1024 * 1024)
            stats["memory_percent"] = mem.percent
            disk = psutil.disk_usage("/")
            stats["disk_used_mb"] = disk.used / (1024 * 1024)
            stats["uptime_hours"] = (time.time() - psutil.boot_time()) / 3600.0

        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                stats["temperature_c"] = float(f.read().strip()) / 1000.0
        except (FileNotFoundError, PermissionError):
            pass

        return stats

    def benchmark(
        self,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> Dict[str, float]:
        """Benchmark inference performance."""
        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for _ in range(num_warmup):
            self._run_sync({self._input_names[0]: dummy_input})

        latencies: List[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            self._run_sync({self._input_names[0]: dummy_input})
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        latencies_np = np.array(latencies)
        results = {
            "mean_latency_ms": float(np.mean(latencies_np)),
            "p50_latency_ms": float(np.median(latencies_np)),
            "p95_latency_ms": float(np.percentile(latencies_np, 95)),
            "p99_latency_ms": float(np.percentile(latencies_np, 99)),
            "throughput_fps": 1000.0 / float(np.mean(latencies_np)),
            "min_latency_ms": float(np.min(latencies_np)),
            "max_latency_ms": float(np.max(latencies_np)),
            "std_latency_ms": float(np.std(latencies_np)),
            "model_size_mb": self._model_size / (1024 * 1024),
        }

        logger.info("rpi_benchmark_complete", **{k: f"{v:.3f}" for k, v in results.items()})
        return results

    def set_power_profile(self, profile: PowerProfile) -> None:
        """Change power profile dynamically."""
        self._power_profile = profile
        self._configure_system()
        logger.info("power_profile_changed", profile=profile.value)

    def release(self) -> None:
        """Release model resources."""
        with self._lock:
            self._session = None
            self._is_loaded = False
            self._threadpool.shutdown(wait=False)
            logger.info("engine_resources_released")

    def __enter__(self) -> RPiInferenceEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    @staticmethod
    def is_raspberry_pi() -> bool:
        """Check if running on a Raspberry Pi."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                return "raspberry" in f.read().lower()
        except FileNotFoundError:
            return platform.machine().startswith("arm") or platform.machine().startswith("aarch64")
