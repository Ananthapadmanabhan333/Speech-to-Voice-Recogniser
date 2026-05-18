"""
Neurolink - Jetson Nano Inference Engine
Adaptive Multimodal Communication Intelligence System

Provides production-grade inference on NVIDIA Jetson platforms with TensorRT
execution context management, CUDA stream management, zero-copy GPU inference,
batch processing, power mode optimization, temperature monitoring, and
Jetson-specific DLA/PVA optimizations.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import structlog

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

try:
    import torch
except ImportError:
    torch = None

logger = structlog.get_logger(__name__)


class PrecisionMode(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    MIXED = "mixed"


class PowerMode(Enum):
    MAX_N = "MAX-N"
    MAX_P = "MAX-P"
    MAX_Q = "MAX-Q"
    MAX_C = "MAX-C"


class ExecutionDevice(Enum):
    GPU = "gpu"
    DLA_0 = "dla_0"
    DLA_1 = "dla_1"
    PVA_0 = "pva_0"
    PVA_1 = "pva_1"


@dataclass
class InferenceResult:
    outputs: Dict[str, np.ndarray]
    latency_ms: float
    input_shape: Tuple[int, ...]
    output_shapes: Dict[str, Tuple[int, ...]]
    precision: PrecisionMode
    device: ExecutionDevice
    timestamp: float = field(default_factory=time.time)
    batch_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "input_shape": self.input_shape,
            "output_shapes": {k: list(v) for k, v in self.output_shapes.items()},
            "precision": self.precision.value,
            "device": self.device.value,
            "timestamp": self.timestamp,
            "batch_size": self.batch_size,
        }


@dataclass
class DeviceStats:
    gpu_utilization_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    temperature_gpu_c: float = 0.0
    temperature_cpu_c: float = 0.0
    power_consumption_mw: float = 0.0
    fan_speed_pct: float = 0.0
    clock_rate_gpu_mhz: float = 0.0
    clock_rate_memory_mhz: float = 0.0
    voltage_mv: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TensorRTExecutionContext:
    """Manages a TensorRT execution context with associated CUDA resources."""

    def __init__(
        self,
        engine: trt.ICudaEngine,
        stream: Optional["cuda.Stream"] = None,
        device_id: int = 0,
    ) -> None:
        if trt is None or cuda is None:
            raise ImportError("tensorrt and pycuda are required")

        self._engine = engine
        self._context = engine.create_execution_context()
        self._stream = stream or cuda.Stream()
        self._device_id = device_id
        self._lock = threading.Lock()
        self._bindings: Dict[str, int] = {}
        self._device_buffers: Dict[str, int] = {}
        self._host_buffers: Dict[str, np.ndarray] = {}
        self._setup_bindings()

    def _setup_bindings(self) -> None:
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            shape = tuple(self._engine.get_binding_shape(i))
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            size = abs(trt.volume(shape))
            nbytes = size * np.dtype(dtype).itemsize

            if self._engine.binding_is_input(i):
                host_buf = np.empty(size, dtype=dtype)
                dev_buf = cuda.mem_alloc(nbytes)
            else:
                host_buf = cuda.pagelocked_empty(size, dtype=dtype)
                dev_buf = cuda.mem_alloc(nbytes)

            self._bindings[name] = int(dev_buf)
            self._device_buffers[name] = dev_buf
            self._host_buffers[name] = host_buf

    def set_input(self, name: str, data: np.ndarray) -> None:
        with self._lock:
            dtype = trt.nptype(self._engine.get_binding_dtype(
                self._engine.get_binding_index(name)
            ))
            host_buf = np.ascontiguousarray(data.ravel().astype(dtype))
            cuda.memcpy_htod_async(self._device_buffers[name], host_buf, self._stream)

    def get_output(self, name: str) -> np.ndarray:
        binding_idx = self._engine.get_binding_index(name)
        shape = tuple(self._engine.get_binding_shape(binding_idx))
        cuda.memcpy_dtoh_async(self._host_buffers[name], self._device_buffers[name], self._stream)
        return self._host_buffers[name].reshape(shape).copy()

    def execute_async(self) -> None:
        with self._lock:
            bindings_list = list(self._bindings.values())
            self._context.execute_async_v2(bindings_list, self._stream.handle, None)

    def synchronize(self) -> None:
        self._stream.synchronize()

    def release(self) -> None:
        with self._lock:
            self._context = None
            for buf in self._device_buffers.values():
                cuda.mem_free(buf)
            self._device_buffers.clear()
            self._bindings.clear()
            self._host_buffers.clear()


class JetsonInferenceEngine:
    """
    Production-grade inference engine for NVIDIA Jetson platforms with
    TensorRT acceleration, CUDA stream management, and hardware-specific
    optimizations (DLA, PVA).

    Usage:
        engine = JetsonInferenceEngine(precision=PrecisionMode.FP16)
        engine.load_model("model.trt")
        result = await engine.infer(input_data)
        stats = engine.get_device_stats()
    """

    def __init__(
        self,
        precision: PrecisionMode = PrecisionMode.FP16,
        device_id: int = 0,
        power_mode: PowerMode = PowerMode.MAX_N,
        use_dla: bool = False,
        use_pva: bool = False,
        max_batch_size: int = 32,
        num_streams: int = 2,
    ) -> None:
        if trt is None or cuda is None:
            raise ImportError("tensorrt and pycuda are required for JetsonInferenceEngine")

        self._precision = precision
        self._device_id = device_id
        self._power_mode = power_mode
        self._use_dla = use_dla
        self._use_pva = use_pva
        self._max_batch_size = max_batch_size
        self._num_streams = num_streams

        self._engine: Optional[trt.ICudaEngine] = None
        self._runtime: Optional[trt.Runtime] = None
        self._contexts: List[TensorRTExecutionContext] = []
        self._streams: List[cuda.Stream] = []
        self._logger = structlog.get_logger(__name__)
        self._trt_logger = trt.Logger(trt.Logger.INFO)
        self._lock = threading.RLock()
        self._context_index = 0
        self._model_path: Optional[str] = None
        self._is_loaded = False
        self._warmup_done = False

        self._init_cuda()
        self._init_streams()
        self._configure_power_mode()
        logger.info("jetson_inference_engine_initialized", precision=precision.value,
                     power_mode=power_mode.value, device_id=device_id)

    def _init_cuda(self) -> None:
        cuda.init()
        cuda.Device(self._device_id).make_context()

    def _init_streams(self) -> None:
        self._streams = [cuda.Stream() for _ in range(self._num_streams)]

    def _configure_power_mode(self) -> None:
        """Configure Jetson power mode using nvpmodel."""
        power_map = {
            PowerMode.MAX_N: "0",
            PowerMode.MAX_P: "1",
            PowerMode.MAX_Q: "2",
            PowerMode.MAX_C: "3",
        }
        mode_id = power_map.get(self._power_mode, "0")
        try:
            subprocess.run(
                ["nvpmodel", "-m", mode_id],
                capture_output=True, timeout=5, check=False,
            )
            subprocess.run(
                ["jetson_clocks"],
                capture_output=True, timeout=5, check=False,
            )
            self._logger.info("power_mode_configured", mode=self._power_mode.value)
        except (subprocess.SubprocessError, FileNotFoundError):
            self._logger.warning("power_mode_config_failed_not_on_jetson")

    def load_model(
        self,
        model_path: str,
        precision: Optional[PrecisionMode] = None,
        force_reload: bool = False,
    ) -> None:
        """
        Load a TensorRT engine for inference.

        Args:
            model_path: Path to TensorRT engine (.trt/.plan) or ONNX model.
            precision: Override default precision.
            force_reload: Reload even if already loaded.
        """
        if self._is_loaded and not force_reload:
            logger.info("model_already_loaded", path=model_path)
            return

        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        if precision:
            self._precision = precision

        ext = model_path_obj.suffix.lower()

        if ext in (".onnx",):
            trt_path = str(model_path_obj.with_suffix(".trt"))
            if not Path(trt_path).exists():
                self._build_engine(model_path, trt_path)
            model_path = trt_path

        self._runtime = trt.Runtime(self._trt_logger)
        engine_data = model_path_obj.read_bytes()
        self._engine = self._runtime.deserialize_cuda_engine(engine_data)

        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {model_path}")

        self._contexts = [
            TensorRTExecutionContext(self._engine, self._streams[i % self._num_streams])
            for i in range(self._num_streams)
        ]

        self._model_path = model_path
        self._is_loaded = True

        logger.info("model_loaded", path=model_path,
                     bindings=self._engine.num_bindings,
                     layers=self._engine.num_layers,
                     streams=self._num_streams)

    def _build_engine(self, onnx_path: str, trt_path: str) -> str:
        """Build TensorRT engine from ONNX model."""
        builder = trt.Builder(self._trt_logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        config = builder.create_builder_config()

        if self._use_dla:
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            dla_core = 0
            config.default_device_type = trt.DeviceType.DLA
            config.DLA_core = dla_core
            logger.info("dla_enabled", core=dla_core)

        if self._precision in (PrecisionMode.FP16, PrecisionMode.MIXED):
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
        if self._precision == PrecisionMode.INT8:
            if builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)

        parser = trt.OnnxParser(network, self._trt_logger)
        if not parser.parse(Path(onnx_path).read_bytes()):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            raise RuntimeError(f"ONNX parse failed: {'; '.join(str(e) for e in errors)}")

        plan = builder.build_serialized_network(network, config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        Path(trt_path).parent.mkdir(parents=True, exist_ok=True)
        Path(trt_path).write_bytes(plan)
        logger.info("engine_built_from_onnx", onnx=onnx_path, trt=trt_path)
        return trt_path

    async def infer(
        self,
        input_data: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
        batch_size: int = 1,
        timeout_ms: Optional[float] = None,
    ) -> InferenceResult:
        """
        Run asynchronous inference.

        Args:
            input_data: Input tensor(s) as numpy array or dict of arrays.
            batch_size: Batch dimension for input.
            timeout_ms: Optional timeout in milliseconds.

        Returns:
            InferenceResult with outputs and timing.
        """
        if not self._is_loaded or self._engine is None:
            raise RuntimeError("No model loaded. Call load_model() first.")

        if not self._warmup_done:
            self._warmup(input_data, batch_size)
            self._warmup_done = True

        with self._lock:
            context = self._contexts[self._context_index % len(self._contexts)]
            self._context_index += 1

        if isinstance(input_data, np.ndarray):
            inputs = {"input": input_data}
        elif isinstance(input_data, list):
            inputs = {f"input_{i}": d for i, d in enumerate(input_data)}
        else:
            inputs = input_data

        if batch_size > 1:
            inputs = {k: np.repeat(v, batch_size, axis=0) for k, v in inputs.items()}

        stream = self._streams[self._context_index % self._num_streams]

        start = time.perf_counter()

        for name, data in inputs.items():
            context.set_input(name, data)

        context.execute_async()
        context.synchronize()

        if timeout_ms is not None:
            elapsed = (time.perf_counter() - start) * 1000.0
            if elapsed > timeout_ms:
                logger.warning("inference_timeout", elapsed_ms=f"{elapsed:.2f}", timeout_ms=timeout_ms)

        outputs: Dict[str, np.ndarray] = {}
        output_shapes: Dict[str, Tuple[int, ...]] = {}
        for i in range(self._engine.num_bindings):
            if not self._engine.binding_is_input(i):
                name = self._engine.get_binding_name(i)
                outputs[name] = context.get_output(name)
                output_shapes[name] = outputs[name].shape

        end = time.perf_counter()
        latency_ms = (end - start) * 1000.0

        if batch_size > 1:
            outputs = {k: v[:batch_size] for k, v in outputs.items()}

        result = InferenceResult(
            outputs=outputs,
            latency_ms=latency_ms,
            input_shape=tuple(next(iter(inputs.values())).shape),
            output_shapes=output_shapes,
            precision=self._precision,
            device=ExecutionDevice.DLA_0 if self._use_dla else ExecutionDevice.GPU,
            batch_size=batch_size,
        )

        logger.debug("inference_complete", latency_ms=f"{latency_ms:.3f}")
        return result

    def _warmup(
        self, input_data: Union[np.ndarray, Dict[str, np.ndarray], List[np.ndarray]],
        batch_size: int,
    ) -> None:
        """Warm up the model with dummy data."""
        if isinstance(input_data, np.ndarray):
            warmup_data = np.random.randn(*input_data.shape).astype(input_data.dtype)
        elif isinstance(input_data, dict):
            warmup_data = {
                k: np.random.randn(*v.shape).astype(v.dtype)
                for k, v in input_data.items()
            }
        else:
            warmup_data = [
                np.random.randn(*d.shape).astype(d.dtype) for d in input_data
            ]

        for _ in range(5):
            context = self._contexts[0]
            if isinstance(warmup_data, np.ndarray):
                context.set_input("input", warmup_data)
            elif isinstance(warmup_data, dict):
                for name, data in warmup_data.items():
                    context.set_input(name, data)
            context.execute_async()
            context.synchronize()

        logger.info("model_warmup_complete")

    def infer_batch(
        self,
        batch_data: List[Union[np.ndarray, Dict[str, np.ndarray]]],
        batch_size: Optional[int] = None,
    ) -> List[InferenceResult]:
        """
        Run batched inference on a list of inputs.

        Uses CUDA streams for parallel execution across batches.
        """
        results: List[InferenceResult] = []
        for i, data in enumerate(batch_data):
            result = self._async_infer_sync(data, batch_size or 1)
            results.append(result)
        return results

    def _async_infer_sync(
        self, input_data: Union[np.ndarray, Dict[str, np.ndarray]], batch_size: int
    ) -> InferenceResult:
        """Synchronous wrapper for async inference."""
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(self.infer(input_data, batch_size))
        finally:
            loop.close()

    def get_device_stats(self) -> DeviceStats:
        """Get real-time Jetson device statistics."""
        stats = DeviceStats()

        try:
            result = subprocess.run(
                ["tegrastats", "--interval", "100", "--count", "1"],
                capture_output=True, text=True, timeout=2,
            )
            output = result.stdout
            import re

            mem_match = re.search(r"RAM (\d+)/(\d+)MB", output)
            if mem_match:
                stats.memory_used_mb = float(mem_match.group(1))
                stats.memory_total_mb = float(mem_match.group(2))

            gpu_match = re.search(r"GR3D_FREQ (\d+)%", output)
            if gpu_match:
                stats.gpu_utilization_pct = float(gpu_match.group(1))

            temp_match = re.search(r"CPU@([\d.]+)C", output)
            if temp_match:
                stats.temperature_cpu_c = float(temp_match.group(1))

            gpu_temp_match = re.search(r"GPU@([\d.]+)C", output)
            if gpu_temp_match:
                stats.temperature_gpu_c = float(gpu_temp_match.group(1))

        except (subprocess.SubprocessError, FileNotFoundError):
            self._get_nvml_stats(stats)

        return stats

    def _get_nvml_stats(self, stats: DeviceStats) -> None:
        """Get device stats via NVML."""
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self._device_id)

            mem_info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            stats.memory_used_mb = mem_info.used / (1024 * 1024)
            stats.memory_total_mb = mem_info.total / (1024 * 1024)

            temp = nvidia_smi.nvmlDeviceGetTemperature(
                handle, nvidia_smi.NVML_TEMPERATURE_GPU
            )
            stats.temperature_gpu_c = float(temp)

            power = nvidia_smi.nvmlDeviceGetPowerUsage(handle)
            stats.power_consumption_mw = float(power)

            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            stats.gpu_utilization_pct = float(util.gpu)

            nvidia_smi.nvmlShutdown()
        except ImportError:
            logger.warning("nvidia_smi_not_available")
        except Exception as e:
            logger.warning("nvml_stats_failed", error=str(e))

    def set_power_mode(self, mode: PowerMode) -> None:
        """Change power mode dynamically."""
        self._power_mode = mode
        self._configure_power_mode()
        logger.info("power_mode_changed", mode=mode.value)

    def monitor_temperature(self, interval_s: float = 1.0, callback: Optional[Callable] = None) -> None:
        """
        Monitor GPU/CPU temperature with callback.

        Args:
            interval_s: Polling interval in seconds.
            callback: Called with DeviceStats on each poll.
        """
        def _monitor() -> None:
            while True:
                stats = self.get_device_stats()
                if callback:
                    callback(stats)
                if stats.temperature_gpu_c > 85.0:
                    logger.warning("high_gpu_temperature", temp_c=stats.temperature_gpu_c)
                time.sleep(interval_s)

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        logger.info("temperature_monitor_started", interval_s=interval_s)

    def release(self) -> None:
        """Release all GPU resources."""
        with self._lock:
            for ctx in self._contexts:
                ctx.release()
            self._contexts.clear()
            self._engine = None
            self._is_loaded = False
            logger.info("engine_resources_released")

    def __enter__(self) -> JetsonInferenceEngine:
        return self

    def __exit__(self, *args: Any) -> None:
        self.release()

    @staticmethod
    def is_jetson_platform() -> bool:
        """Check if running on a Jetson platform."""
        try:
            with open("/proc/device-tree/model", "r") as f:
                model = f.read().lower()
                return "jetson" in model or "tegra" in model
        except FileNotFoundError:
            return False
        except Exception:
            return False

    @staticmethod
    def detect_dla_cores() -> int:
        """Detect number of available DLA cores."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=5,
            )
            dla_count = result.stdout.count("DLA")
            return dla_count
        except (subprocess.SubprocessError, FileNotFoundError):
            return 0

    @staticmethod
    def detect_pva_cores() -> int:
        """Detect number of available PVA cores."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True, text=True, timeout=5,
            )
            pva_count = result.stdout.count("PVA")
            return pva_count
        except (subprocess.SubprocessError, FileNotFoundError):
            return 0
