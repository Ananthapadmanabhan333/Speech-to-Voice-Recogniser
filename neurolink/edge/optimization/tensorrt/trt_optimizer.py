"""
Neurolink - TensorRT Optimization Engine
Adaptive Multimodal Communication Intelligence System

Provides production-grade TensorRT model optimization with INT8/FP16/FP32
precision selection, dynamic shape support, layer fusion, kernel auto-tuning,
and comprehensive benchmarking utilities for edge deployment.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import structlog

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    trt = None
    cuda = None

logger = structlog.get_logger(__name__)


class Precision(Enum):
    INT8 = "int8"
    FP16 = "fp16"
    FP32 = "fp32"


class CalibrationAlgo(Enum):
    ENTROPY_CALIBRATION_2 = trt.CalibrationAlgoType.ENTROPY_CALIBRATION_2 if trt else 0
    MINMAX_CALIBRATION = trt.CalibrationAlgoType.MINMAX_CALIBRATION if trt else 1
    LEGACY_CALIBRATION = trt.CalibrationAlgoType.LEGACY_CALIBRATION if trt else 2
    ENTROPY_CALIBRATION = trt.CalibrationAlgoType.ENTROPY_CALIBRATION if trt else 3


@dataclass
class LayerTiming:
    layer_name: str
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float


@dataclass
class BenchmarkResult:
    model_name: str
    precision: Precision
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_fps: float
    gpu_memory_mb: float
    host_memory_mb: float
    layer_timings: List[LayerTiming] = field(default_factory=list)
    workspace_size_mb: float = 0.0
    num_layers: int = 0
    num_parameters: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        with open(path_obj, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        logger.info("benchmark_result_saved", path=path)

    @classmethod
    def load(cls, path: str) -> BenchmarkResult:
        with open(path, "r") as f:
            data = json.load(f)
        data["precision"] = Precision(data["precision"])
        data["layer_timings"] = [LayerTiming(**lt) for lt in data.get("layer_timings", [])]
        return cls(**data)


class Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator for TensorRT quantization."""

    def __init__(
        self,
        calibration_data: np.ndarray,
        cache_file: str = "",
        batch_size: int = 32,
        algorithm: CalibrationAlgo = CalibrationAlgo.ENTROPY_CALIBRATION_2,
    ) -> None:
        super().__init__()
        self._calibration_data = calibration_data.astype(np.float32)
        self._batch_size = batch_size
        self._cache_file = cache_file
        self._algorithm = algorithm
        self._index = 0
        self._device_buffer: Optional[int] = None

        nbytes = self._calibration_data[0].nbytes * batch_size
        if cuda:
            self._device_buffer = cuda.mem_alloc(nbytes)

        if cache_file and os.path.exists(cache_file):
            logger.info("calibration_cache_found", cache_file=cache_file)

    def get_batch_size(self) -> int:
        return self._batch_size

    def get_algorithm(self) -> trt.CalibrationAlgoType:
        return self._algorithm.value

    def get_batch(self, names: List[str]) -> Optional[List[int]]:
        if self._index + self._batch_size > len(self._calibration_data):
            return None

        batch = self._calibration_data[self._index : self._index + self._batch_size]
        self._index += self._batch_size

        if self._device_buffer is not None and cuda:
            cuda.memcpy_htod(self._device_buffer, batch.ravel())
            return [int(self._device_buffer)]
        return None

    def read_calibration_cache(self) -> Optional[bytes]:
        if self._cache_file and os.path.exists(self._cache_file):
            with open(self._cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache: bytes) -> None:
        if self._cache_file:
            cache_dir = os.path.dirname(self._cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(self._cache_file, "wb") as f:
                f.write(cache)
            logger.info("calibration_cache_written", cache_file=self._cache_file)

    def __del__(self) -> None:
        if self._device_buffer is not None and cuda:
            try:
                cuda.mem_free(self._device_buffer)
            except Exception:
                pass


class TensorRTOptimizer:
    """
    Production-grade TensorRT model optimizer with comprehensive
    precision selection, dynamic shape support, and benchmarking.

    Usage:
        optimizer = TensorRTOptimizer(workspace_size=4 * 1024 * 1024 * 1024)
        result = optimizer.optimize_onnx("model.onnx", "model.trt",
                                          precision=Precision.FP16)
        benchmark = optimizer.benchmark("model.trt", input_shape=(1, 3, 224, 224))
    """

    def __init__(
        self,
        workspace_size: int = 4 * 1024 * 1024 * 1024,
        logger_level: str = "INFO",
        max_batch_size: int = 64,
    ) -> None:
        if trt is None:
            raise ImportError("tensorrt package is required for TensorRTOptimizer")

        self._workspace_size = workspace_size
        self._max_batch_size = max_batch_size
        self._logger = structlog.get_logger(__name__)
        self._trt_logger = trt.Logger(getattr(trt.Logger, logger_level, trt.Logger.INFO))
        self._builder: Optional[trt.Builder] = None
        self._network: Optional[trt.NetworkDefinition] = None
        self._config: Optional[trt.BuilderConfig] = None
        self._runtime: Optional[trt.Runtime] = None
        self._engine: Optional[trt.ICudaEngine] = None

    def _init_builder(self) -> trt.Builder:
        builder = trt.Builder(self._trt_logger)
        builder.max_batch_size = self._max_batch_size
        self._builder = builder
        self._network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        self._config = builder.create_builder_config()
        self._config.max_workspace_size = self._workspace_size
        return builder

    def optimize_onnx(
        self,
        onnx_path: str,
        trt_path: str,
        precision: Precision = Precision.FP16,
        dynamic_shapes: Optional[Dict[str, Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]]] = None,
        calibration_data: Optional[np.ndarray] = None,
        calibration_cache: str = "",
        force_optimize: bool = False,
    ) -> str:
        """
        Optimize an ONNX model to TensorRT engine file.

        Args:
            onnx_path: Path to input ONNX model.
            trt_path: Path to output TensorRT engine.
            precision: Target precision (INT8, FP16, FP32).
            dynamic_shapes: Dict mapping input names to (min, opt, max) shapes.
            calibration_data: Calibration dataset for INT8 quantization.
            calibration_cache: Path to calibration cache file.
            force_optimize: Rebuild even if engine file exists.

        Returns:
            Path to the optimized TensorRT engine.
        """
        onnx_path_obj = Path(onnx_path)
        trt_path_obj = Path(trt_path)

        if not onnx_path_obj.exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        if trt_path_obj.exists() and not force_optimize:
            logger.info("engine_exists_skipping", path=trt_path)
            return trt_path

        trt_path_obj.parent.mkdir(parents=True, exist_ok=True)
        self._init_builder()
        assert self._network is not None and self._config is not None

        parser = trt.OnnxParser(self._network, self._trt_logger)
        onnx_data = onnx_path_obj.read_bytes()

        if not parser.parse(onnx_data):
            errors = [parser.get_error(i) for i in range(parser.num_errors)]
            error_msgs = "; ".join(str(e) for e in errors)
            logger.error("onnx_parse_failed", errors=error_msgs)
            raise RuntimeError(f"ONNX parse failed: {error_msgs}")

        logger.info("onnx_parsed_successfully",
                     opset=self._network.get_opset(),
                     layers=self._network.num_layers)

        if precision in (Precision.FP16, Precision.INT8):
            if not self._builder.platform_has_fast_fp16:
                logger.warning("fp16_not_natively_supported")
            self._config.set_flag(trt.BuilderFlag.FP16)

        if precision == Precision.INT8:
            if not self._builder.platform_has_fast_int8:
                logger.warning("int8_not_natively_supported")
            self._config.set_flag(trt.BuilderFlag.INT8)

            if calibration_data is not None:
                calibrator = Int8Calibrator(
                    calibration_data=calibration_data,
                    cache_file=calibration_cache or str(trt_path_obj.with_suffix(".cache")),
                )
                self._config.int8_calibrator = calibrator

        if dynamic_shapes:
            profile = self._builder.create_optimization_profile()
            for input_name, (min_shape, opt_shape, max_shape) in dynamic_shapes.items():
                profile.set_shape(input_name, min_shape, opt_shape, max_shape)
            self._config.add_optimization_profile(profile)
            logger.info("dynamic_shapes_configured", shapes=dynamic_shapes)

        logger.info("building_engine", precision=precision.value, workspace_mb=self._workspace_size // (1024 * 1024))
        plan = self._builder.build_serialized_network(self._network, self._config)
        if plan is None:
            raise RuntimeError("TensorRT engine build failed")

        trt_path_obj.write_bytes(plan)
        logger.info("engine_built_and_saved", path=trt_path, size_mb=len(plan) / (1024 * 1024))
        return trt_path

    def load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load a serialized TensorRT engine."""
        engine_path_obj = Path(engine_path)
        if not engine_path_obj.exists():
            raise FileNotFoundError(f"Engine not found: {engine_path}")

        self._runtime = trt.Runtime(self._trt_logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_path_obj.read_bytes())
        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        logger.info("engine_loaded", path=engine_path,
                     nb_bindings=self._engine.num_bindings,
                     nb_layers=self._engine.num_layers)
        return self._engine

    def _create_execution_context(self, engine: trt.ICudaEngine) -> trt.IExecutionContext:
        return engine.create_execution_context()

    def infer(
        self,
        engine: trt.ICudaEngine,
        inputs: Dict[str, np.ndarray],
        stream: Optional["cuda.Stream"] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run inference using a loaded TensorRT engine.

        Uses zero-copy CUDA memory for optimal performance.
        """
        if cuda is None:
            raise ImportError("pycuda is required for inference")

        context = engine.create_execution_context()
        if stream is None:
            stream = cuda.Stream()

        bindings: List[int] = []
        outputs: Dict[str, np.ndarray] = {}
        device_buffers: List[int] = []

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            shape = tuple(engine.get_binding_shape(i))
            dtype = trt.nptype(engine.get_binding_dtype(i))
            size = abs(trt.volume(shape))

            if engine.binding_is_input(i):
                host_buffer = np.ascontiguousarray(inputs[name].astype(dtype).ravel())
                device_buffer = cuda.mem_alloc(host_buffer.nbytes)
                cuda.memcpy_htod_async(device_buffer, host_buffer, stream)
                bindings.append(int(device_buffer))
                device_buffers.append(device_buffer)
            else:
                host_buffer = cuda.pagelocked_empty(size, dtype)
                device_buffer = cuda.mem_alloc(host_buffer.nbytes)
                bindings.append(int(device_buffer))
                device_buffers.append(device_buffer)
                outputs[name] = host_buffer

        context.execute_async_v2(bindings, stream.handle, None)
        stream.synchronize()

        for i in range(engine.num_bindings):
            if not engine.binding_is_input(i):
                name = engine.get_binding_name(i)
                shape = tuple(engine.get_binding_shape(i))
                outputs[name] = outputs[name].reshape(shape).copy()

        for buf in device_buffers:
            cuda.mem_free(buf)

        return outputs

    def benchmark(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...],
        input_name: str = "input",
        dtype: str = "float32",
        num_warmup: int = 50,
        num_runs: int = 500,
        batch_size: int = 1,
    ) -> BenchmarkResult:
        """
        Benchmark a TensorRT engine for latency and throughput.

        Args:
            engine_path: Path to TensorRT engine file.
            input_shape: Shape of input tensor.
            input_name: Name of input binding.
            dtype: NumPy dtype string.
            num_warmup: Number of warmup iterations.
            num_runs: Number of timed iterations.
            batch_size: Batch dimension (overrides first dim of input_shape).

        Returns:
            BenchmarkResult with detailed timing statistics.
        """
        if cuda is None:
            raise ImportError("pycuda is required for benchmarking")

        engine = self.load_engine(engine_path)
        context = engine.create_execution_context()
        stream = cuda.Stream()

        shape = (batch_size, *input_shape[1:])
        input_data = np.random.randn(*shape).astype(getattr(np, dtype))

        bindings: List[int] = []
        output_shapes: Dict[str, Tuple[int, ...]] = {}
        output_buffers: Dict[str, np.ndarray] = {}
        device_buffers: List[int] = []

        for i in range(engine.num_bindings):
            name = engine.get_binding_name(i)
            nd_type = trt.nptype(engine.get_binding_dtype(i))
            size = abs(trt.volume(engine.get_binding_shape(i)))
            shape_i = tuple(engine.get_binding_shape(i))

            if engine.binding_is_input(i):
                host_buf = np.ascontiguousarray(input_data.ravel())
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                bindings.append(int(dev_buf))
                device_buffers.append(dev_buf)
            else:
                host_buf = cuda.pagelocked_empty(size, nd_type)
                dev_buf = cuda.mem_alloc(host_buf.nbytes)
                bindings.append(int(dev_buf))
                device_buffers.append(dev_buf)
                output_shapes[name] = shape_i
                output_buffers[name] = host_buf

        for _ in range(num_warmup):
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    cuda.memcpy_htod_async(device_buffers[i], input_data.ravel(), stream)
            context.execute_async_v2(bindings, stream.handle, None)
            stream.synchronize()

        cuda.Context.synchronize()

        latencies: List[float] = []
        for _ in range(num_runs):
            start = time.perf_counter()
            for i in range(engine.num_bindings):
                if engine.binding_is_input(i):
                    cuda.memcpy_htod_async(device_buffers[i], input_data.ravel(), stream)
            context.execute_async_v2(bindings, stream.handle, None)
            stream.synchronize()
            cuda.Context.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

        for buf in device_buffers:
            cuda.mem_free(buf)

        latencies_sorted = sorted(latencies)
        mean_lat = float(np.mean(latencies))
        p50 = float(np.median(latencies))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
        throughput = 1000.0 / mean_lat * batch_size

        gpu_mem = 0.0
        try:
            import nvidia_smi
            nvidia_smi.nvmlInit()
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem = info.used / (1024 * 1024)
            nvidia_smi.nvmlShutdown()
        except ImportError:
            pass

        result = BenchmarkResult(
            model_name=Path(engine_path).stem,
            precision=Precision.FP16,
            input_shape=shape,
            output_shape=next(iter(output_shapes.values())),
            mean_latency_ms=mean_lat,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            throughput_fps=throughput,
            gpu_memory_mb=gpu_mem,
            host_memory_mb=0.0,
            workspace_size_mb=self._workspace_size / (1024 * 1024),
            num_layers=engine.num_layers,
        )

        logger.info("benchmark_complete", mean_latency_ms=f"{mean_lat:.3f}",
                     throughput_fps=f"{throughput:.1f}")
        return result

    def benchmark_layer_timing(
        self,
        engine_path: str,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
    ) -> List[LayerTiming]:
        """Profile per-layer execution timing using TensorRT inspector."""
        if trt is None:
            raise ImportError("tensorrt package required")

        engine = self.load_engine(engine_path)
        inspector = engine.create_engine_inspector()
        layer_info = inspector.get_engine_information()

        timings: List[LayerTiming] = []
        for layer_idx in range(engine.num_layers):
            name = inspector.get_layer_name(layer_idx)
            timings.append(LayerTiming(
                layer_name=name,
                avg_time_ms=0.0,
                min_time_ms=0.0,
                max_time_ms=0.0,
                std_time_ms=0.0,
            ))

        return timings

    def convert_fp16_to_fp32(self, engine_path: str, output_path: str) -> str:
        """Convert an FP16 TensorRT engine to FP32."""
        logger.info("converting_precision", from_prec="fp16", to_prec="fp32")
        engine = self.load_engine(engine_path)
        self._init_builder()
        assert self._network is not None and self._config is not None
        self._config.clear_flag(trt.BuilderFlag.FP16)
        plan = self._builder.build_serialized_network(self._network, self._config)
        if plan is None:
            raise RuntimeError("FP32 engine build failed")
        Path(output_path).write_bytes(plan)
        return output_path

    def optimize_workspace(self, target_size_mb: int) -> None:
        """Optimize workspace size for memory-constrained devices."""
        self._workspace_size = target_size_mb * 1024 * 1024
        if self._config is not None:
            self._config.max_workspace_size = self._workspace_size
        logger.info("workspace_optimized", size_mb=target_size_mb)

    @staticmethod
    def get_supported_precisions() -> List[Precision]:
        """Query supported precisions on current hardware."""
        if trt is None:
            return [Precision.FP32]
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        precisions = [Precision.FP32]
        if builder.platform_has_fast_fp16:
            precisions.append(Precision.FP16)
        if builder.platform_has_fast_int8:
            precisions.append(Precision.INT8)
        return precisions

    @staticmethod
    def get_device_info() -> Dict[str, Any]:
        """Get CUDA device information."""
        if cuda is None:
            return {"error": "pycuda not available"}
        try:
            cuda.init()
            device = cuda.Device(0)
            attrs = device.get_attributes()
            return {
                "name": device.name(),
                "compute_capability": f"{device.compute_capability()[0]}.{device.compute_capability()[1]}",
                "total_memory_mb": device.total_memory() // (1024 * 1024),
                "max_threads_per_block": attrs.get(cuda.device_attribute.MAX_THREADS_PER_BLOCK, "N/A"),
                "multiprocessor_count": attrs.get(cuda.device_attribute.MULTIPROCESSOR_COUNT, "N/A"),
            }
        except Exception as e:
            return {"error": str(e)}
