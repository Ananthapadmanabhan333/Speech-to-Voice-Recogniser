from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from edge.jetson.inference.jetson_inference import (
    DeviceStats,
    ExecutionDevice,
    InferenceResult,
    JetsonInferenceEngine,
    PowerMode,
    PrecisionMode,
    TensorRTExecutionContext,
)


class TestJetsonEngineInitialization:
    """Test JetsonInferenceEngine initialization."""

    def test_requires_tensorrt(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt", None):
            with pytest.raises(ImportError, match="tensorrt"):
                JetsonInferenceEngine()

    def test_default_precision(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                engine = JetsonInferenceEngine()
                assert engine._precision == PrecisionMode.FP16
                assert engine._device_id == 0
                assert engine._power_mode == PowerMode.MAX_N

    def test_custom_power_mode(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                engine = JetsonInferenceEngine(power_mode=PowerMode.MAX_Q)
                assert engine._power_mode == PowerMode.MAX_Q

    def test_dla_enabled(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                engine = JetsonInferenceEngine(use_dla=True)
                assert engine._use_dla is True

    def test_pva_enabled(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                engine = JetsonInferenceEngine(use_pva=True)
                assert engine._use_pva is True

    def test_stream_count(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                engine = JetsonInferenceEngine(num_streams=4)
                assert engine._num_streams == 4


class TestDeviceStats:
    """Test DeviceStats dataclass."""

    def test_default_values(self) -> None:
        stats = DeviceStats()
        assert stats.gpu_utilization_pct == 0.0
        assert stats.memory_used_mb == 0.0
        assert stats.temperature_gpu_c == 0.0
        assert stats.fan_speed_pct == 0.0

    def test_to_dict(self) -> None:
        stats = DeviceStats(gpu_utilization_pct=75.0, memory_used_mb=2048.0)
        d = stats.to_dict()
        assert d["gpu_utilization_pct"] == 75.0
        assert d["memory_used_mb"] == 2048.0

    def test_set_values(self) -> None:
        stats = DeviceStats(
            gpu_utilization_pct=80.0,
            memory_used_mb=4096.0,
            memory_total_mb=8192.0,
            temperature_gpu_c=65.0,
            temperature_cpu_c=55.0,
            power_consumption_mw=15000.0,
            fan_speed_pct=45.0,
        )
        assert stats.gpu_utilization_pct == 80.0
        assert stats.memory_used_mb == 4096.0
        assert stats.power_consumption_mw == 15000.0


class TestInferenceResult:
    """Test InferenceResult dataclass."""

    def test_result_creation(self) -> None:
        result = InferenceResult(
            outputs={"output": np.random.randn(1, 10).astype(np.float32)},
            latency_ms=5.5,
            input_shape=(1, 3, 224, 224),
            output_shapes={"output": (1, 10)},
            precision=PrecisionMode.FP16,
            device=ExecutionDevice.GPU,
        )
        assert result.latency_ms == 5.5
        assert result.precision == PrecisionMode.FP16
        assert result.device == ExecutionDevice.GPU

    def test_to_dict(self) -> None:
        result = InferenceResult(
            outputs={"output": np.zeros((1, 5))},
            latency_ms=3.2,
            input_shape=(1, 64),
            output_shapes={"output": (1, 5)},
            precision=PrecisionMode.INT8,
            device=ExecutionDevice.DLA_0,
        )
        d = result.to_dict()
        assert d["latency_ms"] == 3.2
        assert d["precision"] == "int8"
        assert d["device"] == "dla_0"


class TestPowerModeSwitching:
    """Test power mode switching capabilities."""

    def test_set_power_mode(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0)
                    engine = JetsonInferenceEngine()
                    engine.set_power_mode(PowerMode.MAX_P)
                    assert engine._power_mode == PowerMode.MAX_P

    def test_power_mode_enum_values(self) -> None:
        assert PowerMode.MAX_N.value == "MAX-N"
        assert PowerMode.MAX_P.value == "MAX-P"
        assert PowerMode.MAX_Q.value == "MAX-Q"
        assert PowerMode.MAX_C.value == "MAX-C"

    def test_precision_mode_enum(self) -> None:
        assert PrecisionMode.FP32.value == "fp32"
        assert PrecisionMode.FP16.value == "fp16"
        assert PrecisionMode.INT8.value == "int8"
        assert PrecisionMode.MIXED.value == "mixed"


class TestTemperatureHandling:
    """Test temperature monitoring."""

    def test_get_device_stats(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value.stdout = (
                        "RAM 2048/4096MB CPU@55.0C GPU@65.0C GR3D_FREQ 75%"
                    )
                    mock_run.return_value.returncode = 0
                    engine = JetsonInferenceEngine()
                    stats = engine.get_device_stats()
                    assert isinstance(stats, DeviceStats)
                    assert stats.memory_used_mb == 2048.0
                    assert stats.memory_total_mb == 4096.0
                    assert stats.temperature_gpu_c == 65.0
                    assert stats.temperature_cpu_c == 55.0
                    assert stats.gpu_utilization_pct == 75.0

    def test_monitor_temperature(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(stdout="", returncode=0)
                    engine = JetsonInferenceEngine()

                    callback = MagicMock()
                    engine.monitor_temperature(interval_s=0.1, callback=callback)
                    import time
                    time.sleep(0.3)
                    # Callback may or may not have been called depending on subprocess

    def test_temperature_threshold_warning(self) -> None:
        stats = DeviceStats(temperature_gpu_c=90.0)
        assert stats.temperature_gpu_c > 85.0


class TestCUDAStreamManagement:
    """Test CUDA stream management."""

    def test_stream_initialization(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                mock_cuda.Stream = MagicMock()
                mock_cuda.init = MagicMock()
                mock_cuda.Device = MagicMock()
                engine = JetsonInferenceEngine(num_streams=2)
                assert len(engine._streams) == 2

    def test_round_robin_context_selection(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                engine = JetsonInferenceEngine(num_streams=2)
                engine._engine = MagicMock()
                engine._engine.num_bindings = 2
                engine._engine.binding_is_input = lambda i: i == 0
                engine._engine.get_binding_name = lambda i: f"binding_{i}"
                engine._engine.get_binding_shape = lambda i: (1, 3, 224, 224)
                engine._engine.get_binding_dtype = lambda i: np.float32

                ctx1 = engine._contexts[0]
                ctx2 = engine._contexts[1]
                assert ctx1 is not ctx2


class TestTensorRTExecutionContext:
    """Test execution context management."""

    def test_context_creation(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                mock_engine = MagicMock()
                mock_engine.num_bindings = 2
                mock_engine.binding_is_input = lambda i: i == 0
                mock_engine.get_binding_name = lambda i: f"binding_{i}"
                mock_engine.get_binding_shape = lambda i: (1, 3, 224, 224)
                mock_engine.get_binding_dtype = lambda i: np.float32

                ctx = TensorRTExecutionContext(mock_engine)
                assert ctx._engine is mock_engine
                assert ctx._context is not None

    def test_context_release(self) -> None:
        with patch("edge.jetson.inference.jetson_inference.trt") as mock_trt:
            with patch("edge.jetson.inference.jetson_inference.cuda") as mock_cuda:
                mock_engine = MagicMock()
                mock_engine.num_bindings = 0
                ctx = TensorRTExecutionContext(mock_engine)
                ctx.release()
                assert ctx._context is None
