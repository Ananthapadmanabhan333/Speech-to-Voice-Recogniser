from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

from edge.raspberry_pi.inference.rpi_inference import (
    InferenceBackend,
    InferenceResult,
    MemoryMappedModel,
    PowerProfile,
    RPiInferenceEngine,
    RPiModel,
    RPiSystemInfo,
)


class TestRPiEngineInitialization:
    """Test RPiInferenceEngine initialization."""

    def test_requires_onnxruntime(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort", None):
            with pytest.raises(ImportError, match="onnxruntime"):
                RPiInferenceEngine()

    def test_default_backend(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            assert engine._backend == InferenceBackend.ONNX_RUNTIME_CPU
            assert engine._power_profile == PowerProfile.BALANCED

    def test_custom_power_profile(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine(power_profile=PowerProfile.PERFORMANCE)
            assert engine._power_profile == PowerProfile.PERFORMANCE

    def test_coral_tpu_config(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine(use_coral=True)
            assert engine._use_coral is True

    def test_mmap_enabled_by_default(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            assert engine._enable_mmap is True

    def test_thread_counts(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine(inter_op_threads=2, intra_op_threads=4)
            assert engine._inter_op_threads == 2
            assert engine._intra_op_threads == 4


class TestRPiModelDetection:
    """Test Raspberry Pi model detection."""

    def test_rpi_model_enum(self) -> None:
        assert RPiModel.RPI_ZERO.value == "rpi_zero"
        assert RPiModel.RPI_4.value == "rpi_4"
        assert RPiModel.RPI_5.value == "rpi_5"

    def test_identify_model_rpi4(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            model = engine._identify_model("B03114")
            assert model == RPiModel.RPI_4

    def test_identify_model_rpi5(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            model = engine._identify_model("B04114")
            assert model == RPiModel.RPI_5

    def test_identify_model_unknown(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            model = engine._identify_model("ZZZZZZ")
            assert model == RPiModel.UNKNOWN


class TestRPiSystemInfo:
    """Test RPiSystemInfo dataclass."""

    def test_default_values(self) -> None:
        info = RPiSystemInfo()
        assert info.model == RPiModel.UNKNOWN
        assert info.cpu_count == 0
        assert info.has_coral_tpu is False
        assert info.has_neon is False

    def test_to_dict(self) -> None:
        info = RPiSystemInfo(
            model=RPiModel.RPI_4,
            cpu_count=4,
            memory_total_mb=4096,
            has_neon=True,
        )
        d = info.to_dict()
        assert d["model"] == RPiModel.RPI_4
        assert d["cpu_count"] == 4
        assert d["has_neon"] is True

    def test_set_values(self) -> None:
        info = RPiSystemInfo(
            model=RPiModel.RPI_5,
            hardware_version="B04114",
            cpu_count=4,
            cpu_type="Cortex-A76",
            memory_total_mb=8192,
            arm_version=8,
            has_neon=True,
            has_coral_tpu=False,
            has_gpu=True,
        )
        assert info.model == RPiModel.RPI_5
        assert info.cpu_count == 4
        assert info.arm_version == 8


class TestMemoryMappedModel:
    """Test memory-mapped model loading."""

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            MemoryMappedModel("/nonexistent/model.onnx")

    def test_read_with_context(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model.onnx"
        model_path.write_bytes(b"model data content")
        with MemoryMappedModel(str(model_path)) as mmap_model:
            data = mmap_model.read()
            assert data == b"model data content"

    def test_read_outside_context(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model2.onnx"
        model_path.write_bytes(b"test")
        mmap_model = MemoryMappedModel(str(model_path))
        with pytest.raises(RuntimeError, match="not mapped"):
            mmap_model.read()

    def test_size_property(self, tmp_path: Path) -> None:
        model_path = tmp_path / "model3.onnx"
        expected_size = 4096
        model_path.write_bytes(b"x" * expected_size)
        mmap_model = MemoryMappedModel(str(model_path))
        assert mmap_model.size == expected_size


class TestInferenceResult:
    """Test InferenceResult dataclass."""

    def test_result_creation(self) -> None:
        result = InferenceResult(
            outputs={"output": np.random.randn(1, 10).astype(np.float32)},
            latency_ms=12.5,
            input_shape=(1, 3, 224, 224),
            output_shapes={"output": (1, 10)},
            backend=InferenceBackend.ONNX_RUNTIME_CPU,
        )
        assert result.latency_ms == 12.5
        assert result.backend == InferenceBackend.ONNX_RUNTIME_CPU

    def test_to_dict(self) -> None:
        result = InferenceResult(
            outputs={"output": np.zeros((1, 5))},
            latency_ms=8.2,
            input_shape=(1, 64),
            output_shapes={"output": (1, 5)},
            backend=InferenceBackend.ONNX_RUNTIME_CORAL,
            batch_size=2,
        )
        d = result.to_dict()
        assert d["latency_ms"] == 8.2
        assert d["backend"] == "onnxruntime_coral"
        assert d["batch_size"] == 2


class TestPowerAwareScheduling:
    """Test power-aware scheduling."""

    def test_default_power_profile(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            assert engine._power_profile == PowerProfile.BALANCED

    def test_set_power_profile_performance(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            with patch("pathlib.Path.write_text") as mock_write:
                engine = RPiInferenceEngine()
                engine.set_power_profile(PowerProfile.PERFORMANCE)
                assert engine._power_profile == PowerProfile.PERFORMANCE

    def test_set_power_profile_powersave(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            with patch("pathlib.Path.write_text") as mock_write:
                engine = RPiInferenceEngine()
                engine.set_power_profile(PowerProfile.POWER_SAVE)
                assert engine._power_profile == PowerProfile.POWER_SAVE

    def test_power_profile_enum(self) -> None:
        assert PowerProfile.PERFORMANCE.value == "performance"
        assert PowerProfile.BALANCED.value == "balanced"
        assert PowerProfile.POWER_SAVE.value == "power_save"


class TestRPiBenchmark:
    """Test RPi benchmark functionality."""

    def test_benchmark_requires_model(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            with pytest.raises(RuntimeError, match="No model loaded"):
                engine.benchmark()

    def test_get_system_stats(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            with patch("edge.raspberry_pi.inference.rpi_inference.psutil") as mock_psutil:
                mock_psutil.cpu_percent.return_value = 45.0
                mock_psutil.virtual_memory.return_value = MagicMock(
                    used=2 * 1024**3,
                    percent=50.0,
                )
                mock_psutil.disk_usage.return_value = MagicMock(used=10 * 1024**3)
                mock_psutil.boot_time.return_value = 1000.0

                engine = RPiInferenceEngine()
                stats = engine.get_system_stats()
                assert "cpu_percent" in stats
                assert "memory_used_mb" in stats
                assert "temperature_c" in stats


class TestCoralTPUIntegration:
    """Test Coral TPU integration."""

    def test_check_coral(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            with patch("builtins.__import__", side_effect=ImportError):
                engine = RPiInferenceEngine()
                assert engine._is_coral_available is False
                assert engine._system_info.has_coral_tpu is False

    def test_providers_without_coral(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine(use_coral=False)
            providers = engine._get_providers()
            assert "CPUExecutionProvider" in providers
            assert len(providers) == 1

    def test_platform_detection(self) -> None:
        result = RPiInferenceEngine.is_raspberry_pi()
        # On non-RPi hardware, this will return False or check architecture
        assert isinstance(result, bool)


class TestModelLoading:
    """Test model loading scenarios."""

    def test_load_nonexistent_model(self) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            engine = RPiInferenceEngine()
            with pytest.raises(FileNotFoundError):
                engine.load_model("/nonexistent/model.onnx")

    def test_load_with_mmap(self, tmp_path: Path) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = []
            mock_session.get_outputs.return_value = []
            mock_ort.InferenceSession.return_value = mock_session

            model_path = tmp_path / "model.onnx"
            model_path.write_bytes(b"onnx model data")

            engine = RPiInferenceEngine(enable_mmap=True)
            engine.load_model(str(model_path))
            assert engine._is_loaded is True

    def test_load_without_mmap(self, tmp_path: Path) -> None:
        with patch("edge.raspberry_pi.inference.rpi_inference.ort") as mock_ort:
            mock_session = MagicMock()
            mock_session.get_inputs.return_value = []
            mock_session.get_outputs.return_value = []
            mock_ort.InferenceSession.return_value = mock_session

            model_path = tmp_path / "model.onnx"
            model_path.write_bytes(b"onnx model data")

            engine = RPiInferenceEngine(enable_mmap=False)
            engine.load_model(str(model_path))
            assert engine._is_loaded is True
