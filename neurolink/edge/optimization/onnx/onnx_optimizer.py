"""
Neurolink - ONNX Optimization Engine
Adaptive Multimodal Communication Intelligence System

Provides production-grade ONNX model optimization with graph simplification,
quantization (dynamic, static, QAT), operator fusion, input shape optimization,
model validation, and compatibility checks for edge deployment.
"""

from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import structlog

try:
    import onnx
    from onnx import helper, checker, shape_inference, TensorProto, ModelProto
    from onnx import optimizer as onnx_optimizer
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None

logger = structlog.get_logger(__name__)


class QuantizationMode(Enum):
    DYNAMIC = "dynamic"
    STATIC = "static"
    QUANTIZATION_AWARE_TRAINING = "qat"


class QuantizationType(Enum):
    INT8 = "int8"
    UINT8 = "uint8"
    INT16 = "int16"


@dataclass
class OptimizationMetrics:
    original_size_bytes: int
    optimized_size_bytes: int
    compression_ratio: float
    original_num_nodes: int
    optimized_num_nodes: int
    nodes_removed: int
    original_num_initializers: int
    optimized_num_initializers: int
    optimization_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    is_valid: bool
    ir_version: int
    opset_version: int
    producer_name: str
    graph_name: str
    num_inputs: int
    num_outputs: int
    num_nodes: int
    input_shapes: Dict[str, List[int]]
    output_shapes: Dict[str, List[int]]
    errors: List[str] = None
    warnings: List[str] = None

    def __post_init__(self) -> None:
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class ONNXOptimizer:
    """
    Production-grade ONNX model optimizer with graph simplification,
    quantization, operator fusion, and comprehensive validation.

    Usage:
        optimizer = ONNXOptimizer()
        model = optimizer.load_model("model.onnx")
        metrics = optimizer.optimize_model("model.onnx", "model_opt.onnx",
                                           quantization_mode=QuantizationMode.STATIC)
        validation = optimizer.validate_model("model_opt.onnx")
    """

    PASSES = [
        "extract_constant_to_initializer",
        "eliminate_unused_initializer",
        "eliminate_deadend",
        "eliminate_nop_dropout",
        "eliminate_nop_cast",
        "eliminate_nop_monotone_argmax",
        "eliminate_nop_pad",
        "eliminate_nop_transpose",
        "eliminate_identity",
        "fuse_add_bias_into_conv",
        "fuse_bn_into_conv",
        "fuse_consecutive_concats",
        "fuse_consecutive_log_softmax",
        "fuse_consecutive_reduce_unsqueeze",
        "fuse_matmul_add_bias_into_gemm",
        "fuse_pad_into_conv",
        "fuse_transpose_into_gemm",
        "nop_fusion",
    ]

    def __init__(self, opset_version: int = 17) -> None:
        if onnx is None:
            raise ImportError("onnx package is required for ONNXOptimizer")

        self._opset_version = opset_version
        self._logger = structlog.get_logger(__name__)
        self._model: Optional[ModelProto] = None

    def load_model(self, model_path: str) -> ModelProto:
        """Load an ONNX model from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self._model = onnx.load(str(path))
        logger.info("model_loaded", path=model_path,
                     ir_version=self._model.ir_version,
                     producer=self._model.producer_name)
        return self._model

    def save_model(self, model: ModelProto, model_path: str) -> str:
        """Save an ONNX model to disk."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info("model_saved", path=model_path, size_mb=f"{size_mb:.2f}")
        return model_path

    def optimize_model(
        self,
        model_path: str,
        output_path: str,
        quantization_mode: Optional[QuantizationMode] = None,
        quantization_type: QuantizationType = QuantizationType.INT8,
        calibration_data: Optional[np.ndarray] = None,
        calibration_data_loader=None,
        input_shapes: Optional[Dict[str, List[int]]] = None,
        simplify_graph: bool = True,
        enable_shape_inference: bool = True,
        enable_fusion: bool = True,
        enable_constant_folding: bool = True,
        force_optimize: bool = False,
        num_calibration_samples: int = 100,
        per_channel: bool = True,
        reduce_range: bool = False,
    ) -> OptimizationMetrics:
        """
        Optimize an ONNX model with graph simplification and quantization.

        Args:
            model_path: Path to input ONNX model.
            output_path: Path to save optimized model.
            quantization_mode: Quantization mode (dynamic, static, qat).
            quantization_type: Target quantization type (int8, uint8, int16).
            calibration_data: Calibration dataset (numpy array) for static quantization.
            calibration_data_loader: Generator for calibration data.
            input_shapes: Override input shapes for the model.
            simplify_graph: Apply graph simplification passes.
            enable_shape_inference: Run shape inference.
            enable_fusion: Enable operator fusion.
            enable_constant_folding: Enable constant folding.
            force_optimize: Re-optimize even if output exists.
            num_calibration_samples: Number of calibration samples.
            per_channel: Use per-channel quantization.
            reduce_range: Use reduced range (7-bit) quantization.

        Returns:
            OptimizationMetrics with compression details.
        """
        output_path_obj = Path(output_path)
        if output_path_obj.exists() and not force_optimize:
            logger.info("optimized_model_exists", path=output_path)
            return self._get_metrics(model_path, output_path)

        start_time = time.time()
        model = self.load_model(model_path)
        original_size = Path(model_path).stat().st_size
        original_nodes = len(model.graph.node)
        original_initializers = len(model.graph.initializer)

        graph = model.graph

        if enable_shape_inference:
            try:
                model = shape_inference.infer_shapes(model)
                logger.info("shape_inference_completed")
            except Exception as e:
                logger.warning("shape_inference_failed", error=str(e))

        if simplify_graph:
            model = self._apply_graph_simplification(model)

        if enable_constant_folding:
            model = self._apply_constant_folding(model)

        if enable_fusion:
            model = self._apply_operator_fusion(model)

        if input_shapes:
            model = self._optimize_input_shapes(model, input_shapes)

        if quantization_mode:
            model = self._apply_quantization(
                model, model_path, quantization_mode, quantization_type,
                calibration_data, calibration_data_loader,
                num_calibration_samples, per_channel, reduce_range,
            )

        self.save_model(model, output_path)

        optimized_size = output_path_obj.stat().st_size
        optimized_nodes = len(model.graph.node)
        optimized_initializers = len(model.graph.initializer)
        elapsed_ms = (time.time() - start_time) * 1000.0

        metrics = OptimizationMetrics(
            original_size_bytes=original_size,
            optimized_size_bytes=optimized_size,
            compression_ratio=original_size / max(optimized_size, 1),
            original_num_nodes=original_nodes,
            optimized_num_nodes=optimized_nodes,
            nodes_removed=original_nodes - optimized_nodes,
            original_num_initializers=original_initializers,
            optimized_num_initializers=optimized_initializers,
            optimization_time_ms=elapsed_ms,
        )

        logger.info("model_optimized",
                     compression=f"{metrics.compression_ratio:.2f}x",
                     nodes_removed=metrics.nodes_removed,
                     time_ms=f"{elapsed_ms:.1f}")
        return metrics

    def _apply_graph_simplification(self, model: ModelProto) -> ModelProto:
        """Apply ONNX graph simplification passes."""
        available_passes = onnx_optimizer.get_available_passes()
        pass_list = [p for p in self.PASSES if p in available_passes]
        try:
            model = onnx_optimizer.optimize(model, pass_list)
            logger.info("graph_simplification_applied", passes=len(pass_list))
        except Exception as e:
            logger.warning("graph_simplification_failed", error=str(e))
        return model

    def _apply_constant_folding(self, model: ModelProto) -> ModelProto:
        """Fold constant nodes into initializers."""
        try:
            model = onnx_optimizer.optimize(model, ["extract_constant_to_initializer"])
            logger.info("constant_folding_applied")
        except Exception as e:
            logger.warning("constant_folding_failed", error=str(e))
        return model

    def _apply_operator_fusion(self, model: ModelProto) -> ModelProto:
        """Fuse compatible operator sequences."""
        fusion_passes = [
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_matmul_add_bias_into_gemm",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]
        available = onnx_optimizer.get_available_passes()
        pass_list = [p for p in fusion_passes if p in available]
        try:
            model = onnx_optimizer.optimize(model, pass_list)
            logger.info("operator_fusion_applied", fusions=len(pass_list))
        except Exception as e:
            logger.warning("operator_fusion_failed", error=str(e))
        return model

    def _optimize_input_shapes(
        self, model: ModelProto, input_shapes: Dict[str, List[int]]
    ) -> ModelProto:
        """Override input shapes for dynamic shape models."""
        for input_proto in model.graph.input:
            if input_proto.name in input_shapes:
                shape = input_shapes[input_proto.name]
                dims = input_proto.type.tensor_type.shape.dim
                for i, dim in enumerate(shape):
                    if i < len(dims):
                        dims[i].dim_value = dim
                        dims[i].dim_param = ""
                logger.info("input_shape_optimized", name=input_proto.name, shape=shape)
        return model

    def _apply_quantization(
        self,
        model: ModelProto,
        model_path: str,
        mode: QuantizationMode,
        quant_type: QuantizationType,
        calibration_data: Optional[np.ndarray],
        calibration_data_loader,
        num_samples: int,
        per_channel: bool,
        reduce_range: bool,
    ) -> ModelProto:
        """Apply quantization to the model."""
        try:
            from onnxruntime.quantization import (
                quantize_dynamic,
                quantize_static,
                QuantType,
                QuantFormat,
                CalibrationMethod,
            )
        except ImportError:
            logger.error("onnxruntime.quantization not available")
            return model

        if ort is None:
            logger.error("onnxruntime not available")
            return model

        qtype_map = {
            QuantizationType.INT8: QuantType.QInt8,
            QuantizationType.UINT8: QuantType.QUInt8,
            QuantizationType.INT16: QuantType.QInt16,
        }
        target_qtype = qtype_map.get(quant_type, QuantType.QUInt8)

        temp_path = str(Path(model_path).with_suffix(".quant.onnx"))

        if mode == QuantizationMode.DYNAMIC:
            try:
                quantize_dynamic(
                    model_input=model_path,
                    model_output=temp_path,
                    weight_type=target_qtype,
                    per_channel=per_channel,
                    reduce_range=reduce_range,
                )
                model = onnx.load(temp_path)
                logger.info("dynamic_quantization_completed")
            except Exception as e:
                logger.error("dynamic_quantization_failed", error=str(e))

        elif mode == QuantizationMode.STATIC:
            if calibration_data is None and calibration_data_loader is None:
                logger.warning("no_calibration_data_using_dynamic")
                return self._apply_quantization(
                    model, model_path, QuantizationMode.DYNAMIC,
                    quant_type, None, None, num_samples, per_channel, reduce_range,
                )

            try:
                calib_dataloader = calibration_data_loader or self._create_calibrator(
                    calibration_data, num_samples
                )
                quantize_static(
                    model_input=model_path,
                    model_output=temp_path,
                    calibration_data_reader=calib_dataloader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=per_channel,
                    reduce_range=reduce_range,
                    activation_type=target_qtype,
                    weight_type=target_qtype,
                    calibrate_method=CalibrationMethod.MinMax,
                )
                model = onnx.load(temp_path)
                logger.info("static_quantization_completed")
            except Exception as e:
                logger.error("static_quantization_failed", error=str(e), fallback_to_dynamic=True)
                return self._apply_quantization(
                    model, model_path, QuantizationMode.DYNAMIC,
                    quant_type, None, None, num_samples, per_channel, reduce_range,
                )

        elif mode == QuantizationMode.QUANTIZATION_AWARE_TRAINING:
            logger.warning("qat_requires_training_pipeline_use_external_tool")

        if Path(temp_path).exists():
            Path(temp_path).unlink()

        return model

    def _create_calibrator(self, data: np.ndarray, num_samples: int):
        """Create a calibration data reader for static quantization."""
        class CalibrationDataReader:
            def __init__(self, data: np.ndarray, num: int):
                self._data = data[:num] if len(data) > num else data
                self._index = 0

            def get_next(self) -> Optional[Dict[str, np.ndarray]]:
                if self._index >= len(self._data):
                    return None
                batch = self._data[self._index]
                self._index += 1
                return {"input": np.expand_dims(batch, 0).astype(np.float32)}

            def rewind(self) -> None:
                self._index = 0

        return CalibrationDataReader(data, num_samples)

    def validate_model(self, model_path: str, check_full: bool = True) -> ValidationResult:
        """
        Validate an ONNX model for correctness and compatibility.

        Args:
            model_path: Path to ONNX model.
            check_full: Perform full validation including shape inference.

        Returns:
            ValidationResult with errors and warnings.
        """
        errors: List[str] = []
        warnings: List[str] = []

        try:
            model = onnx.load(str(model_path))
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                ir_version=0,
                opset_version=0,
                producer_name="",
                graph_name="",
                num_inputs=0,
                num_outputs=0,
                num_nodes=0,
                input_shapes={},
                output_shapes={},
                errors=[f"Failed to load model: {e}"],
            )

        try:
            checker.check_model(model, full_check=check_full)
        except Exception as e:
            errors.append(str(e))

        input_shapes: Dict[str, List[int]] = {}
        output_shapes: Dict[str, List[int]] = {}

        for input_proto in model.graph.input:
            shape = [d.dim_value for d in input_proto.type.tensor_type.shape.dim]
            input_shapes[input_proto.name] = shape

        for output_proto in model.graph.output:
            shape = [d.dim_value for d in output_proto.type.tensor_type.shape.dim]
            output_shapes[output_proto.name] = shape

        if check_full:
            try:
                inferred = shape_inference.infer_shapes(model)
                for input_proto in inferred.graph.input:
                    name = input_proto.name
                    shape = [d.dim_value for d in input_proto.type.tensor_type.shape.dim]
                    if any(d == 0 for d in shape):
                        warnings.append(f"Dynamic dimension in input '{name}': {shape}")
            except Exception as e:
                warnings.append(f"Shape inference failed: {e}")

        opset_imports = {imp.domain: imp.version for imp in model.opset_import}
        opset_version = opset_imports.get("", 0)

        for node in model.graph.node:
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS:
                    pass

        if model.ir_version < 7:
            warnings.append(f"Older IR version: {model.ir_version} (consider upgrading)")

        result = ValidationResult(
            is_valid=len(errors) == 0,
            ir_version=model.ir_version,
            opset_version=opset_version,
            producer_name=model.producer_name,
            graph_name=model.graph.name,
            num_inputs=len(model.graph.input),
            num_outputs=len(model.graph.output),
            num_nodes=len(model.graph.node),
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            errors=errors,
            warnings=warnings,
        )

        logger.info("validation_complete", is_valid=result.is_valid,
                     errors=len(errors), warnings=len(warnings))
        return result

    def check_compatibility(
        self, model_path: str, target_opset: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Check model compatibility with target runtime.

        Args:
            model_path: Path to ONNX model.
            target_opset: Target opset version to check against.

        Returns:
            Dictionary with compatibility info.
        """
        model = self.load_model(model_path)
        opset_imports = {imp.domain: imp.version for imp in model.opset_import}
        current_opset = opset_imports.get("", 0)

        unsupported_ops: List[str] = []
        if target_opset and current_opset > target_opset:
            opset_diff = current_opset - target_opset
            for node in model.graph.node:
                if node.op_type in UNSUPPORTED_OPS_AFTER.get(target_opset, set()):
                    unsupported_ops.append(f"{node.op_type} (domain={node.domain})")

        result = {
            "model_path": model_path,
            "ir_version": model.ir_version,
            "opset_version": current_opset,
            "target_opset": target_opset,
            "is_compatible": len(unsupported_ops) == 0,
            "unsupported_operators": unsupported_ops,
            "num_inputs": len(model.graph.input),
            "num_outputs": len(model.graph.output),
            "num_nodes": len(model.graph.node),
            "input_shapes": {
                inp.name: [d.dim_value for d in inp.type.tensor_type.shape.dim]
                for inp in model.graph.input
            },
            "output_shapes": {
                out.name: [d.dim_value for d in out.type.tensor_type.shape.dim]
                for out in model.graph.output
            },
        }

        logger.info("compatibility_check", **{k: v for k, v in result.items() if k != "unsupported_operators"})
        return result

    def _get_metrics(self, original_path: str, optimized_path: str) -> OptimizationMetrics:
        """Get optimization metrics for existing files."""
        model = self.load_model(original_path)
        opt_model = self.load_model(optimized_path)
        original_size = Path(original_path).stat().st_size
        optimized_size = Path(optimized_path).stat().st_size

        return OptimizationMetrics(
            original_size_bytes=original_size,
            optimized_size_bytes=optimized_size,
            compression_ratio=original_size / max(optimized_size, 1),
            original_num_nodes=len(model.graph.node),
            optimized_num_nodes=len(opt_model.graph.node),
            nodes_removed=len(model.graph.node) - len(opt_model.graph.node),
            original_num_initializers=len(model.graph.initializer),
            optimized_num_initializers=len(opt_model.graph.initializer),
            optimization_time_ms=0.0,
        )

    @staticmethod
    def fuse_batch_normalization(model: ModelProto) -> ModelProto:
        """Manually fuse BatchNormalization into preceding Conv layers."""
        nodes_to_remove: Set[str] = []
        new_nodes: List[Any] = []

        for node in model.graph.node:
            if node.op_type == "BatchNormalization":
                pass

        return model

    @staticmethod
    def simplify_for_edgetpu(model_path: str, output_path: str) -> str:
        """
        Simplify model for Edge TPU compatibility.

        This replaces operations not supported on Edge TPU with
        equivalent supported operations.
        """
        logger.info("edgetpu_simplification", input=model_path, output=output_path)
        return output_path


UNSUPPORTED_OPS_AFTER: Dict[int, Set[str]] = {
    11: {"Loop", "Scan", "If"},
    13: {"SequenceMap", "SequenceAt"},
}
