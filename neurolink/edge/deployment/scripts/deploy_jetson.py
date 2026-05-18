#!/usr/bin/env python3
"""
Neurolink - Jetson Deployment Script
Adaptive Multimodal Communication Intelligence System

Command-line interface for Jetson model deployment with optimization,
device discovery, verification, and performance benchmarking.

Usage:
    python deploy_jetson.py --model model.onnx --device-id jetson-001 --precision fp16
    python deploy_jetson.py --list-devices
    python deploy_jetson.py --benchmark --model model.trt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from edge.jetson.deployment.jetson_deploy import JetsonDeployment, DeploymentHandle
from edge.jetson.inference.jetson_inference import JetsonInferenceEngine, PrecisionMode, PowerMode
from edge.jetson.optimization.jetson_optimizer import JetsonOptimizer, JetsonPowerMode

logger = structlog.get_logger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neurolink Jetson Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model model.onnx --device-id jetson-001 --precision fp16
  %(prog)s --list-devices
  %(prog)s --benchmark --model model.trt --num-runs 500
  %(prog)s --optimize --model model.onnx --output model_opt.onnx --precision int8
        """,
    )

    parser.add_argument("--model", type=str, help="Path to model file (ONNX or TensorRT)")
    parser.add_argument("--output", type=str, help="Output path for optimized model")
    parser.add_argument("--device-id", type=str, default="jetson-001", help="Jetson device identifier")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32", "int8"],
                        default="fp16", help="Target precision")
    parser.add_argument("--power-mode", type=str,
                        choices=["MAX-N", "MAX-P", "MAX-Q", "MAX-C"],
                        default="MAX-N", help="Jetson power mode")

    parser.add_argument("--optimize", action="store_true", help="Optimize model before deployment")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--list-devices", action="store_true", help="List available Jetson devices")
    parser.add_argument("--verify", action="store_true", help="Verify deployment after deploy")
    parser.add_argument("--containerized", action="store_true", default=True,
                        help="Deploy in Docker container")
    parser.add_argument("--port", type=int, default=8501, help="Service port")
    parser.add_argument("--num-runs", type=int, default=200, help="Benchmark iterations")
    parser.add_argument("--calibration-data", type=str, help="Path to calibration data (npy)")
    parser.add_argument("--version-label", type=str, help="Model version label")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser


def list_devices() -> None:
    """List available Jetson devices on the network."""
    print("\n" + "=" * 60)
    print("  Neurolink - Available Jetson Devices")
    print("=" * 60)

    info = JetsonOptimizer.get_device_info() if hasattr(JetsonOptimizer, "get_device_info") else {}
    print(f"\n  Local System:")
    print(f"    GPU: {info.get('name', 'N/A')}")
    print(f"    Compute Capability: {info.get('compute_capability', 'N/A')}")
    print(f"    Memory: {info.get('total_memory_mb', 'N/A')} MB")

    if JetsonInferenceEngine.is_jetson_platform():
        dla = JetsonInferenceEngine.detect_dla_cores()
        pva = JetsonInferenceEngine.detect_pva_cores()
        print(f"    DLA Cores: {dla}")
        print(f"    PVA Cores: {pva}")

    print(f"\n  Supported Precisions:")
    from edge.optimization.tensorrt.trt_optimizer import Precision
    for p in Precision:
        print(f"    - {p.value}")

    print(f"\n  Power Modes: MAX-N, MAX-P, MAX-Q, MAX-C")
    print("=" * 60)


def optimize_model(args: argparse.Namespace) -> str:
    """Optimize a model for Jetson deployment."""
    print(f"\n  Optimizing model for Jetson...")
    print(f"    Input:  {args.model}")
    print(f"    Precision: {args.precision}")

    output_path = args.output or str(
        Path(args.model).with_suffix(".jetson_opt.trt")
    )

    try:
        optimizer = JetsonOptimizer()
        system_info = optimizer.detect_system_info()
        print(f"    Device: {system_info.model}")

        plan = optimizer.create_optimization_plan(
            args.model,
            JetsonPowerMode[args.power_mode.replace("-", "_")],
            args.precision,
        )

        calib_data = None
        if args.calibration_data and args.precision == "int8":
            import numpy as np
            calib_data = np.load(args.calibration_data)
            print(f"    Calibration data loaded: {calib_data.shape}")

        output_path = optimizer.optimize_for_jetson(
            args.model,
            output_path,
            plan=plan,
            calibration_data=calib_data,
        )

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"    Output: {output_path} ({size_mb:.2f} MB)")
        print(f"    Optimization complete!")

    except ImportError as e:
        print(f"  Warning: Optimization libraries not available: {e}")
        print(f"  Using model as-is.")
        output_path = args.model

    return output_path


def benchmark_model(args: argparse.Namespace, model_path: str) -> None:
    """Run performance benchmarks on a Jetson model."""
    print(f"\n  Benchmarking model on Jetson...")
    print(f"    Model: {model_path}")
    print(f"    Iterations: {args.num_runs}")

    if not Path(model_path).exists():
        print(f"  Error: Model not found: {model_path}")
        return

    try:
        engine = JetsonInferenceEngine(
            precision=PrecisionMode(args.precision),
            power_mode=PowerMode(args.power_mode),
        )
        engine.load_model(model_path)

        results = engine.get_device_stats()
        print(f"\n  Device Stats:")
        print(f"    GPU Memory: {results.memory_used_mb:.0f} / {results.memory_total_mb:.0f} MB")
        print(f"    GPU Temp: {results.temperature_gpu_c:.1f}°C")

        from edge.optimization.tensorrt.trt_optimizer import TensorRTOptimizer
        trt_opt = TensorRTOptimizer()
        benchmark = trt_opt.benchmark(
            model_path,
            input_shape=(1, 3, 224, 224),
            num_warmup=50,
            num_runs=args.num_runs,
        )

        print(f"\n  Benchmark Results:")
        print(f"    Mean Latency:   {benchmark.mean_latency_ms:.3f} ms")
        print(f"    P50 Latency:    {benchmark.p50_latency_ms:.3f} ms")
        print(f"    P95 Latency:    {benchmark.p95_latency_ms:.3f} ms")
        print(f"    P99 Latency:    {benchmark.p99_latency_ms:.3f} ms")
        print(f"    Throughput:     {benchmark.throughput_fps:.1f} FPS")
        print(f"    GPU Memory:     {benchmark.gpu_memory_mb:.0f} MB")

        result_path = Path(model_path).with_suffix(".benchmark.json")
        benchmark.save(str(result_path))
        print(f"\n  Benchmark saved to: {result_path}")

        engine.release()

    except ImportError as e:
        print(f"  Error: Benchmarking failed - {e}")
    except Exception as e:
        print(f"  Error during benchmark: {e}")


def deploy_model(args: argparse.Namespace, model_path: str) -> Optional[DeploymentHandle]:
    """Deploy a model to Jetson device."""
    print(f"\n  Deploying model to Jetson...")
    print(f"    Device:  {args.device_id}")
    print(f"    Model:   {model_path}")
    print(f"    Port:    {args.port}")

    try:
        deployer = JetsonDeployment()
        handle = deployer.deploy_model(
            model_path=model_path,
            device_id=args.device_id,
            precision=args.precision,
            containerized=args.containerized,
            port=args.port,
            version_label=args.version_label,
        )

        print(f"\n  Deployment Successful!")
        print(f"    Deployment ID: {handle.deployment_id}")
        print(f"    Status:         {handle.status.value}")
        print(f"    Endpoint:       {handle.endpoint}")
        print(f"    Service:        {handle.service_name}")

        if args.verify:
            verify_deployment(deployer, handle)

        return handle

    except Exception as e:
        print(f"  Error deploying model: {e}")
        return None


def verify_deployment(deployer: JetsonDeployment, handle: DeploymentHandle) -> None:
    """Verify a deployment is working."""
    print(f"\n  Verifying deployment...")
    report = deployer.monitor_health(handle)
    print(f"    Status:     {report.status.value}")
    print(f"    Checks:     {report.checks_passed}/{report.checks_total}")


def main() -> None:
    parser = setup_parser()
    args = parser.parse_args()

    if args.verbose:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.dev.ConsoleRenderer(),
            ],
        )

    has_action = any([
        args.list_devices, args.benchmark, args.optimize, args.model,
    ])

    if not has_action:
        parser.print_help()
        sys.exit(1)

    if args.list_devices:
        list_devices()
        return

    model_path = args.model
    if not model_path:
        print("Error: --model is required for deploy/benchmark/optimize actions")
        sys.exit(1)

    if args.optimize:
        model_path = optimize_model(args)

    if args.benchmark:
        benchmark_model(args, model_path)

    if args.model and not args.benchmark:
        deploy_model(args, model_path)


if __name__ == "__main__":
    main()
