#!/usr/bin/env python3
"""
Neurolink - Raspberry Pi Deployment Script
Adaptive Multimodal Communication Intelligence System

Command-line interface for Raspberry Pi model deployment with optimization,
SSH-based device connection, service installation, and verification.

Usage:
    python deploy_rpi.py --model model.onnx --device-ip 192.168.1.100
    python deploy_rpi.py --benchmark --model model.onnx
    python deploy_rpi.py --optimize --model model.onnx --output model_rpi.onnx
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from edge.raspberry_pi.deployment.rpi_deploy import (
    RPiDeployment, RPiDeploymentHandle, DeploymentMethod, RPiDeviceModel,
)
from edge.raspberry_pi.inference.rpi_inference import (
    RPiInferenceEngine, InferenceBackend, PowerProfile,
)
from edge.raspberry_pi.optimization.rpi_optimizer import (
    RPiOptimizer, TargetBackend,
)

logger = structlog.get_logger(__name__)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Neurolink Raspberry Pi Deployment Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model model.onnx --device-ip 192.168.1.100
  %(prog)s --benchmark --model model.onnx --num-runs 100
  %(prog)s --optimize --model model.onnx --output model_rpi.onnx
  %(prog)s --list-devices --subnet 192.168.1.0/24
  %(prog)s --model model.onnx --device-ip 192.168.1.100 --use-docker
        """,
    )

    parser.add_argument("--model", type=str, help="Path to ONNX model file")
    parser.add_argument("--output", type=str, help="Output path for optimized model")
    parser.add_argument("--device-ip", type=str, help="Raspberry Pi IP address")
    parser.add_argument("--device-model", type=str,
                        choices=["rpi_zero", "rpi_3", "rpi_4", "rpi_5", "rpi_400"],
                        default="rpi_4", help="Raspberry Pi model")
    parser.add_argument("--ssh-user", type=str, default="pi", help="SSH username")
    parser.add_argument("--ssh-key", type=str, help="SSH key path")
    parser.add_argument("--port", type=int, default=8502, help="Service port")

    parser.add_argument("--optimize", action="store_true", help="Optimize model for RPi")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--list-devices", type=str, nargs="?",
                        const="192.168.1.0/24", help="Scan subnet for RPi devices")
    parser.add_argument("--verify", action="store_true", help="Verify deployment")
    parser.add_argument("--use-docker", action="store_true", help="Deploy via Docker")
    parser.add_argument("--use-coral", action="store_true", help="Use Coral TPU if available")
    parser.add_argument("--power-profile", type=str,
                        choices=["performance", "balanced", "power_save"],
                        default="balanced", help="RPi power profile")
    parser.add_argument("--num-runs", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    return parser


def list_devices(subnet: str) -> None:
    """Scan network for Raspberry Pi devices."""
    print("\n" + "=" * 60)
    print(f"  Scanning {subnet} for Raspberry Pi devices...")
    print("=" * 60)

    try:
        import subprocess
        result = subprocess.run(
            ["nmap", "-sn", subnet, "--open"],
            capture_output=True, text=True, timeout=120,
        )

        devices = []
        current_ip = ""
        for line in result.stdout.splitlines():
            if "Nmap scan report for" in line:
                current_ip = line.split()[-1].strip("()")
            if "Raspberry" in line or "pi" in line.lower():
                devices.append((current_ip, line.strip()))

        if devices:
            for ip, desc in devices:
                print(f"  {ip:20s} {desc}")
        else:
            print("  No Raspberry Pi devices found.")

    except (subprocess.SubprocessError, FileNotFoundError):
        print("  nmap not available. Install nmap or specify --device-ip directly.")

    print("=" * 60)


def optimize_model(args: argparse.Namespace) -> str:
    """Optimize a model for Raspberry Pi deployment."""
    print(f"\n  Optimizing model for Raspberry Pi...")
    print(f"    Input:  {args.model}")
    print(f"    Target: {args.device_model}")

    output_path = args.output or str(
        Path(args.model).with_suffix(".rpi_opt.onnx")
    )

    try:
        model_enum = RPiDeviceModel[args.device_model.upper().replace("-", "_")]
        target_enum = RPiDeviceModel[args.device_model.upper().replace("-", "_")]

        optimizer = RPiOptimizer(
            rpi_model=target_enum,
            target_backend=TargetBackend.ONNX_CPU,
        )

        result = optimizer.optimize_for_rpi(
            args.model,
            output_path,
            force_optimize=True,
        )

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        print(f"\n  Optimization Results:")
        print(f"    Output:          {output_path} ({size_mb:.2f} MB)")
        print(f"    Compression:     {result.compression_ratio:.2f}x")
        print(f"    Expected Speedup: {result.expected_speedup:.2f}x")
        print(f"    Modifications:   {len(result.modifications)}")
        for mod in result.modifications:
            print(f"      - {mod}")

        print(f"\n  Optimization Report:")
        print(optimizer.get_optimization_report(result))

    except ImportError as e:
        print(f"  Warning: Optimization libraries not available: {e}")
        output_path = args.model

    return output_path


def benchmark_model(args: argparse.Namespace, model_path: str) -> None:
    """Run performance benchmarks on RPi model."""
    print(f"\n  Benchmarking model on this system...")
    print(f"    Model: {model_path}")
    print(f"    Iterations: {args.num_runs}")

    if not Path(model_path).exists():
        print(f"  Error: Model not found: {model_path}")
        return

    try:
        engine = RPiInferenceEngine(
            backend=InferenceBackend.ONNX_RUNTIME_CPU,
            power_profile=PowerProfile(args.power_profile),
            use_coral=args.use_coral,
        )

        if RPiInferenceEngine.is_raspberry_pi():
            stats = engine.get_system_stats()
            print(f"\n  System Stats:")
            print(f"    CPU: {stats.get('cpu_percent', 0):.1f}%")
            print(f"    Memory: {stats.get('memory_used_mb', 0):.0f} MB")
            print(f"    Temperature: {stats.get('temperature_c', 0):.1f}°C")

        engine.load_model(model_path)

        results = engine.benchmark(
            input_shape=(1, 3, 224, 224),
            num_warmup=10,
            num_runs=args.num_runs,
        )

        print(f"\n  Benchmark Results:")
        print(f"    Mean Latency:   {results['mean_latency_ms']:.3f} ms")
        print(f"    P50 Latency:    {results['p50_latency_ms']:.3f} ms")
        print(f"    P95 Latency:    {results['p95_latency_ms']:.3f} ms")
        print(f"    P99 Latency:    {results['p99_latency_ms']:.3f} ms")
        print(f"    Throughput:     {results['throughput_fps']:.1f} FPS")
        print(f"    Model Size:     {results['model_size_mb']:.2f} MB")

        result_path = Path(model_path).with_suffix(".rpi_benchmark.json")
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Benchmark saved to: {result_path}")

        engine.release()

    except ImportError as e:
        print(f"  Error: Benchmarking failed - {e}")
    except Exception as e:
        print(f"  Error during benchmark: {e}")


def deploy_model(args: argparse.Namespace, model_path: str) -> Optional[RPiDeploymentHandle]:
    """Deploy a model to Raspberry Pi device."""
    if not args.device_ip:
        print("Error: --device-ip is required for deployment")
        return None

    print(f"\n  Deploying model to Raspberry Pi...")
    print(f"    Device:  {args.device_ip}")
    print(f"    Model:   {model_path}")
    print(f"    Port:    {args.port}")
    print(f"    Method:  {'Docker' if args.use_docker else 'SSH'}")

    try:
        deployer = RPiDeployment(
            ssh_key_path=args.ssh_key,
            default_username=args.ssh_user,
        )

        model_enum = RPiDeviceModel[args.device_model.upper().replace("-", "_")]

        handle = deployer.deploy_to_device(
            model_path=model_path,
            device_ip=args.device_ip,
            device_model=model_enum,
            deployment_method=DeploymentMethod.DOCKER if args.use_docker else DeploymentMethod.SSH,
            port=args.port,
            ssh_username=args.ssh_user,
            use_docker=args.use_docker,
        )

        print(f"\n  Deployment Successful!")
        print(f"    Deployment ID: {handle.deployment_id}")
        print(f"    Status:        {handle.status}")
        print(f"    Service:       {handle.service_name}")

        if args.verify:
            verify_deployment(deployer, handle)

        return handle

    except Exception as e:
        print(f"  Error deploying model: {e}")
        return None


def verify_deployment(deployer: RPiDeployment, handle: RPiDeploymentHandle) -> None:
    """Verify a deployment is working."""
    print(f"\n  Verifying deployment...")
    result = deployer.verify_deployment(handle)
    print(f"    Status:  {result.get('status', 'unknown')}")
    for check, passed in result.get("checks", {}).items():
        print(f"    {check}: {'OK' if passed else 'FAIL'}")


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
        list_devices(args.list_devices if isinstance(args.list_devices, str) else "192.168.1.0/24")
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
