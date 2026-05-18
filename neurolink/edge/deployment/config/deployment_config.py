"""
Neurolink - Deployment Configuration
Adaptive Multimodal Communication Intelligence System

Centralized deployment configuration with device types, precision options,
model registry, deployment profiles, resource constraints, network
configuration, and security settings for edge deployment.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)


class DeviceType(Enum):
    JETSON_NANO = "jetson_nano"
    JETSON_TX2 = "jetson_tx2"
    JETSON_XAVIER = "jetson_xavier"
    JETSON_ORIN = "jetson_orin"
    JETSON_ORIN_NX = "jetson_orin_nx"
    RPI_ZERO = "rpi_zero"
    RPI_3 = "rpi_3"
    RPI_4 = "rpi_4"
    RPI_5 = "rpi_5"
    RPI_400 = "rpi_400"
    RPI_CM4 = "rpi_cm4"
    RPI_CM5 = "rpi_cm5"
    CORAL_TPU = "coral_tpu"
    X86_CPU = "x86_cpu"
    X86_GPU = "x86_gpu"
    UNKNOWN = "unknown"


class Precision(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    INT8 = "int8"
    INT4 = "int4"
    MIXED = "mixed"
    BF16 = "bf16"


class OptimizationLevel(Enum):
    NONE = "none"
    BASIC = "basic"
    EXTENDED = "extended"
    AGGRESSIVE = "aggressive"


class DeploymentMode(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DISASTER_RECOVERY = "disaster_recovery"


class SecurityMode(Enum):
    NONE = "none"
    BASIC = "basic"
    ENCRYPTED = "encrypted"
    HARDENED = "hardened"


@dataclass
class HardwareSpec:
    device_type: DeviceType
    cpu_cores: int = 0
    cpu_arch: str = ""
    gpu_name: str = ""
    gpu_compute_capability: str = ""
    memory_mb: int = 0
    storage_mb: int = 0
    has_dla: bool = False
    num_dla_cores: int = 0
    has_pva: bool = False
    num_pva_cores: int = 0
    has_neon: bool = False
    has_coral_tpu: bool = False
    max_power_watts: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def for_device(cls, device_type: DeviceType) -> HardwareSpec:
        specs = _DEVICE_SPECS.get(device_type)
        if specs:
            return cls(**specs)
        return cls(device_type=device_type)


_DEVICE_SPECS: Dict[DeviceType, Dict[str, Any]] = {
    DeviceType.JETSON_NANO: {
        "device_type": DeviceType.JETSON_NANO,
        "cpu_cores": 4,
        "cpu_arch": "armv8",
        "gpu_name": "NVIDIA Maxwell",
        "gpu_compute_capability": "5.3",
        "memory_mb": 4096,
        "has_dla": False,
        "has_pva": False,
        "max_power_watts": 10.0,
    },
    DeviceType.JETSON_TX2: {
        "device_type": DeviceType.JETSON_TX2,
        "cpu_cores": 6,
        "cpu_arch": "armv8",
        "gpu_name": "NVIDIA Pascal",
        "gpu_compute_capability": "6.2",
        "memory_mb": 8192,
        "has_dla": False,
        "has_pva": True,
        "num_pva_cores": 2,
        "max_power_watts": 15.0,
    },
    DeviceType.JETSON_XAVIER: {
        "device_type": DeviceType.JETSON_XAVIER,
        "cpu_cores": 8,
        "cpu_arch": "armv8.2",
        "gpu_name": "NVIDIA Volta",
        "gpu_compute_capability": "7.2",
        "memory_mb": 16384,
        "has_dla": True,
        "num_dla_cores": 2,
        "has_pva": True,
        "num_pva_cores": 2,
        "max_power_watts": 30.0,
    },
    DeviceType.JETSON_ORIN: {
        "device_type": DeviceType.JETSON_ORIN,
        "cpu_cores": 12,
        "cpu_arch": "armv8.2",
        "gpu_name": "NVIDIA Ampere",
        "gpu_compute_capability": "8.7",
        "memory_mb": 32768,
        "has_dla": True,
        "num_dla_cores": 2,
        "has_pva": True,
        "num_pva_cores": 2,
        "max_power_watts": 60.0,
    },
    DeviceType.JETSON_ORIN_NX: {
        "device_type": DeviceType.JETSON_ORIN_NX,
        "cpu_cores": 8,
        "cpu_arch": "armv8.2",
        "gpu_name": "NVIDIA Ampere",
        "gpu_compute_capability": "8.7",
        "memory_mb": 16384,
        "has_dla": True,
        "num_dla_cores": 1,
        "has_pva": True,
        "num_pva_cores": 1,
        "max_power_watts": 25.0,
    },
    DeviceType.RPI_4: {
        "device_type": DeviceType.RPI_4,
        "cpu_cores": 4,
        "cpu_arch": "armv8",
        "memory_mb": 4096,
        "has_neon": True,
        "max_power_watts": 7.5,
    },
    DeviceType.RPI_5: {
        "device_type": DeviceType.RPI_5,
        "cpu_cores": 4,
        "cpu_arch": "armv8",
        "memory_mb": 8192,
        "has_neon": True,
        "max_power_watts": 15.0,
    },
    DeviceType.RPI_ZERO: {
        "device_type": DeviceType.RPI_ZERO,
        "cpu_cores": 1,
        "cpu_arch": "armv6",
        "memory_mb": 512,
        "has_neon": False,
        "max_power_watts": 1.5,
    },
}


@dataclass
class ResourceConstraints:
    max_memory_mb: int = 0
    max_cpu_percent: float = 90.0
    max_gpu_memory_mb: int = 0
    max_model_size_mb: int = 0
    max_latency_ms: float = 100.0
    min_throughput_fps: float = 10.0
    max_temperature_c: float = 85.0
    max_power_watts: float = 0.0
    max_storage_mb: int = 0
    max_batch_size: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkConfig:
    protocol: str = "http"
    host: str = "0.0.0.0"
    port: int = 8501
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    grpc_enabled: bool = False
    grpc_port: int = 8500
    max_connections: int = 100
    request_timeout_s: float = 30.0
    keep_alive_enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityConfig:
    mode: SecurityMode = SecurityMode.NONE
    api_key_required: bool = False
    api_keys: List[str] = field(default_factory=list)
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    token_expiry_minutes: int = 60
    encryption_key: str = ""
    rate_limit_per_minute: int = 1000
    allowed_ips: List[str] = field(default_factory=list)
    enable_audit_logging: bool = True
    enable_model_checksum: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["api_keys"] = ["***hidden***"] if self.api_keys else []
        d["jwt_secret"] = "***hidden***" if self.jwt_secret else ""
        d["encryption_key"] = "***hidden***" if self.encryption_key else ""
        return d


@dataclass
class DeploymentProfile:
    name: str
    device_type: DeviceType
    precision: Precision = Precision.FP16
    optimization_level: OptimizationLevel = OptimizationLevel.EXTENDED
    deployment_mode: DeploymentMode = DeploymentMode.PRODUCTION
    resource_constraints: ResourceConstraints = field(default_factory=ResourceConstraints)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    power_mode: str = "MAX-N"
    batch_size: int = 1
    num_inference_threads: int = 2
    enable_model_warmup: bool = True
    enable_monitoring: bool = True
    enable_auto_restart: bool = True
    enable_ota_updates: bool = False
    log_level: str = "INFO"
    metrics_export_port: int = 0
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> DeploymentProfile:
        with open(path, "r") as f:
            data = json.load(f)
        data["device_type"] = DeviceType(data["device_type"])
        data["precision"] = Precision(data["precision"])
        data["optimization_level"] = OptimizationLevel(data["optimization_level"])
        data["deployment_mode"] = DeploymentMode(data["deployment_mode"])
        data["resource_constraints"] = ResourceConstraints(**data["resource_constraints"])
        data["network"] = NetworkConfig(**data["network"])
        data["security"] = SecurityConfig(**data["security"])
        return cls(**data)

    @classmethod
    def create_jetson_default(cls, device_type: DeviceType = DeviceType.JETSON_ORIN_NX) -> DeploymentProfile:
        return cls(
            name=f"jetson_{device_type.value}_default",
            device_type=device_type,
            precision=Precision.FP16,
            optimization_level=OptimizationLevel.EXTENDED,
            power_mode="MAX-N",
            batch_size=32,
            num_inference_threads=4,
            resource_constraints=ResourceConstraints(
                max_memory_mb=2048,
                max_gpu_memory_mb=2048,
                max_model_size_mb=500,
                max_latency_ms=50.0,
                min_throughput_fps=30.0,
                max_temperature_c=85.0,
            ),
            network=NetworkConfig(port=8501),
            security=SecurityConfig(mode=SecurityMode.BASIC, api_key_required=True),
        )

    @classmethod
    def create_rpi_default(cls, device_type: DeviceType = DeviceType.RPI_4) -> DeploymentProfile:
        return cls(
            name=f"rpi_{device_type.value}_default",
            device_type=device_type,
            precision=Precision.FP32,
            optimization_level=OptimizationLevel.BASIC,
            power_mode="balanced",
            batch_size=1,
            num_inference_threads=2,
            resource_constraints=ResourceConstraints(
                max_memory_mb=256,
                max_model_size_mb=100,
                max_latency_ms=200.0,
                min_throughput_fps=5.0,
                max_temperature_c=80.0,
            ),
            network=NetworkConfig(port=8502),
            security=SecurityConfig(mode=SecurityMode.NONE),
        )


@dataclass
class ModelRegistryEntry:
    model_id: str
    model_path: str
    version: str
    device_type: DeviceType
    precision: Precision
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    size_mb: float
    accuracy: float = 0.0
    latency_ms: float = 0.0
    checksum: str = ""
    status: str = "active"
    registered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Registry for managing deployed models and their versions."""

    def __init__(self, registry_path: str) -> None:
        self._path = Path(registry_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: Dict[str, ModelRegistryEntry] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r") as f:
                data = json.load(f)
            for entry in data.get("entries", []):
                entry["device_type"] = DeviceType(entry["device_type"])
                entry["precision"] = Precision(entry["precision"])
                entry["input_shape"] = tuple(entry["input_shape"])
                entry["output_shape"] = tuple(entry["output_shape"])
                model_entry = ModelRegistryEntry(**entry)
                self._entries[model_entry.model_id] = model_entry

    def _save(self) -> None:
        data = {
            "entries": [
                {**asdict(e), "device_type": e.device_type.value, "precision": e.precision.value}
                for e in self._entries.values()
            ]
        }
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def register(self, entry: ModelRegistryEntry) -> None:
        self._entries[entry.model_id] = entry
        self._save()
        logger.info("model_registered", model_id=entry.model_id, version=entry.version)

    def unregister(self, model_id: str) -> None:
        self._entries.pop(model_id, None)
        self._save()

    def get(self, model_id: str) -> Optional[ModelRegistryEntry]:
        return self._entries.get(model_id)

    def list(self, device_type: Optional[DeviceType] = None) -> List[ModelRegistryEntry]:
        if device_type:
            return [e for e in self._entries.values() if e.device_type == device_type]
        return list(self._entries.values())

    def find_best_for_device(self, device_type: DeviceType) -> Optional[ModelRegistryEntry]:
        candidates = [
            e for e in self._entries.values()
            if e.device_type == device_type and e.status == "active"
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.latency_ms)


class DeploymentConfig:
    """
    Centralized deployment configuration manager.

    Provides factory methods for default profiles, resource constraint
    validation, and configuration persistence for edge deployments.
    """

    def __init__(self, config_dir: Optional[str] = None) -> None:
        self._config_dir = Path(config_dir or str(Path.home() / ".neurolink" / "config"))
        self._config_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, DeploymentProfile] = {}
        self._registry = ModelRegistry(str(self._config_dir / "model_registry.json"))
        self._logger = structlog.get_logger(__name__)
        self._load_profiles()

    def _load_profiles(self) -> None:
        profiles_dir = self._config_dir / "profiles"
        if profiles_dir.exists():
            for profile_file in profiles_dir.glob("*.json"):
                try:
                    profile = DeploymentProfile.load(str(profile_file))
                    self._profiles[profile.name] = profile
                except Exception as e:
                    logger.warning("failed_to_load_profile", file=str(profile_file), error=str(e))

    def get_profile(self, name: str) -> Optional[DeploymentProfile]:
        return self._profiles.get(name)

    def add_profile(self, profile: DeploymentProfile) -> None:
        self._profiles[profile.name] = profile
        profile_dir = self._config_dir / "profiles"
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile.save(str(profile_dir / f"{profile.name}.json"))
        logger.info("profile_added", name=profile.name)

    def list_profiles(self) -> List[str]:
        return list(self._profiles.keys())

    def get_registry(self) -> ModelRegistry:
        return self._registry

    def validate_constraints(
        self,
        model_size_mb: float,
        device_type: DeviceType,
        profile: Optional[DeploymentProfile] = None,
    ) -> Dict[str, Any]:
        """Validate model against device resource constraints."""
        constraints = profile.resource_constraints if profile else ResourceConstraints()
        spec = HardwareSpec.for_device(device_type)

        validation: Dict[str, Any] = {
            "valid": True,
            "checks": [],
            "warnings": [],
            "errors": [],
        }

        if constraints.max_model_size_mb > 0 and model_size_mb > constraints.max_model_size_mb:
            validation["checks"].append({
                "check": "model_size",
                "passed": False,
                "message": f"Model size {model_size_mb:.0f}MB exceeds limit {constraints.max_model_size_mb}MB",
            })
            validation["errors"].append("model_size_exceeded")
            validation["valid"] = False
        else:
            validation["checks"].append({
                "check": "model_size",
                "passed": True,
                "message": f"Model size {model_size_mb:.0f}MB within limit",
            })

        if spec.memory_mb > 0 and constraints.max_memory_mb > spec.memory_mb:
            validation["warnings"].append(
                f"Requested memory {constraints.max_memory_mb}MB exceeds device capacity {spec.memory_mb}MB"
            )
            validation["checks"].append({
                "check": "memory",
                "passed": False,
                "message": "Requested memory exceeds device capacity",
            })
        else:
            validation["checks"].append({
                "check": "memory",
                "passed": True,
                "message": "Memory within device capacity",
            })

        return validation

    def suggest_precision(self, device_type: DeviceType) -> Precision:
        """Suggest optimal precision for a device type."""
        precision_map = {
            DeviceType.JETSON_ORIN: Precision.FP16,
            DeviceType.JETSON_ORIN_NX: Precision.FP16,
            DeviceType.JETSON_XAVIER: Precision.FP16,
            DeviceType.JETSON_NANO: Precision.FP16,
            DeviceType.JETSON_TX2: Precision.FP16,
            DeviceType.RPI_5: Precision.FP32,
            DeviceType.RPI_4: Precision.FP32,
            DeviceType.RPI_3: Precision.FP32,
            DeviceType.RPI_ZERO: Precision.FP32,
        }
        return precision_map.get(device_type, Precision.FP32)

    def save(self) -> None:
        """Save all configuration to disk."""
        for profile in self._profiles.values():
            profile_dir = self._config_dir / "profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            profile.save(str(profile_dir / f"{profile.name}.json"))
        logger.info("configuration_saved", profiles=len(self._profiles))

    @classmethod
    def from_defaults(cls) -> DeploymentConfig:
        """Create configuration with default profiles."""
        config = cls()
        config.add_profile(DeploymentProfile.create_jetson_default(DeviceType.JETSON_ORIN_NX))
        config.add_profile(DeploymentProfile.create_jetson_default(DeviceType.JETSON_NANO))
        config.add_profile(DeploymentProfile.create_rpi_default(DeviceType.RPI_4))
        config.add_profile(DeploymentProfile.create_rpi_default(DeviceType.RPI_5))
        return config


def get_device_spec(device_type: DeviceType) -> HardwareSpec:
    return HardwareSpec.for_device(device_type)


def get_supported_precisions(device_type: DeviceType) -> List[Precision]:
    """Get supported precision formats for a device type."""
    base = [Precision.FP32]

    jetson_devices = {
        DeviceType.JETSON_NANO, DeviceType.JETSON_TX2,
        DeviceType.JETSON_XAVIER, DeviceType.JETSON_ORIN,
        DeviceType.JETSON_ORIN_NX,
    }

    if device_type in jetson_devices:
        base.append(Precision.FP16)
        base.append(Precision.INT8)
        if device_type in (DeviceType.JETSON_ORIN, DeviceType.JETSON_ORIN_NX):
            base.append(Precision.BF16)

    return base
