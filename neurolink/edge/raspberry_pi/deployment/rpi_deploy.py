"""
Neurolink - Raspberry Pi Deployment Engine
Adaptive Multimodal Communication Intelligence System

Provides production-grade deployment pipeline for Raspberry Pi platforms
with SSH-based deployment, Docker container deployment, systemd service
setup, auto-start configuration, resource monitoring, log collection,
and remote update mechanisms.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import structlog

logger = structlog.get_logger(__name__)


class RPiDeviceModel(Enum):
    RPI_ZERO = "rpi_zero"
    RPI_3 = "rpi_3"
    RPI_4 = "rpi_4"
    RPI_5 = "rpi_5"
    RPI_400 = "rpi_400"
    UNKNOWN = "unknown"


class DeploymentMethod(Enum):
    SSH = "ssh"
    DOCKER = "docker"
    LOCAL = "local"


class ServiceStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class RPiDeploymentHandle:
    deployment_id: str
    device_ip: str
    device_model: RPiDeviceModel
    model_path: str
    deployment_method: DeploymentMethod
    service_name: str = ""
    port: int = 0
    status: str = "pending"
    deployed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_seen: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


@dataclass
class RPiSystemMetrics:
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    temperature_c: float = 0.0
    disk_used_mb: float = 0.0
    disk_percent: float = 0.0
    uptime_hours: float = 0.0
    load_average: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    wifi_signal_dbm: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SSHConnection:
    """Manages SSH connections to Raspberry Pi devices."""

    def __init__(
        self,
        host: str,
        port: int = 22,
        username: str = "pi",
        key_path: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        self._host = host
        self._port = port
        self._username = username
        self._key_path = key_path
        self._password = password

    def run_command(self, command: str, timeout_s: int = 30) -> Tuple[int, str, str]:
        """Run a command on the remote device via SSH."""
        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(self._port),
        ]

        if self._key_path:
            ssh_cmd.extend(["-i", self._key_path])

        ssh_cmd.append(f"{self._username}@{self._host}")
        ssh_cmd.append(command)

        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=timeout_s,
        )
        return result.returncode, result.stdout, result.stderr

    def copy_file(self, local_path: str, remote_path: str, timeout_s: int = 60) -> bool:
        """Copy a file to the remote device via SCP."""
        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=no",
            "-P", str(self._port),
        ]

        if self._key_path:
            scp_cmd.extend(["-i", self._key_path])

        scp_cmd.extend([local_path, f"{self._username}@{self._host}:{remote_path}"])

        result = subprocess.run(scp_cmd, capture_output=True, timeout=timeout_s)
        return result.returncode == 0

    def test_connection(self) -> bool:
        """Test SSH connection to the device."""
        code, _, _ = self.run_command("echo 'neurolink_connection_test'")
        return code == 0

    def close(self) -> None:
        pass


class RPiDeployment:
    """
    Production-grade deployment manager for Raspberry Pi platforms
    with SSH, Docker, and local deployment methods, systemd service
    management, health monitoring, and remote updates.

    Usage:
        deployer = RPiDeployment()
        handle = deployer.deploy_to_device("model.onnx", "192.168.1.100")
        deployer.monitor_resources(handle)
        deployer.update_deployment(handle, "model_v2.onnx")
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        ssh_key_path: Optional[str] = None,
        default_username: str = "pi",
    ) -> None:
        self._data_dir = Path(data_dir or str(Path.home() / ".neurolink" / "rpi_deployments"))
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._ssh_key_path = ssh_key_path
        self._default_username = default_username
        self._deployments: Dict[str, RPiDeploymentHandle] = {}
        self._monitoring_threads: Dict[str, threading.Thread] = {}
        self._logger = structlog.get_logger(__name__)

    def deploy_to_device(
        self,
        model_path: str,
        device_ip: str,
        device_model: Optional[RPiDeviceModel] = None,
        deployment_method: DeploymentMethod = DeploymentMethod.SSH,
        port: int = 8502,
        ssh_port: int = 22,
        ssh_username: Optional[str] = None,
        use_docker: bool = False,
        config: Optional[Dict[str, Any]] = None,
    ) -> RPiDeploymentHandle:
        """
        Deploy a model to a Raspberry Pi device.

        Args:
            model_path: Path to optimized model file.
            device_ip: IP address of the RPi device.
            device_model: RPi model (auto-detected if not specified).
            deployment_method: Deployment method (SSH, Docker, Local).
            port: Service port for inference endpoint.
            ssh_port: SSH port on the device.
            ssh_username: SSH username (default: pi).
            use_docker: Deploy in Docker container.
            config: Additional deployment configuration.

        Returns:
            RPiDeploymentHandle for managing the deployment.
        """
        model_path_obj = Path(model_path)
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        deployment_id = f"rpi_{device_ip.replace('.', '_')}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

        handle = RPiDeploymentHandle(
            deployment_id=deployment_id,
            device_ip=device_ip,
            device_model=device_model or RPiDeviceModel.UNKNOWN,
            model_path=str(model_path_obj.resolve()),
            deployment_method=deployment_method,
            port=port,
            config=config or {},
        )

        self._deployments[deployment_id] = handle

        if deployment_method == DeploymentMethod.SSH:
            self._deploy_via_ssh(handle, ssh_port, ssh_username or self._default_username)
        elif deployment_method == DeploymentMethod.DOCKER:
            self._deploy_via_docker(handle, ssh_port, ssh_username or self._default_username)
        elif deployment_method == DeploymentMethod.LOCAL:
            self._deploy_local(handle)
        else:
            raise ValueError(f"Unsupported deployment method: {deployment_method}")

        handle.status = "active"
        handle.last_seen = datetime.utcnow().isoformat()
        handle.save(str(self._data_dir / f"{deployment_id}.json"))

        logger.info("deployment_completed",
                     device_ip=device_ip,
                     deployment_id=deployment_id,
                     method=deployment_method.value)
        return handle

    def _deploy_via_ssh(
        self, handle: RPiDeploymentHandle, ssh_port: int, username: str,
    ) -> None:
        """Deploy model via SSH to Raspberry Pi."""
        ssh = SSHConnection(
            host=handle.device_ip,
            port=ssh_port,
            username=username,
            key_path=self._ssh_key_path,
        )

        logger.info("establishing_ssh_connection", ip=handle.device_ip)

        if not ssh.test_connection():
            raise ConnectionError(f"Cannot connect to {handle.device_ip} via SSH")

        remote_dir = f"/opt/neurolink/models/{handle.deployment_id}"
        code, out, err = ssh.run_command(f"mkdir -p {remote_dir}")
        if code != 0:
            raise RuntimeError(f"Failed to create remote directory: {err}")

        success = ssh.copy_file(handle.model_path, f"{remote_dir}/model.onnx")
        if not success:
            raise RuntimeError("Failed to copy model to device")

        self._install_service_via_ssh(ssh, handle, remote_dir)
        self._setup_auto_start_via_ssh(ssh, handle)

        handle.service_name = f"neurolink-{handle.deployment_id}"
        logger.info("ssh_deployment_completed", ip=handle.device_ip)

    def _install_service_via_ssh(
        self, ssh: SSHConnection, handle: RPiDeploymentHandle, remote_dir: str,
    ) -> None:
        """Install systemd service via SSH."""
        service_content = f"""[Unit]
Description=Neurolink Inference Service - {handle.deployment_id}
After=network.target

[Service]
Type=simple
ExecStartPre=/bin/sleep 2
ExecStart=/usr/bin/python3 -c "
import onnxruntime as ort
import numpy as np
import json
from http.server import HTTPServer, BaseHTTPRequestHandler

class InferenceHandler(BaseHTTPRequestHandler):
    model = ort.InferenceSession('{remote_dir}/model.onnx')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = np.frombuffer(body, dtype=np.float32).reshape(
            {self._get_input_shape()})
        outputs = self.model.run(None, {{self.model.get_inputs()[0].name: data}})
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        self.wfile.write(outputs[0].tobytes())

    def _get_input_shape(self):
        return tuple(self.model.get_inputs()[0].shape)

HTTPServer(('0.0.0.0', {handle.port}), InferenceHandler).serve_forever()
"
Restart=always
RestartSec=5
User=root
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=multi-user.target
"""

        handle.service_name = f"neurolink-{handle.deployment_id}"
        service_path = f"/etc/systemd/system/{handle.service_name}.service"

        ssh.run_command(f"cat > {service_path} << 'SERVICEEOF'\n{service_content}\nSERVICEEOF")

        ssh.run_command("systemctl daemon-reload")
        ssh.run_command(f"systemctl enable {handle.service_name}")
        ssh.run_command(f"systemctl start {handle.service_name}")

    def _setup_auto_start_via_ssh(
        self, ssh: SSHConnection, handle: RPiDeploymentHandle,
    ) -> None:
        """Configure auto-start for the service."""
        ssh.run_command(f"systemctl enable {handle.service_name}")

    def _deploy_via_docker(
        self, handle: RPiDeploymentHandle, ssh_port: int, username: str,
    ) -> None:
        """Deploy model via Docker on Raspberry Pi."""
        ssh = SSHConnection(
            host=handle.device_ip,
            port=ssh_port,
            username=username,
            key_path=self._ssh_key_path,
        )

        if not ssh.test_connection():
            raise ConnectionError(f"Cannot connect to {handle.device_ip}")

        remote_dir = f"/opt/neurolink/models/{handle.deployment_id}"
        ssh.run_command(f"mkdir -p {remote_dir}")
        ssh.copy_file(handle.model_path, f"{remote_dir}/model.onnx")

        dockerfile = f"""FROM arm64v8/python:3.9-slim
RUN pip install onnxruntime numpy
COPY {remote_dir}/model.onnx /models/model.onnx
EXPOSE {handle.port}
CMD ["python3", "-c", "
import onnxruntime as ort
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler

class Handler(BaseHTTPRequestHandler):
    model = ort.InferenceSession('/models/model.onnx')
    def do_POST(self):
        body = self.rfile.read(int(self.headers['Content-Length']))
        data = np.frombuffer(body, dtype=np.float32).reshape(tuple(self.model.get_inputs()[0].shape))
        outputs = self.model.run(None, {{self.model.get_inputs()[0].name: data}})
        self.send_response(200)
        self.send_header('Content-Type', 'application/octet-stream')
        self.end_headers()
        self.wfile.write(outputs[0].tobytes())

HTTPServer(('0.0.0.0', {handle.port}), Handler).serve_forever()
"]
"""
        ssh.run_command(f"cat > {remote_dir}/Dockerfile << 'DOCKEREOF'\n{dockerfile}\nDOCKEREOF")

        ssh.run_command(
            f"cd {remote_dir} && docker build -t neurolink-{handle.deployment_id} ."
        )
        ssh.run_command(
            f"docker run -d --restart unless-stopped "
            f"--name neurolink-{handle.deployment_id} "
            f"-p {handle.port}:{handle.port} "
            f"neurolink-{handle.deployment_id}"
        )

        handle.service_name = f"docker-neurolink-{handle.deployment_id}"
        logger.info("docker_deployment_completed", ip=handle.device_ip)

    def _deploy_local(self, handle: RPiDeploymentHandle) -> None:
        """Deploy model locally (same machine)."""
        local_model_dir = Path(f"/opt/neurolink/models/{handle.deployment_id}")
        local_model_dir.mkdir(parents=True, exist_ok=True)
        dest = local_model_dir / "model.onnx"
        shutil.copy2(handle.model_path, dest)
        handle.service_name = f"neurolink-{handle.deployment_id}"

        logger.info("local_deployment_completed", path=str(dest))

    def get_deployment(self, deployment_id: str) -> Optional[RPiDeploymentHandle]:
        return self._deployments.get(deployment_id)

    def list_deployments(self) -> List[RPiDeploymentHandle]:
        return list(self._deployments.values())

    def get_device_metrics(
        self, handle: RPiDeploymentHandle,
    ) -> Optional[RPiSystemMetrics]:
        """Get system metrics from deployed Raspberry Pi."""
        if handle.deployment_method == DeploymentMethod.SSH:
            return self._get_remote_metrics(handle)
        return None

    def _get_remote_metrics(self, handle: RPiDeploymentHandle) -> Optional[RPiSystemMetrics]:
        """Get metrics from remote device via SSH."""
        ssh = SSHConnection(
            host=handle.device_ip,
            username=self._default_username,
            key_path=self._ssh_key_path,
        )

        try:
            code, stdout, _ = ssh.run_command(
                "python3 -c \""
                "import psutil, json; "
                "print(json.dumps({"
                "'cpu_percent': psutil.cpu_percent(interval=0.5), "
                "'memory_percent': psutil.virtual_memory().percent, "
                "'memory_used_mb': psutil.virtual_memory().used / 1024 / 1024, "
                "'disk_used_mb': psutil.disk_usage('/').used / 1024 / 1024, "
                "'disk_percent': psutil.disk_usage('/').percent, "
                "'uptime_hours': (__import__('time').time() - psutil.boot_time()) / 3600, "
                "'load_average': list(__import__('os').getloadavg()), "
                "}))\""
            )

            if code == 0:
                data = json.loads(stdout)
                metrics = RPiSystemMetrics(**data)

                temp_code, temp_out, _ = ssh.run_command("cat /sys/class/thermal/thermal_zone0/temp")
                if temp_code == 0:
                    metrics.temperature_c = float(temp_out.strip()) / 1000.0

                return metrics

        except Exception as e:
            logger.warning("failed_to_get_metrics", error=str(e))

        return None

    def monitor_resources(
        self,
        handle: RPiDeploymentHandle,
        interval_s: float = 30.0,
        callback: Optional[Callable[[RPiSystemMetrics], None]] = None,
        threshold_temp: float = 80.0,
    ) -> None:
        """
        Start resource monitoring on deployed device.

        Args:
            handle: Deployment handle.
            interval_s: Polling interval in seconds.
            callback: Called with metrics on each poll.
            threshold_temp: Temperature threshold for warning.
        """
        def _monitor() -> None:
            while True:
                metrics = self.get_device_metrics(handle)
                if metrics:
                    handle.metrics = metrics.to_dict()
                    handle.last_seen = datetime.utcnow().isoformat()

                    if metrics.temperature_c > threshold_temp:
                        logger.warning("high_temperature",
                                       device=handle.device_ip,
                                       temp_c=metrics.temperature_c)

                    if callback:
                        callback(metrics)

                time.sleep(interval_s)

        thread = threading.Thread(target=_monitor, daemon=True)
        thread.start()
        self._monitoring_threads[handle.deployment_id] = thread
        logger.info("resource_monitor_started", ip=handle.device_ip, interval_s=interval_s)

    def collect_logs(
        self,
        handle: RPiDeploymentHandle,
        lines: int = 100,
        output_path: Optional[str] = None,
    ) -> str:
        """Collect service logs from deployed device."""
        if handle.deployment_method != DeploymentMethod.SSH:
            logger.warning("log_collection_only_supported_for_ssh")
            return ""

        ssh = SSHConnection(
            host=handle.device_ip,
            username=self._default_username,
            key_path=self._ssh_key_path,
        )

        code, stdout, _ = ssh.run_command(
            f"journalctl -u {handle.service_name} --no-pager -n {lines}"
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            Path(output_path).write_text(stdout)
            logger.info("logs_saved", path=output_path)

        return stdout

    def update_deployment(
        self,
        handle: RPiDeploymentHandle,
        new_model_path: str,
        restart_service: bool = True,
    ) -> RPiDeploymentHandle:
        """
        Update a deployed model (OTA update).

        Args:
            handle: Existing deployment handle.
            new_model_path: Path to new model file.
            restart_service: Restart the service after update.

        Returns:
            Updated deployment handle.
        """
        logger.info("ota_update_initiated",
                     device=handle.device_ip,
                     new_model=new_model_path)

        if handle.deployment_method == DeploymentMethod.SSH:
            ssh = SSHConnection(
                host=handle.device_ip,
                username=self._default_username,
                key_path=self._ssh_key_path,
            )

            remote_dir = f"/opt/neurolink/models/{handle.deployment_id}"
            success = ssh.copy_file(new_model_path, f"{remote_dir}/model.onnx")
            if not success:
                raise RuntimeError("Failed to copy updated model")

            if restart_service:
                ssh.run_command(f"systemctl restart {handle.service_name}")

        handle.model_path = str(Path(new_model_path).resolve())
        handle.last_seen = datetime.utcnow().isoformat()
        handle.save(str(self._data_dir / f"{handle.deployment_id}.json"))

        logger.info("ota_update_completed", device=handle.device_ip)
        return handle

    def stop_deployment(self, handle: RPiDeploymentHandle) -> None:
        """Stop a deployment."""
        if handle.deployment_method == DeploymentMethod.SSH:
            ssh = SSHConnection(
                host=handle.device_ip,
                username=self._default_username,
                key_path=self._ssh_key_path,
            )
            ssh.run_command(f"systemctl stop {handle.service_name}")
            ssh.run_command(f"systemctl disable {handle.service_name}")

        handle.status = "stopped"
        handle.save(str(self._data_dir / f"{handle.deployment_id}.json"))
        logger.info("deployment_stopped", device=handle.device_ip)

    def restart_deployment(self, handle: RPiDeploymentHandle) -> None:
        """Restart a deployment."""
        if handle.deployment_method == DeploymentMethod.SSH:
            ssh = SSHConnection(
                host=handle.device_ip,
                username=self._default_username,
                key_path=self._ssh_key_path,
            )
            ssh.run_command(f"systemctl restart {handle.service_name}")

        handle.status = "active"
        handle.last_seen = datetime.utcnow().isoformat()
        logger.info("deployment_restarted", device=handle.device_ip)

    def verify_deployment(self, handle: RPiDeploymentHandle) -> Dict[str, Any]:
        """Verify a deployment is working correctly."""
        verification: Dict[str, Any] = {
            "deployment_id": handle.deployment_id,
            "device_ip": handle.device_ip,
            "status": "unknown",
            "checks": {},
        }

        try:
            if handle.deployment_method == DeploymentMethod.SSH:
                ssh = SSHConnection(
                    host=handle.device_ip,
                    username=self._default_username,
                    key_path=self._ssh_key_path,
                )

                code, out, _ = ssh.run_command(
                    f"systemctl is-active {handle.service_name}"
                )
                is_active = code == 0 and "active" in out
                verification["checks"]["service_active"] = is_active

                code, out, _ = ssh.run_command(
                    f"ss -tlnp | grep {handle.port} || netstat -tlnp | grep {handle.port}"
                )
                is_listening = code == 0
                verification["checks"]["port_listening"] = is_listening

                metrics = self._get_remote_metrics(handle)
                if metrics:
                    verification["metrics"] = metrics.to_dict()
                    verification["checks"]["memory_ok"] = metrics.memory_percent < 90.0
                    verification["checks"]["cpu_ok"] = metrics.cpu_percent < 90.0
                    verification["checks"]["temperature_ok"] = metrics.temperature_c < 80.0

                passed = sum(1 for v in verification["checks"].values() if v)
                total = len(verification["checks"])
                verification["status"] = "healthy" if passed == total else "degraded"

        except Exception as e:
            verification["status"] = "error"
            verification["error"] = str(e)

        verification["checks_passed"] = sum(1 for v in verification["checks"].values() if v)
        verification["checks_total"] = len(verification["checks"])
        logger.info("deployment_verified",
                     device=handle.device_ip,
                     status=verification["status"])
        return verification
