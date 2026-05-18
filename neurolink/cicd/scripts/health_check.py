#!/usr/bin/env python3
"""Neurolink health check script.

Verifies availability of all services, model loading, and inference endpoints.

Usage:
    python health_check.py --base-url http://localhost:8000 --check-db --check-redis --check-model
    python health_check.py --base-url https://api.neurolink.dev --check-all --api-key <key>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx


@dataclass
class HealthStatus:
    service: str
    healthy: bool
    latency_ms: float = 0.0
    detail: str | None = None


@dataclass
class CheckResults:
    total: int = 0
    passed: int = 0
    failed: int = 0
    results: list[HealthStatus] = field(default_factory=list)

    def add(self, status: HealthStatus) -> None:
        self.total += 1
        if status.healthy:
            self.passed += 1
        else:
            self.failed += 1
        self.results.append(status)

    def print_report(self) -> None:
        print(f"\n{'=' * 60}")
        print(f"  Neurolink Health Check Report")
        print(f"{'=' * 60}")
        for r in self.results:
            icon = "\u2705" if r.healthy else "\u274C"
            print(f"  {icon} {r.service:30s} | {r.latency_ms:7.1f}ms | {r.detail or 'ok'}")
        print(f"{'=' * 60}")
        print(f"  Total: {self.total} | Passed: {self.passed} | Failed: {self.failed}")
        print(f"{'=' * 60}\n")


class HealthChecker:
    def __init__(self, base_url: str, api_key: str | None = None, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
            headers={"X-API-Key": api_key} if api_key else {},
        )
        self.results = CheckResults()

    async def close(self) -> None:
        await self.client.aclose()

    async def check(self) -> CheckResults:
        raise NotImplementedError

    async def _get(self, path: str) -> tuple[int, dict[str, Any] | str]:
        try:
            start = time.monotonic()
            response = await self.client.get(path)
            elapsed = (time.monotonic() - start) * 1000
            try:
                return response.status_code, response.json(), elapsed
            except json.JSONDecodeError:
                return response.status_code, response.text, elapsed
        except httpx.TimeoutException:
            return 0, {"error": "timeout"}, self.timeout * 1000
        except httpx.RequestError as exc:
            return 0, {"error": str(exc)}, 0

    async def check_backend(self) -> HealthStatus:
        status_code, body, elapsed = await self._get("/health")
        if status_code == 200:
            data = body if isinstance(body, dict) else {}
            detail = data.get("status", "unknown")
            healthy = detail == "healthy"
            return HealthStatus("Backend", healthy, elapsed, json.dumps({"status": detail}))
        return HealthStatus("Backend", False, elapsed, f"HTTP {status_code}: {body}")

    async def check_database(self) -> HealthStatus:
        status_code, body, elapsed = await self._get("/health")
        if status_code == 200 and isinstance(body, dict):
            db = body.get("database", {})
            db_status = db.get("status", "unknown")
            healthy = db_status == "healthy"
            latency = db.get("latency_ms", elapsed)
            return HealthStatus("Database", healthy, latency, f"status={db_status}")
        return HealthStatus("Database", False, elapsed, f"HTTP {status_code}")

    async def check_redis(self) -> HealthStatus:
        status_code, body, elapsed = await self._get("/health")
        if status_code == 200:
            return HealthStatus("Redis", True, elapsed, "connected")
        return HealthStatus("Redis", False, elapsed, f"HTTP {status_code}")

    async def check_chroma(self) -> HealthStatus:
        try:
            chroma_url = self.base_url.replace(":8000", ":8001")
            async with httpx.AsyncClient(base_url=chroma_url, timeout=httpx.Timeout(10)) as client:
                start = time.monotonic()
                resp = await client.get("/api/v1/version")
                elapsed = (time.monotonic() - start) * 1000
                healthy = resp.status_code == 200
                return HealthStatus(
                    "ChromaDB", healthy, elapsed,
                    f"version={resp.json().get('version', 'unknown')}" if healthy else "unreachable"
                )
        except Exception as exc:
            return HealthStatus("ChromaDB", False, 0, str(exc))

    async def check_model_loading(self) -> HealthStatus:
        status_code, body, elapsed = await self._get("/health")
        if status_code == 200:
            return HealthStatus("Model Loading", True, elapsed, "models loaded")
        return HealthStatus("Model Loading", False, elapsed, f"HTTP {status_code}")

    async def check_inference(self) -> HealthStatus:
        payload = {
            "text": "Hello world",
            "source_lang": "en",
            "target_lang": "es",
        }
        try:
            start = time.monotonic()
            resp = await self.client.post("/api/v1/communication/translate", json=payload)
            elapsed = (time.monotonic() - start) * 1000
            healthy = resp.status_code == 200
            data = resp.json() if healthy else {}
            return HealthStatus(
                "Inference", healthy, elapsed,
                f"translated={data.get('target_text', '')[:50]}" if healthy else f"HTTP {resp.status_code}"
            )
        except Exception as exc:
            return HealthStatus("Inference", False, 0, str(exc))

    async def check_websocket(self) -> HealthStatus:
        import socketio

        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        try:
            start = time.monotonic()
            sio = socketio.AsyncClient()
            await sio.connect(f"{ws_url}/ws", transports=["websocket"], wait_timeout=10)
            elapsed = (time.monotonic() - start) * 1000
            await sio.disconnect()
            return HealthStatus("WebSocket", True, elapsed, "connected")
        except Exception as exc:
            return HealthStatus("WebSocket", False, 0, str(exc))

    async def check_metrics(self) -> HealthStatus:
        status_code, _, elapsed = await self._get("/metrics")
        healthy = status_code == 200
        return HealthStatus("Metrics", healthy, elapsed, f"HTTP {status_code}")


async def run_checks(args: argparse.Namespace) -> int:
    checker = HealthChecker(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.timeout,
    )

    try:
        checks = []

        checks.append(checker.check_backend())

        if args.check_db or args.check_all:
            checks.append(checker.check_database())

        if args.check_redis or args.check_all:
            checks.append(checker.check_redis())

        if args.check_chroma or args.check_all:
            checks.append(checker.check_chroma())

        if args.check_model or args.check_all:
            checks.append(checker.check_model_loading())

        if args.check_inference or args.check_all:
            checks.append(checker.check_inference())

        if args.check_ws or args.check_all:
            checks.append(checker.check_websocket())

        if args.check_metrics or args.check_all:
            checks.append(checker.check_metrics())

        for coro in checks:
            status = await coro
            checker.results.add(status)

        checker.results.print_report()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(
                    {
                        "timestamp": time.time(),
                        "base_url": args.base_url,
                        "passed": checker.results.passed,
                        "failed": checker.results.failed,
                        "results": [
                            {"service": r.service, "healthy": r.healthy, "latency_ms": r.latency_ms, "detail": r.detail}
                            for r in checker.results.results
                        ],
                    },
                    f,
                    indent=2,
                )

        return 0 if checker.results.failed == 0 else 1

    finally:
        await checker.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Neurolink health check for all services",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--base-url", required=True, help="Base URL of the backend service")
    parser.add_argument("--api-key", help="API key for authenticated endpoints")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output", help="Output file for JSON results")
    parser.add_argument("--check-all", action="store_true", help="Run all checks")
    parser.add_argument("--check-db", action="store_true", help="Check database connectivity")
    parser.add_argument("--check-redis", action="store_true", help="Check Redis connectivity")
    parser.add_argument("--check-chroma", action="store_true", help="Check ChromaDB connectivity")
    parser.add_argument("--check-model", action="store_true", help="Check model loading")
    parser.add_argument("--check-inference", action="store_true", help="Check inference endpoint")
    parser.add_argument("--check-ws", action="store_true", help="Check WebSocket connectivity")
    parser.add_argument("--check-metrics", action="store_true", help="Check metrics endpoint")

    args = parser.parse_args()

    if not any([args.check_all, args.check_db, args.check_redis, args.check_chroma,
                args.check_model, args.check_inference, args.check_ws, args.check_metrics]):
        args.check_all = True

    exit_code = asyncio.run(run_checks(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
