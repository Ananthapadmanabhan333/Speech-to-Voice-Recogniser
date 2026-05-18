from __future__ import annotations

import asyncio
import time
from statistics import mean, median
from typing import Any, Dict, List, Optional

import numpy as np
import pytest

from ai.gesture_engine.classification.gesture_classifier import GestureClassifier
from ai.multimodal_fusion.embeddings.fusion_embeddings import MultimodalEmbeddingFusion


# Performance thresholds in milliseconds
PERFORMANCE_THRESHOLDS = {
    "gesture_p50_ms": 50,
    "gesture_p95_ms": 150,
    "gesture_p99_ms": 300,
    "speech_p50_ms": 500,
    "speech_p95_ms": 2000,
    "multimodal_p50_ms": 100,
    "multimodal_p95_ms": 300,
    "e2e_p50_ms": 1000,
    "e2e_p95_ms": 5000,
}


@pytest.fixture(autouse=True)
def set_seed() -> None:
    np.random.seed(42)


def _compute_percentiles(latencies: List[float]) -> Dict[str, float]:
    arr = np.array(latencies)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


class TestGestureInferenceLatency:
    """Test gesture inference latency."""

    @pytest.fixture(scope="class")
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=10, device="cpu")

    def test_gesture_inference_latency(self, classifier: GestureClassifier) -> None:
        sequences = [
            np.random.randn(np.random.randint(10, 60), 21, 3).astype(np.float32)
            for _ in range(50)
        ]

        latencies: List[float] = []
        for seq in sequences:
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nGesture Inference Latency (ms):")
        print(f"  P50: {percentiles['p50']:.2f}")
        print(f"  P95: {percentiles['p95']:.2f}")
        print(f"  P99: {percentiles['p99']:.2f}")
        print(f"  Mean: {percentiles['mean']:.2f}")
        print(f"  Min: {percentiles['min']:.2f}")
        print(f"  Max: {percentiles['max']:.2f}")

        assert percentiles["p50"] < PERFORMANCE_THRESHOLDS["gesture_p50_ms"], \
            f"P50 {percentiles['p50']:.2f}ms > {PERFORMANCE_THRESHOLDS['gesture_p50_ms']}ms"
        assert percentiles["p95"] < PERFORMANCE_THRESHOLDS["gesture_p95_ms"], \
            f"P95 {percentiles['p95']:.2f}ms > {PERFORMANCE_THRESHOLDS['gesture_p95_ms']}ms"

    def test_gesture_min_sequence_latency(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(5, 21, 3).astype(np.float32)
        latencies: List[float] = []
        for _ in range(20):
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nGesture Min Sequence Latency (ms): P50={percentiles['p50']:.2f}")
        assert percentiles["p50"] < PERFORMANCE_THRESHOLDS["gesture_p50_ms"]

    def test_gesture_max_sequence_latency(self, classifier: GestureClassifier) -> None:
        seq = np.random.randn(150, 21, 3).astype(np.float32)
        latencies: List[float] = []
        for _ in range(20):
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nGesture Max Sequence Latency (ms): P50={percentiles['p50']:.2f}")
        assert percentiles["p95"] < PERFORMANCE_THRESHOLDS["gesture_p95_ms"]


class TestMultimodalFusionLatency:
    """Test multimodal fusion latency."""

    @pytest.fixture(scope="class")
    def fusion(self) -> MultimodalEmbeddingFusion:
        return MultimodalEmbeddingFusion(device="cpu")

    def test_multimodal_fusion_latency(self, fusion: MultimodalEmbeddingFusion) -> None:
        gestures = [np.random.randn(128).astype(np.float32) for _ in range(50)]
        speeches = [np.random.randn(512).astype(np.float32) for _ in range(50)]

        latencies: List[float] = []
        for g, s in zip(gestures, speeches):
            start = time.perf_counter()
            fusion.fuse_embeddings(gesture_emb=g, speech_emb=s)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nMultimodal Fusion Latency (ms):")
        print(f"  P50: {percentiles['p50']:.2f}")
        print(f"  P95: {percentiles['p95']:.2f}")
        print(f"  P99: {percentiles['p99']:.2f}")
        print(f"  Mean: {percentiles['mean']:.2f}")

        assert percentiles["p50"] < PERFORMANCE_THRESHOLDS["multimodal_p50_ms"], \
            f"P50 {percentiles['p50']:.2f}ms > {PERFORMANCE_THRESHOLDS['multimodal_p50_ms']}ms"

    def test_single_modality_latency(self, fusion: MultimodalEmbeddingFusion) -> None:
        gesture = np.random.randn(128).astype(np.float32)
        latencies: List[float] = []
        for _ in range(30):
            start = time.perf_counter()
            fusion.fuse_embeddings(gesture_emb=gesture)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nSingle Modality Fusion Latency (ms): P50={percentiles['p50']:.2f}")
        assert percentiles["p50"] < PERFORMANCE_THRESHOLDS["multimodal_p50_ms"]

    def test_all_modalities_latency(self, fusion: MultimodalEmbeddingFusion) -> None:
        latencies: List[float] = []
        for _ in range(30):
            gesture = np.random.randn(128).astype(np.float32)
            speech = np.random.randn(512).astype(np.float32)
            facial = np.random.randn(128).astype(np.float32)
            context = np.random.randn(256).astype(np.float32)
            start = time.perf_counter()
            fusion.fuse_embeddings(
                gesture_emb=gesture,
                speech_emb=speech,
                facial_emb=facial,
                context_emb=context,
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nAll Modalities Fusion Latency (ms): P50={percentiles['p50']:.2f}")
        assert percentiles["p95"] < PERFORMANCE_THRESHOLDS["multimodal_p95_ms"]


class TestConcurrentLatency:
    """Test latency under concurrent load."""

    @pytest.mark.asyncio
    async def test_concurrent_gesture_inference(self) -> None:
        classifier = GestureClassifier(num_classes=10, device="cpu")

        async def infer() -> float:
            seq = np.random.randn(30, 21, 3).astype(np.float32)
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            return (time.perf_counter() - start) * 1000

        tasks = [infer() for _ in range(10)]
        latencies = await asyncio.gather(*tasks)
        percentiles = _compute_percentiles(list(latencies))

        print(f"\nConcurrent Gesture Latency (ms): P50={percentiles['p50']:.2f}, P95={percentiles['p95']:.2f}")
        assert percentiles["p95"] < PERFORMANCE_THRESHOLDS["gesture_p95_ms"] * 3

    @pytest.mark.asyncio
    async def test_concurrent_multimodal_fusion(self) -> None:
        fusion = MultimodalEmbeddingFusion(device="cpu")

        async def fuse() -> float:
            g = np.random.randn(128).astype(np.float32)
            s = np.random.randn(512).astype(np.float32)
            start = time.perf_counter()
            fusion.fuse_embeddings(gesture_emb=g, speech_emb=s)
            return (time.perf_counter() - start) * 1000

        tasks = [fuse() for _ in range(10)]
        latencies = await asyncio.gather(*tasks)
        percentiles = _compute_percentiles(list(latencies))

        print(f"\nConcurrent Fusion Latency (ms): P50={percentiles['p50']:.2f}, P95={percentiles['p95']:.2f}")
        assert percentiles["p95"] < PERFORMANCE_THRESHOLDS["multimodal_p95_ms"] * 3


class TestEndToEndLatency:
    """Test end-to-end pipeline latency."""

    def test_gesture_to_embedding_latency(self) -> None:
        """Simulate gesture detection -> classification -> embedding pipeline."""
        classifier = GestureClassifier(num_classes=10, device="cpu")
        latencies: List[float] = []

        for _ in range(20):
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            seq = np.random.randn(30, 21, 3).astype(np.float32)

            start = time.perf_counter()

            # Simulate detection (simplified: just classification)
            classifier.classify_gesture(seq)

            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        percentiles = _compute_percentiles(latencies)
        print(f"\nGesture Pipeline E2E Latency (ms): P50={percentiles['p50']:.2f}, P95={percentiles['p95']:.2f}")

    def test_latency_stability(self) -> None:
        """Test that latency doesn't degrade over repeated calls."""
        classifier = GestureClassifier(num_classes=10, device="cpu")
        seq = np.random.randn(30, 21, 3).astype(np.float32)

        # Warmup
        for _ in range(5):
            classifier.classify_gesture(seq)

        # Measure batches
        batch1: List[float] = []
        batch2: List[float] = []
        batch3: List[float] = []

        for _ in range(15):
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            batch1.append((time.perf_counter() - start) * 1000)

        for _ in range(15):
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            batch2.append((time.perf_counter() - start) * 1000)

        for _ in range(15):
            start = time.perf_counter()
            classifier.classify_gesture(seq)
            batch3.append((time.perf_counter() - start) * 1000)

        m1, m2, m3 = mean(batch1), mean(batch2), mean(batch3)
        print(f"\nLatency Stability: batch1={m1:.2f}ms, batch2={m2:.2f}ms, batch3={m3:.2f}ms")

        # Latency should not increase by more than 50% over time
        assert m3 < m1 * 1.5, f"Latency degraded: {m1:.2f} -> {m3:.2f} ms"
