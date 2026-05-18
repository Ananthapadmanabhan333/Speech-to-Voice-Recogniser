from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.gesture_engine.classification.gesture_classifier import (
    GestureClassifier,
    GestureLSTM,
    TemporalCNN,
    TemporalConvBlock,
)


@pytest.fixture(autouse=True)
def set_seed() -> None:
    torch.manual_seed(42)
    np.random.seed(42)


class TestModelForwardPass:
    """Test model forward pass shapes and values."""

    @pytest.fixture
    def classifier(self) -> GestureClassifier:
        return GestureClassifier(num_classes=10, device="cpu")

    def test_temporal_cnn_forward_shape(self) -> None:
        model = TemporalCNN(input_dim=63, hidden_dims=[128, 64])
        x = torch.randn(4, 30, 63)
        out = model(x)
        assert out.shape == (4, 30, 64)

    def test_temporal_cnn_with_varying_seq_len(self) -> None:
        model = TemporalCNN(input_dim=63, hidden_dims=[64, 32])
        x = torch.randn(2, 50, 63)
        out = model(x)
        assert out.shape == (2, 50, 32)

    def test_gesture_lstm_forward_shape(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=128, num_layers=2, num_classes=10)
        x = torch.randn(4, 30, 64)
        out = model(x)
        assert out.shape == (4, 10)

    def test_full_pipeline_forward(self, classifier: GestureClassifier) -> None:
        seq = torch.randn(1, 30, 63)
        cnn_out = classifier._temporal_cnn(seq)
        assert cnn_out.shape == (1, 30, 128)
        logits = classifier._lstm(cnn_out)
        assert logits.shape == (1, 10)

    def test_temporal_conv_block(self) -> None:
        block = TemporalConvBlock(in_channels=63, out_channels=128, kernel_size=3)
        x = torch.randn(4, 63, 30)
        out = block(x)
        assert out.shape == (4, 128, 30)

    def test_all_channels_preserved(self) -> None:
        model = TemporalCNN(input_dim=63, hidden_dims=[128, 256, 128])
        x = torch.randn(2, 20, 63)
        out = model(x)
        assert out.shape[-1] == 128


class TestTrainingStep:
    """Test training step functionality."""

    def test_loss_computation(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        x = torch.randn(8, 20, 64)
        targets = torch.randint(0, 5, (8,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        assert loss.item() > 0.0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_backward_pass(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        x = torch.randn(4, 20, 64)
        targets = torch.randint(0, 5, (4,))
        logits = model(x)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()
                break

    def test_training_reduces_loss(self) -> None:
        model = GestureLSTM(input_dim=16, hidden_dim=32, num_layers=1, num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        x = torch.randn(16, 10, 16)
        targets = torch.randint(0, 3, (16,))

        losses = []
        for _ in range(10):
            logits = model(x)
            loss = F.cross_entropy(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0] + 0.1  # Should not diverge


class TestValidation:
    """Test validation mode and gradients."""

    def test_eval_mode_disables_grad(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        model.eval()
        x = torch.randn(2, 15, 64)
        with torch.no_grad():
            out = model(x)
        assert out.requires_grad is False

    def test_train_mode_enables_grad(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        model.train()
        x = torch.randn(2, 15, 64)
        out = model(x)
        assert out.requires_grad is True

    def test_accuracy_computation(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        model.eval()
        x = torch.randn(10, 20, 64)
        targets = torch.randint(0, 5, (10,))
        with torch.no_grad():
            logits = model(x)
            preds = logits.argmax(dim=1)
            accuracy = (preds == targets).float().mean()
        assert 0.0 <= accuracy <= 1.0


class TestCheckpointing:
    """Test model checkpoint save/load."""

    def test_save_and_load_state_dict(self, tmp_path: Path) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        path = tmp_path / "model.pt"
        torch.save(model.state_dict(), path)
        assert path.exists()

        model2 = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=1, num_classes=5)
        model2.load_state_dict(torch.load(path))
        x = torch.randn(2, 15, 64)
        with torch.no_grad():
            out1 = model(x)
            out2 = model2(x)
        assert torch.allclose(out1, out2)

    def test_save_full_checkpoint(self, tmp_path: Path) -> None:
        classifier = GestureClassifier(num_classes=5, device="cpu")
        path = tmp_path / "full_checkpoint.pt"
        classifier.save(path)
        assert path.exists()

        classifier2 = GestureClassifier(num_classes=5, device="cpu", model_path=path)
        x = np.random.randn(20, 21, 3).astype(np.float32)
        r1 = classifier.classify_gesture(x)
        r2 = classifier2.classify_gesture(x)
        assert r1.gesture_id == r2.gesture_id

    def test_checkpoint_contains_all_keys(self, tmp_path: Path) -> None:
        classifier = GestureClassifier(num_classes=5, device="cpu")
        path = tmp_path / "checkpoint.pt"
        classifier.save(path)
        checkpoint = torch.load(path)
        assert "temporal_cnn" in checkpoint
        assert "lstm" in checkpoint
        assert "temperature" in checkpoint
        assert "gesture_labels" in checkpoint


class TestONNXExport:
    """Test ONNX model export."""

    def test_trace_forward(self) -> None:
        model = GestureLSTM(input_dim=63, hidden_dim=128, num_layers=1, num_classes=10)
        model.eval()
        x = torch.randn(1, 30, 63)
        with torch.no_grad():
            traced = torch.jit.trace(model, x)
            out = traced(x)
        assert out.shape == (1, 10)

    def test_onnx_export(self, tmp_path: Path) -> None:
        model = GestureLSTM(input_dim=63, hidden_dim=128, num_layers=1, num_classes=10)
        model.eval()
        x = torch.randn(1, 30, 63)
        path = tmp_path / "model.onnx"
        torch.onnx.export(
            model,
            x,
            path,
            input_names=["input"],
            output_names=["output"],
            opset_version=14,
        )
        assert path.exists()
        assert path.stat().st_size > 0

    def test_onnx_export_with_dynamic_axes(self, tmp_path: Path) -> None:
        model = GestureLSTM(input_dim=63, hidden_dim=128, num_layers=1, num_classes=10)
        model.eval()
        x = torch.randn(1, 30, 63)
        path = tmp_path / "dynamic_model.onnx"
        torch.onnx.export(
            model,
            x,
            path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch", 1: "sequence"},
                "output": {0: "batch"},
            },
            opset_version=14,
        )
        assert path.exists()


class TestInferenceLatency:
    """Test inference latency."""

    def test_temporal_cnn_latency(self) -> None:
        model = TemporalCNN(input_dim=63, hidden_dims=[128, 256, 128])
        model.eval()
        x = torch.randn(1, 30, 63)
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(10):
                model(x)
            elapsed = (time.perf_counter() - start) * 1000
        avg_ms = elapsed / 10
        assert avg_ms < 1000  # Should complete well under 1 second

    def test_lstm_latency(self) -> None:
        model = GestureLSTM(input_dim=63, hidden_dim=256, num_layers=2, num_classes=50)
        model.eval()
        x = torch.randn(1, 30, 63)
        with torch.no_grad():
            start = time.perf_counter()
            for _ in range(10):
                model(x)
            elapsed = (time.perf_counter() - start) * 1000
        avg_ms = elapsed / 10
        assert avg_ms < 1000

    def test_full_pipeline_latency(self) -> None:
        classifier = GestureClassifier(num_classes=10, device="cpu")
        seq = np.random.randn(30, 21, 3).astype(np.float32)
        start = time.perf_counter()
        for _ in range(5):
            classifier.classify_gesture(seq)
        elapsed = (time.perf_counter() - start) * 1000
        avg_ms = elapsed / 5
        assert avg_ms < 3000


class TestGradientFlow:
    """Test gradient flow through the network."""

    def test_temporal_cnn_gradients(self) -> None:
        model = TemporalCNN(input_dim=63, hidden_dims=[128, 64])
        x = torch.randn(2, 20, 63)
        out = model(x).mean()
        out.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_lstm_gradients(self) -> None:
        model = GestureLSTM(input_dim=64, hidden_dim=64, num_layers=2, num_classes=5)
        x = torch.randn(2, 15, 64)
        out = model(x).mean()
        out.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradient_flow_all_layers(self) -> None:
        classifier = GestureClassifier(num_classes=5, device="cpu")
        classifier._temporal_cnn.train()
        classifier._lstm.train()

        x = torch.randn(2, 20, 63)
        cnn_out = classifier._temporal_cnn(x)
        logits = classifier._lstm(cnn_out)
        loss = logits.mean()
        loss.backward()

        has_grad = False
        for param in classifier._temporal_cnn.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradient in temporal CNN"

        has_grad = False
        for param in classifier._lstm.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradient in LSTM"
