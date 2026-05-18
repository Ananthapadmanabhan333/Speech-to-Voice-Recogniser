"""
Main evaluation orchestrator for Neurolink.

Provides the EvaluationPipeline class that coordinates evaluation of all
ML models, runs cross-validation, performance benchmarks, generates
HTML evaluation reports, supports model comparison, and regression testing.
"""

import gc
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a single evaluation benchmark."""

    model_name: str
    model_type: str
    metrics: Dict[str, float]
    inference_time_ms: float
    fps: float
    model_size_mb: float
    num_parameters: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation pipeline."""

    output_dir: Path = Path("evaluation_reports")
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    cv_folds: int = 5
    cv_stratified: bool = True
    benchmark_warmup: int = 50
    benchmark_runs: int = 200
    batch_size: int = 64
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    regression_threshold: float = 0.01
    track_inference_time: bool = True
    generate_html_report: bool = True


class CrossValidator:
    """Cross-validation runner for model evaluation."""

    def __init__(
        self,
        model_class: Type[nn.Module],
        model_params: Dict[str, Any],
        n_folds: int = 5,
        stratified: bool = True,
        seed: int = 42,
    ):
        self.model_class = model_class
        self.model_params = model_params
        self.n_folds = n_folds
        self.stratified = stratified
        self.seed = seed

    def validate(
        self,
        dataset: Dataset,
        trainer_fn: Callable[..., Dict[str, float]],
    ) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
        """Run k-fold cross-validation.

        Args:
            dataset: Full dataset.
            trainer_fn: Training function that returns metrics dict.

        Returns:
            (fold_results, aggregated_metrics)
        """
        import random
        random.seed(self.seed)

        indices = list(range(len(dataset)))
        labels = []
        for i in range(len(dataset)):
            try:
                _, label = dataset[i] if len(dataset[i]) >= 2 else (None, 0)
                labels.append(label if isinstance(label, (int, np.integer)) else 0)
            except (TypeError, IndexError):
                labels.append(0)

        # Stratified fold assignment
        from collections import defaultdict
        label_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in zip(indices, labels):
            label_indices[lbl].append(idx)

        folds: List[List[int]] = [[] for _ in range(self.n_folds)]
        fold_idx = 0
        for lbl, lbl_idxs in label_indices.items():
            random.shuffle(lbl_idxs)
            for i, idx in enumerate(lbl_idxs):
                folds[fold_idx % self.n_folds].append(idx)
                fold_idx += 1

        fold_results: List[Dict[str, float]] = []
        for fold in range(self.n_folds):
            val_idx = folds[fold]
            train_idx = [i for f in range(self.n_folds) if f != fold for i in folds[f]]

            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)

            train_loader = DataLoader(
                train_subset, batch_size=32, shuffle=True
            )
            val_loader = DataLoader(
                val_subset, batch_size=32, shuffle=False
            )

            model = self.model_class(**self.model_params)
            metrics = trainer_fn(model, train_loader, val_loader)
            metrics["fold"] = fold
            fold_results.append(metrics)

            logger.info(
                f"CV fold {fold + 1}/{self.n_folds}: "
                f"acc={metrics.get('val_accuracy', metrics.get('accuracy', 0)):.4f}"
            )

        # Aggregate
        aggregated: Dict[str, float] = {}
        for key in fold_results[0]:
            if key != "fold":
                values = [r[key] for r in fold_results if key in r]
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))

        logger.info(
            f"Cross-validation results: "
            f"accuracy_mean={aggregated.get('val_accuracy_mean', aggregated.get('accuracy_mean', 0)):.4f}"
        )

        return fold_results, aggregated


class ModelComparator:
    """Compare multiple models on the same evaluation dataset."""

    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.results: List[BenchmarkResult] = []

    def add_result(self, result: BenchmarkResult):
        self.results.append(result)

    def compare(self) -> Dict[str, Any]:
        """Generate comparison report."""
        if not self.results:
            return {"models": [], "summary": {}}

        sorted_results = sorted(
            self.results,
            key=lambda r: r.metrics.get("accuracy", r.metrics.get("f1_score", 0)),
            reverse=True,
        )

        comparison = {
            "models": [
                {
                    "name": r.model_name,
                    "type": r.model_type,
                    "metrics": r.metrics,
                    "inference_time_ms": r.inference_time_ms,
                    "fps": r.fps,
                    "model_size_mb": r.model_size_mb,
                    "num_parameters": r.num_parameters,
                }
                for r in sorted_results
            ],
            "summary": {
                "best_model": sorted_results[0].model_name,
                "best_accuracy": sorted_results[0].metrics.get(
                    "accuracy", sorted_results[0].metrics.get("f1_score", 0)
                ),
                "fastest_model": min(
                    sorted_results, key=lambda r: r.inference_time_ms
                ).model_name,
                "smallest_model": min(
                    sorted_results, key=lambda r: r.model_size_mb
                ).model_name,
            },
        }

        return comparison


class EvaluationPipeline:
    """Main evaluation orchestrator for all models.

    Coordinates:
      - Model loading and evaluation
      - Cross-validation
      - Performance benchmarks (FPS, latency)
      - HTML report generation
      - Model comparison
      - Regression testing
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        self.device = torch.device(self.config.device)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.comparator = ModelComparator(self.device)

        import random as _random
        import numpy as _np
        _random.seed(self.config.seed)
        _np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        logger.info(f"EvaluationPipeline initialized (device={self.device})")

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        model_name: str,
        evaluator_fn: Callable[..., Dict[str, float]],
    ) -> BenchmarkResult:
        """Evaluate a single model and return benchmark results.

        Args:
            model: The trained model.
            dataloader: Evaluation data loader.
            model_name: Name for the model.
            evaluator_fn: Function that computes metrics given model and dataloader.

        Returns:
            BenchmarkResult with all computed metrics.
        """
        model.to(self.device)
        model.eval()

        # Metrics evaluation
        metrics = evaluator_fn(model, dataloader)

        # Benchmark inference
        inference_time, fps = self._benchmark_inference(model, dataloader)

        # Model size
        model_size = self._get_model_size_mb(model)
        num_params = sum(p.numel() for p in model.parameters())

        result = BenchmarkResult(
            model_name=model_name,
            model_type=type(model).__name__,
            metrics=metrics,
            inference_time_ms=inference_time,
            fps=fps,
            model_size_mb=model_size,
            num_parameters=num_params,
        )

        self.comparator.add_result(result)
        logger.info(
            f"Evaluated '{model_name}': accuracy={metrics.get('accuracy', 0):.4f}, "
            f"fps={fps:.1f}, size={model_size:.1f}MB, params={num_params:,}"
        )

        return result

    def _benchmark_inference(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Tuple[float, float]:
        """Benchmark model inference time and FPS."""
        if not self.config.track_inference_time:
            return 0.0, 0.0

        model.eval()
        latencies: List[float] = []

        # Warmup
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.benchmark_warmup:
                    break
                inputs = batch[0].to(self.device)
                _ = model(inputs)

        # Benchmark
        count = 0
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= self.config.benchmark_runs:
                    break
                inputs = batch[0].to(self.device)
                count += inputs.size(0)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(inputs)

                if self.device.type == "cuda":
                    torch.cuda.synchronize()

                latencies.append(time.perf_counter() - start)

        if not latencies:
            return 0.0, 0.0

        avg_latency = float(np.mean(latencies) * 1000.0)
        fps = count / sum(latencies) if sum(latencies) > 0 else 0.0

        return avg_latency, fps

    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get the memory size of model parameters in MB."""
        param_size = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )
        buffer_size = sum(
            b.numel() * b.element_size() for b in model.buffers()
        )
        total_bytes = param_size + buffer_size
        return total_bytes / (1024 * 1024)

    def run_cross_validation(
        self,
        model_class: Type[nn.Module],
        model_params: Dict[str, Any],
        dataset: Dataset,
        trainer_fn: Callable[..., Dict[str, float]],
        model_name: str,
    ) -> Dict[str, Any]:
        """Run cross-validation for a model.

        Args:
            model_class: The model class.
            model_params: Parameters to instantiate the model.
            dataset: Full dataset for cross-validation.
            trainer_fn: Training function.
            model_name: Name for the model.

        Returns:
            Dict with fold_results and aggregated_metrics.
        """
        cv = CrossValidator(
            model_class,
            model_params,
            n_folds=self.config.cv_folds,
            stratified=self.config.cv_stratified,
            seed=self.config.seed,
        )

        fold_results, aggregated = cv.validate(dataset, trainer_fn)

        # Save CV results
        cv_output = self.config.output_dir / "cross_validation"
        cv_output.mkdir(parents=True, exist_ok=True)

        results_data = {
            "model_name": model_name,
            "cv_folds": self.config.cv_folds,
            "fold_results": fold_results,
            "aggregated": aggregated,
        }
        with open(cv_output / f"{model_name}_cv.json", "w") as f:
            json.dump(results_data, f, indent=2, default=str)

        logger.info(
            f"Cross-validation for '{model_name}' completed. "
            f"Aggregated results saved to {cv_output}"
        )

        return results_data

    def run_regression_tests(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        baseline_metrics: Dict[str, float],
        model_name: str,
    ) -> bool:
        """Run regression tests to verify model performance hasn't degraded.

        Args:
            model: The model to test.
            dataloader: Evaluation data loader.
            baseline_metrics: Previous metrics to compare against.
            model_name: Name for the model.

        Returns:
            True if all metrics pass regression thresholds.
        """
        model.to(self.device)
        model.eval()
        threshold = self.config.regression_threshold

        all_preds: List[int] = []
        all_labels: List[int] = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch[0].to(self.device)
                labels = batch[-1].to(self.device)
                logits = model(inputs)
                preds = logits.argmax(dim=-1)
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        current_accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        baseline_acc = baseline_metrics.get("accuracy", 0)

        passed = current_accuracy >= baseline_acc - threshold

        regression_report = {
            "model_name": model_name,
            "baseline_accuracy": baseline_acc,
            "current_accuracy": float(current_accuracy),
            "threshold": threshold,
            "passed": bool(passed),
            "degradation": float(baseline_acc - current_accuracy),
        }

        reg_path = self.config.output_dir / "regression_tests"
        reg_path.mkdir(parents=True, exist_ok=True)
        with open(reg_path / f"{model_name}_regression.json", "w") as f:
            json.dump(regression_report, f, indent=2)

        logger.info(
            f"Regression test for '{model_name}': "
            f"baseline={baseline_acc:.4f}, current={current_accuracy:.4f}, "
            f"threshold={threshold:.4f}, passed={passed}"
        )

        if not passed:
            logger.warning(
                f"REGRESSION FAILED: '{model_name}' degraded by "
                f"{regression_report['degradation']:.4f}"
            )

        return passed

    def generate_report(self, report_name: str = "evaluation_report") -> Path:
        """Generate an HTML evaluation report.

        Includes: model comparison table, per-model metrics,
        cross-validation results, benchmark results.
        """
        from datetime import datetime as _dt

        comparison = self.comparator.compare()
        output_path = self.config.output_dir / f"{report_name}.html"

        html_parts = [
            "<!DOCTYPE html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="UTF-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
            f"<title>Neurolink Evaluation Report - {_dt.now().strftime('%Y-%m-%d %H:%M')}</title>",
            "<style>",
            "* { box-sizing: border-box; margin: 0; padding: 0; }",
            "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; "
            "background: #f5f5f5; color: #333; padding: 20px; }",
            "h1 { font-size: 1.8em; margin-bottom: 10px; color: #1a1a2e; }",
            "h2 { font-size: 1.3em; margin: 20px 0 10px; color: #16213e; }",
            ".summary-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }",
            ".card { background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }",
            ".card h3 { font-size: 0.9em; color: #888; margin-bottom: 5px; }",
            ".card .value { font-size: 1.5em; font-weight: bold; color: #1a1a2e; }",
            "table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin: 15px 0; }",
            "th { background: #1a1a2e; color: white; padding: 12px 15px; text-align: left; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.5px; }",
            "td { padding: 10px 15px; border-bottom: 1px solid #eee; }",
            "tr:hover { background: #f8f9fa; }",
            ".badge { display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; font-weight: 500; }",
            ".badge-best { background: #d4edda; color: #155724; }",
            ".badge-fast { background: #cce5ff; color: #004085; }",
            ".badge-small { background: #fff3cd; color: #856404; }",
            ".metric-cell { font-family: 'SF Mono', 'Consolas', monospace; }",
            "footer { margin-top: 30px; font-size: 0.8em; color: #888; text-align: center; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>Neurolink Evaluation Report</h1>",
            f"<p>Generated: {_dt.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]

        # Summary cards
        if comparison["models"]:
            best = comparison["summary"]
            html_parts.append('<div class="summary-cards">')
            cards = [
                ("Best Model", best.get("best_model", "N/A")),
                ("Best Accuracy", f'{best.get("best_accuracy", 0):.4f}'),
                ("Fastest Model", best.get("fastest_model", "N/A")),
                ("Smallest Model", best.get("smallest_model", "N/A")),
            ]
            for title, value in cards:
                html_parts.append(
                    f'<div class="card"><h3>{title}</h3>'
                    f'<div class="value">{value}</div></div>'
                )
            html_parts.append("</div>")

        # Model comparison table
        html_parts.append("<h2>Model Comparison</h2>")
        html_parts.append("<table>")
        html_parts.append("<tr><th>Model</th><th>Type</th><th>Accuracy</th><th>F1</th><th>Latency (ms)</th><th>FPS</th><th>Size (MB)</th><th>Parameters</th></tr>")

        for model_data in comparison["models"]:
            metrics = model_data["metrics"]
            html_parts.append(
                f"<tr>"
                f"<td><strong>{model_data['name']}</strong></td>"
                f"<td>{model_data['type']}</td>"
                f'<td class="metric-cell">{metrics.get("accuracy", 0):.4f}</td>'
                f'<td class="metric-cell">{metrics.get("f1_score", metrics.get("precision", 0)):.4f}</td>'
                f'<td class="metric-cell">{model_data["inference_time_ms"]:.2f}</td>'
                f'<td class="metric-cell">{model_data["fps"]:.1f}</td>'
                f'<td class="metric-cell">{model_data["model_size_mb"]:.1f}</td>'
                f'<td class="metric-cell">{model_data["num_parameters"]:,}</td>'
                f"</tr>"
            )
        html_parts.append("</table>")

        # Detailed metrics
        html_parts.append("<h2>Detailed Model Metrics</h2>")
        for model_data in comparison["models"]:
            html_parts.append(f"<h3>{model_data['name']}</h3>")
            html_parts.append("<table>")
            html_parts.append("<tr><th>Metric</th><th>Value</th></tr>")
            for key, value in model_data["metrics"].items():
                html_parts.append(
                    f"<tr><td>{key}</td><td>{value:.4f}</td></tr>"
                )
            html_parts.append("</table>")

        html_parts.extend([
            "<footer>",
            f"<p>Neurolink Evaluation Pipeline &mdash; Report generated automatically</p>",
            "</footer>",
            "</body>",
            "</html>",
        ])

        html_content = "\n".join(html_parts)
        with open(output_path, "w") as f:
            f.write(html_content)

        logger.info(f"HTML evaluation report generated: {output_path}")
        return output_path

    def save_results(self, name: str = "evaluation_results"):
        """Save all evaluation results to JSON."""
        comparison = self.comparator.compare()
        output_path = self.config.output_dir / f"{name}.json"
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        logger.info(f"Evaluation results saved to {output_path}")
        return output_path

    def evaluate_all(
        self,
        models: Dict[str, nn.Module],
        dataloaders: Dict[str, DataLoader],
        evaluator_fn: Callable[..., Dict[str, float]],
        run_cv: bool = False,
        cv_dataset: Optional[Dataset] = None,
    ) -> Dict[str, Any]:
        """Evaluate all models and generate complete report.

        Args:
            models: Dict mapping model_name -> model instance.
            dataloaders: Dict mapping model_name -> dataloader.
            evaluator_fn: Function that computes metrics.
            run_cv: Whether to run cross-validation.
            cv_dataset: Dataset for cross-validation (required if run_cv=True).

        Returns:
            Dict of all evaluation results organized by model.
        """
        all_results: Dict[str, Any] = {}

        for model_name, model in models.items():
            if model_name not in dataloaders:
                logger.warning(f"No dataloader for '{model_name}', skipping")
                continue

            result = self.evaluate_model(
                model, dataloaders[model_name], model_name, evaluator_fn
            )
            all_results[model_name] = {
                "metrics": result.metrics,
                "inference_time_ms": result.inference_time_ms,
                "fps": result.fps,
                "model_size_mb": result.model_size_mb,
                "num_parameters": result.num_parameters,
            }

        # Generate report
        if self.config.generate_html_report:
            report_path = self.generate_report()
            all_results["report_path"] = str(report_path)

        # Save results
        self.save_results()

        return all_results
