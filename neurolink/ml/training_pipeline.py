"""
Main training orchestrator for Neurolink.

Provides the TrainingPipeline class that manages the training of all ML models,
configuration management, distributed training (DDP), hyperparameter
optimization (Optuna), experiment tracking (MLflow), and model registry.
"""

import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the training pipeline."""

    # General
    experiment_name: str = "neurolink_training"
    output_dir: Path = Path("outputs")
    seed: int = 42
    log_level: str = "INFO"
    debug: bool = False

    # Distributed training
    distributed: bool = False
    num_nodes: int = 1
    gpus_per_node: int = 1
    master_addr: str = "localhost"
    master_port: str = "29500"
    backend: str = "nccl"

    # Hyperparameter optimization
    hpo_enabled: bool = False
    hpo_trials: int = 50
    hpo_direction: str = "maximize"
    hpo_storage: Optional[str] = None

    # Experiment tracking
    tracking_enabled: bool = True
    mlflow_tracking_uri: Optional[str] = None
    mlflow_experiment_name: Optional[str] = None

    # Model registry
    registry_path: Path = Path("model_registry")
    register_best: bool = True
    max_registered: int = 10

    # Checkpointing
    checkpoint_interval: int = 5
    keep_last_n_checkpoints: int = 3
    resume_from: Optional[str] = None

    # Data
    data_dir: Path = Path("data")

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelRegistration:
    """Metadata for a registered model."""

    name: str
    version: str
    model_type: str
    metrics: Dict[str, float]
    path: Path
    timestamp: str
    experiment_id: str
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModelRegistry:
    """Simple model registry for tracking trained model versions."""

    def __init__(self, registry_path: Path, max_models: int = 10):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.max_models = max_models
        self._registrations: Dict[str, List[ModelRegistration]] = {}
        self._load()

    def _load(self):
        reg_file = self.registry_path / "registry.json"
        if reg_file.exists():
            with open(reg_file, "r") as f:
                data = json.load(f)
            for name, regs in data.items():
                self._registrations[name] = [
                    ModelRegistration(**r) for r in regs
                ]

    def _save(self):
        data = {
            name: [r.to_dict() for r in regs]
            for name, regs in self._registrations.items()
        }
        with open(self.registry_path / "registry.json", "w") as f:
            json.dump(data, f, indent=2)

    def register(
        self,
        name: str,
        model: nn.Module,
        metrics: Dict[str, float],
        tags: Optional[Dict[str, str]] = None,
    ) -> ModelRegistration:
        """Register a trained model."""
        version = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
        model_path = self.registry_path / name / version
        model_path.mkdir(parents=True, exist_ok=True)

        registration = ModelRegistration(
            name=name,
            version=version,
            model_type=type(model).__name__,
            metrics=metrics,
            path=model_path,
            timestamp=datetime.now().isoformat(),
            experiment_id=os.environ.get("MLFLOW_RUN_ID", ""),
            tags=tags or {},
        )

        # Save model
        model_save_path = model_path / "model.pt"
        torch.save(model.state_dict(), model_save_path)

        # Save metadata
        with open(model_path / "metadata.json", "w") as f:
            json.dump(registration.to_dict(), f, indent=2)

        if name not in self._registrations:
            self._registrations[name] = []
        self._registrations[name].append(registration)

        # Enforce max
        if len(self._registrations[name]) > self.max_models:
            removed = self._registrations[name].pop(0)
            import shutil
            shutil.rmtree(removed.path, ignore_errors=True)

        self._save()
        logger.info(f"Registered model '{name}' v{version} with metrics: {metrics}")
        return registration

    def get_best(self, name: str, metric: str = "val_accuracy") -> Optional[ModelRegistration]:
        """Get the best registered model by metric."""
        if name not in self._registrations:
            return None
        sorted_regs = sorted(
            self._registrations[name],
            key=lambda r: r.metrics.get(metric, 0),
            reverse=True,
        )
        return sorted_regs[0] if sorted_regs else None

    def list_models(self, name: Optional[str] = None) -> List[ModelRegistration]:
        """List registered models, optionally filtered by name."""
        if name:
            return self._registrations.get(name, [])
        return [r for regs in self._registrations.values() for r in regs]


class ExperimentTracker:
    """Experiment tracking via MLflow."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._mlflow_available = False

        if config.tracking_enabled:
            try:
                import mlflow
                self._mlflow = mlflow
                self._mlflow_available = True

                tracking_uri = config.mlflow_tracking_uri or (
                    config.output_dir / "mlruns"
                ).as_uri()
                mlflow.set_tracking_uri(tracking_uri)

                experiment_name = config.mlflow_experiment_name or config.experiment_name
                mlflow.set_experiment(experiment_name)
                logger.info(f"MLflow tracking enabled: {tracking_uri} / {experiment_name}")
            except ImportError:
                logger.warning("MLflow not installed, tracking disabled")
                self._mlflow_available = False

    def start_run(self, run_name: Optional[str] = None):
        if self._mlflow_available:
            self._mlflow.start_run(
                run_name=run_name or f"run_{uuid.uuid4().hex[:8]}"
            )

    def end_run(self):
        if self._mlflow_available:
            self._mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        if self._mlflow_available:
            self._mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        if self._mlflow_available:
            self._mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, path: Union[str, Path]):
        if self._mlflow_available:
            self._mlflow.log_artifact(str(path))

    def log_model(self, model: nn.Module, artifact_path: str):
        if self._mlflow_available:
            self._mlflow.pytorch.log_model(model, artifact_path)


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._optuna_available = False

        if config.hpo_enabled:
            try:
                import optuna
                self._optuna = optuna
                self._optuna_available = True
                logger.info(
                    f"Optuna HPO enabled: {config.hpo_trials} trials"
                )
            except ImportError:
                logger.warning("Optuna not installed, HPO disabled")

    def optimize(
        self,
        objective_fn: callable,
        search_space: Dict[str, Any],
        n_trials: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization.

        Args:
            objective_fn: Function that takes a trial and returns a metric.
            search_space: Dict mapping param name to optuna distribution.
            n_trials: Number of trials (default: config value).

        Returns:
            Best parameters found.
        """
        if not self._optuna_available:
            raise RuntimeError("Optuna is not available")

        study = self._optuna.create_study(
            direction=self.config.hpo_direction,
            storage=self.config.hpo_storage,
            study_name=f"{self.config.experiment_name}_hpo",
            load_if_exists=True,
        )

        study.optimize(
            objective_fn,
            n_trials=n_trials or self.config.hpo_trials,
            show_progress_bar=True,
        )

        logger.info(
            f"HPO completed: best value={study.best_value:.4f}, "
            f"best params={study.best_params}"
        )
        return study.best_params


class DistributedTrainer:
    """Manages distributed data parallel (DDP) training setup."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._is_initialized = False

    def setup(self, rank: int, world_size: int):
        """Initialize the distributed process group."""
        if not self.config.distributed:
            return

        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = self.config.master_port

        dist.init_process_group(
            backend=self.config.backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )
        self._is_initialized = True
        self._rank = rank
        self._world_size = world_size

        torch.cuda.set_device(rank)
        logger.info(f"Distributed setup: rank={rank}, world_size={world_size}")

    def cleanup(self):
        """Cleanup the distributed process group."""
        if self._is_initialized:
            dist.destroy_process_group()
            self._is_initialized = False

    def wrap_model(self, model: nn.Module, rank: int) -> nn.Module:
        """Wrap model in DDP."""
        if not self.config.distributed:
            return model
        return DDP(model.to(rank), device_ids=[rank])

    @property
    def rank(self) -> int:
        return getattr(self, "_rank", 0)

    @property
    def world_size(self) -> int:
        return getattr(self, "_world_size", 1)

    @property
    def is_master(self) -> bool:
        return self.rank == 0


class TrainingPipeline:
    """Main training orchestrator for all Neurolink models.

    Manages:
      - Configuration via YAML
      - Distributed training (DDP)
      - Hyperparameter optimization (Optuna)
      - Experiment tracking (MLflow)
      - Model registry
      - Metrics dashboard logging
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = ExperimentTracker(self.config)
        self.hpo = HyperparameterOptimizer(self.config)
        self.dist_trainer = DistributedTrainer(self.config)
        self.registry = ModelRegistry(
            self.config.registry_path, self.config.max_registered
        )

        self._setup_logging()
        self._set_seed()

        logger.info(
            f"TrainingPipeline initialized (experiment='{self.config.experiment_name}')"
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper()),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    def _set_seed(self):
        import random
        import numpy as np
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)

    def train_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        model_name: str,
        trainer_fn: callable,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        hpo_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Train a single model with full pipeline support.

        Args:
            model: The PyTorch model to train.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            model_name: Name for registration and tracking.
            trainer_fn: Function that runs training loop.
                       Signature: (model, train_loader, val_loader, **kwargs) -> metrics
            trainer_kwargs: Additional kwargs for trainer_fn.
            hpo_params: Hyperparameters to log (from HPO).

        Returns:
            Training results dict with metrics.
        """
        logger.info(f"Starting training for model '{model_name}'")
        self.tracker.start_run(run_name=model_name)

        try:
            # Log configuration
            params = {
                "model_name": model_name,
                "model_type": type(model).__name__,
                "distributed": self.config.distributed,
            }
            if hpo_params:
                params.update(hpo_params)
            self.tracker.log_params(params)

            # Training
            kwargs = trainer_kwargs or {}
            kwargs.update({
                "model": model,
                "train_loader": train_loader,
                "val_loader": val_loader,
            })

            metrics = trainer_fn(**kwargs)

            # Log metrics
            self.tracker.log_metrics(metrics)

            # Register model if best
            if self.config.register_best:
                self.registry.register(
                    name=model_name,
                    model=model,
                    metrics=metrics,
                    tags={"experiment": self.config.experiment_name},
                )

            logger.info(
                f"Training completed for '{model_name}': {metrics}"
            )

            return metrics

        finally:
            self.tracker.end_run()

    def train_distributed(
        self,
        model_fn: callable,
        dataset_fn: callable,
        trainer_fn: callable,
        model_name: str,
        world_size: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Train a model using distributed data parallel (DDP).

        Args:
            model_fn: Callable that returns a new model instance.
            dataset_fn: Callable that returns (train_dataset, val_dataset).
            trainer_fn: Callable that trains the model.
            model_name: Name for registration.
            world_size: Number of GPUs (default: gpus_per_node).

        Returns:
            Training metrics.
        """
        if not self.config.distributed:
            logger.warning("Distributed training not enabled, running single GPU")
            model = model_fn()
            train_loader, val_loader = dataset_fn()
            return self.train_model(model, train_loader, val_loader, model_name, trainer_fn)

        world_size = world_size or self.config.gpus_per_node

        def _train_worker(rank: int, world_size: int):
            self.dist_trainer.setup(rank, world_size)

            model = model_fn().to(rank)
            model = self.dist_trainer.wrap_model(model, rank)

            train_dataset, val_dataset = dataset_fn()
            train_sampler = DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=32,
                sampler=train_sampler,
                num_workers=2,
            )
            val_loader = DataLoader(
                val_dataset, batch_size=32, shuffle=False, num_workers=2
            )

            metrics = trainer_fn(model, train_loader, val_loader)

            if self.dist_trainer.is_master:
                self.registry.register(model_name, model, metrics)

            self.dist_trainer.cleanup()
            return metrics

        mp.spawn(_train_worker, args=(world_size,), nprocs=world_size, join=True)
        return {"status": "completed_distributed"}

    def optimize_hyperparameters(
        self,
        model_class: Type[nn.Module],
        dataset_fn: callable,
        trainer_fn: callable,
        search_space: Dict[str, Any],
        model_name: str,
        fixed_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run hyperparameter optimization with Optuna.

        Args:
            model_class: The model class to instantiate with different params.
            dataset_fn: Callable that returns (train_loader, val_loader).
            trainer_fn: Callable that trains the model.
            search_space: Optuna search space dict.
            model_name: Name for the model.
            fixed_params: Fixed parameters to pass to model constructor.

        Returns:
            Best parameters found.
        """
        def objective(trial):
            # Sample hyperparameters
            params = fixed_params or {}
            for name, distribution in search_space.items():
                if isinstance(distribution, dict):
                    if distribution["type"] == "int":
                        params[name] = trial.suggest_int(
                            name, distribution["low"], distribution["high"]
                        )
                    elif distribution["type"] == "float":
                        params[name] = trial.suggest_float(
                            name, distribution["low"], distribution["high"],
                            log=distribution.get("log", False),
                        )
                    elif distribution["type"] == "categorical":
                        params[name] = trial.suggest_categorical(
                            name, distribution["choices"]
                        )
                else:
                    params[name] = trial.suggest_uniform(name, *distribution)

            model = model_class(**params)
            train_loader, val_loader = dataset_fn()
            metrics = trainer_fn(model, train_loader, val_loader)

            # Return the metric to optimize
            metric_key = "val_accuracy" if "val_accuracy" in metrics else list(metrics.keys())[0]
            return metrics[metric_key]

        best_params = self.hpo.optimize(objective, search_space)
        return best_params

    def save_config(self, path: Optional[Union[str, Path]] = None):
        """Save pipeline configuration to YAML."""
        path = Path(path or self.config.output_dir / "pipeline_config.yaml")

        try:
            import yaml
            config_dict = asdict(self.config)
            # Convert Path objects to strings
            config_dict = json.loads(json.dumps(config_dict, default=str))
            with open(path, "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            logger.info(f"Pipeline config saved to {path}")
        except ImportError:
            logger.warning("PyYAML not installed, saving as JSON")
            json_path = path.with_suffix(".json")
            config_dict = asdict(self.config)
            with open(json_path, "w") as f:
                json.dump(config_dict, f, indent=2, default=str)

    @classmethod
    def from_config(cls, config_path: Union[str, Path]) -> "TrainingPipeline":
        """Load pipeline from a YAML config file."""
        config_path = Path(config_path)
        try:
            import yaml
            with open(config_path, "r") as f:
                config_dict = yaml.safe_load(f)
        except ImportError:
            with open(config_path, "r") as f:
                config_dict = json.load(f)

        config = PipelineConfig(**config_dict)
        return cls(config)
