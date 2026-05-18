"""
Few-shot personalization adapter for Neurolink.

Provides the FewShotAdapter implementing MAML (Model-Agnostic Meta-Learning)
for fast adaptation of base models to individual users with minimal samples.
"""

import copy
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

logger = logging.getLogger(__name__)


@dataclass
class FewShotConfig:
    """Configuration for few-shot personalization."""

    # MAML parameters
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    inner_steps: int = 5
    outer_steps: int = 1000
    meta_batch_size: int = 4

    # Task sampling
    support_size: int = 5  # shots per class
    query_size: int = 15
    num_ways: int = 5  # number of classes per task

    # Adaptation
    adaptation_lr: float = 0.005
    adaptation_steps: int = 10
    adaptation_batch_size: int = 4

    # Optimization
    weight_decay: float = 1e-4
    first_order: bool = True  # use first-order MAML (FOMAML)

    # Data
    val_split: float = 0.2

    # Features
    feature_dim: int = 512
    hidden_dim: int = 256

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MAMLModel(nn.Module):
    """Base model used for MAML meta-learning.

    This is a generic representation learning model that can be adapted
    to new tasks/classes with few gradient steps.
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.feature_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(
        self, x: torch.Tensor, params: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass optionally using custom parameters (for MAML inner loop)."""
        if params is None:
            params = OrderedDict(self.named_parameters())

        def _named_param(name: str) -> torch.Tensor:
            if name in params:
                return params[name]
            raise KeyError(f"Parameter {name} not found")

        h = F.linear(x, _named_param("feature_net.0.weight"), _named_param("feature_net.0.bias"))
        h = F.relu(h)
        h = F.linear(h, _named_param("feature_net.2.weight"), _named_param("feature_net.2.bias"))
        h = F.relu(h)
        h = F.linear(h, _named_param("feature_net.4.weight"), _named_param("feature_net.4.bias"))
        h = F.linear(h, _named_param("classifier.weight"), _named_param("classifier.bias"))
        return h


class MAML(nn.Module):
    """MAML meta-learner. Supports 5-shot, N-way few-shot learning.

    Implements the Model-Agnostic Meta-Learning algorithm (Finn et al. 2017)
    with optional first-order approximation (FOMAML).
    """

    def __init__(self, config: FewShotConfig, base_model: nn.Module):
        super().__init__()
        self.config = config
        self.base_model = base_model
        self.meta_optimizer = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=config.outer_lr,
            weight_decay=config.weight_decay,
        )

        logger.info(
            f"MAML initialized (inner_lr={config.inner_lr}, "
            f"outer_lr={config.outer_lr}, "
            f"inner_steps={config.inner_steps}, "
            f"first_order={config.first_order})"
        )

    def _inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: int,
    ) -> Dict[str, torch.Tensor]:
        """Perform inner loop gradient adaptation for a single task.

        Args:
            support_x: (support_size * num_ways, feature_dim)
            support_y: (support_size * num_ways,)
            num_steps: Number of inner gradient steps.

        Returns:
            Adapted parameters.
        """
        params = OrderedDict(
            {name: param.clone() for name, param in self.base_model.named_parameters()}
        )

        for _ in range(num_steps):
            logits = self._forward_with_params(support_x, params)
            loss = F.cross_entropy(logits, support_y)

            grads = torch.autograd.grad(
                loss,
                params.values(),
                create_graph=not self.config.first_order,
            )

            for (name, param), grad in zip(params.items(), grads):
                if grad is not None:
                    params[name] = param - self.config.inner_lr * grad

        return params

    def _forward_with_params(
        self, x: torch.Tensor, params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass using custom parameter dictionary."""
        if hasattr(self.base_model, 'forward'):
            # Default to the base model's forward signature
            pass
        # Explicit computation for our MAMLModel
        h = F.linear(x, params["feature_net.0.weight"], params["feature_net.0.bias"])
        h = F.relu(h)
        h = F.linear(h, params["feature_net.2.weight"], params["feature_net.2.bias"])
        h = F.relu(h)
        h = F.linear(h, params["feature_net.4.weight"], params["feature_net.4.bias"])
        logits = F.linear(h, params["classifier.weight"], params["classifier.bias"])
        return logits

    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """Perform a single meta-training step on a batch of tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples.

        Returns:
            Average meta-loss across tasks.
        """
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for support_x, support_y, query_x, query_y in tasks:
            # Inner loop adaptation
            adapted_params = self._inner_loop(
                support_x, support_y, self.config.inner_steps
            )

            # Compute meta-loss on query set
            query_logits = self._forward_with_params(query_x, adapted_params)
            task_loss = F.cross_entropy(query_logits, query_y)
            meta_loss = meta_loss + task_loss

        # Average
        meta_loss = meta_loss / len(tasks)

        # Outer loop update
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=5.0)
        self.meta_optimizer.step()

        return meta_loss.item()

    @torch.no_grad()
    def meta_eval(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> float:
        """Evaluate meta-learning performance on a set of tasks.

        Args:
            tasks: List of (support_x, support_y, query_x, query_y) tuples.

        Returns:
            Average query accuracy across tasks.
        """
        accuracies = []
        for support_x, support_y, query_x, query_y in tasks:
            adapted_params = self._inner_loop(
                support_x, support_y, self.config.inner_steps
            )
            query_logits = self._forward_with_params(query_x, adapted_params)
            preds = query_logits.argmax(dim=-1)
            acc = (preds == query_y).float().mean().item()
            accuracies.append(acc)

        return float(np.mean(accuracies))


class FewShotAdapter:
    """Few-shot personalization adapter using MAML.

    Provides high-level API for:
      - Meta-training on diverse user data
      - Fast adaptation to a new user with few samples
      - Online/continual adaptation loop
    """

    def __init__(self, config: Optional[FewShotConfig] = None):
        self.config = config or FewShotConfig()
        self.device = torch.device(self.config.device)
        self.maml: Optional[MAML] = None
        logger.info(
            f"FewShotAdapter initialized "
            f"(support={self.config.support_size}-shot, "
            f"{self.config.num_ways}-way)"
        )

    def build_model(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
    ) -> MAMLModel:
        """Build and return a new MAML-compatible base model."""
        model = MAMLModel(feature_dim, hidden_dim, num_classes)
        return model.to(self.device)

    def setup_meta_learning(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        """Initialize the MAML meta-learner with a new base model."""
        base_model = self.build_model(feature_dim, hidden_dim, num_classes)
        self.maml = MAML(self.config, base_model)
        logger.info("MAML meta-learner set up")

    def create_task(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a support/query split task from features and labels.

        Args:
            features: (N, feature_dim)
            labels: (N,)

        Returns:
            (support_x, support_y, query_x, query_y)
        """
        unique_classes = labels.unique()
        if len(unique_classes) < self.config.num_ways:
            raise ValueError(
                f"Need at least {self.config.num_ways} classes, got {len(unique_classes)}"
            )

        selected_classes = unique_classes[torch.randperm(len(unique_classes))][
            : self.config.num_ways
        ]

        support_x: List[torch.Tensor] = []
        support_y: List[torch.Tensor] = []
        query_x: List[torch.Tensor] = []
        query_y: List[torch.Tensor] = []

        for i, cls in enumerate(selected_classes):
            cls_mask = labels == cls
            cls_features = features[cls_mask]

            indices = torch.randperm(len(cls_features))
            n_support = min(self.config.support_size, len(cls_features) // 2)
            n_query = min(self.config.query_size, len(cls_features) - n_support)

            support_idx = indices[:n_support]
            query_idx = indices[n_support : n_support + n_query]

            support_x.append(cls_features[support_idx])
            support_y.append(torch.full((n_support,), i, dtype=torch.long))
            query_x.append(cls_features[query_idx])
            query_y.append(torch.full((n_query,), i, dtype=torch.long))

        return (
            torch.cat(support_x, dim=0).to(self.device),
            torch.cat(support_y, dim=0).to(self.device),
            torch.cat(query_x, dim=0).to(self.device),
            torch.cat(query_y, dim=0).to(self.device),
        )

    def meta_train(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_iterations: int = 1000,
        verbose: bool = True,
    ) -> List[float]:
        """Run meta-training on the provided dataset.

        Args:
            features: (N, feature_dim) tensor of all features.
            labels: (N,) tensor of class labels.
            num_iterations: Number of meta-training iterations.
            verbose: Whether to log progress.

        Returns:
            List of meta-loss values per iteration.
        """
        if self.maml is None:
            raise RuntimeError(
                "MAML not initialized. Call setup_meta_learning() first."
            )

        losses = []
        for step in range(num_iterations):
            tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
            for _ in range(self.config.meta_batch_size):
                task = self.create_task(features, labels)
                tasks.append(task)

            loss = self.maml.meta_train_step(tasks)
            losses.append(loss)

            if verbose and (step + 1) % 100 == 0:
                logger.info(
                    f"Meta-train step {step + 1}/{num_iterations}: loss={loss:.4f}"
                )

        logger.info(f"Meta-training completed: final loss={losses[-1]:.4f}")
        return losses

    def adapt_to_user(
        self,
        base_model: nn.Module,
        user_samples: torch.Tensor,
        user_labels: torch.Tensor,
        num_steps: Optional[int] = None,
        lr: Optional[float] = None,
    ) -> nn.Module:
        """Adapt a base model to a specific user using few samples.

        Performs online gradient descent on the user's samples to
        produce a personalized model.

        Args:
            base_model: The pre-trained base model.
            user_samples: (N, feature_dim) user's data.
            user_labels: (N,) user's labels.
            num_steps: Number of adaptation steps (default: config value).
            lr: Learning rate for adaptation (default: config value).

        Returns:
            Personalized model (new instance with adapted parameters).
        """
        num_steps = num_steps or self.config.adaptation_steps
        lr = lr or self.config.adaptation_lr

        # Clone the model for personalization
        personal_model = copy.deepcopy(base_model).to(self.device)
        personal_model.train()

        optimizer = torch.optim.SGD(
            personal_model.parameters(), lr=lr
        )

        dataset = TensorDataset(user_samples.to(self.device), user_labels.to(self.device))
        loader = DataLoader(dataset, batch_size=self.config.adaptation_batch_size, shuffle=True)

        for step in range(num_steps):
            epoch_loss = 0.0
            batches = 0
            for x, y in loader:
                optimizer.zero_grad()
                logits = personal_model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(personal_model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
                batches += 1

            if (step + 1) % 5 == 0:
                logger.debug(
                    f"Adaptation step {step + 1}/{num_steps}: loss={epoch_loss / batches:.4f}"
                )

        logger.info(
            f"Model adapted to user with {len(user_samples)} samples "
            f"({num_steps} steps, lr={lr})"
        )

        return personal_model

    def online_adaptation_loop(
        self,
        model: nn.Module,
        get_user_samples: Callable[[], Tuple[torch.Tensor, torch.Tensor]],
        max_steps: int = 50,
        convergence_threshold: float = 0.01,
    ) -> nn.Module:
        """Online adaptation loop that continuously adapts to a user.

        Args:
            model: The base model to adapt.
            get_user_samples: Callable that returns (features, labels) tuples
                             of new user data as it becomes available.
            max_steps: Maximum number of adaptation steps.
            convergence_threshold: Stop when loss change is below this.

        Returns:
            The adapted model.
        """
        adapted_model = copy.deepcopy(model).to(self.device)
        adapted_model.train()

        optimizer = torch.optim.SGD(
            adapted_model.parameters(), lr=self.config.adaptation_lr
        )

        prev_loss = float("inf")
        for step in range(max_steps):
            features, labels = get_user_samples()
            if len(features) == 0:
                logger.info("No new user samples, stopping adaptation")
                break

            features = features.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits = adapted_model(features)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            loss_val = loss.item()
            loss_change = abs(prev_loss - loss_val)

            logger.debug(
                f"Online adapt step {step + 1}: loss={loss_val:.4f}, "
                f"change={loss_change:.4f}"
            )

            if loss_change < convergence_threshold:
                logger.info(f"Converged at step {step + 1}")
                break

            prev_loss = loss_val

        logger.info(f"Online adaptation completed after {step + 1} steps")
        return adapted_model

    def save(self, path: Union[str, Path]):
        """Save the adapter state including MAML meta-learner."""
        if self.maml is None:
            raise RuntimeError("No MAML meta-learner to save")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "maml_state": self.maml.state_dict(),
                "config": self.config,
            },
            path / "few_shot_adapter.pt",
        )
        logger.info(f"FewShotAdapter saved to {path}")

    def load(self, path: Union[str, Path]):
        """Load the adapter state."""
        path = Path(path) / "few_shot_adapter.pt"
        if not path.exists():
            raise FileNotFoundError(f"Adapter checkpoint not found: {path}")

        state = torch.load(path, map_location=self.device)
        self.config = state.get("config", self.config)

        base_model = MAMLModel(
            self.config.feature_dim,
            self.config.hidden_dim,
            self.config.num_ways,
        )
        self.maml = MAML(self.config, base_model)
        self.maml.load_state_dict(state["maml_state"])
        logger.info(f"FewShotAdapter loaded from {path}")


import numpy as np
