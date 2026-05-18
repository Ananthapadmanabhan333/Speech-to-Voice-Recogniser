from __future__ import annotations

import random
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class RLError(Exception):
    """Raised when RL adaptation fails."""


@dataclass
class Experience:
    """A single experience tuple for RL."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelUpdate:
    """Result of RL model update."""

    policy_loss: float
    value_loss: float
    entropy: float
    learning_rate: float
    update_time: float


class AdaptationLearner:
    """RL-based adaptation learner for gesture recognition and phrase prediction.

    Uses policy gradient (REINFORCE) with reward modeling to adapt system
    behavior based on user feedback. Supports online learning with
    exploration vs exploitation strategy.

    Features:
    - Reward modeling for gesture recognition accuracy
    - Policy gradient for phrase prediction
    - Online learning loop with experience replay
    - Exploration vs exploitation (epsilon-greedy)
    - Feedback-driven adaptation
    """

    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 20,
        hidden_dim: int = 128,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        min_epsilon: float = 0.01,
        memory_size: int = 10000,
        batch_size: int = 32,
        update_interval: int = 10,
    ):
        """Initialize adaptation learner.

        Args:
            state_dim: State representation dimension.
            action_dim: Action space dimension.
            hidden_dim: Hidden layer dimension.
            learning_rate: Policy gradient learning rate.
            gamma: Discount factor.
            epsilon: Initial exploration rate.
            epsilon_decay: Epsilon decay per update.
            min_epsilon: Minimum exploration rate.
            memory_size: Experience replay buffer size.
            batch_size: Training batch size.
            update_interval: Steps between model updates.
        """
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._hidden_dim = hidden_dim
        self._lr = learning_rate
        self._gamma = gamma
        self._epsilon = epsilon
        self._epsilon_decay = epsilon_decay
        self._min_epsilon = min_epsilon
        self._batch_size = batch_size
        self._update_interval = update_interval

        # Experience replay buffer
        self._memory: deque = deque(maxlen=memory_size)

        # Policy network parameters (simple linear policy)
        self._policy_weights = np.random.randn(state_dim, action_dim).astype(np.float32) * 0.01
        self._value_weights = np.random.randn(state_dim, 1).astype(np.float32) * 0.01

        # Optimizer state (Adam-like)
        self._policy_m = np.zeros_like(self._policy_weights)
        self._policy_v = np.zeros_like(self._policy_weights)
        self._value_m = np.zeros_like(self._value_weights)
        self._value_v = np.zeros_like(self._value_weights)
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._step = 0

        # Training state
        self._total_steps = 0
        self._episode_rewards: List[float] = []

        # Model performance tracking
        self._performance: Dict[str, float] = {
            "avg_reward": 0.0,
            "policy_loss": 0.0,
            "accuracy": 0.0,
        }

        logger.info(
            "adaptation_learner_initialized",
            state_dim=state_dim,
            action_dim=action_dim,
            epsilon=epsilon,
            gamma=gamma,
        )

    async def learn_from_feedback(
        self,
        user_feedback: Dict[str, Any],
        context: Dict[str, Any],
    ) -> ModelUpdate:
        """Learn from user feedback using policy gradient.

        Converts feedback into a reward signal and updates the policy.

        Args:
            user_feedback: Dict with feedback data.
                Keys: 'intent_accuracy', 'gesture_accuracy', 'phrase_quality',
                      'satisfaction', 'correction'
            context: Dict with contextual information.

        Returns:
            ModelUpdate with training metrics.
        """
        try:
            # Convert feedback to reward
            reward = self._compute_reward(user_feedback)

            # Create state from context
            state = self._encode_state(context)

            # Sample action (system adaptation decision)
            action = self._select_action(state)

            # Create experience
            next_state = self._encode_state({**context, **user_feedback})
            done = user_feedback.get("session_end", False)

            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
            )

            # Store experience
            self._memory.append(experience)
            self._total_steps += 1

            # Update model periodically
            if self._total_steps % self._update_interval == 0 and len(self._memory) >= self._batch_size:
                update = self._update_model()
                self._decay_epsilon()
                return update

            return ModelUpdate(
                policy_loss=0.0,
                value_loss=0.0,
                entropy=0.0,
                learning_rate=self._lr,
                update_time=0.0,
            )

        except Exception as e:
            logger.error("rl_learning_failed", error=str(e))
            raise RLError(f"RL learning failed: {e}") from e

    def get_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> Tuple[int, float]:
        """Get action from policy.

        Args:
            state: State vector.
            deterministic: If True, always take best action.

        Returns:
            (action_index, action_probability).
        """
        logits = state @ self._policy_weights
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / (np.sum(exp_logits) + 1e-8)

        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int(np.random.choice(self._action_dim, p=probs))

        return action, float(probs[action])

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics.

        Returns:
            Dict with avg_reward, policy_loss, accuracy.
        """
        return dict(self._performance)

    def reset(self) -> None:
        """Reset learner state."""
        self._memory.clear()
        self._episode_rewards.clear()
        self._total_steps = 0
        self._policy_weights = np.random.randn(self._state_dim, self._action_dim).astype(np.float32) * 0.01
        self._value_weights = np.random.randn(self._state_dim, 1).astype(np.float32) * 0.01
        logger.info("adaptation_learner_reset")

    def _compute_reward(self, feedback: Dict[str, Any]) -> float:
        """Compute scalar reward from user feedback.

        Args:
            feedback: User feedback dict.

        Returns:
            Reward value.
        """
        reward = 0.0

        # Intent accuracy (positive reward)
        if "intent_accuracy" in feedback:
            reward += feedback["intent_accuracy"] * 1.0

        # Gesture accuracy
        if "gesture_accuracy" in feedback:
            reward += feedback["gesture_accuracy"] * 1.0

        # Phrase prediction quality
        if "phrase_quality" in feedback:
            reward += feedback["phrase_quality"] * 0.8

        # User satisfaction
        if "satisfaction" in feedback:
            reward += feedback["satisfaction"] * 1.5

        # Correction (negative reward for mistakes)
        if "correction" in feedback and feedback["correction"] is True:
            reward -= 0.5

        # Explicit positive/negative feedback
        if "rating" in feedback:
            reward += (feedback["rating"] - 0.5) * 2.0  # Normalize to [-1, 1]

        return float(np.clip(reward, -2.0, 2.0))

    def _encode_state(self, context: Dict[str, Any]) -> np.ndarray:
        """Encode context dict into a fixed-size state vector.

        Args:
            context: Context dictionary.

        Returns:
            State vector of shape (state_dim,).
        """
        state = np.zeros(self._state_dim, dtype=np.float32)

        # Encode available fields
        for i, (key, value) in enumerate(context.items()):
            if i >= self._state_dim:
                break
            if isinstance(value, (int, float)):
                state[i] = float(value)
            elif isinstance(value, str):
                # Simple string hash encoding
                state[i % self._state_dim] = hash(value) % 1000 / 1000.0
            elif isinstance(value, dict):
                # Recursively flatten small dicts
                for j, (k2, v2) in enumerate(value.items()):
                    if i + j < self._state_dim and isinstance(v2, (int, float)):
                        state[i + j] = float(v2)

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 1e-6:
            state = state / norm

        return state

    def _select_action(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state.

        Returns:
            Selected action index.
        """
        if random.random() < self._epsilon:
            return random.randint(0, self._action_dim - 1)
        else:
            action, _ = self.get_action(state, deterministic=True)
            return action

    def _update_model(self) -> ModelUpdate:
        """Update policy and value networks using experience replay.

        Uses REINFORCE with baseline (actor-critic style).

        Returns:
            ModelUpdate with training metrics.
        """
        start_time = time.time()

        # Sample batch
        batch = random.sample(list(self._memory), self._batch_size)

        # Prepare batch data
        states = np.array([exp.state for exp in batch])
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = np.array([exp.next_state for exp in batch])
        dones = np.array([exp.done for exp in batch])

        # Compute returns (discounted)
        returns = np.zeros_like(rewards)
        running_return = 0.0
        for t in reversed(range(len(batch))):
            running_return = rewards[t] + self._gamma * running_return * (1 - dones[t])
            returns[t] = running_return

        # Compute baseline (value function)
        values = states @ self._value_weights
        advantages = returns - values.flatten()

        # Policy gradient (REINFORCE)
        logits = states @ self._policy_weights
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / (np.sum(exp_logits, axis=1, keepdims=True) + 1e-8)

        # Compute policy loss
        action_probs = probs[np.arange(len(batch)), actions]
        policy_loss = -np.mean(advantages * np.log(action_probs + 1e-8))

        # Compute value loss
        value_loss = np.mean((returns - values.flatten()) ** 2)

        # Entropy bonus (for exploration)
        entropy = -np.mean(np.sum(probs * np.log(probs + 1e-8), axis=1))

        # Combined loss
        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

        # Update policy weights (Adam optimizer)
        self._step += 1
        lr_t = self._lr * np.sqrt(1 - self._beta2 ** self._step) / (1 - self._beta1 ** self._step)

        # Policy gradient
        policy_grad = states.T @ (advantages[:, np.newaxis] * (action_probs[:, np.newaxis] < 0).astype(float))
        policy_grad = policy_grad / self._batch_size

        # Rewards to gradients: d(log pi) = (1[a=a_i] - pi) * advantages
        one_hot = np.zeros((len(batch), self._action_dim))
        one_hot[np.arange(len(batch)), actions] = 1
        policy_grad = (states.T @ ((one_hot - probs) * advantages[:, np.newaxis])) / self._batch_size

        # Adam update
        self._policy_m = self._beta1 * self._policy_m + (1 - self._beta1) * policy_grad
        self._policy_v = self._beta2 * self._policy_v + (1 - self._beta2) * (policy_grad ** 2)
        m_hat = self._policy_m / (1 - self._beta1 ** self._step)
        v_hat = self._policy_v / (1 - self._beta2 ** self._step)
        self._policy_weights -= lr_t * m_hat / (np.sqrt(v_hat) + 1e-8)

        # Value gradient
        value_grad = (states.T @ (returns - values.flatten())) / self._batch_size

        self._value_m = self._beta1 * self._value_m + (1 - self._beta1) * value_grad
        self._value_v = self._beta2 * self._value_v + (1 - self._beta2) * (value_grad ** 2)
        m_hat_v = self._value_m / (1 - self._beta1 ** self._step)
        v_hat_v = self._value_v / (1 - self._beta2 ** self._step)
        self._value_weights -= lr_t * m_hat_v / (np.sqrt(v_hat_v) + 1e-8)

        # Update metrics
        avg_reward = float(np.mean(rewards))
        self._performance["avg_reward"] = 0.9 * self._performance.get("avg_reward", 0.0) + 0.1 * avg_reward
        self._performance["policy_loss"] = float(policy_loss)
        self._performance["accuracy"] = float(np.mean(advantages > 0))

        update_time = time.time() - start_time

        logger.debug(
            "model_updated",
            policy_loss=float(policy_loss),
            value_loss=float(value_loss),
            entropy=float(entropy),
            avg_reward=avg_reward,
            epsilon=self._epsilon,
        )

        return ModelUpdate(
            policy_loss=float(policy_loss),
            value_loss=float(value_loss),
            entropy=float(entropy),
            learning_rate=lr_t,
            update_time=update_time,
        )

    def _decay_epsilon(self) -> None:
        """Decay exploration rate."""
        self._epsilon = max(self._min_epsilon, self._epsilon * self._epsilon_decay)
