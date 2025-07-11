"""
PPO (Proximal Policy Optimization)
---------------------------------
A popular RL algorithm that balances exploration and exploitation by limiting policy updates to stay within a trust region.

Example usage:
    >>> import torch
    >>> from langvision.concepts.ppo import PPO
    >>> class MyPPO(PPO):
    ...     def step(self, policy, old_log_probs, states, actions, rewards, optimizer):
    ...         super().step(policy, old_log_probs, states, actions, rewards, optimizer)
    >>> policy = torch.nn.Linear(2, 2)
    >>> old_log_probs = torch.tensor([0.5, 0.5])
    >>> states = torch.randn(2, 2)
    >>> actions = torch.tensor([0, 1])
    >>> rewards = torch.tensor([1.0, 0.5])
    >>> optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    >>> ppo = MyPPO()
    >>> ppo.step(policy, old_log_probs, states, actions, rewards, optimizer)
"""
from abc import ABC, abstractmethod
from typing import Any
import torch

class PPO(ABC):
    """
    Abstract base class for Proximal Policy Optimization (PPO).
    """
    @abstractmethod
    def step(self, policy: torch.nn.Module, old_log_probs: torch.Tensor, states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        """
        Perform a PPO update step.
        """
        # Forward pass
        logits = policy(states)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]
        # Calculate ratio
        ratio = torch.exp(selected_log_probs - old_log_probs)
        # Calculate surrogate loss
        advantage = rewards - rewards.mean()
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
        loss = -torch.min(surr1, surr2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 