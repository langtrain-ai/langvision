"""
GRPO (Generalized Reinforcement Policy Optimization)
---------------------------------------------------
A family of algorithms for optimizing policies in reinforcement learning, generalizing methods like PPO and DPO.

Example usage:
    >>> from langvision.concepts.grpo import GRPO
    >>> class MyGRPO(GRPO):
    ...     def optimize(self, policy: Any, rewards: list) -> None:
    ...         # Implement GRPO optimization
    ...         pass
"""
from abc import ABC, abstractmethod
from typing import Any, List

class GRPO(ABC):
    """
    Abstract base class for Generalized Reinforcement Policy Optimization (GRPO).
    """
    @abstractmethod
    def optimize(self, policy: Any, rewards: List[Any]) -> None:
        """
        Optimize the policy using provided rewards.
        """
        pass 