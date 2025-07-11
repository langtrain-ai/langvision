"""
RLVR (Reinforcement Learning with Value Ranking)
-----------------------------------------------
A method that uses value-based ranking to guide policy optimization in RL settings.

Example usage:
    >>> from langvision.concepts.rlvr import RLVR
    >>> class MyRLVR(RLVR):
    ...     def rank_and_update(self, values: list) -> None:
    ...         # Implement value ranking and update
    ...         pass
"""
from abc import ABC, abstractmethod
from typing import Any, List

class RLVR(ABC):
    """
    Abstract base class for Reinforcement Learning with Value Ranking (RLVR).
    """
    @abstractmethod
    def rank_and_update(self, values: List[Any]) -> None:
        """
        Rank values and update the policy accordingly.
        """
        pass 