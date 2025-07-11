"""
CCoT (Contrastive Chain-of-Thought)
----------------------------------
Extends CoT by using contrastive learning to distinguish between correct and incorrect reasoning chains.

Example usage:
    >>> from langvision.concepts.ccot import CCoT
    >>> class MyCCoT(CCoT):
    ...     def contrastive_train(self, positive_chains, negative_chains):
    ...         return super().contrastive_train(positive_chains, negative_chains)
    >>> ccot = MyCCoT()
    >>> pos = [['Step 1: Think', 'Step 2: Solve']]
    >>> neg = [['Step 1: Guess', 'Step 2: Wrong']]
    >>> ccot.contrastive_train(pos, neg)
"""
from abc import ABC, abstractmethod
from typing import Any, List

class CCoT(ABC):
    """
    Abstract base class for Contrastive Chain-of-Thought (CCoT).
    """
    @abstractmethod
    def contrastive_train(self, positive_chains: List[Any], negative_chains: List[Any]) -> None:
        """
        Train using positive and negative reasoning chains.
        """
        # Simple example: print contrastive pairs (toy logic)
        for pos, neg in zip(positive_chains, negative_chains):
            print(f"Positive: {pos} | Negative: {neg}") 