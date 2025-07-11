"""
CoT (Chain-of-Thought)
----------------------
A prompting or training method that encourages models to reason step-by-step, improving performance on complex tasks.

Example usage:
    >>> from langvision.concepts.cot import CoT
    >>> class MyCoT(CoT):
    ...     def generate_chain(self, prompt: str) -> list:
    ...         return super().generate_chain(prompt)
    >>> cot = MyCoT()
    >>> chain = cot.generate_chain('What is 2 + 2?')
    >>> print(chain)
"""
from abc import ABC, abstractmethod
from typing import Any, List

class CoT(ABC):
    """
    Abstract base class for Chain-of-Thought (CoT).
    """
    @abstractmethod
    def generate_chain(self, prompt: str) -> List[Any]:
        """
        Generate a chain of thought for a given prompt.
        """
        # Simple example: split prompt into steps (toy logic)
        steps = [f"Step {i+1}: {word}" for i, word in enumerate(prompt.split())]
        return steps 