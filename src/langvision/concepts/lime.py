"""
LIME (Local Interpretable Model-agnostic Explanations)
-----------------------------------------------------
A technique for explaining model predictions by approximating them locally with interpretable models.

Example usage:
    >>> from langvision.concepts.lime import LIME
    >>> import torch
    >>> class MyLIME(LIME):
    ...     def explain(self, model, input_data):
    ...         return super().explain(model, input_data)
    >>> model = torch.nn.Linear(2, 1)
    >>> lime = MyLIME()
    >>> explanation = lime.explain(model, [[0.5, 1.0], [1.0, 2.0]])
    >>> print(explanation)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class LIME(ABC):
    """
    Abstract base class for Local Interpretable Model-agnostic Explanations (LIME).
    """
    @abstractmethod
    def explain(self, model: Any, input_data: Any) -> Dict[str, Any]:
        """
        Generate a local explanation for the model's prediction on input_data using lime if available.
        """
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("Please install the 'lime' package to use LIME explanations.")
        import numpy as np
        X = np.array(input_data)
        explainer = LimeTabularExplainer(X, mode="regression")
        explanation = explainer.explain_instance(X[0], lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy())
        return {"explanation": explanation.as_list()} 