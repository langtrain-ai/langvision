"""
SHAP (SHapley Additive exPlanations)
------------------------------------
A unified approach to interpreting model predictions using Shapley values from cooperative game theory.

Example usage:
    >>> from langvision.concepts.shap import SHAP
    >>> import torch
    >>> class MySHAP(SHAP):
    ...     def explain(self, model, input_data):
    ...         return super().explain(model, input_data)
    >>> model = torch.nn.Linear(2, 1)
    >>> shap = MySHAP()
    >>> explanation = shap.explain(model, [[0.5, 1.0], [1.0, 2.0]])
    >>> print(explanation)
"""
from abc import ABC, abstractmethod
from typing import Any, Dict

class SHAP(ABC):
    """
    Abstract base class for SHapley Additive exPlanations (SHAP).
    """
    @abstractmethod
    def explain(self, model: Any, input_data: Any) -> Dict[str, Any]:
        """
        Generate SHAP values for the model's prediction on input_data using shap if available.
        """
        try:
            import shap
        except ImportError:
            raise ImportError("Please install the 'shap' package to use SHAP explanations.")
        import numpy as np
        X = np.array(input_data)
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        return {"shap_values": shap_values.values.tolist()} 