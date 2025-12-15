"""
Langvision API Module

This module provides the client interface to the Langvision server.
All heavy operations (training, inference, evaluation) run on the server.
"""

from .client import (
    # Client
    LangvisionClient,
    ServerConfig,
    JobResult,
    JobStatus,
    get_client,
    
    # Convenience functions
    train,
    generate,
    
    # Exceptions
    LangvisionAPIError,
    AuthenticationError,
    RateLimitError,
    ServerError,
)

from .cloud import (
    CloudTrainer,
    CloudInference,
    CloudTrainingConfig,
    cloud_train,
    cloud_generate,
)

__all__ = [
    # Client
    "LangvisionClient",
    "ServerConfig",
    "JobResult",
    "JobStatus",
    "get_client",
    
    # Cloud
    "CloudTrainer",
    "CloudInference",
    "CloudTrainingConfig",
    "cloud_train",
    "cloud_generate",
    
    # Convenience
    "train",
    "generate",
    
    # Exceptions
    "LangvisionAPIError",
    "AuthenticationError",
    "RateLimitError", 
    "ServerError",
]

