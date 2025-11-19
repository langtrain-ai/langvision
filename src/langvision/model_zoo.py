"""
Model Zoo for Langvision - Pre-trained Vision Transformer models.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Default model configurations
DEFAULT_MODELS = {
    "vit_tiny_patch16_224": {
        "name": "vit_tiny_patch16_224",
        "type": "vision_transformer",
        "size": "5.4M",
        "description": "Vision Transformer Tiny (16x16 patches, 224x224 input)",
        "config": {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 192,
            "depth": 12,
            "num_heads": 3,
            "mlp_ratio": 4.0,
            "num_classes": 1000
        }
    },
    "vit_small_patch16_224": {
        "name": "vit_small_patch16_224",
        "type": "vision_transformer",
        "size": "22.1M",
        "description": "Vision Transformer Small (16x16 patches, 224x224 input)",
        "config": {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 384,
            "depth": 12,
            "num_heads": 6,
            "mlp_ratio": 4.0,
            "num_classes": 1000
        }
    },
    "vit_base_patch16_224": {
        "name": "vit_base_patch16_224",
        "type": "vision_transformer",
        "size": "86.4M",
        "description": "Vision Transformer Base (16x16 patches, 224x224 input)",
        "config": {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "num_classes": 1000
        }
    },
    "vit_large_patch16_224": {
        "name": "vit_large_patch16_224",
        "type": "vision_transformer",
        "size": "304.3M",
        "description": "Vision Transformer Large (16x16 patches, 224x224 input)",
        "config": {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "num_classes": 1000
        }
    }
}


def get_available_models() -> List[Dict[str, Any]]:
    """Get list of all available models."""
    return list(DEFAULT_MODELS.values())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(DEFAULT_MODELS.keys())}")
    
    return DEFAULT_MODELS[model_name]


def download_model(model_name: str, output_dir: str = "./models", force: bool = False) -> str:
    """Download a pre-trained model (placeholder implementation)."""
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(DEFAULT_MODELS.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # For now, just save the model configuration
    # In a real implementation, this would download actual model weights
    model_info = DEFAULT_MODELS[model_name]
    output_path = os.path.join(output_dir, f"{model_name}.json")
    
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"Model already exists at {output_path}. Use --force to overwrite.")
    
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return output_path


def list_models_by_type(model_type: str = None) -> List[Dict[str, Any]]:
    """List models filtered by type."""
    models = get_available_models()
    if model_type:
        models = [m for m in models if m.get('type') == model_type]
    return models


def search_models(query: str) -> List[Dict[str, Any]]:
    """Search models by name or description."""
    models = get_available_models()
    query_lower = query.lower()
    
    results = []
    for model in models:
        if (query_lower in model.get('name', '').lower() or 
            query_lower in model.get('description', '').lower()):
            results.append(model)
    
    return results


class ModelZoo:
    """Model Zoo manager for Langvision."""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List available models."""
        return list_models_by_type(model_type)
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        return get_model_info(model_name)
    
    def download(self, model_name: str, force: bool = False) -> str:
        """Download a model."""
        return download_model(model_name, str(self.cache_dir), force)
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for models."""
        return search_models(query)
    
    def is_downloaded(self, model_name: str) -> bool:
        """Check if model is downloaded."""
        model_path = self.cache_dir / f"{model_name}.json"
        return model_path.exists()
    
    def get_downloaded_models(self) -> List[str]:
        """Get list of downloaded models."""
        downloaded = []
        for model_file in self.cache_dir.glob("*.json"):
            model_name = model_file.stem
            if model_name in DEFAULT_MODELS:
                downloaded.append(model_name)
        return downloaded
