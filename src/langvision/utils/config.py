"""
Configuration utilities for Langvision.
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path


default_config = {
    'model': {
        'name': 'vit_base',
        'img_size': 224,
        'patch_size': 16,
        'in_chans': 3,
        'num_classes': 1000,
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'dropout': 0.1,
        'attention_dropout': 0.1,
    },
    'data': {
        'dataset': 'cifar10',
        'data_dir': './data',
        'batch_size': 64,
        'num_workers': 2,
        'pin_memory': True,
        'persistent_workers': False,
    },
    'training': {
        'epochs': 10,
        'learning_rate': 1e-3,
        'optimizer': 'adam',
        'weight_decay': 0.01,
        'scheduler': 'cosine',
        'warmup_epochs': 0,
        'min_lr': 1e-6,
        'gradient_clip': None,
    },
    'lora': {
        'r': 4,
        'alpha': 1.0,
        'dropout': 0.1,
        'target_modules': ['attention.qkv', 'attention.proj', 'mlp.fc1', 'mlp.fc2'],
    },
    'callbacks': {
        'early_stopping': {
            'enabled': False,
            'patience': 5,
            'min_delta': 0.001,
        },
        'checkpointing': {
            'enabled': True,
            'save_best': True,
            'save_last': True,
        },
    },
    'logging': {
        'level': 'info',
        'log_interval': 100,
        'save_interval': 5,
    },
    'output': {
        'output_dir': './outputs',
        'save_name': 'vit_lora_best.pth',
    },
    'device': {
        'device': 'cuda',
        'cuda_deterministic': False,
        'cuda_benchmark': True,
    },
    'misc': {
        'seed': 42,
        'log_level': 'info',
    },
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML or JSON file."""
    config_path = Path(config_path)
    
    # Create parent directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        elif config_path.suffix.lower() == '.json':
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two configurations, with override_config taking precedence."""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure."""
    required_sections = ['model', 'data', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate model section
    model_config = config['model']
    required_model_keys = ['img_size', 'patch_size', 'num_classes']
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model configuration key: {key}")
    
    # Validate data section
    data_config = config['data']
    required_data_keys = ['dataset', 'batch_size']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data configuration key: {key}")
    
    # Validate training section
    training_config = config['training']
    required_training_keys = ['epochs', 'learning_rate']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training configuration key: {key}")
    
    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get configuration value using dot notation (e.g., 'model.img_size')."""
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value


def set_config_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set configuration value using dot notation (e.g., 'model.img_size')."""
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value 