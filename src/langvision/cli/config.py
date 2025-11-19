#!/usr/bin/env python3
"""
Configuration CLI for Langvision.
"""

import argparse
import json
import logging
import os
import sys
import yaml
from pathlib import Path

# Import from langvision modules
try:
    from langvision.utils.config import default_config, load_config, save_config
except ImportError as e:
    print(f"❌ Error importing langvision modules: {e}")
    print("Please ensure langvision is properly installed:")
    print("  pip install langvision")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments for configuration operations."""
    parser = argparse.ArgumentParser(
        description='Langvision Configuration Manager - Create, validate, and manage config files',
        epilog='''\nExamples:\n  langvision config create --output config.yaml\n  langvision config validate config.yaml\n  langvision config show --format json\n''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Configuration commands')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new configuration file')
    create_parser.add_argument('--output', type=str, default='config.yaml', help='Output configuration file')
    create_parser.add_argument('--template', type=str, choices=['basic', 'advanced', 'custom'], default='basic', help='Configuration template')
    create_parser.add_argument('--dataset', type=str, default='cifar10', help='Default dataset')
    create_parser.add_argument('--model', type=str, default='vit_base', help='Default model type')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration file')
    validate_parser.add_argument('config_file', type=str, help='Configuration file to validate')
    validate_parser.add_argument('--strict', action='store_true', help='Use strict validation')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show default configuration')
    show_parser.add_argument('--format', type=str, default='yaml', choices=['yaml', 'json'], help='Output format')
    show_parser.add_argument('--section', type=str, help='Show specific configuration section')
    
    # Convert command
    convert_parser = subparsers.add_parser('convert', help='Convert between configuration formats')
    convert_parser.add_argument('input_file', type=str, help='Input configuration file')
    convert_parser.add_argument('--output', type=str, help='Output configuration file')
    convert_parser.add_argument('--format', type=str, choices=['yaml', 'json'], help='Output format')
    
    # Diff command
    diff_parser = subparsers.add_parser('diff', help='Compare two configuration files')
    diff_parser.add_argument('config1', type=str, help='First configuration file')
    diff_parser.add_argument('config2', type=str, help='Second configuration file')
    diff_parser.add_argument('--format', type=str, default='table', choices=['table', 'json'], help='Output format')
    
    # Misc
    parser.add_argument('--log_level', type=str, default='info', help='Logging level')
    
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')


def create_config_template(template_type, dataset, model):
    """Create a configuration template."""
    if template_type == 'basic':
        config = {
            'model': {
                'name': model,
                'img_size': 224,
                'patch_size': 16,
                'num_classes': 10 if dataset == 'cifar10' else 100,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4.0
            },
            'data': {
                'dataset': dataset,
                'data_dir': './data',
                'batch_size': 64,
                'num_workers': 2
            },
            'training': {
                'epochs': 10,
                'learning_rate': 1e-3,
                'optimizer': 'adam',
                'weight_decay': 0.01,
                'scheduler': 'cosine'
            },
            'lora': {
                'rank': 4,
                'alpha': 1.0,
                'dropout': 0.1
            }
        }
    elif template_type == 'advanced':
        config = {
            'model': {
                'name': model,
                'img_size': 224,
                'patch_size': 16,
                'num_classes': 10 if dataset == 'cifar10' else 100,
                'embed_dim': 768,
                'depth': 12,
                'num_heads': 12,
                'mlp_ratio': 4.0,
                'dropout': 0.1,
                'attention_dropout': 0.1
            },
            'data': {
                'dataset': dataset,
                'data_dir': './data',
                'batch_size': 64,
                'num_workers': 4,
                'pin_memory': True,
                'persistent_workers': True
            },
            'training': {
                'epochs': 50,
                'learning_rate': 1e-4,
                'optimizer': 'adamw',
                'weight_decay': 0.05,
                'scheduler': 'cosine',
                'warmup_epochs': 5,
                'min_lr': 1e-6,
                'gradient_clip': 1.0
            },
            'lora': {
                'rank': 16,
                'alpha': 32,
                'dropout': 0.1,
                'target_modules': ['attention.qkv', 'attention.proj', 'mlp.fc1', 'mlp.fc2']
            },
            'callbacks': {
                'early_stopping': {
                    'enabled': True,
                    'patience': 10,
                    'min_delta': 0.001
                },
                'checkpointing': {
                    'enabled': True,
                    'save_best': True,
                    'save_last': True
                }
            },
            'logging': {
                'level': 'info',
                'log_interval': 100,
                'save_interval': 5
            }
        }
    else:  # custom
        config = default_config.copy()
        config['data']['dataset'] = dataset
        config['model']['name'] = model
    
    return config


def validate_config(config_file, strict=False):
    """Validate a configuration file."""
    try:
        config = load_config(config_file)
        
        # Basic validation
        required_sections = ['model', 'data', 'training']
        for section in required_sections:
            if section not in config:
                return False, f"Missing required section: {section}"
        
        # Model validation
        model_config = config['model']
        required_model_keys = ['img_size', 'patch_size', 'num_classes']
        for key in required_model_keys:
            if key not in model_config:
                return False, f"Missing required model key: {key}"
        
        # Data validation
        data_config = config['data']
        required_data_keys = ['dataset', 'batch_size']
        for key in required_data_keys:
            if key not in data_config:
                return False, f"Missing required data key: {key}"
        
        # Training validation
        training_config = config['training']
        required_training_keys = ['epochs', 'learning_rate']
        for key in required_training_keys:
            if key not in training_config:
                return False, f"Missing required training key: {key}"
        
        if strict:
            # Additional strict validation
            if training_config['learning_rate'] <= 0:
                return False, "Learning rate must be positive"
            if training_config['epochs'] <= 0:
                return False, "Epochs must be positive"
            if data_config['batch_size'] <= 0:
                return False, "Batch size must be positive"
        
        return True, "Configuration is valid"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def main():
    """Main function for configuration operations."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'create':
            logger.info(f"📝 Creating {args.template} configuration template...")
            config = create_config_template(args.template, args.dataset, args.model)
            
            # Save configuration
            if args.output.endswith('.json'):
                with open(args.output, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                with open(args.output, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Configuration saved to {args.output}")
            
        elif args.command == 'validate':
            logger.info(f"🔍 Validating configuration: {args.config_file}")
            is_valid, message = validate_config(args.config_file, args.strict)
            
            if is_valid:
                logger.info(f"✅ {message}")
            else:
                logger.error(f"❌ {message}")
                return 1
                
        elif args.command == 'show':
            logger.info("📋 Showing default configuration...")
            config = default_config
            
            if args.section:
                if args.section in config:
                    config = {args.section: config[args.section]}
                else:
                    logger.error(f"❌ Section '{args.section}' not found")
                    return 1
            
            if args.format == 'json':
                print(json.dumps(config, indent=2))
            else:
                print(yaml.dump(config, default_flow_style=False, indent=2))
                
        elif args.command == 'convert':
            logger.info(f"🔄 Converting configuration: {args.input_file}")
            
            # Load input config
            config = load_config(args.input_file)
            
            # Determine output format
            if args.format:
                output_format = args.format
            elif args.output:
                output_format = 'json' if args.output.endswith('.json') else 'yaml'
            else:
                output_format = 'yaml'
            
            # Determine output file
            if args.output:
                output_file = args.output
            else:
                base_name = os.path.splitext(args.input_file)[0]
                output_file = f"{base_name}.{output_format}"
            
            # Save in new format
            if output_format == 'json':
                with open(output_file, 'w') as f:
                    json.dump(config, f, indent=2)
            else:
                with open(output_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Configuration converted to {output_file}")
            
        elif args.command == 'diff':
            logger.info(f"🔍 Comparing configurations: {args.config1} vs {args.config2}")
            
            config1 = load_config(args.config1)
            config2 = load_config(args.config2)
            
            # Simple diff implementation
            def get_differences(dict1, dict2, path=""):
                differences = []
                all_keys = set(dict1.keys()) | set(dict2.keys())
                
                for key in all_keys:
                    current_path = f"{path}.{key}" if path else key
                    
                    if key not in dict1:
                        differences.append(f"+ {current_path}: {dict2[key]}")
                    elif key not in dict2:
                        differences.append(f"- {current_path}: {dict1[key]}")
                    elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                        differences.extend(get_differences(dict1[key], dict2[key], current_path))
                    elif dict1[key] != dict2[key]:
                        differences.append(f"~ {current_path}: {dict1[key]} -> {dict2[key]}")
                
                return differences
            
            differences = get_differences(config1, config2)
            
            if args.format == 'json':
                print(json.dumps(differences, indent=2))
            else:
                if differences:
                    print("Configuration differences:")
                    for diff in differences:
                        print(f"  {diff}")
                else:
                    print("No differences found")
                    
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
