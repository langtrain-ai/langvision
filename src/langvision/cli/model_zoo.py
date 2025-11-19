#!/usr/bin/env python3
"""
Model Zoo CLI for Langvision.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Import from langvision modules
try:
    from langvision.model_zoo import get_available_models, get_model_info, download_model
except ImportError as e:
    print(f"❌ Error importing langvision modules: {e}")
    print("Please ensure langvision is properly installed:")
    print("  pip install langvision")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments for model zoo operations."""
    parser = argparse.ArgumentParser(
        description='Langvision Model Zoo - Browse, download, and manage pre-trained models',
        epilog='''\nExamples:\n  langvision model-zoo list\n  langvision model-zoo info vit_base_patch16_224\n  langvision model-zoo download vit_base_patch16_224\n''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Model zoo commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all available models')
    list_parser.add_argument('--format', type=str, default='table', choices=['table', 'json'], help='Output format')
    list_parser.add_argument('--filter', type=str, help='Filter models by name or type')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get detailed information about a model')
    info_parser.add_argument('model_name', type=str, help='Name of the model')
    info_parser.add_argument('--format', type=str, default='table', choices=['table', 'json'], help='Output format')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download a pre-trained model')
    download_parser.add_argument('model_name', type=str, help='Name of the model to download')
    download_parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save the model')
    download_parser.add_argument('--force', action='store_true', help='Force download even if model exists')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for models')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--format', type=str, default='table', choices=['table', 'json'], help='Output format')
    
    # Misc
    parser.add_argument('--log_level', type=str, default='info', help='Logging level')
    
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')


def print_model_table(models):
    """Print models in a formatted table."""
    if not models:
        print("No models found.")
        return
    
    # Calculate column widths
    name_width = max(len(model.get('name', '')) for model in models) + 2
    type_width = max(len(model.get('type', '')) for model in models) + 2
    size_width = max(len(str(model.get('size', ''))) for model in models) + 2
    
    # Print header
    print(f"{'Name':<{name_width}} {'Type':<{type_width}} {'Size':<{size_width}} {'Description'}")
    print("-" * (name_width + type_width + size_width + 50))
    
    # Print models
    for model in models:
        name = model.get('name', '')
        model_type = model.get('type', '')
        size = model.get('size', '')
        description = model.get('description', '')
        print(f"{name:<{name_width}} {model_type:<{type_width}} {size:<{size_width}} {description}")


def print_model_info(model_info, format='table'):
    """Print detailed model information."""
    if format == 'json':
        print(json.dumps(model_info, indent=2))
        return
    
    print(f"\n🤖 Model: {model_info.get('name', 'Unknown')}")
    print("=" * 50)
    
    for key, value in model_info.items():
        if key == 'name':
            continue
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    if 'config' in model_info:
        print(f"\n📋 Configuration:")
        config = model_info['config']
        for key, value in config.items():
            print(f"  {key}: {value}")


def main():
    """Main function for model zoo operations."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == 'list':
            logger.info("📋 Fetching available models...")
            models = get_available_models()
            
            if args.filter:
                models = [m for m in models if args.filter.lower() in m.get('name', '').lower()]
            
            if args.format == 'json':
                print(json.dumps(models, indent=2))
            else:
                print(f"\n🎯 Available Models ({len(models)} total):")
                print_model_table(models)
                
        elif args.command == 'info':
            logger.info(f"🔍 Getting information for model: {args.model_name}")
            model_info = get_model_info(args.model_name)
            print_model_info(model_info, args.format)
            
        elif args.command == 'download':
            logger.info(f"⬇️ Downloading model: {args.model_name}")
            output_path = download_model(args.model_name, args.output_dir, force=args.force)
            logger.info(f"✅ Model downloaded to: {output_path}")
            
        elif args.command == 'search':
            logger.info(f"🔍 Searching for: {args.query}")
            models = get_available_models()
            results = [m for m in models if args.query.lower() in m.get('name', '').lower() or 
                      args.query.lower() in m.get('description', '').lower()]
            
            if args.format == 'json':
                print(json.dumps(results, indent=2))
            else:
                print(f"\n🔍 Search Results for '{args.query}' ({len(results)} found):")
                print_model_table(results)
                
    except Exception as e:
        logger.error(f"❌ Operation failed: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
