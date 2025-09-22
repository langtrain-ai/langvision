import argparse
import sys
from .train import main as train_main
from .finetune import main as finetune_main
from .evaluate import main as evaluate_main
from .export import main as export_main
from .model_zoo import main as model_zoo_main
from .config import main as config_main

__version__ = "0.1.0"  # Keep in sync with package version in pyproject.toml

def print_banner():
    banner = r"""
\033[1;36m
 _                     __     ___     _             
| |    __ _ _ __   __ _\ \   / (_)___(_) ___  _ __  
| |   / _` | '_ \ / _` |\ \ / /| / __| |/ _ \| '_ \ 
| |__| (_| | | | | (_| | \ V / | \__ \ | (_) | | | |
|_____\__,_|_| |_|\__, |  \_/  |_|___/_|\___/|_| |_|
                  |___/                             
\033[0m
"""
    print(banner)
    print("\033[1;33mLANGVISION\033[0m: Modular Vision LLMs with Efficient LoRA Fine-Tuning")
    print(f"\033[1;35mVersion:\033[0m {__version__}")
    print("\033[1;32mDocs:\033[0m https://github.com/langtrain-ai/langtrain/tree/main/docs    \033[1;34mPyPI:\033[0m https://pypi.org/project/langvision/\n")

def main():
    print_banner()
    parser = argparse.ArgumentParser(
        prog="langvision",
        description="Langvision: Modular Vision LLMs with Efficient LoRA Fine-Tuning.\n\nUse subcommands to train or finetune vision models."
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-commands')

    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a VisionTransformer model')
    train_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Finetune subcommand
    finetune_parser = subparsers.add_parser('finetune', help='Finetune a VisionTransformer model with LoRA and LLM concepts')
    finetune_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Evaluate subcommand
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    evaluate_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Export subcommand
    export_parser = subparsers.add_parser('export', help='Export a model to various formats (ONNX, TorchScript)')
    export_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Model Zoo subcommand
    model_zoo_parser = subparsers.add_parser('model-zoo', help='Browse and download pre-trained models')
    model_zoo_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Config subcommand
    config_parser = subparsers.add_parser('config', help='Manage configuration files')
    config_parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == 'train':
        sys.argv = [sys.argv[0]] + args.args
        train_main()
    elif args.command == 'finetune':
        sys.argv = [sys.argv[0]] + args.args
        finetune_main()
    elif args.command == 'evaluate':
        sys.argv = [sys.argv[0]] + args.args
        evaluate_main()
    elif args.command == 'export':
        sys.argv = [sys.argv[0]] + args.args
        export_main()
    elif args.command == 'model-zoo':
        sys.argv = [sys.argv[0]] + args.args
        model_zoo_main()
    elif args.command == 'config':
        sys.argv = [sys.argv[0]] + args.args
        config_main()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()