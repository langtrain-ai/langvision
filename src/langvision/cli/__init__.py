import argparse
import sys
from .train import main as train_main
from .finetune import main as finetune_main

__version__ = "0.1.0"  # Keep in sync with package version

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
    print("\033[1;33mLANGVISION\033[0m: Modular Vision LLMs with Efficient LoRA Fine-Tuning\n")
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
    # Accepts all arguments from train.py
    train_parser.add_argument('args', nargs=argparse.REMAINDER)

    # Finetune subcommand
    finetune_parser = subparsers.add_parser('finetune', help='Finetune a VisionTransformer model with LoRA and LLM concepts')
    # Accepts all arguments from finetune.py
    finetune_parser.add_argument('args', nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.command == 'train':
        sys.argv = [sys.argv[0]] + args.args
        train_main()
    elif args.command == 'finetune':
        sys.argv = [sys.argv[0]] + args.args
        finetune_main()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()