import argparse
import sys
from .train import main as train_main
from .finetune import main as finetune_main
from .evaluate import main as evaluate_main
from .export import main as export_main
from .model_zoo import main as model_zoo_main
from .config import main as config_main

__version__ = "0.1.0"  # Keep in sync with package version in pyproject.toml

# ANSI color codes
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"

def print_banner():
    """Print a stylish welcome banner for Langvision CLI."""
    c = Colors
    
    logo = f"""
{c.CYAN}{c.BOLD}
    ╭─────────────────────────────────────────────────────────────╮
    │                                                             │
    │   ██╗      █████╗ ███╗   ██╗ ██████╗ ██╗   ██╗██╗███████╗  │
    │   ██║     ██╔══██╗████╗  ██║██╔════╝ ██║   ██║██║██╔════╝  │
    │   ██║     ███████║██╔██╗ ██║██║  ███╗██║   ██║██║███████╗  │
    │   ██║     ██╔══██║██║╚██╗██║██║   ██║╚██╗ ██╔╝██║╚════██║  │
    │   ███████╗██║  ██║██║ ╚████║╚██████╔╝ ╚████╔╝ ██║███████║  │
    │   ╚══════╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝   ╚═══╝  ╚═╝╚══════╝  │
    │                                                             │
    ╰─────────────────────────────────────────────────────────────╯
{c.RESET}"""
    
    print(logo)
    print(f"    {c.BOLD}{c.WHITE}Efficient LoRA Fine-Tuning for Vision LLMs{c.RESET}")
    print(f"    {c.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{c.RESET}\n")
    
    print(f"    {c.YELLOW}⚡ Version:{c.RESET}  {c.WHITE}{__version__}{c.RESET}")
    print(f"    {c.GREEN}📖 Docs:{c.RESET}     {c.BLUE}https://github.com/langtrain-ai/langvision{c.RESET}")
    print(f"    {c.MAGENTA}📦 PyPI:{c.RESET}     {c.BLUE}https://pypi.org/project/langvision/{c.RESET}")
    print()

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