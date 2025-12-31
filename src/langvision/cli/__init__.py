import argparse
import sys
import getpass
from .train import main as train_main
from .finetune import main as finetune_main
from .evaluate import main as evaluate_main
from .export import main as export_main
from .model_zoo import main as model_zoo_main
from .config import main as config_main
from .auth import (
    LangvisionAuth, 
    AuthenticationError, 
    UsageLimitError,
    login as auth_login,
    logout as auth_logout,
    is_authenticated,
    get_auth
)

__version__ = "0.1.0"  # Keep in sync with package version in pyproject.toml

# ANSI color codes
class Colors:
    CYAN = "\033[96m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    MAGENTA = "\033[95m"
    BLUE = "\033[94m"
    RED = "\033[91m"
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
    print(f"    {c.GREEN}📖 Docs:{c.RESET}     {c.BLUE}https://langtrain.xyz/docs{c.RESET}")
    print(f"    {c.MAGENTA}📦 PyPI:{c.RESET}     {c.BLUE}https://pypi.org/project/langvision/{c.RESET}")
    print()


def print_auth_status():
    """Print authentication status."""
    c = Colors
    auth = get_auth()
    
    if auth.is_authenticated:
        usage = auth.check_usage_limits()
        print(f"    {c.GREEN}🔐 Status:{c.RESET}   {c.GREEN}Authenticated{c.RESET}")
        print(f"    {c.YELLOW}📊 Usage:{c.RESET}    {usage['commands_used']}/{usage['commands_limit']} commands this month")
        remaining = usage['commands_remaining']
        if remaining < 100:
            print(f"    {c.RED}⚠️  Warning:{c.RESET}  Only {remaining} commands remaining!")
    else:
        print(f"    {c.RED}🔒 Status:{c.RESET}   {c.RED}Not authenticated{c.RESET}")
        print(f"    {c.DIM}    Run 'langvision auth login' to authenticate{c.RESET}")
    print()


def check_auth_and_usage(command_type: str = "general") -> bool:
    """
    Check authentication and usage limits before running a command.
    Returns True if allowed to proceed, False otherwise.
    """
    c = Colors
    auth = get_auth()
    
    if not auth.is_authenticated:
        print(f"\n    {c.RED}╭──────────────────────────────────────────────────────────╮{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  {c.BOLD}🔐 Authentication Required{c.RESET}                              {c.RED}│{c.RESET}")
        print(f"    {c.RED}├──────────────────────────────────────────────────────────┤{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  Langvision requires an API key to work.                 {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  Get your API key at: {c.CYAN}https://langtrain.xyz{c.RESET}              {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  Then run:                                               {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}    {c.GREEN}langvision auth login{c.RESET}                               {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  Or set the environment variable:                        {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}    {c.GREEN}export LANGVISION_API_KEY=lv-xxxx-xxxx{c.RESET}              {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}╰──────────────────────────────────────────────────────────╯{c.RESET}\n")
        return False
    
    if not auth.validate_api_key():
        print(f"\n    {c.RED}❌ Invalid API key format.{c.RESET}")
        print(f"    {c.DIM}API keys should start with 'lv-' and be at least 20 characters.{c.RESET}")
        print(f"    {c.DIM}Get a valid key at: https://langtrain.xyz{c.RESET}\n")
        return False
    
    usage = auth.check_usage_limits()
    if not usage["within_limits"]:
        print(f"\n    {c.RED}╭──────────────────────────────────────────────────────────╮{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  {c.BOLD}⚠️  Usage Limit Exceeded{c.RESET}                                {c.RED}│{c.RESET}")
        print(f"    {c.RED}├──────────────────────────────────────────────────────────┤{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  You've used {usage['commands_used']}/{usage['commands_limit']} commands this month.             {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}  Upgrade your plan to continue:                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}    {c.CYAN}https://billing.langtrain.xyz{c.RESET}                        {c.RED}│{c.RESET}")
        print(f"    {c.RED}│{c.RESET}                                                          {c.RED}│{c.RESET}")
        print(f"    {c.RED}╰──────────────────────────────────────────────────────────╯{c.RESET}\n")
        return False
    
    # Record usage
    auth.record_usage(command_type)
    return True


def handle_auth_command(args):
    """Handle auth subcommand."""
    c = Colors
    
    if not args.auth_action:
        print(f"\n    {c.YELLOW}Usage:{c.RESET} langvision auth <login|logout|status|usage>\n")
        return
    
    if args.auth_action == 'login':
        print(f"\n    {c.CYAN}🔐 Langvision Authentication{c.RESET}")
        print(f"    {c.DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━{c.RESET}\n")
        print(f"    Get your API key at: {c.BLUE}https://langtrain.xyz{c.RESET}\n")
        
        api_key = getpass.getpass(f"    Enter your API key: ")
        
        if auth_login(api_key):
            print(f"\n    {c.GREEN}✓ Successfully authenticated!{c.RESET}\n")
        else:
            print(f"\n    {c.RED}✗ Invalid API key format.{c.RESET}")
            print(f"    {c.DIM}API keys should start with 'lv-' and be at least 20 characters.{c.RESET}\n")
    
    elif args.auth_action == 'logout':
        auth_logout()
        print(f"\n    {c.GREEN}✓ Successfully logged out.{c.RESET}\n")
    
    elif args.auth_action == 'status':
        print()
        print_auth_status()
    
    elif args.auth_action == 'usage':
        auth = get_auth()
        if not auth.is_authenticated:
            print(f"\n    {c.RED}Not authenticated. Run 'langvision auth login' first.{c.RESET}\n")
            return
        
        usage = auth.check_usage_limits()
        print(f"\n    {c.CYAN}📊 Usage Summary{c.RESET}")
        print(f"    {c.DIM}━━━━━━━━━━━━━━━━{c.RESET}\n")
        print(f"    Commands this month:  {usage['commands_used']} / {usage['commands_limit']}")
        print(f"    Remaining:            {usage['commands_remaining']}")
        print(f"    Training runs:        {usage['training_runs']}")
        print(f"    Fine-tune runs:       {usage['finetune_runs']}")
        print()
        
        # Progress bar
        used_pct = min(100, int((usage['commands_used'] / usage['commands_limit']) * 100))
        bar_width = 30
        filled = int(bar_width * used_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)
        color = c.GREEN if used_pct < 70 else (c.YELLOW if used_pct < 90 else c.RED)
        print(f"    [{color}{bar}{c.RESET}] {used_pct}%\n")
    
    else:
        print(f"\n    {c.RED}Unknown auth action: {args.auth_action}{c.RESET}")
        print(f"    {c.YELLOW}Usage:{c.RESET} langvision auth <login|logout|status|usage>\n")


def main():
    print_banner()
    print_auth_status()
    
    parser = argparse.ArgumentParser(
        prog="langvision",
        description="Langvision: Modular Vision LLMs with Efficient LoRA Fine-Tuning.\n\nUse subcommands to train or finetune vision models."
    )
    parser.add_argument('--version', action='version', version=f'%(prog)s {__version__}')
    subparsers = parser.add_subparsers(dest='command', help='Sub-commands')

    # Auth subcommand (doesn't require authentication)
    auth_parser = subparsers.add_parser('auth', help='Manage authentication')
    auth_parser.add_argument('auth_action', nargs='?', choices=['login', 'logout', 'status', 'usage'],
                            help='Authentication action')

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

    if not args.command:
        parser.print_help()
        return

    # Auth command doesn't require authentication
    if args.command == 'auth':
        handle_auth_command(args)
        return

    # All other commands require authentication
    if args.command == 'train':
        if not check_auth_and_usage("train"):
            return
        sys.argv = [sys.argv[0]] + args.args
        train_main()
    elif args.command == 'finetune':
        if not check_auth_and_usage("finetune"):
            return
        sys.argv = [sys.argv[0]] + args.args
        finetune_main()
    elif args.command == 'evaluate':
        if not check_auth_and_usage("evaluate"):
            return
        sys.argv = [sys.argv[0]] + args.args
        evaluate_main()
    elif args.command == 'export':
        if not check_auth_and_usage("export"):
            return
        sys.argv = [sys.argv[0]] + args.args
        export_main()
    elif args.command == 'model-zoo':
        if not check_auth_and_usage("model-zoo"):
            return
        sys.argv = [sys.argv[0]] + args.args
        model_zoo_main()
    elif args.command == 'config':
        if not check_auth_and_usage("config"):
            return
        sys.argv = [sys.argv[0]] + args.args
        config_main()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()