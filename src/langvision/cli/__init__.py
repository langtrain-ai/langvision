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
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint
from rich.layout import Layout
from rich.align import Align
from rich.prompt import Prompt
from rich.theme import Theme
from rich import box

__version__ = "0.1.42"  # Keep in sync with pyproject.toml

# Langtrain Branding Theme
langtrain_theme = Theme({
    "primary": "bold #3B82F6",    # Langtrain Blue
    "secondary": "#06B6D4",       # Cyan
    "accent": "#8B5CF6",          # Purple
    "success": "#10B981",         # Green
    "warning": "#F59E0B",         # Amber
    "error": "#EF4444",           # Red
    "muted": "dim white",
    "info": "white"
})

console = Console(theme=langtrain_theme)

def print_banner():
    """Print the unified Langtrain banner."""
    header = Table.grid(padding=1, expand=True)
    header.add_column(justify="right", ratio=1)
    header.add_column(justify="left", ratio=2)
    
    # Custom L Logo
    logo_art = r"""
            ________
           /       /
          /       /
         /       /
        /       /
       /       /
      /       /
     /       /      ________
    /       /      /       /
   /_______/      /_______/

   L A N G T R A I N
"""
    
    # Text side
    text_content = Text()
    text_content.append("\nLangvision", style="bold primary")
    text_content.append(f"\nv{__version__}", style="dim white")
    text_content.append("\nEnterprise Vision Modules", style="muted")
    
    header.add_row(Text(logo_art, style="primary"), text_content)
    
    console.print(Panel(
        header,
        style="primary",
        border_style="primary",
        box=box.ROUNDED,
        padding=(0, 2)
    ))

def print_auth_status():
    """Print authentication status using Rich tables."""
    auth = get_auth()
    
    if auth.is_authenticated:
        try:
            usage = auth.check_usage_limits()
            
            # Create a clean status table
            grid = Table.grid(padding=(0, 2))
            grid.add_column(justify="left")
            grid.add_column(justify="left")
            
            grid.add_row(
                "[success]● Authenticated[/]",
                f"[muted]Usage: {usage['commands_used']}/{usage['commands_limit']} commands[/]"
            )
            
            remaining = usage['commands_remaining']
            if remaining < 100:
                grid.add_row(
                    "[warning]![/]", 
                    f"[warning]Low balance: {remaining} commands left[/]"
                )
                
            console.print(Panel(
                grid,
                title="[bold]System Status[/]",
                title_align="left",
                border_style="success",
                padding=(0, 1),
                width=60
            ))
        except Exception:
             console.print("[muted]● Authenticated (Offline Mode)[/]")

    else:
        console.print(Panel(
            "[warning]● Not Authenticated[/]\n[muted]Run 'langvision auth login' to connect[/]",
            border_style="warning",
            width=60,
            padding=(0, 1)
        ))
    console.print()

def check_auth_and_usage(command_type: str = "general") -> bool:
    """
    Check authentication and usage limits using Rich panels.
    """
    auth = get_auth()
    
    if not auth.is_authenticated:
        console.print(Panel(
            "[bold error]Authentication Required[/]\n\n"
            "This Langtrain tool requires an active session.\n"
            "1. Get your key at: [underline primary]https://langtrain.xyz[/]\n"
            "2. Run: [bold success]langvision auth login[/]\n\n"
            "[muted]Or set LANGVISION_API_KEY env var[/]",
            border_style="error",
            title="🔒 Access Denied"
        ))
        return False
    
    if not auth.validate_api_key():
        console.print(Panel(
            "[bold error]Invalid API Key[/]\n\n"
            "Your key format is incorrect. Keys must start with 'lv-'.",
            border_style="error"
        ))
        return False
    
    try:
        usage = auth.check_usage_limits()
        if not usage["within_limits"]:
            console.print(Panel(
                f"[bold error]Usage Limit Exceeded[/]\n\n"
                f"You have used {usage['commands_used']}/{usage['commands_limit']} commands.\n"
                "Please upgrade your plan at: [underline primary]https://billing.langtrain.xyz[/]",
                border_style="error"
            ))
            return False
            
        auth.record_usage(command_type)
        return True
    except Exception as e:
        # Fail open if usage check fails (e.g. offline)
        return True

def handle_auth_command(args):
    """Handle auth subcommand with Rich."""
    
    if not args.auth_action:
        console.print("[warning]Usage:[/langvision auth [login|logout|status|usage]")
        return
    
    if args.auth_action == 'login':
        console.print(Panel(
            "Please enter your Langvision API Key.\n"
            "[muted]You can find this in your dashboard at https://langtrain.xyz[/]",
            title="🔐 Authentication",
            border_style="secondary"
        ))
        
        api_key = getpass.getpass("API Key: ")
        
        if auth_login(api_key):
            console.print("[bold success]✓ Successfully authenticated![/]")
        else:
            console.print("[bold error]✗ Invalid API key format.[/]")
    
    elif args.auth_action == 'logout':
        auth_logout()
        console.print("[bold success]✓ Successfully logged out.[/]")
    
    elif args.auth_action == 'status':
        print_auth_status()
    
    elif args.auth_action == 'usage':
        auth = get_auth()
        if not auth.is_authenticated:
            console.print("[error]Not authenticated.[/]")
            return
        
        usage = auth.check_usage_limits()
        
        table = Table(title="Usage Summary", box=None)
        table.add_column("Metric", style="secondary")
        table.add_column("Value", style="bold info")
        
        table.add_row("Monthly Usage", f"{usage['commands_used']} / {usage['commands_limit']}")
        table.add_row("Training Runs", str(usage['training_runs']))
        table.add_row("Fine-tune Runs", str(usage['finetune_runs']))
        
        console.print(table)
        
        # Simple progress bar
        pct = usage['commands_used'] / usage['commands_limit']
        width = 40
        filled = int(width * pct)
        bar = f"[{'#' * filled}{'.' * (width - filled)}]"
        color = "success" if pct < 0.8 else "error"
        console.print(f"[{color}]{bar}[/] {int(pct*100)}%")

def main():
    print_banner()
    if len(sys.argv) == 1:
        print_auth_status() # Only show status on bare command

    
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