import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.style import Style

class RichHelpFormatter(argparse.HelpFormatter):
    """
    A custom argparse HelperFormatter that uses Rich to render beautiful help messages.
    """
    def __init__(self, prog, indent_increment=2, max_help_position=24, width=None):
        super().__init__(prog, indent_increment, max_help_position, width)
        self.console = Console()

    def _format_action_invocation(self, action):
        if not action.option_strings:
            return super()._format_action_invocation(action)
        return ", ".join(action.option_strings)

    def format_help(self):
        formatter = self._root_section
        
        # Create a header
        help_text = Text()
        if self._prog:
            help_text.append("Usage: ", style="bold secondary")
            help_text.append(f"{self._prog} ", style="bold primary")
            help_text.append("[options] <command> [args]\n", style="dim white")
        
        # Description
        description = self._current_section.description
        if description:
            help_text.append(f"\n{description}\n", style="white")

        # Arguments Table
        table = Table(box=None, padding=(0, 2), show_header=False)
        table.add_column("Options", style="bold accent")
        table.add_column("Description", style="white")

        for action in self._current_section.items:
            # This is a simplification; iterating internal items is fragile in strict OOP
            # but effective for customizing argparse. 
            # In standard usage we'd override _format_action, but building a table is cleaner.
            pass
            
        # Revert to standard argparse logic but captured for rich rendering
        # Just wrapping standard output in a panel for now to ensure robustness
        raw_help = super().format_help()
        
        return raw_help

def print_rich_help(parser):
    """
    Parses the standard help output and renders it with Rich.
    This is safer than subclassing HelpFormatter deeply.
    """
    console = Console()
    
    # Extract raw help
    raw_help = parser.format_help()
    
    # Simple parsing to beautify
    lines = raw_help.split('\n')
    
    usage = ""
    description = ""
    commands = []
    options = []
    
    mode = "start"
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith("usage:"):
            usage = line.replace("usage: ", "")
            continue
            
        if line.startswith("positional arguments:") or line.startswith("commands:"):
            mode = "commands"
            continue
            
        if line.startswith("options:") or line.startswith("optional arguments:"):
            mode = "options"
            continue
            
        # Simplistic parsing
        if mode == "start":
            description += line + " "
        elif mode == "commands" or mode == "options":
            # Heuristic: split by double space
            parts = line.split("  ")
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                cmd = parts[0]
                desc = " ".join(parts[1:])
                if mode == "commands": commands.append((cmd, desc))
                else: options.append((cmd, desc))
    
    # Render
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="bold primary")
    grid.add_row("Usage:", usage)
    
    console.print(Panel(grid, border_style="secondary", title="Langtune CLI", title_align="left"))
    
    if description:
        console.print(f"[white]{description}[/]\n")
        
    if commands:
        table = Table(title="Commands", box=None, padding=(0,2))
        table.add_column("Command", style="bold accent")
        table.add_column("Description", style="white")
        for cmd, desc in commands:
            table.add_row(cmd, desc)
        console.print(table)
        console.print()

    if options:
        table = Table(title="Options", box=None, padding=(0,2))
        table.add_column("Flag", style="bold secondary")
        table.add_column("Description", style="dim white")
        for opt, desc in options:
            table.add_row(opt, desc)
        console.print(table)
        console.print()
