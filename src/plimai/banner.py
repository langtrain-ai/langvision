import sys
import platform
from datetime import datetime
import re

# ANSI color codes
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"
WHITE = "\033[97m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"

BOX_WIDTH = 74


def strip_ansi(text: str) -> str:
    """Remove ANSI codes for accurate width calculation."""
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)


def strip_emoji(text: str) -> str:
    """Replace emojis with placeholder for width alignment."""
    # Count emojis properly - each emoji takes ~2 characters width
    emoji_pattern = re.compile(r'[🚀🔗🛠️🐍🕒✨]')
    emoji_count = len(emoji_pattern.findall(text))
    text_without_emoji = emoji_pattern.sub('XX', text)  # Replace with 2 chars
    return text_without_emoji


def pad_line(line: str, width: int = BOX_WIDTH) -> str:
    """Center a line in the box, accounting for ANSI and emoji."""
    # Get the actual display width
    clean_line = strip_emoji(strip_ansi(line))
    display_width = len(clean_line)
    
    # Calculate padding
    available_width = width - 2  # Account for box borders │
    if display_width >= available_width:
        return f"│{line[:available_width]}│"
    
    padding = available_width - display_width
    left = padding // 2
    right = padding - left
    return f"│{' ' * left}{line}{' ' * right}│"


def get_ascii_logo() -> list[str]:
    """Return ASCII art for Plimai logo - clean blocky style."""
    lines = [
        "██████╗ ██╗     ██╗███╗   ███╗ █████╗ ██╗",
        "██╔══██╗██║     ██║████╗ ████║██╔══██╗██║",
        "██████╔╝██║     ██║██╔████╔██║███████║██║",
        "██╔═══╝ ██║     ██║██║╚██╔╝██║██╔══██║██║",
        "██║     ███████╗██║██║ ╚═╝ ██║██║  ██║██║",
        "╚═╝     ╚══════╝╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝",
    ]
    
    # Ensure all lines are the same width for proper centering
    max_width = max(len(line) for line in lines)
    normalized_lines = [line.ljust(max_width) for line in lines]
    
    return [f"{BOLD}{WHITE}{line}{RESET}" for line in normalized_lines]


def get_info_block(version: str) -> list[str]:
    """Return info block content."""
    github_url = "https://github.com/yourorg/plimai"
    author = f"{MAGENTA}Author: Pritesh Raj (@priteshraj){RESET}"
    slogan = f"{GREEN}The Modular Vision LLMs & LoRA Fine-Tuning Framework{RESET}"
    python_info = f"Python {platform.python_version()} | {platform.system()} {platform.release()}"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return [
        f"🚀  {YELLOW}{BOLD}Plimai{RESET}{CYAN}: Vision LLMs with Efficient LoRA Fine-Tuning",
        slogan,
        "",
        f"🔗  GitHub: {YELLOW}{github_url}{RESET}",
        f"🛠️   {YELLOW}Version: {version}{RESET}",
        f"{author}",
        f"🐍  {python_info}",
        f"🕒  {timestamp}"
    ]


def print_banner(version="0.1.0"):
    """Print a styled and centered ASCII banner for Plimai."""
    border_top = f"{CYAN}┌{'─' * BOX_WIDTH}┐{RESET}"
    border_bottom = f"{CYAN}└{'─' * BOX_WIDTH}┘{RESET}"

    content = [''] + get_ascii_logo() + [''] + get_info_block(version) + ['']
    banner = [border_top] + [pad_line(line) for line in content] + [border_bottom]

    print("\n".join(banner))


if __name__ == "__main__":
    print_banner()