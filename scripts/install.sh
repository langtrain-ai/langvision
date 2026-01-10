#!/bin/bash

# Langvision Installer
# Installs Langvision with a dedicated virtual environment

set -e

# Branding Colors
PRIMARY="\033[1;34m"    # Blue
SECONDARY="\033[0;36m"  # Cyan
ACCENT="\033[0;35m"     # Purple
SUCCESS="\033[0;32m"    # Green
WARNING="\033[0;33m"    # Yellow
ERROR="\033[0;31m"      # Red
MUTED="\033[0;90m"      # Grey
RESET="\033[0m"

INSTALL_DIR="$HOME/.langtrain/langvision"
BIN_DIR="$HOME/.local/bin"
VENV_DIR="$INSTALL_DIR/venv"

cat << "EOF"
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
EOF
echo -e "\n${SECONDARY}Langvision Installer${RESET}"
echo -e "${MUTED}Setting up your computer vision workspace...${RESET}\n"

# 1. Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${ERROR}Error: Python 3 is required but not found.${RESET}"
    exit 1
fi

echo -e "${ACCENT}• Checking system requirements...${RESET}"

# 2. Create Directory
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"

# 3. Create Venv
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${ACCENT}• Creating isolated environment...${RESET}"
    python3 -m venv "$VENV_DIR"
else
    echo -e "${ACCENT}• Updating existing environment...${RESET}"
fi

# 4. Install
echo -e "${ACCENT}• Installing Langvision modules...${RESET}"
"$VENV_DIR/bin/pip" install -U pip > /dev/null 2>&1
"$VENV_DIR/bin/pip" install -e . > /dev/null 2>&1 || "$VENV_DIR/bin/pip" install langvision --upgrade > /dev/null 2>&1

# 5. Link Binary
echo -e "${ACCENT}• Configuring shell access...${RESET}"
ln -sf "$VENV_DIR/bin/langvision" "$BIN_DIR/langvision"

# 6. Path Check
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo -e "\n${WARNING}Warning: $BIN_DIR is not in your PATH.${RESET}"
    echo -e "Add this to your shell config file (.zshrc or .bashrc):"
    echo -e "${MUTED}export PATH=\"\$HOME/.local/bin:\$PATH\"${RESET}"
fi

echo -e "\n${SUCCESS}✔ Installation Complete${RESET}"
echo -e "\n${PRIMARY}Get started:${RESET}"
echo -e "  ${SECONDARY}langvision auth login${RESET}   Connect your account"
echo -e "  ${SECONDARY}langvision model-zoo${RESET}    Browse available models\n"
