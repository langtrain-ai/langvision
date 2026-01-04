import os
import subprocess
import sys
import toml

def run_command(command):
    try:
        subprocess.run(command, check=True, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {command}")
        sys.exit(1)

def main():
    # 1. Bump version
    print("🚀 Bumping version...")
    run_command("bumpver update --patch --no-commit")

    # 2. Get new version
    with open("pyproject.toml", "r") as f:
        config = toml.load(f)
    version = config["project"]["version"]
    print(f"📦 New version: {version}")

    # 3. Configure Git
    run_command('git config user.name "github-actions[bot]"')
    run_command('git config user.email "github-actions[bot]@users.noreply.github.com"')

    # 4. Commit and Tag
    run_command("git add .")
    run_command(f'git commit -m "ci: bump version {version}"')
    run_command(f"git tag v{version}")
    
    # 5. Push
    print("⬆️ Pushing changes...")
    run_command("git push")
    run_command("git push --tags")

if __name__ == "__main__":
    main()
