import toml
import re
import sys

pyproject_path = "pyproject.toml"
try:
    data = toml.load(pyproject_path)
    version = data["project"]["version"]
    # Match versions like 0.1.2 or 0.1.2a1, 0.1.2b1, etc.
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)([a-zA-Z0-9]*)$", version)
    if not match:
        print(f"Error: Version '{version}' is not in a recognized format (X.Y.Z or X.Y.Z<suffix>)")
        sys.exit(1)
    major, minor, patch, suffix = match.groups()
    patch = int(patch) + 1
    new_version = f"{major}.{minor}.{patch}{suffix}"
    data["project"]["version"] = new_version
    with open(pyproject_path, "w") as f:
        toml.dump(data, f)
    print(f"Bumped version: {version} -> {new_version}")
except Exception as e:
    print(f"Error bumping version: {e}")
    sys.exit(1)
