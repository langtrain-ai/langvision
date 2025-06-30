import toml

pyproject_path = "pyproject.toml"
data = toml.load(pyproject_path)
version = data["project"]["version"]
major, minor, patch = map(int, version.split('.'))
patch += 1
new_version = f"{major}.{minor}.{patch}"
data["project"]["version"] = new_version
with open(pyproject_path, "w") as f:
    toml.dump(data, f)
print(f"Bumped version: {version} -> {new_version}")
