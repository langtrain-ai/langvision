#!/usr/bin/env python3
"""
Fail if any import inside the package points at a module or name that doesn't
exist. Static (no package install needed), so it catches what a refactor or a
"sync" commit silently deletes, before users hit ModuleNotFoundError.

Imports wrapped in `try: ... except ImportError` are optional by design and
skipped.

Usage: python scripts/check_imports.py <src-dir> <package>
"""
import ast
import os
import sys

root, pkg = sys.argv[1], sys.argv[2]


def module_file(mod):
    p = os.path.join(root, *mod.split("."))
    if os.path.isfile(p + ".py"):
        return p + ".py"
    if os.path.isfile(os.path.join(p, "__init__.py")):
        return os.path.join(p, "__init__.py")
    return None


_names = {}


def defined_names(mod):
    """Names a module defines or imports at any depth, plus its submodules."""
    if mod in _names:
        return _names[mod]
    f = module_file(mod)
    if f is None:
        return None
    names = set()
    _names[mod] = names
    for node in ast.walk(ast.parse(open(f, encoding="utf-8").read())):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                names.update(x.id for x in ast.walk(target) if isinstance(x, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
    return names


def guarded(tree):
    """ImportFrom nodes inside a try whose handlers catch ImportError."""
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        caught = set()
        for h in node.handlers:
            if h.type is None:
                caught.add("*")
            for x in ast.walk(h.type) if h.type else []:
                if isinstance(x, ast.Name):
                    caught.add(x.id)
        if caught & {"ImportError", "ModuleNotFoundError", "Exception", "*"}:
            for stmt in node.body:
                out.update(id(x) for x in ast.walk(stmt) if isinstance(x, ast.ImportFrom))
    return out


problems = []
for dirpath, _, files in os.walk(os.path.join(root, pkg)):
    for fn in sorted(files):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(dirpath, fn)
        dotted = os.path.relpath(path, root)[:-3].replace(os.sep, ".")
        package = dotted[: -len(".__init__")] if fn == "__init__.py" else dotted.rsplit(".", 1)[0]
        try:
            tree = ast.parse(open(path, encoding="utf-8").read())
        except SyntaxError as e:
            problems.append(f"{path}: syntax error: {e}")
            continue
        skip = guarded(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or id(node) in skip:
                continue
            if node.level:
                parts = package.split(".")
                if node.level > 1:
                    parts = parts[: -(node.level - 1)]
                mod = ".".join(parts + ([node.module] if node.module else []))
            elif node.module and (node.module == pkg or node.module.startswith(pkg + ".")):
                mod = node.module
            else:
                continue
            names = defined_names(mod)
            if names is None:
                problems.append(f"{path}:{node.lineno}: no module {mod}")
                continue
            for alias in node.names:
                if alias.name != "*" and alias.name not in names and not module_file(f"{mod}.{alias.name}"):
                    problems.append(f"{path}:{node.lineno}: {mod} has no {alias.name}")

for p in problems:
    print(p)
print(f"{len(problems)} broken import(s) in {pkg}")
sys.exit(1 if problems else 0)
