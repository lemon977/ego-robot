"""Conservative project-local Python import closure, without importing models."""
from __future__ import annotations

import ast
from collections import deque
from pathlib import Path


def local_import_closure(root: Path, seeds: list[Path]) -> list[Path]:
    root = root.resolve()
    src = root / "src"

    def locate(name):
        if not name.startswith("chaoyang"):
            return None
        base = src.joinpath(*name.split("."))
        for path in (base.with_suffix(".py"), base / "__init__.py"):
            if path.is_file():
                resolved = path.resolve()
                if root not in resolved.parents:
                    raise RuntimeError("SOURCE_CLOSURE_ESCAPES_PROJECT:" + str(path))
                return resolved
        return None

    queue, seen = deque(seeds), set()
    while queue:
        path = queue.popleft().resolve()
        if path in seen:
            continue
        if root not in path.parents or not path.is_file():
            raise RuntimeError("SOURCE_CLOSURE_MISSING_OR_EXTERNAL:" + str(path))
        seen.add(path)
        for parent in path.parents:
            if parent == src:
                break
            init = parent / "__init__.py"
            if init.is_file() and init not in seen:
                queue.append(init)
        module = ".".join(path.relative_to(src).with_suffix("").parts)
        if module.endswith(".__init__"):
            module = module[:-9]
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    parent = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
                    prefix = ".".join(parent[:len(parent) - node.level + 1])
                    base = prefix + ("." + base if base else "")
                names = [base, *(base + "." + alias.name for alias in node.names if alias.name != "*")]
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "import_module" and node.args
                  and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                names = [node.args[0].value]
            for name in names:
                if dependency := locate(name):
                    queue.append(dependency)
    return sorted(seen)
