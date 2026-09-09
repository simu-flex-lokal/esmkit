"""The dependency direction: esmkit.core must never import a component."""


import ast
import os

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "esmkit", "core")


def _imported_modules(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = {0: "", 1: "esmkit.core.", 2: "esmkit."}.get(node.level, "esmkit.")
            names.add(prefix + (node.module or ""))
    return names


def test_core_never_imports_components():
    offenders = []
    for filename in sorted(os.listdir(CORE)):
        if not filename.endswith(".py"):
            continue
        for name in _imported_modules(os.path.join(CORE, filename)):
            if name.startswith("esmkit.components") or name == "esmkit":
                offenders.append("{} imports {}".format(filename, name))

    assert not offenders, (
        "esmkit.core must not depend on any component - components are "
        "plugins that register themselves: " + "; ".join(offenders)
    )
