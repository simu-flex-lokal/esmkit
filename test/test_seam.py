# -*- coding: utf-8 -*-
"""
The seam that keeps the kit generic.

`esmkit.core` is the framework: spec, registry, builder, solver, results.
`esmkit.components` are plugins which register themselves. The moment core
imports a component, the kit stops being a kit and starts being a model of
one particular thing - so it is checked rather than merely intended.
"""

import ast
import os

CORE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "esmkit", "core")


def _imported_modules(path):
    """Every module name a file imports, including inside functions."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level counts the leading dots: 1 is esmkit.core, 2 is esmkit
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


def test_the_core_solves_without_any_component_module():
    """Proof rather than inspection: the framework imports and builds a
    model with only the stock factories registered."""
    import pandas as pd

    from esmkit.core.results import node_results, solve
    from esmkit.core.spec import SystemSpec, build_system

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=2.0)

    index = pd.date_range("2010-01-01", periods=6, freq="h")
    es, nodes = build_system(spec, {}, index)
    model, _ = solve(es)
    assert node_results(model, nodes)["demand"]["in_elec"].sum() == 12.0
