# -*- coding: utf-8 -*-
"""
Result extraction from a solved solph model into the kit's result shape.

solph's own `processing.results()` returns a flow-centric nested dict, which
is the right structure for generic post-processing but awkward when the unit
of interest is the *kit component* rather than the solph node - a grid
connection is two nodes, and a custom component may report states that are
not flows at all. This module bridges the two without replacing either.

A component reports its own shape by implementing `results(model, index)`;
everything else falls back to generic flow extraction. That hook is why this
module needs no knowledge of any particular component type.
"""

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof import solph


def solve(model_or_system, solver=None, tee=False, solverOpts=None):
    """
    Solves an energy system with the kit's solver handling.

    Deliberately not `solph.Model.solve()`: models carrying free state
    variables are numerically badly conditioned (glpk fails on them
    outright, HiGHS needs the interior point method with crossover switched
    off), and that tuning lives in `solverutils.manageSolverOpts`.
    `solph.Model` is a pyomo ConcreteModel, so the kit's solver path applies
    to it unchanged.

    Parameters
    ----------
    model_or_system: solph.Model or solph.EnergySystem, required

    Returns
    -------
    (solph.Model, solver results)
    """
    from .solverutils import solve_model

    model = model_or_system
    if isinstance(model_or_system, solph.EnergySystem):
        model = solph.Model(model_or_system)
    results = solve_model(model, solver=solver, tee=tee, solverOpts=solverOpts)
    return model, results


def objective_value(model):
    return po.value(model.objective)


def flow_series(model, source, target, index=None):
    """Time series [kW] of a single directed edge."""
    values = np.array([po.value(model.flow[source, target, t]) for t in model.TIMESTEPS])
    return pd.Series(values, index=index)


def node_results(model, nodes, index=None):
    """
    Per-component results of a built system.

    Parameters
    ----------
    model: solph.Model, required
        Solved model.
    nodes: dict, required
        The name -> node(s) mapping returned by `build_system`.
    index: pandas.DatetimeIndex, optional

    Returns
    -------
    dict of component name -> result dict. A node implementing
    `results(model, index)` reports whatever shape it chooses; every other
    component reports its flows and, where applicable, its invested capacity
    and storage content.
    """
    results = {}
    for name, node in nodes.items():
        group = node if isinstance(node, list) else [node]
        collected = {}
        for item in group:
            own = getattr(item, "results", None)
            if callable(own):
                collected = own(model, index)
                break
            single = _generic_node_results(model, item, index)
            labels = collected.pop("labels", []) + single.pop("labels", [])
            collected.update(single)
            collected["labels"] = labels
        results[name] = collected
    return results


def _generic_node_results(model, node, index=None):
    """Flows in and out of a stock solph node, plus capacity/SOC."""
    out = {}
    label = str(node.label)
    for bus in getattr(node, "inputs", {}):
        out["in_" + str(bus.label)] = flow_series(model, bus, node, index)
    for bus in getattr(node, "outputs", {}):
        out["out_" + str(bus.label)] = flow_series(model, node, bus, index)

    if isinstance(node, solph.components.GenericStorage):
        block = _storage_block(model, node)
        if block is not None:
            content = np.array(
                [po.value(block.storage_content[node, t]) for t in model.TIMEPOINTS]
            )
            out["storage_content"] = pd.Series(content)

    capacity = _invested_capacity(model, node)
    if capacity is not None:
        out["capacity"] = capacity

    # a kit component may expand into several solph nodes (a grid connection
    # becomes an import Source and an export Sink) whose results are merged
    # under the component name, so labels are collected rather than replaced
    out.setdefault("labels", []).append(label)
    return out


def _storage_block(model, node):
    for attr in ("GenericStorageBlock", "GenericInvestmentStorageBlock"):
        block = getattr(model, attr, None)
        if block is not None and node in getattr(block, "STORAGES", []):
            return block
        if block is not None and hasattr(block, "storage_content"):
            try:
                block.storage_content[node, 0]
                return block
            except (KeyError, ValueError):
                continue
    return None


def _invested_capacity(model, node):
    """Optimized capacity of a node with an Investment, else None."""
    block = getattr(model, "InvestmentFlowBlock", None)
    if block is not None and hasattr(block, "invest"):
        for bus in getattr(node, "outputs", {}):
            try:
                return po.value(block.invest[node, bus, 0])
            except (KeyError, ValueError):
                pass
        for bus in getattr(node, "inputs", {}):
            try:
                return po.value(block.invest[bus, node, 0])
            except (KeyError, ValueError):
                pass
    block = getattr(model, "GenericInvestmentStorageBlock", None)
    if block is not None and hasattr(block, "invest"):
        try:
            return po.value(block.invest[node, 0])
        except (KeyError, ValueError):
            pass
    return None
