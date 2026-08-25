"""Solving a built system and reading the flows, states and invested
capacities back out of the solved pyomo model."""

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof import solph


def solve(model_or_system, solver=None, tee=False, solverOpts=None):
    """Solve an energy system or an already-built model.

    Args:
        model_or_system: a ``solph.EnergySystem`` (wrapped in a ``solph.Model``
            here) or a model that has already been built and modified.
        solver: solver name; auto-detected when omitted.
        tee: stream the solver log.
        solverOpts: explicit solver options; overrides the defaults.

    Returns:
        ``(model, results)`` - the solved model and the solver results object.
    """
    from .solverutils import solve_model

    model = model_or_system
    if isinstance(model_or_system, solph.EnergySystem):
        model = solph.Model(model_or_system)
    results = solve_model(model, solver=solver, tee=tee, solverOpts=solverOpts)
    return model, results


def objective_value(model):
    """Return the objective value of the solved model."""
    return po.value(model.objective)


def flow_series(model, source, target, index=None):
    """Return the flow from ``source`` to ``target`` as a pandas Series."""
    values = np.array([po.value(model.flow[source, target, t]) for t in model.TIMESTEPS])
    return pd.Series(values, index=index)


def node_results(model, nodes, index=None):
    """Collect per-component result series from a solved model.

    A node that defines its own ``results(model, index)`` method reports
    through it - that is the hook custom components such as
    ``ThermalZone5R1C`` use to expose their internal states. Everything else
    falls back to generic in/out flow series.

    Args:
        model: the solved model.
        nodes: the node mapping returned by ``build_system``.
        index: index to put on the returned series. It reaches the flow
            series only - ``storage_content`` is a state at the interval
            bounds, so it is one step longer and comes back unindexed.

    Returns:
        ``{component name: {series name: values}}``.
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
    out = {}
    label = str(node.label)
    for bus in getattr(node, "inputs", {}):
        out["in_" + str(bus.label)] = flow_series(model, bus, node, index)
    for bus in getattr(node, "outputs", {}):
        out["out_" + str(bus.label)] = flow_series(model, node, bus, index)

    if isinstance(node, solph.components.GenericStorage):
        block = _storage_block(model, node)
        if block is not None:
            # Over TIMEPOINTS, not TIMESTEPS: the state is defined at the
            # interval bounds, so this series has one more value than the flows.
            content = np.array(
                [po.value(block.storage_content[node, t]) for t in model.TIMEPOINTS]
            )
            out["storage_content"] = pd.Series(content)

    capacity = _invested_capacity(model, node)
    if capacity is not None:
        out["capacity"] = capacity

    out.setdefault("labels", []).append(label)
    return out


def _storage_block(model, node):
    """Find the storage block a node lives in.

    solph offers no lookup from node to block, and the dispatch and
    investment variants are separate blocks, so the block is identified by
    membership if possible and by trial indexing otherwise.
    """
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
    """Return a node's invested capacity, or None if nothing was invested.

    The investment variable is indexed by the flow (or by the storage), and
    there is no lookup for which index belongs to a node, so each candidate
    index is tried until one exists.
    """
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
