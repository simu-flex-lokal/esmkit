# -*- coding: utf-8 -*-
"""
Base classes for custom solph components.

Almost nothing needs these: every stock component of the kit (grid, PV,
storage, heat pump, demand, metering) maps onto a stock solph object and is
built by a `@factory` in `components/stock.py`. A custom component is only
necessary when the physics is not expressible as flows between buses - free
state variables, coupled algebraic nodes, bounds on something that is not a
flow. The 5R1C thermal zone is the kit's only such case, and doubles as the
reference implementation.

A custom component is always two classes that have to find each other:

  * a **node** (`EsmComponent`) that hangs in the energy system graph and
    carries the parameterization;
  * a **block** (`EsmBlock`) that holds the pyomo variables and constraints,
    created once for all nodes of that type.

`constraint_group()` on the node is the link: it returns the block class,
and solph instantiates that block while assembling the model.

Deriving from these bases rather than from `Node`/`ScalarBlock` directly
buys two things: `CONSTRAINT_GROUP` is inherited instead of being repeated
(see `assert_constraint_groups` for why its absence is so costly), and the
abstract methods make a half-implemented component fail at instantiation
rather than at solve time.
"""

from abc import ABC, abstractmethod

from oemof import solph
from oemof.network.network import Node
from pyomo.core.base.block import ScalarBlock


class EsmComponent(Node, ABC):
    """
    A solph node whose physics lives in an `EsmBlock`.

    Subclasses implement `constraint_group()` and are otherwise ordinary
    `oemof.network.Node`s: the bus coupling is declared through the
    `inputs`/`outputs` passed to `Node.__init__`, and read inside the block
    as `m.flow[bus, node, t]` / `m.flow[node, bus, t]`. Do not create your
    own flow variables - solph already made them for those edges.

    A subclass may additionally implement `results(model, index)` to report
    its own result shape; `node_results` picks it up automatically and
    falls back to generic flow extraction when it is absent.
    """

    @abstractmethod
    def constraint_group(self):
        """The `EsmBlock` subclass holding this component's constraints."""


class EsmBlock(ScalarBlock, ABC):
    """
    Constraint block of all nodes of one custom component type.

    `CONSTRAINT_GROUP` is what makes solph pick the block up:
    `oemof.solph._models.Model.__init__` filters the energy system's groups
    with `hasattr(i, "CONSTRAINT_GROUP")`. Note that the test is for
    existence, not for truth - setting it to False does not switch anything
    off, it only obscures the intent.

    Subclasses implement `_create(group)`, where `group` is the list of
    nodes of this type in the system, and optionally
    `_objective_expression()`.
    """

    CONSTRAINT_GROUP = True

    @abstractmethod
    def _create(self, group=None):
        """Builds variables and constraints for all nodes in `group`."""


def assert_constraint_groups(nodes):
    """
    Raises if a custom component's block would be ignored by solph.

    Without `CONSTRAINT_GROUP` on the block, solph skips it in silence: the
    model builds, the solver reports optimality, and the component imposes
    no constraints whatsoever. Deriving the block from `EsmBlock` makes
    that unreachable, but nothing stops a component from subclassing
    `ScalarBlock` directly, so the guard is checked on every built system.

    Stock solph blocks are exempt because they never take the `hasattr`
    path: `Model.CONSTRAINT_GROUPS` already lists them, and the attribute
    scan only collects the custom blocks on top of that list.

    Parameters
    ----------
    nodes: iterable or dict, required
        Created nodes; values may be single nodes or lists of nodes, as
        returned by `build_system`.
    """
    values = nodes.values() if isinstance(nodes, dict) else nodes
    for value in values:
        for node in value if isinstance(value, (list, tuple)) else [value]:
            constraint_group = getattr(node, "constraint_group", None)
            if constraint_group is None:
                continue
            # stock solph nodes inherit constraint_group() -> None, which is
            # how they declare that they have no block of their own
            block = constraint_group()
            if block is None or block in solph.Model.CONSTRAINT_GROUPS:
                continue
            if not hasattr(block, "CONSTRAINT_GROUP"):
                raise TypeError(
                    "{} of component '{}' has no CONSTRAINT_GROUP attribute, so "
                    "solph would silently ignore its constraints and the "
                    "component would impose nothing on the solution. Derive it "
                    "from esmkit.EsmBlock.".format(
                        getattr(block, "__name__", block), node.label
                    )
                )
