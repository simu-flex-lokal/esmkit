"""Base classes and the guard that keeps custom components from silently
imposing no constraints on the model."""

from abc import ABC, abstractmethod

from oemof import solph
from oemof.network.network import Node
from pyomo.core.base.block import ScalarBlock


class EsmComponent(Node, ABC):
    """Base class for a custom solph node that carries its own pyomo block."""

    @abstractmethod
    def constraint_group(self):
        """Return the block class holding this component's constraints."""
        pass


class EsmBlock(ScalarBlock, ABC):
    """Base class for a component's pyomo block, pre-set as a constraint group."""

    CONSTRAINT_GROUP = True

    @abstractmethod
    def _create(self, group=None):
        """Build the variables and constraints for the nodes in ``group``."""
        pass


def assert_constraint_groups(nodes):
    """Raise if any node's block would be skipped by solph's constraint discovery.

    solph collects constraints by iterating ``solph.Model.CONSTRAINT_GROUPS``
    and only picks up blocks flagged with ``CONSTRAINT_GROUP = True``. A block
    without that flag is dropped without a warning: the model still solves, but
    the component constrains nothing at all.

    Args:
        nodes: the node mapping (or iterable) returned by ``build_system``;
            values may be single nodes or lists of nodes.
    """
    values = nodes.values() if isinstance(nodes, dict) else nodes
    for value in values:
        for node in value if isinstance(value, (list, tuple)) else [value]:
            constraint_group = getattr(node, "constraint_group", None)
            if constraint_group is None:
                continue

            block = constraint_group()
            # Blocks already registered with solph are wired up by definition.
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
