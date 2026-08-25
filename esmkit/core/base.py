from abc import ABC, abstractmethod

from oemof import solph
from oemof.network.network import Node
from pyomo.core.base.block import ScalarBlock


class EsmComponent(Node, ABC):
    @abstractmethod
    def constraint_group(self):
        pass


class EsmBlock(ScalarBlock, ABC):
    CONSTRAINT_GROUP = True

    @abstractmethod
    def _create(self, group=None):
        pass


def assert_constraint_groups(nodes):
    values = nodes.values() if isinstance(nodes, dict) else nodes
    for value in values:
        for node in value if isinstance(value, (list, tuple)) else [value]:
            constraint_group = getattr(node, "constraint_group", None)
            if constraint_group is None:
                continue

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
