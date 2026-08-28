"""esmkit - a component kit for building oemof-solph energy system models.

Describe the system as a ``SystemSpec`` (buses, components, scalars),
turn it into an energy system with ``build_system(spec, inputs, timeindex)``,
then ``solve()`` it and read the flows back with ``node_results()``.
"""

from . import components
from .components import ThermalZone5R1C, ThermalZone5R1CBlock
from .core.analysis import bus_flows, net_grid_exchange
from .core.base import EsmBlock, EsmComponent, assert_constraint_groups
from .core.investment import annuity_factor
from .core.registry import (
    COMPONENT_FACTORIES,
    build_component,
    bus,
    capacity,
    factory,
    investment,
    profile,
)
from .core.results import flow_series, node_results, objective_value, solve
from .core.solverutils import detect_solver, manageSolverOpts
from .core.spec import (
    SystemSpec,
    build_system,
    capacity_params,
    check_inputs,
    required_inputs,
    resolve_profile,
    validate,
)

__all__ = [
    "SystemSpec",
    "build_system",
    "validate",
    "check_inputs",
    "required_inputs",
    "resolve_profile",
    "capacity_params",
    "solve",
    "node_results",
    "flow_series",
    "objective_value",
    "bus_flows",
    "net_grid_exchange",
    "ThermalZone5R1C",
    "ThermalZone5R1CBlock",
    "EsmComponent",
    "EsmBlock",
    "assert_constraint_groups",
    "COMPONENT_FACTORIES",
    "build_component",
    "factory",
    "bus",
    "profile",
    "capacity",
    "investment",
    "annuity_factor",
    "detect_solver",
    "manageSolverOpts",
]
