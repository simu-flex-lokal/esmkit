# -*- coding: utf-8 -*-
"""
esmkit - Energy System Model Kit.

Builds oemof-solph energy systems out of plain, serializable specifications.
A `SystemSpec` holds topology and scalar parameters; time series are
referenced by name (`"@elecPrice"`) and resolved against an inputs mapping at
build time, so one spec can be stored, diffed and varied across thousands of
systems.

    from esmkit import SystemSpec, build_system, solve, node_results

    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price="@price")
    spec.add_component("load", "demand", bus="elec", profile="@demand")

    es, nodes = build_system(spec, inputs, timeindex)
    model, _ = solve(es)
    results = node_results(model, nodes, index=timeindex)

Units throughout: kW, kWh, EUR, degC, hours.

Importing this package registers every component type shipped with the kit,
including the 5R1C thermal zone. The kit itself knows nothing about
buildings: the zone takes explicit heat transfer coefficients, gains and a
comfort band, never a building configuration.
"""

from . import components  # noqa: F401  (imported for factory registration)
from .components import ThermalZone5R1C, ThermalZone5R1CBlock
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
