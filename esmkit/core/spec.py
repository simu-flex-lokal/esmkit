# -*- coding: utf-8 -*-
"""
Declarative energy system specification and the builder which turns it into
an oemof-solph EnergySystem.

A `SystemSpec` is deliberately a plain, serializable description of the
*topology plus scalar parameters* - never of the time series. Profiles are
referenced by name (`"@elecPrice"`) and resolved against an inputs mapping
at build time. That boundary is what keeps a spec small enough to store,
diff and vary across thousands of systems, instead of dragging 8760-element
arrays through the description.

The inputs mapping is anything dict-like: `Mapping[str, float | Sequence[float]]`.
The kit never inspects it beyond the keys a spec asks for, which is what
`required_inputs` and `check_inputs` report.
"""

import copy
import json

import numpy as np
import pandas as pd
from oemof import solph

from .base import assert_constraint_groups
from .registry import COMPONENT_FACTORIES, build_component

#: Prefix marking a value which is resolved from the inputs mapping.
PROFILE_PREFIX = "@"

#: Schema version written into `to_dict()`; bumped when the spec layout changes.
SPEC_VERSION = "1"


class SystemSpec(object):
    """
    Topology and scalar parameters of an energy system.

    Parameters
    ----------
    buses: dict, required
        Bus name -> dict, currently only {"carrier": str} which is
        documentation for the reader; solph buses are carrier agnostic.
    components: dict, required
        Component name -> dict with a "type" key naming a factory in
        `registry.COMPONENT_FACTORIES` plus that factory's parameters.
        Bus-valued parameters hold bus *names*, profile-valued parameters
        either a number, a sequence, or a "@key" reference into the inputs
        mapping.
    """

    def __init__(self, buses=None, components=None):
        self.buses = dict(buses or {})
        self.components = dict(components or {})

    # --- authoring helpers ------------------------------------------

    def add_bus(self, name, carrier=None):
        if name in self.buses:
            raise ValueError("Duplicate bus '{}'".format(name))
        self.buses[name] = {"carrier": carrier}
        return name

    def add_component(self, name, type, **params):
        if name in self.components:
            raise ValueError("Duplicate component '{}'".format(name))
        self.components[name] = dict(params, type=type)
        return name

    def copy(self):
        return SystemSpec(copy.deepcopy(self.buses), copy.deepcopy(self.components))

    # --- serialization ------------------------------------------------

    def to_dict(self):
        return {"version": SPEC_VERSION,
                "buses": copy.deepcopy(self.buses),
                "components": copy.deepcopy(self.components)}

    @classmethod
    def from_dict(cls, data):
        version = data.get("version", SPEC_VERSION)
        if str(version) != SPEC_VERSION:
            raise ValueError(
                "Spec version '{}' cannot be read by esmkit, which speaks "
                "version '{}'".format(version, SPEC_VERSION)
            )
        return cls(buses=data.get("buses"), components=data.get("components"))

    def to_json(self, **kwargs):
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, text):
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return "SystemSpec({} buses, {} components)".format(
            len(self.buses), len(self.components)
        )


def capacity_params(value):
    """
    A number becomes a fixed capacity, a dict an investment decision.

    The two forms a capacity parameter may take, in one place, so callers
    which offer the same choice per component do not each reinvent it.
    """
    if isinstance(value, dict):
        return dict(value)
    return {"capacity": value}


def resolve_profile(value, inputs, n_steps, name=""):
    """
    Resolves a spec value into a number or a float array of length n_steps.

    A string starting with "@" is looked up in the inputs mapping; anything
    else is passed through (scalars stay scalars so solph can broadcast
    them itself).
    """
    if isinstance(value, str):
        if not value.startswith(PROFILE_PREFIX):
            raise ValueError(
                "String parameter '{}' for {} must be an input reference "
                "starting with '{}'".format(value, name, PROFILE_PREFIX)
            )
        key = value[len(PROFILE_PREFIX):]
        if key not in inputs or inputs[key] is None:
            raise KeyError(
                "Profile reference '{}' of {} is not in the inputs "
                "mapping".format(value, name)
            )
        value = inputs[key]

    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    if isinstance(value, (list, tuple, np.ndarray)):
        values = np.asarray(value, dtype=float).ravel()
        if len(values) != n_steps:
            raise ValueError(
                "Profile of {} has {} values, expected {}".format(
                    name, len(values), n_steps
                )
            )
        return values
    return value


def required_inputs(spec):
    """
    Input keys a spec will look up when it is built.

    Every `"@key"` a component carries is resolved out of the inputs mapping
    by `resolve_profile`, so this is the complete list of what a caller has
    to provide - or has to have simulated - before `build_system` can
    succeed.

    Parameters
    ----------
    spec: SystemSpec or dict, required

    Returns
    -------
    Sorted list of input keys, without the "@" prefix.
    """
    if isinstance(spec, dict):
        spec = SystemSpec.from_dict(spec)

    keys = set()
    for params in spec.components.values():
        for value in params.values():
            if isinstance(value, str) and value.startswith(PROFILE_PREFIX):
                keys.add(value[len(PROFILE_PREFIX):])
    return sorted(keys)


def _is_bus_key(key):
    """Spec parameter names which hold a bus name: `bus`, `bus_in`, `heat_bus`."""
    return key == "bus" or key.startswith("bus_") or key.endswith("_bus")


def validate(spec):
    """
    Checks a spec against the registry before anything is built.

    Reports every problem at once rather than failing on the first one, so
    a hand-written or machine-generated spec can be fixed in one pass.

    Raises
    ------
    ValueError listing unknown component types, components without a type,
    and bus names referenced by a component but never declared.
    """
    if isinstance(spec, dict):
        spec = SystemSpec.from_dict(spec)

    problems = []
    for name, params in spec.components.items():
        ctype = params.get("type")
        if ctype is None:
            problems.append("component '{}' has no 'type'".format(name))
        elif ctype not in COMPONENT_FACTORIES:
            problems.append(
                "component '{}' has unknown type '{}' (available: {})".format(
                    name, ctype, ", ".join(sorted(COMPONENT_FACTORIES))
                )
            )
        for key, value in params.items():
            if not _is_bus_key(key):
                continue
            if value is not None and value not in spec.buses:
                problems.append(
                    "component '{}' refers to undeclared bus '{}'".format(name, value)
                )

    if problems:
        raise ValueError("Invalid spec: " + "; ".join(problems))
    return spec


def check_inputs(spec, inputs, n_steps):
    """
    Checks that the inputs mapping satisfies a spec, before building.

    `resolve_profile` would report the same problems, but one at a time and
    halfway through model construction. This says up front which keys are
    missing and which have the wrong length.

    Raises
    ------
    KeyError if a required key is missing or None.
    ValueError if a provided series has a length other than `n_steps`.
    """
    missing = []
    wrong_length = []
    for key in required_inputs(spec):
        if key not in inputs or inputs[key] is None:
            missing.append(key)
            continue
        value = inputs[key]
        if isinstance(value, (pd.Series, pd.DataFrame)):
            value = value.values
        if hasattr(value, "__len__") and len(value) != n_steps:
            wrong_length.append("{} ({} of {})".format(key, len(value), n_steps))

    if missing:
        raise KeyError(
            "Inputs mapping is missing: {}".format(", ".join(sorted(missing)))
        )
    if wrong_length:
        raise ValueError(
            "Inputs with the wrong number of steps: {}".format(
                ", ".join(sorted(wrong_length))
            )
        )


def build_system(spec, inputs, timeindex):
    """
    Builds a solph EnergySystem from a spec and an inputs mapping.

    Parameters
    ----------
    spec: SystemSpec or dict, required
    inputs: Mapping, required
        Source of every `"@key"` profile reference. Values are floats or
        sequences of length `len(timeindex)`.
    timeindex: pandas.DatetimeIndex, required
        The model horizon. Explicit rather than derived, because the kit
        has no opinion on where a caller keeps its time axis.

    Returns
    -------
    (solph.EnergySystem, dict) - the system and a mapping of component name
    to the created solph node(s); components which expand into several nodes
    (a grid connection becomes a Source and a Sink) map to a list.
    """
    if isinstance(spec, dict):
        spec = SystemSpec.from_dict(spec)

    n_steps = len(timeindex)
    validate(spec)
    check_inputs(spec, inputs, n_steps)

    # infer_last_interval=True is required: solph 0.6 defaults to False,
    # which would turn 8760 time stamps into 8759 intervals and silently
    # drop the last hour of the year.
    es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)

    # step size in hours; investment costing is scaled by the horizon in
    # hours rather than in steps, so it stays correct below hourly resolution
    step_size_h = float(es.timeincrement[0])

    buses = {}
    for name in spec.buses:
        buses[name] = solph.Bus(label=name)
    es.add(*buses.values())

    nodes = {}
    for name, params in spec.components.items():
        params = dict(params)
        ctype = params.pop("type")
        created = build_component(
            ctype,
            name,
            params,
            buses=buses,
            inputs=inputs,
            n_steps=n_steps,
            step_size_h=step_size_h,
        )
        es.add(*created)
        nodes[name] = created[0] if len(created) == 1 else created

    assert_constraint_groups(nodes)

    return es, nodes
