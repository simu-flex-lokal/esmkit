"""The serialisable system description: buses, components and their scalars,
plus the builder that turns one into an oemof-solph energy system."""

import copy
import json

import numpy as np
import pandas as pd
from oemof import solph

from .base import assert_constraint_groups
from .registry import COMPONENT_FACTORIES, build_component


# Marks a parameter as a reference into the ``inputs`` mapping ("@price").
PROFILE_PREFIX = "@"


# Bumped whenever the serialised layout changes; from_dict refuses others.
SPEC_VERSION = "1"


class SystemSpec(object):
    """Topology and scalar parameters of an energy system.

    Holds no time series: a parameter that varies over time is stored as a
    ``"@key"`` string resolved against the ``inputs`` mapping at build time.
    That is what keeps a spec JSON-serialisable and exchangeable.
    """

    def __init__(self, buses=None, components=None):
        self.buses = dict(buses or {})
        self.components = dict(components or {})

    def add_bus(self, name, carrier=None):
        """Declare a bus and return its name."""
        if name in self.buses:
            raise ValueError("Duplicate bus '{}'".format(name))
        self.buses[name] = {"carrier": carrier}
        return name

    def add_component(self, name, type, **params):
        """Declare a component of the given registered type and return its name."""
        if name in self.components:
            raise ValueError("Duplicate component '{}'".format(name))
        self.components[name] = dict(params, type=type)
        return name

    def copy(self):
        """Return an independent deep copy of this spec."""
        return SystemSpec(copy.deepcopy(self.buses), copy.deepcopy(self.components))

    def to_dict(self):
        """Return the spec as a plain, JSON-ready dict."""
        return {"version": SPEC_VERSION,
                "buses": copy.deepcopy(self.buses),
                "components": copy.deepcopy(self.components)}

    @classmethod
    def from_dict(cls, data):
        """Rebuild a spec from a dict, rejecting an unreadable spec version."""
        version = data.get("version", SPEC_VERSION)
        if str(version) != SPEC_VERSION:
            raise ValueError(
                "Spec version '{}' cannot be read by esmkit, which speaks "
                "version '{}'".format(version, SPEC_VERSION)
            )
        return cls(buses=data.get("buses"), components=data.get("components"))

    def to_json(self, **kwargs):
        """Serialise the spec to JSON; keyword arguments go to ``json.dumps``."""
        return json.dumps(self.to_dict(), sort_keys=True, **kwargs)

    @classmethod
    def from_json(cls, text):
        """Rebuild a spec from its JSON representation."""
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return "SystemSpec({} buses, {} components)".format(
            len(self.buses), len(self.components)
        )


def capacity_params(value):
    """Normalise a capacity given as a scalar or as a dict into a params dict."""
    if isinstance(value, dict):
        return dict(value)
    return {"capacity": value}


def resolve_profile(value, inputs, n_steps, name=""):
    """Resolve a parameter into a float array, or pass a scalar through.

    Args:
        value: a ``"@key"`` reference, an array-like, or a plain scalar.
        inputs: mapping the reference is looked up in.
        n_steps: length every resolved series must have.
        name: label used in error messages.

    Returns:
        A 1-D numpy array for series-valued parameters, the value itself for
        scalars.
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
    """Return the sorted input keys the spec references via ``"@key"``."""
    if isinstance(spec, dict):
        spec = SystemSpec.from_dict(spec)

    keys = set()
    for params in spec.components.values():
        for value in params.values():
            if isinstance(value, str) and value.startswith(PROFILE_PREFIX):
                keys.add(value[len(PROFILE_PREFIX):])
    return sorted(keys)


def _is_bus_key(key):
    """Report whether a parameter name is a bus reference by convention.

    A parameter refers to a bus if it is called ``bus``, starts with ``bus_``
    or ends with ``_bus``; there is no other declaration of bus-ness.
    """
    return key == "bus" or key.startswith("bus_") or key.endswith("_bus")


def validate(spec):
    """Check component types and bus references, and return the spec object."""
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
    """Raise if a referenced input is missing or has the wrong number of steps."""
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
    """Turn a spec plus its input time series into a solph energy system.

    Runs validate -> check_inputs -> create buses -> build components ->
    assert_constraint_groups.

    Args:
        spec: a ``SystemSpec`` or its dict form.
        inputs: mapping of input key to time series, for ``"@key"`` references.
        timeindex: pandas ``DatetimeIndex`` of the horizon.

    Returns:
        ``(es, nodes)`` - the ``solph.EnergySystem`` and a mapping of component
        name to the node it created (a list where it created several).
    """
    if isinstance(spec, dict):
        spec = SystemSpec.from_dict(spec)

    n_steps = len(timeindex)
    validate(spec)
    check_inputs(spec, inputs, n_steps)

    # infer_last_interval=True lets solph close the horizon from the index
    # spacing, so len(timeindex) steps are modelled rather than len - 1.
    es = solph.EnergySystem(timeindex=timeindex, infer_last_interval=True)

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
