# -*- coding: utf-8 -*-
"""
The component registry: the mechanism only.

A spec `type` name is mapped onto a factory function, which returns the
solph nodes for one kit component. The factories themselves are plugins and
live in `esmkit.components` - this module must stay free of them so that
`core` never depends on any particular technology.

Each factory returns a *list* of solph nodes, because some kit components
expand into more than one: a grid connection with feed-in is a Source plus a
Sink, since solph prices energy per directed edge.

Writing a factory means using the four helpers below. They are public API:
`bus` resolves a bus name from the spec, `profile` resolves a scalar or a
`"@key"` reference against the inputs mapping, and `capacity`/`investment`
turn capacity parameters into either a number or a `solph.Investment`.
"""

from oemof import solph

from .investment import annuity_factor

#: registered factories, filled by @factory below
COMPONENT_FACTORIES = {}


def factory(name):
    """Registers a component factory under its spec type name."""

    def register(func):
        if name in COMPONENT_FACTORIES:
            raise ValueError("Duplicate component factory '{}'".format(name))
        COMPONENT_FACTORIES[name] = func
        return func

    return register


def build_component(ctype, name, params, buses, inputs, n_steps, step_size_h):
    """Dispatches to the registered factory of `ctype`."""
    if ctype not in COMPONENT_FACTORIES:
        raise ValueError(
            "Unknown component type '{}' for '{}'. Available: {}".format(
                ctype, name, sorted(COMPONENT_FACTORIES)
            )
        )
    return COMPONENT_FACTORIES[ctype](
        name, params, buses, inputs, n_steps, step_size_h
    )


def bus(buses, params, key, name):
    """Looks up a bus by the name given in the spec."""
    if key not in params:
        raise ValueError("Component '{}' needs a '{}'".format(name, key))
    label = params.pop(key)
    if label not in buses:
        raise ValueError(
            "Component '{}' refers to unknown bus '{}'".format(name, label)
        )
    return buses[label]


def profile(params, key, inputs, n_steps, name, default=None):
    """Resolves a spec value into a number or an array of length n_steps."""
    from .spec import resolve_profile

    value = params.pop(key, default)
    if value is None:
        return None
    return resolve_profile(value, inputs, n_steps, "{}.{}".format(name, key))


def investment(params, hours, wacc):
    """
    Turns capex/lifetime spec entries into a solph Investment.

    solph expects an already annualized cost per unit (`ep_costs`), so the
    annuity and the horizon scaling stay on the kit's side, which is what
    keeps a 72 hour design study trading capex against opex correctly.

    `hours` is the length of the horizon in hours, not the number of steps,
    so the scaling stays correct at sub-hourly resolution.
    """
    capex = params.pop("capex_per_unit", None)
    if capex is None:
        return None
    lifetime = params.pop("lifetime")
    opex_fix_share = params.pop("opex_fix_share", 0.0)
    year_fraction = hours / 8760.0
    ep_costs = capex * (annuity_factor(lifetime, wacc) + opex_fix_share) * year_fraction
    return solph.Investment(
        ep_costs=ep_costs,
        minimum=params.pop("min_capacity", 0.0),
        maximum=params.pop("max_capacity", None),
    )


def capacity(params, hours, wacc):
    """Either a fixed nominal capacity or a free Investment."""
    option = investment(params, hours, wacc)
    if option is not None:
        return option
    return params.pop("capacity", None)
