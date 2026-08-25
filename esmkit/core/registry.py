"""Registry of component factories plus the helpers a factory uses to read its
parameters.

Every helper here (``bus``, ``profile``, ``capacity``, ``investment``) *pops*
the keys it reads out of ``params``, so whatever a factory has not consumed is
what is left in ``params`` afterwards - components can hand that remainder
straight to a constructor as ``**params``.
"""

from oemof import solph

from .investment import annuity_factor


COMPONENT_FACTORIES = {}


def factory(name):
    """Register the decorated function as the builder for component type ``name``.

    The builder is called as ``func(name, params, buses, inputs, n_steps,
    step_size_h)`` and must return a *list* of solph nodes (a component may
    expand into several, as ``grid`` does).

    Args:
        name: the component ``type`` used in a spec. Must not already be taken.
    """
    def register(func):
        if name in COMPONENT_FACTORIES:
            raise ValueError("Duplicate component factory '{}'".format(name))
        COMPONENT_FACTORIES[name] = func
        return func

    return register


def build_component(ctype, name, params, buses, inputs, n_steps, step_size_h):
    """Call the registered factory for ``ctype`` and return the nodes it creates.

    Args:
        ctype: registered component type.
        name: component name; becomes the label of the node(s).
        params: the component's parameters, minus ``type``. Mutated - the
            factory pops what it consumes.
        buses: mapping of bus name to the ``solph.Bus`` already created.
        inputs: mapping of input key to time series, for ``"@key"`` references.
        n_steps: number of time steps every profile must have.
        step_size_h: length of one time step in hours.

    Returns:
        The list of solph nodes returned by the factory.
    """
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
    """Pop a required bus reference from ``params`` and return the bus object."""
    if key not in params:
        raise ValueError("Component '{}' needs a '{}'".format(name, key))
    label = params.pop(key)
    if label not in buses:
        raise ValueError(
            "Component '{}' refers to unknown bus '{}'".format(name, label)
        )
    return buses[label]


def profile(params, key, inputs, n_steps, name, default=None):
    """Pop a parameter from ``params`` and resolve it into a scalar or a series."""
    # Imported here rather than at module level: spec imports registry.
    from .spec import resolve_profile

    value = params.pop(key, default)
    if value is None:
        return None
    return resolve_profile(value, inputs, n_steps, "{}.{}".format(name, key))


def investment(params, hours, wacc):
    """Build a ``solph.Investment`` from the capex keys, or None if there is none.

    Args:
        params: component parameters; ``capex_per_unit``, ``lifetime``,
            ``opex_fix_share``, ``min_capacity`` and ``max_capacity`` are popped.
        hours: length of the modelled horizon in hours.
        wacc: weighted average cost of capital used for the annuity.
    """
    capex = params.pop("capex_per_unit", None)
    if capex is None:
        return None
    lifetime = params.pop("lifetime")
    opex_fix_share = params.pop("opex_fix_share", 0.0)
    # Annualised costs must be scaled to the modelled horizon, otherwise a
    # short run would pay a full year of capex against a fraction of the yield.
    year_fraction = hours / 8760.0
    ep_costs = capex * (annuity_factor(lifetime, wacc) + opex_fix_share) * year_fraction
    return solph.Investment(
        ep_costs=ep_costs,
        minimum=params.pop("min_capacity", 0.0),
        maximum=params.pop("max_capacity", None),
    )


def capacity(params, hours, wacc):
    """Return an ``Investment`` if capex was given, else the fixed ``capacity``."""
    option = investment(params, hours, wacc)
    if option is not None:
        return option
    return params.pop("capacity", None)
