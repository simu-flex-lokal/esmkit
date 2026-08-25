from oemof import solph

from .investment import annuity_factor


COMPONENT_FACTORIES = {}


def factory(name):
    def register(func):
        if name in COMPONENT_FACTORIES:
            raise ValueError("Duplicate component factory '{}'".format(name))
        COMPONENT_FACTORIES[name] = func
        return func

    return register


def build_component(ctype, name, params, buses, inputs, n_steps, step_size_h):
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
    if key not in params:
        raise ValueError("Component '{}' needs a '{}'".format(name, key))
    label = params.pop(key)
    if label not in buses:
        raise ValueError(
            "Component '{}' refers to unknown bus '{}'".format(name, label)
        )
    return buses[label]


def profile(params, key, inputs, n_steps, name, default=None):
    from .spec import resolve_profile

    value = params.pop(key, default)
    if value is None:
        return None
    return resolve_profile(value, inputs, n_steps, "{}.{}".format(name, key))


def investment(params, hours, wacc):
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
    option = investment(params, hours, wacc)
    if option is not None:
        return option
    return params.pop("capacity", None)
