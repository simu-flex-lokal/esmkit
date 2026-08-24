# -*- coding: utf-8 -*-
"""
Component factories mapping spec types onto stock oemof-solph objects.

Everything here is a plugin: `core` never imports this module, the package
`__init__` does, and importing it is what registers the types. Adding a kind
of component to the kit means adding one factory - no changes to the
builder, the registry or the solver.

Each factory returns a *list* of solph nodes, because some kit components
expand into more than one: a grid connection with feed-in is a Source plus a
Sink, since solph prices energy per directed edge.
"""

from oemof import solph

from ..core import registry
from ..core.registry import factory


@factory("demand")
def _demand(name, params, buses, inputs, n_steps, step_size_h):
    """Inflexible load profile [kW]."""
    bus = registry.bus(buses, params, "bus", name)
    values = registry.profile(params, "profile", inputs, n_steps, name)
    # solph expresses a fixed profile as fix * nominal_capacity
    return [
        solph.components.Sink(
            label=name, inputs={bus: solph.Flow(fix=values, nominal_capacity=1.0)}
        )
    ]


@factory("source")
def _source(name, params, buses, inputs, n_steps, step_size_h):
    """Unlimited supply at a price [EUR/kWh] - the generic heat/cool supply."""
    bus = registry.bus(buses, params, "bus", name)
    price = registry.profile(params, "price", inputs, n_steps, name, default=0.0)
    nominal = params.pop("capacity", None)
    return [
        solph.components.Source(
            label=name,
            outputs={bus: solph.Flow(variable_costs=price, nominal_capacity=nominal)},
        )
    ]


@factory("grid")
def _grid(name, params, buses, inputs, n_steps, step_size_h):
    """
    Grid connection: import at a price and optionally export at a
    remuneration. Two solph nodes, because solph prices directed edges.
    """
    bus = registry.bus(buses, params, "bus", name)
    import_price = registry.profile(
        params, "import_price", inputs, n_steps, name, default=0.0
    )
    export_price = registry.profile(params, "export_price", inputs, n_steps, name)
    max_import = params.pop("max_import", None)
    max_export = params.pop("max_export", None)

    nodes = [
        solph.components.Source(
            label=name + "_import",
            outputs={
                bus: solph.Flow(
                    variable_costs=import_price, nominal_capacity=max_import
                )
            },
        )
    ]
    if export_price is not None:
        # negative variable costs = revenue
        nodes.append(
            solph.components.Sink(
                label=name + "_export",
                inputs={
                    bus: solph.Flow(
                        variable_costs=-1.0 * export_price
                        if not hasattr(export_price, "__len__")
                        else [-p for p in export_price],
                        nominal_capacity=max_export,
                    )
                },
            )
        )
    return nodes


@factory("meter")
def _meter(name, params, buses, inputs, n_steps, step_size_h):
    """
    Sub-meter between two buses: a lossless (or efficiency-scaled) transfer
    which prices the metered throughput. This is the primitive of a
    cascading metering concept, such as the reduced grid fee branch of
    section 14a EnWG.
    """
    up = registry.bus(buses, params, "bus_up", name)
    down = registry.bus(buses, params, "bus_down", name)
    price = registry.profile(params, "price", inputs, n_steps, name, default=0.0)
    efficiency = params.pop("efficiency", 1.0)
    max_power = params.pop("max_power", None)
    return [
        solph.components.Converter(
            label=name,
            inputs={up: solph.Flow(variable_costs=price)},
            outputs={down: solph.Flow(nominal_capacity=max_power)},
            conversion_factors={down: efficiency},
        )
    ]


@factory("pv")
def _pv(name, params, buses, inputs, n_steps, step_size_h):
    """PV generator driven by a specific yield profile [kW/kWp]."""
    bus = registry.bus(buses, params, "bus", name)
    yield_profile = registry.profile(params, "specific_yield", inputs, n_steps, name)
    curtailable = params.pop("curtailable", True)
    wacc = params.pop("wacc", 0.0)
    nominal = registry.capacity(params, n_steps * step_size_h, wacc)
    # curtailable: the profile is an upper bound; otherwise it is fixed
    limit = {"maximum": yield_profile} if curtailable else {"fix": yield_profile}
    return [
        solph.components.Source(
            label=name,
            outputs={bus: solph.Flow(nominal_capacity=nominal, **limit)},
        )
    ]


@factory("heat_pump")
def _heat_pump(name, params, buses, inputs, n_steps, step_size_h):
    """
    Heat pump coupling an electricity and a heat bus at a time varying COP.

    A COP of zero means the machine is off below its cut-off temperature.
    solph divides by the conversion factor, so those hours are handled by
    forcing the heat output to zero instead.
    """
    bus_in = registry.bus(buses, params, "bus_in", name)
    bus_out = registry.bus(buses, params, "bus_out", name)
    cop = registry.profile(params, "cop", inputs, n_steps, name)
    wacc = params.pop("wacc", 0.0)
    nominal = registry.capacity(params, n_steps * step_size_h, wacc)

    if hasattr(cop, "__len__"):
        conversion = [1.0 / c if c > 0 else 0.0 for c in cop]
        available = [1.0 if c > 0 else 0.0 for c in cop]
    else:
        conversion = 1.0 / cop if cop > 0 else 0.0
        available = 1.0 if cop > 0 else 0.0

    return [
        solph.components.Converter(
            label=name,
            inputs={bus_in: solph.Flow()},
            outputs={
                bus_out: solph.Flow(nominal_capacity=nominal, maximum=available)
                if nominal is not None
                else solph.Flow()
            },
            # conversion_factors are given per input: electricity = heat / COP
            conversion_factors={bus_in: conversion},
        )
    ]


@factory("battery")
def _battery(name, params, buses, inputs, n_steps, step_size_h):
    """Electrical storage. `balanced=True` reproduces a periodic SOC wrap."""
    return _storage(name, params, buses, inputs, n_steps, step_size_h)


@factory("thermal_storage")
def _thermal_storage(name, params, buses, inputs, n_steps, step_size_h):
    """
    Hot water storage. The standby heat loss maps onto solph's
    `fixed_losses_absolute`, which also supports relative losses.
    """
    params.setdefault("fixed_losses_absolute", params.pop("standby_loss_kW", 0.0))
    return _storage(name, params, buses, inputs, n_steps, step_size_h)


def _storage(name, params, buses, inputs, n_steps, step_size_h):
    """
    Shared storage factory. Charging and discharging bus are given
    separately, so a storage can bridge two buses (e.g. charge behind a
    sub-meter and discharge in front of it). Both are mandatory, also when
    they name the same bus.
    """
    bus_in = registry.bus(buses, params, "bus_in", name)
    bus_out = registry.bus(buses, params, "bus_out", name)
    wacc = params.pop("wacc", 0.0)
    nominal = registry.capacity(params, n_steps * step_size_h, wacc)
    losses = registry.profile(
        params, "fixed_losses_absolute", inputs, n_steps, name, default=0.0
    )

    kwargs = dict(
        loss_rate=params.pop("self_discharge_per_h", 0.0),
        fixed_losses_absolute=losses,
        inflow_conversion_factor=params.pop("charge_efficiency", 0.95),
        outflow_conversion_factor=params.pop("discharge_efficiency", 0.95),
        min_storage_level=params.pop("soc_min", 0.0),
        max_storage_level=params.pop("soc_max", 1.0),
        balanced=params.pop("balanced", True),
        inputs={
            bus_in: solph.Flow(
                nominal_capacity=params.pop("charge_power_limit", None)
            )
        },
        outputs={
            bus_out: solph.Flow(
                nominal_capacity=params.pop("discharge_power_limit", None)
            )
        },
        nominal_capacity=nominal,
    )
    initial = params.pop("initial_soc", None)
    if initial is not None:
        kwargs["initial_storage_level"] = initial
        kwargs["balanced"] = False

    return [solph.components.GenericStorage(label=name, **kwargs)]
