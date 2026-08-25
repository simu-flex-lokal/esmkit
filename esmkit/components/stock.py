"""Stock solph components - the everyday building blocks of a system spec.

Units are kW, kWh and EUR/kWh throughout, and any parameter may be given
either as a scalar or as an ``"@key"`` reference resolved against the
``inputs`` mapping.
"""

from oemof import solph

from ..core import registry
from ..core.registry import factory


@factory("demand")
def _demand(name, params, buses, inputs, n_steps, step_size_h):
    """Fixed consumption: a ``Sink`` whose inflow is pinned to a profile.

    Consumes ``bus`` and ``profile`` (kW per step).
    """
    bus = registry.bus(buses, params, "bus", name)
    values = registry.profile(params, "profile", inputs, n_steps, name)

    return [
        solph.components.Sink(
            label=name, inputs={bus: solph.Flow(fix=values, nominal_capacity=1.0)}
        )
    ]


@factory("source")
def _source(name, params, buses, inputs, n_steps, step_size_h):
    """Unconstrained supply: a ``Source`` feeding one bus.

    Consumes ``bus``, ``price`` (EUR/kWh, free by default) and ``capacity``
    (kW, unlimited by default).
    """
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
    """Grid connection: an import ``Source``, plus an export ``Sink`` when
    ``export_price`` is given.

    Consumes ``bus``, ``import_price`` and ``export_price`` (EUR/kWh) and
    ``max_import`` / ``max_export`` (kW).
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
        nodes.append(
            solph.components.Sink(
                label=name + "_export",
                inputs={
                    bus: solph.Flow(
                        # Revenue is a cost with the opposite sign, so the
                        # export price is negated; the branch only picks the
                        # right negation for a scalar or for a sequence.
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
    """Metered link between two buses: a ``Converter`` from up to down.

    Consumes ``bus_up``, ``bus_down``, ``price`` (EUR/kWh, charged on the
    inflow), ``efficiency`` and ``max_power`` (kW on the outflow).
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
    """Photovoltaics: a ``Source`` driven by a specific yield profile.

    Consumes ``bus``, ``specific_yield`` (kW per kW installed),
    ``curtailable`` (``maximum`` on the flow when true, ``fix`` when not),
    ``wacc`` and either ``capacity`` (kW) or the capex keys that turn the
    size into a solph ``Investment``.
    """
    bus = registry.bus(buses, params, "bus", name)
    yield_profile = registry.profile(params, "specific_yield", inputs, n_steps, name)
    curtailable = params.pop("curtailable", True)
    wacc = params.pop("wacc", 0.0)
    nominal = registry.capacity(params, n_steps * step_size_h, wacc)

    limit = {"maximum": yield_profile} if curtailable else {"fix": yield_profile}
    return [
        solph.components.Source(
            label=name,
            outputs={bus: solph.Flow(nominal_capacity=nominal, **limit)},
        )
    ]


@factory("heat_pump")
def _heat_pump(name, params, buses, inputs, n_steps, step_size_h):
    """Heat pump: a ``Converter`` lifting ``bus_in`` power onto ``bus_out``.

    Consumes ``bus_in``, ``bus_out``, ``cop`` (-), ``wacc`` and either
    ``capacity`` (kW of output) or the capex keys that turn the size into a
    solph ``Investment``.
    """
    bus_in = registry.bus(buses, params, "bus_in", name)
    bus_out = registry.bus(buses, params, "bus_out", name)
    cop = registry.profile(params, "cop", inputs, n_steps, name)
    wacc = params.pop("wacc", 0.0)
    nominal = registry.capacity(params, n_steps * step_size_h, wacc)

    # solph keys conversion_factors on the flow they belong to, and here that
    # is the input flow, so the factor is the electricity per unit of heat,
    # 1/COP. Hours whose COP is not positive have no defined inverse, so they
    # are switched off through maximum=available instead.
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

            conversion_factors={bus_in: conversion},
        )
    ]


@factory("battery")
def _battery(name, params, buses, inputs, n_steps, step_size_h):
    """Electricity storage: a ``GenericStorage`` with the shared storage
    parameters listed on ``_storage``."""
    return _storage(name, params, buses, inputs, n_steps, step_size_h)


@factory("thermal_storage")
def _thermal_storage(name, params, buses, inputs, n_steps, step_size_h):
    """Heat storage: as ``battery``, plus a constant standby loss.

    Consumes ``standby_loss_kW`` on top of the ``_storage`` parameters.
    """
    # standby_loss_kW is only a domain-flavoured alias for solph's
    # fixed_losses_absolute; both name the same kW lost per step.
    params.setdefault("fixed_losses_absolute", params.pop("standby_loss_kW", 0.0))
    return _storage(name, params, buses, inputs, n_steps, step_size_h)


def _storage(name, params, buses, inputs, n_steps, step_size_h):
    """Build the ``GenericStorage`` behind ``battery`` and ``thermal_storage``.

    Consumes ``bus_in``, ``bus_out``, ``self_discharge_per_h`` (1/h),
    ``fixed_losses_absolute`` (kW), ``charge_efficiency`` and
    ``discharge_efficiency``, ``soc_min`` / ``soc_max`` (fractions of the
    capacity), ``charge_power_limit`` / ``discharge_power_limit`` (kW),
    ``balanced``, ``initial_soc``, ``wacc`` and either ``capacity`` (kWh) or
    the capex keys that turn the size into a solph ``Investment``.
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
        # A given starting level only means something on an open horizon, so
        # it replaces solph's default periodic balance (end == start) rather
        # than being fought by it.
        kwargs["initial_storage_level"] = initial
        kwargs["balanced"] = False

    return [solph.components.GenericStorage(label=name, **kwargs)]
