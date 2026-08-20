# -*- coding: utf-8 -*-
"""
An energy system with no building in it.

A small district heating scheme: a gas-fired CHP unit serves an electricity
and a heat demand, backed by a boiler and a priced grid connection. It
exists to make one point concrete - esmkit is an energy system kit, not a
building model. Nothing here knows about envelopes, weather or occupants,
and no component of the kit requires them.

It also doubles as the tutorial from `docs/kit.md` on adding a technology:
the CHP is registered from *outside* the package in one function, and is
then usable in a spec exactly like a built-in type.

Run:
    SOLVER=highs uv run python examples/district_chp.py
"""

import numpy as np
import pandas as pd
from oemof import solph

from esmkit import (
    SystemSpec,
    build_system,
    bus,
    capacity,
    factory,
    node_results,
    objective_value,
    solve,
)


# --- the whole addition -------------------------------------------------


@factory("chp")
def _chp(name, params, buses, inputs, n_steps, step_size_h):
    """
    Gas-fired combined heat and power unit.

    Spec parameters
    ---------------
    bus_fuel, bus_elec, bus_heat: str, required
        Bus names.
    capacity: float, optional
        Fuel input capacity [kW]. Alternatively capex_per_unit/lifetime/
        max_capacity for an investment decision.
    electrical_efficiency, thermal_efficiency: float, optional
    """
    fuel = bus(buses, params, "bus_fuel", name)
    elec = bus(buses, params, "bus_elec", name)
    heat = bus(buses, params, "bus_heat", name)
    wacc = params.pop("wacc", 0.0)
    nominal = capacity(params, n_steps * step_size_h, wacc)

    return [
        solph.components.Converter(
            label=name,
            inputs={fuel: solph.Flow(nominal_capacity=nominal)},
            outputs={elec: solph.Flow(), heat: solph.Flow()},
            conversion_factors={
                elec: params.pop("electrical_efficiency", 0.35),
                heat: params.pop("thermal_efficiency", 0.50),
            },
        )
    ]


# --- using it -----------------------------------------------------------


def main():
    n = 48
    index = pd.date_range("2010-01-01", periods=n, freq="h")

    # expensive grid electricity in the evening: the CHP should run then
    inputs = {
        "elecPrice": np.where((index.hour >= 17) & (index.hour < 21), 0.45, 0.12),
        "heatDemand": np.full(n, 4.0),
        "elecDemand": np.full(n, 1.5),
    }

    spec = SystemSpec()
    spec.add_bus("gas", carrier="gas")
    spec.add_bus("elec", carrier="electricity")
    spec.add_bus("heat", carrier="heat")

    spec.add_component("gas_grid", "source", bus="gas", price=0.08)
    spec.add_component("grid", "grid", bus="elec", import_price="@elecPrice")
    spec.add_component("boiler_backup", "source", bus="heat", price=0.20)
    spec.add_component("heat_demand", "demand", bus="heat", profile="@heatDemand")
    spec.add_component("elec_demand", "demand", bus="elec", profile="@elecDemand")

    # the new component, used exactly like any built-in one
    spec.add_component(
        name="chp",
        type="chp",
        bus_fuel="gas",
        bus_elec="elec",
        bus_heat="heat",
        capacity=12.0,
        electrical_efficiency=0.35,
        thermal_efficiency=0.50,
    )

    es, nodes = build_system(spec, inputs, index)
    model, _ = solve(es)
    results = node_results(model, nodes, index=index)

    fuel = results["chp"]["in_gas"]
    power = results["chp"]["out_elec"]
    heat = results["chp"]["out_heat"]
    imported = results["grid"]["out_elec"]
    backup = results["boiler_backup"]["out_heat"]

    print("objective            : {:8.2f} EUR".format(objective_value(model)))
    print("gas input            : {:8.2f} kWh".format(fuel.sum()))
    print("electricity generated: {:8.2f} kWh".format(power.sum()))
    print("heat generated       : {:8.2f} kWh".format(heat.sum()))
    print("backup boiler heat   : {:8.2f} kWh".format(backup.sum()))
    print("grid import          : {:8.2f} kWh".format(imported.sum()))
    print()
    # the conversion factors hold by construction - solph derives both
    # outputs from the single fuel input
    print(
        "electricity == fuel * 0.35 : {:.6f} == {:.6f}".format(
            power.sum(), fuel.sum() * 0.35
        )
    )
    print(
        "heat        == fuel * 0.50 : {:.6f} == {:.6f}".format(
            heat.sum(), fuel.sum() * 0.50
        )
    )
    print(
        "heat covered by the CHP    : {:.1%}".format(
            heat.sum() / (heat.sum() + backup.sum())
        )
    )


if __name__ == "__main__":
    main()
