"""A third-party component: a CHP unit registered from outside the library.

Nothing in ``esmkit`` knows about combined heat and power. This script defines
a ``chp`` component in user code, registers it with ``@factory("chp")`` and then
uses it in a spec exactly like a built-in type - no fork, no patch, no import
of anything private. That is the extension point: a factory is a plain function
returning solph nodes, and the registry is global.

The system is a small district: a gas grid, an electricity grid with a
time-varying import price, a backup boiler, a heat and an electricity demand,
and the CHP tying gas, power and heat together. Run it with
``SOLVER=highs uv run python examples/district_chp.py``.
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


@factory("chp")
def _chp(name, params, buses, inputs, n_steps, step_size_h):
    """Build a gas-fired CHP unit: one converter, fuel in, power and heat out.

    Parameters: ``bus_fuel``, ``bus_elec``, ``bus_heat``, a fixed ``capacity``
    (or the ``capex_per_unit``/``lifetime`` investment keys plus ``wacc``), and
    the ``electrical_efficiency`` / ``thermal_efficiency`` of the unit.
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


def main():
    n = 48
    index = pd.date_range("2010-01-01", periods=n, freq="h")

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
