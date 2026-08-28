# esmkit — Energy System Model Kit

`esmkit` is a small component kit for building [oemof-solph](https://oemof-solph.readthedocs.io)
energy system optimisation models. You describe a system as a `SystemSpec` — buses, components
and their scalar parameters, with every time-varying parameter stored as a `"@key"` reference
instead of the data itself. That keeps a spec JSON-serialisable and exchangeable: topology and
sizing travel as one small document, the time series travel separately. `build_system(spec,
inputs, timeindex)` turns it into a solph `EnergySystem`, `solve()` runs it, and `node_results()`
reads the flows back as pandas objects. The kit ships the stock components you need for a
building or district model (grid, PV, storage, heat pump, meter, demand) plus one genuinely
custom solph component: `ThermalZone5R1C`, a 5R1C thermal zone with free temperature states, so
a building's comfort band and thermal mass become a dispatchable flexibility resource inside the
same optimisation as storage and prices.

## Install

```bash
pip install esmkit[highs]          # HiGHS, the free solver, comes with the extra
uv sync --group dev --extra highs  # in a clone of this repo
```

Requires Python >= 3.12.

## Quickstart

```python
import numpy as np
import pandas as pd
from esmkit import SystemSpec, build_system, node_results, objective_value, solve

index = pd.date_range("2010-01-01", periods=24, freq="h")
sun = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)

spec = SystemSpec()
spec.add_bus("elec", carrier="electricity")
spec.add_component("grid", "grid", bus="elec", import_price=0.35, export_price=0.08)
spec.add_component("pv", "pv", bus="elec", specific_yield="@sun", capacity=3.0)
spec.add_component("demand", "demand", bus="elec", profile=2.0)

es, nodes = build_system(spec, {"sun": sun}, index)
model, _ = solve(es)
results = node_results(model, nodes, index=index)

print("pv generation:", round(results["pv"]["out_elec"].sum(), 2), "kWh")
print("grid import:  ", round(results["grid"]["out_elec"].sum(), 2), "kWh")
print("grid export:  ", round(results["grid"]["in_elec"].sum(), 2), "kWh")
print("cost:         ", round(objective_value(model), 2), "EUR")
```

```
pv generation: 22.79 kWh
grid import:   29.45 kWh
grid export:   4.23 kWh
cost:          9.97 EUR
```

## Built-in components

Parameters without a default are required. Any parameter may be given as a `"@key"` reference
if it varies over time.

| type | parameters |
| --- | --- |
| `demand` | `bus`, `profile` (the fixed load) |
| `source` | `bus`, `price` = `0.0`, `capacity` = unlimited |
| `grid` | `bus`, `import_price` = `0.0`, `export_price` = none (no export sink is created), `max_import` = unlimited, `max_export` = unlimited |
| `meter` | `bus_up`, `bus_down`, `price` = `0.0`, `efficiency` = `1.0`, `max_power` = unlimited |
| `pv` | `bus`, `specific_yield` (per unit of capacity), `curtailable` = `True`, plus capacity or investment keys |
| `heat_pump` | `bus_in`, `bus_out`, `cop` (steps with `cop <= 0` are unavailable), plus capacity or investment keys |
| `battery` | `bus_in`, `bus_out`, `self_discharge_per_h` = `0.0`, `fixed_losses_absolute` = `0.0`, `charge_efficiency` = `0.95`, `discharge_efficiency` = `0.95`, `soc_min` = `0.0`, `soc_max` = `1.0`, `charge_power_limit` / `discharge_power_limit` = unlimited, `balanced` = `True`, `initial_soc` = none (setting it makes the storage unbalanced), plus capacity or investment keys |
| `thermal_storage` | as `battery`, plus `standby_loss_kW` = `0.0` (fills `fixed_losses_absolute`) |
| `zone5r1c` | `heat_bus`, `cool_bus` = none (cooling is then fictitious and free), series `T_e`, `gain_mass`, `gain_surface`, `comfort_lb`, `comfort_ub`, scalars `H_ms`, `H_is`, `H_door`, `C_m`, `H` (a dict over `Walls`, `Roof`, `Floor`, `Windows`, `Ventilation`), `max_load`, `max_load_violation_penalty` = `100.0`, `initial_T_m` = none (the state is then cyclic), `design_capacity` = `max_load` |

## Input references

A string parameter starting with `@` is a reference into the `inputs` mapping passed to
`build_system`; anything else must be a scalar or an array of exactly `len(timeindex)` values.
`required_inputs(spec)` returns the sorted keys a spec needs — that is the data contract between
a spec and whoever supplies its time series. `check_inputs(spec, inputs, n_steps)` verifies the
mapping and names what is missing or wrongly sized; `build_system` calls it before building
anything.

## Investment

Any component that takes a `capacity` accepts `capex_per_unit` + `lifetime` instead, which turns
its size into a decision variable. Optional: `opex_fix_share` (fixed O&M as a share of capex),
`min_capacity`, `max_capacity` and `wacc` (default `0.0`). The annualised cost is scaled to the
modelled horizon, so a one-week run is not charged a full year of capex. See
`esmkit/core/registry.py`; the sized result comes back as `results[name]["capacity"]`.

## Adding your own component

Register a factory from anywhere — the library never needs to be touched:

```python
from esmkit import bus, capacity, factory

@factory("chp")
def _chp(name, params, buses, inputs, n_steps, step_size_h):
    ...  # return a list of solph nodes
```

`examples/district_chp.py` is a complete worked example. If your component carries its own pyomo
constraints, derive the block from `esmkit.EsmBlock` (or set `CONSTRAINT_GROUP = True` on it) —
solph silently drops a block without that flag, so the model still solves but your component
constrains nothing. `build_system` raises rather than let that happen.

## Solvers

`detect_solver()` honours the `SOLVER` environment variable first, then tries `gurobi`, `cplex`,
`scip`, `cbc`, `highs` in that order — commercial solvers first, HiGHS last as the free fallback.
Install one with the `[highs]` or `[gurobi]` extra, or bring your own (`apt install coinor-cbc`).
HiGHS is driven through `pyomo.contrib.appsi` and retries with progressively more conservative
settings if a run does not terminate optimally. `glpk` is explicitly refused: it is a MILP solver,
but it fails on badly conditioned models such as the 5R1C zone. See `esmkit/core/solverutils.py`.

## Reading the results

`node_results(model, nodes, index)` returns `{component: {series: values}}`. Two helpers turn
that into the series people actually ask for, both pure pandas:

```python
from esmkit import bus_flows, net_grid_exchange

flows = bus_flows(results, "elec")     # signed frame, positive = into the bus
net = net_grid_exchange(results)       # one series [kW], import positive
```

`net_grid_exchange` is the series a grid powerflow computation consumes per building.

## Plotting

`esmkit.plotting` draws a solved system: `plot_dispatch` (every flow on a bus, stacked),
`plot_storage`, `plot_grid_exchange` (with the tariff behind it) and `plot_price_response`.
Every function takes an optional `ax` and returns the axes it drew on; nothing calls
`plt.show()`. It needs matplotlib, which the kit itself does not require:

```bash
pip install esmkit[plots]
```

```python
from esmkit import plotting

with plotting.use_style():
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    plotting.plot_dispatch(results, "elec", ax=a1, window=("2010-01-04", "2010-01-11"))
    plotting.plot_grid_exchange(results, price=tariff, ax=a2, window=("2010-01-04", "2010-01-11"))
```

## Documentation

- [docs/01_grid_pv_battery.ipynb](docs/01_grid_pv_battery.ipynb) — grid, PV and a battery
- [docs/02_zone_heatpump.ipynb](docs/02_zone_heatpump.ipynb) — the 5R1C zone on a heat pump
- [docs/03_zone_heatpump_buffer_pv_battery.ipynb](docs/03_zone_heatpump_buffer_pv_battery.ipynb) — the full building
- [docs/model-deviations.md](docs/model-deviations.md) — known, deliberate deviations of the zone from its source papers

## Development

```bash
uv sync --group dev --extra highs
SOLVER=highs uv run pytest
```

## License

MIT. See [LICENSE](LICENSE).
