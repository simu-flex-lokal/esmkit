# The kit

esmkit builds [oemof-solph](https://oemof-solph.readthedocs.io/) energy systems out of plain,
serializable descriptions. Everything except the 5R1C thermal zone is stock solph — the kit
contributes the description format, the assembly and the solver handling around it.

- [1. Minimal example](#1-minimal-example)
- [2. Specs](#2-specs)
- [3. The components](#3-the-components)
- [4. Results](#4-results)
- [5. Tutorial: adding a component](#5-tutorial-adding-a-component)
- [6. Gotchas](#6-gotchas)

Units throughout: **kW**, **kWh**, **EUR**, **°C**, **hours**.

---

## 1. Minimal example

```python
import numpy as np
import pandas as pd
from esmkit import SystemSpec, build_system, solve, node_results

index = pd.date_range("2010-01-01", periods=24, freq="h")
inputs = {"price": np.full(24, 0.35), "load": np.full(24, 2.0)}

spec = SystemSpec()
spec.add_bus("elec", carrier="electricity")
spec.add_component("grid", "grid", bus="elec", import_price="@price")
spec.add_component("demand", "demand", bus="elec", profile="@load")

es, nodes = build_system(spec, inputs, index)
model, _ = solve(es)
results = node_results(model, nodes, index=index)

results["grid"]["out_elec"]      # import [kW]
results["demand"]["in_elec"]     # load [kW]
```

Three arguments, three contracts: the **spec** (topology and scalars), the **inputs** mapping
(everything time-varying), and the **time index**.

---

## 2. Specs

A `SystemSpec` describes topology plus scalar parameters. It is plain data:

```python
{
  "version": "1",
  "buses": {"elec": {"carrier": "electricity"}, "heat": {"carrier": "heat"}},
  "components": {
    "grid":   {"type": "grid",      "bus": "elec", "import_price": 0.35, "export_price": 0.08},
    "hp":     {"type": "heat_pump", "bus_in": "elec", "bus_out": "heat", "cop": "@cop"},
    "demand": {"type": "demand",    "bus": "elec", "profile": "@elecLoad"}
  }
}
```

**Time series never live in the spec.** A `"@key"` string is resolved against the inputs mapping
at build time. That boundary is deliberate: it keeps a spec small enough to store, diff and vary
across thousands of systems, and it is what makes `SystemSpec.to_json()` round-trip. Scalars pass
through untouched, so `profile=1.0` is a valid constant 1 kW demand.

`required_inputs(spec)` returns exactly which keys a spec will look up, which makes that contract
inspectable before anything is built:

```python
>>> required_inputs(spec)
['cop', 'elecLoad']
```

### What is checked, and when

`build_system` runs two checks before it constructs anything, both callable on their own:

- `validate(spec)` — unknown component types, components without a type, and bus names a
  component refers to but nobody declared. Every problem is reported at once, so a generated spec
  can be fixed in one pass.
- `check_inputs(spec, inputs, n_steps)` — which required keys are missing, and which series have
  the wrong length.

The `version` field is what lets a reader refuse a layout it does not understand instead of
misinterpreting it. It is checked on `from_dict`/`from_json`.

### Capacities

Every capacity parameter takes either a number (fixed) or investment parameters:

```python
spec.add_component("pv", "pv", bus="elec", specific_yield="@pv_yield",
                   capex_per_unit=1200.0, lifetime=25.0, max_capacity=20.0, wacc=0.06)
```

`capacity_params(value)` turns whichever of the two forms a caller holds into the right keyword
arguments. The annuity and the horizon scaling stay on the kit's side, because solph expects an
already-annualized `ep_costs`:

$$\text{ep\_costs} = \text{capex}\cdot\big(a(n,i) + \text{opex}_{\text{fix}}\big)\cdot
\underbrace{\tfrac{H}{8760}}_{\text{year fraction}}, \qquad
a(n,i) = \frac{i(1+i)^n}{(1+i)^n - 1}$$

The year fraction is why a 72-hour design study still trades capex against opex correctly, at any
resolution.

---

## 3. The components

Every factory but the last returns stock solph objects:

| spec `type` | solph | notes |
|---|---|---|
| `demand` | `Sink(Flow(fix=…, nominal_capacity=1))` | inflexible load |
| `source` | `Source(Flow(variable_costs=…))` | generic priced supply |
| `grid` | `Source` **+** `Sink` | two nodes: solph prices directed edges |
| `meter` | `Converter` between two buses | the sub-metering primitive |
| `pv` | `Source(Flow(maximum=yield, …))` | `curtailable=False` uses `fix=` instead |
| `heat_pump` | `Converter(conversion_factors={bus_in: 1/COP})` | couples two carriers |
| `battery` | `GenericStorage` | `balanced=True` ≙ periodic SOC wrap |
| `thermal_storage` | `GenericStorage(fixed_losses_absolute=…)` | standby loss |
| `zone5r1c` | **`ThermalZone5R1C`** | the one custom component — see [`zone5r1c.md`](zone5r1c.md) |

`esmkit.core` never imports any of them: components are plugins that register themselves when
`esmkit.components` is imported, and `test/test_seam.py` enforces that direction. Someone using
esmkit as a plain ESM tool simply never writes `type: "zone5r1c"` in a spec.

---

## 4. Results

`node_results(model, nodes)` returns one dict per *kit component*, not per solph node — a grid
connection's import Source and export Sink are merged under the name `"grid"`:

- `in_<bus>` / `out_<bus>` — flows [kW]
- `capacity` — invested capacity, where an investment was declared
- `storage_content` — SOC [kWh] for storages
- `labels` — the solph node labels the component expanded into

A component may report its own shape instead, by implementing `results(model, index)`. That hook
is why result extraction needs no knowledge of any particular component type; the thermal zone
uses it to report `timeseries` / `static` / `max_load_violation`.

---

## 5. Tutorial: adding a component

Most additions are **one factory function**, because solph already has the component. A combined
heat and power unit, for instance:

```python
from oemof import solph
from esmkit import bus, capacity, factory


@factory("chp")
def _chp(name, params, buses, inputs, n_steps, step_size_h):
    """Gas-fired CHP: one fuel input, electricity and heat output."""
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
```

That is the whole change, and it works from outside the package. `build_component` picks the
factory up, so `spec.add_component("chp", "chp", bus_fuel="gas", bus_elec="elec",
bus_heat="heat")` works immediately, including profile references and investment parameters.
Runnable: [`examples/district_chp.py`](../examples/district_chp.py).

The four helpers a factory uses are public API: `bus` resolves a bus name, `profile` resolves a
scalar or a `"@key"` reference, and `capacity` / `investment` turn capacity parameters into a
number or a `solph.Investment`.

### When you actually need a custom solph component

Only when the physics is not expressible as flows between buses — free state variables, coupled
algebraic nodes, bounds on something that is not a flow. The 5R1C zone is the kit's only such
case.

A custom component is always **two classes that have to find each other**: a node that hangs in
the graph, and a block that holds the pyomo variables and constraints for *all* nodes of that
type. `constraint_group()` is the link. Derive them from the bases rather than from
`Node`/`ScalarBlock` directly:

```python
import pyomo.environ as po
from oemof.solph import Flow
from esmkit import EsmBlock, EsmComponent


class MyComponent(EsmComponent):
    def __init__(self, label, bus, **params):
        super().__init__(label=label, inputs={bus: Flow()})
        ...
    def constraint_group(self):
        return MyComponentBlock


class MyComponentBlock(EsmBlock):
    def _create(self, group=None):
        if group is None:
            return
        m = self.parent_block()
        self.UNITS = po.Set(initialize=list(group), ordered=True)
        self.state = po.Var(self.UNITS, m.TIMESTEPS, within=po.Reals)
        self.balance = po.Constraint(self.UNITS, m.TIMESTEPS, rule=...)

    def _objective_expression(self):
        return sum(...)
```

`EsmBlock` carries `CONSTRAINT_GROUP = True`, so the attribute solph looks for cannot be
forgotten, and both bases declare their central method abstract — a node without
`constraint_group()` or a block without `_create()` fails at instantiation instead of at solve
time. If you bypass the bases anyway, `build_system` still refuses to return a system whose
custom block would be ignored (`assert_constraint_groups`).

Read the bus coupling as `m.flow[bus, node, t]` (into the node) or `m.flow[node, bus, t]` (out of
it); do not create your own flow variables.

---

## 6. Gotchas

- **`CONSTRAINT_GROUP = True` is mandatory on a custom block.** `solph.Model.__init__` collects
  custom constraint groups with `if hasattr(i, "CONSTRAINT_GROUP")`, on top of the stock blocks
  already listed in `Model.CONSTRAINT_GROUPS`. Without the attribute your block is **silently
  ignored** — the model solves, the objective looks plausible, and your component imposes no
  constraints whatsoever. This is the single nastiest failure mode here, which is why
  `EsmBlock` carries the attribute and `build_system` rejects a block that lacks it. Note the
  test is `hasattr`, not truth: `CONSTRAINT_GROUP = False` switches nothing off.
- **`infer_last_interval=True` is mandatory.** solph 0.6 defaults it to `False`, which turns 8760
  time stamps into 8759 intervals and silently drops the last hour of the year. `build_system`
  sets it; if you construct an `EnergySystem` by hand, you must too.
- **Do not use `solph.Model.solve()`.** A model with free state variables is badly conditioned:
  `glpk` fails on it outright, and HiGHS needs the interior point method with crossover switched
  off. That tuning lives in `solverutils.py` and is applied by `esmkit.solve()`, which works
  because `solph.Model` is a pyomo `ConcreteModel`.
- **`Flow()` without `nominal_capacity` is unbounded above but non-negative** — solph sets
  `lb = 0` for unidirectional flows and no upper bound.
- **A bus with no supply fails silently**: the balance just forces every withdrawal to zero. If a
  component mysteriously does nothing, check that something injects into every bus it touches.
  `oemof.network.graph.create_nx_graph(es)` gives you the topology to check.
- **`max` is deprecated in favour of `maximum`** in solph 0.6 flows; using both raises a
  `FutureWarning` and one silently overwrites the other.
- **Storages are periodically balanced by default** (`balanced=True`). Passing `initial_soc`
  switches it off.
- **Degenerate optima are real.** With a lossless storage or a constant price, shifting energy is
  cost-neutral and the solver may return any of many optima. Tests that assert on a *trajectory*
  need a strictly convex setup (lossy storage, varying price); tests on aggregates do not.
- **Set `$SOLVER=highs`** unless you have gurobi or cplex.
