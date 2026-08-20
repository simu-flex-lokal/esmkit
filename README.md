[![Build Status](https://github.com/simu-flex-lokal/esmkit/actions/workflows/test.yml/badge.svg)](https://github.com/simu-flex-lokal/esmkit/actions/workflows/test.yml)

# esmkit — Energy System Model Kit

A component kit for [oemof-solph](https://oemof-solph.readthedocs.io/) energy systems. A system
is described as plain, serializable data — a **spec** holding topology and scalar parameters —
which `build_system` turns into a ready-to-solve solph model. Time series never live in the spec:
they are referenced by name (`"@elecPrice"`) and resolved against an inputs mapping at build time,
so one spec can be stored, diffed and varied across thousands of systems.

Units throughout: **kW**, **kWh**, **EUR**, **°C**, **hours**.

## Features

- **Specs instead of wiring.** `SystemSpec` is a JSON round-trippable description of buses,
  components and their parameters. `required_inputs(spec)` says which time series a spec will
  ask for, before anything is built.
- **A registry of stock components.** `grid`, `pv`, `battery`, `thermal_storage`, `heat_pump`,
  `demand`, `source` and `meter` map onto stock solph objects. Adding a technology is one
  `@factory` function.
- **A custom-component API that cannot be got wrong.** `EsmComponent` / `EsmBlock` carry the
  `CONSTRAINT_GROUP` attribute solph silently requires, and `build_system` refuses to return a
  system whose block would be ignored.
- **A 5R1C thermal zone** (`zone5r1c`) as the worked example of a non-trivial custom component:
  three temperature nodes with free states inside a comfort band, so building thermal mass is a
  dispatchable flexibility resource in the same solve as storage, PV and prices.
- **Solver handling that works on badly conditioned models**, including a HiGHS retry chain.

## Installation

```bash
pip install esmkit[highs]
```

### Solver

The kit needs a MILP/LP solver. At solve time it tries, in order: the solver named by `$SOLVER`,
then `gurobi`, `cplex`, `scip`, `cbc`, `highs`. Neither `gurobipy` nor `highspy` is installed by
default — both are optional-dependency extras (`esmkit[highs]`, `esmkit[gurobi]`). Set
`SOLVER=highs` unless you have gurobi or cplex.

### Development

```bash
uv sync --group dev --extra highs
SOLVER=highs uv run pytest test/
```

## Examples

Runnable demonstrations live in [`examples/`](examples/). Documentation is in
[`docs/`](docs/README.md).

## Origin and attribution

esmkit was extracted from the energy system layer of
[tsib](https://github.com/simu-flex-lokal/tsib) (Time Series Initialization for Buildings),
originally developed at FZJ IEK-3. The 5R1C thermal zone implements the MILP formulation of
Schütz et al. (2017) as used by Kotzur (2018); its known deviations from those sources are
documented in [`docs/model-deviations.md`](docs/model-deviations.md).

esmkit knows nothing about buildings. Building identity, weather, envelope physics and occupancy
stay in tsib, which feeds this kit explicit parameters and time series.

## License

MIT — see [LICENSE](LICENSE).
