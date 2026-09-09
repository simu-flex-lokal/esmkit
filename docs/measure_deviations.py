"""Measures what each remaining 5R1C deviation is worth, and what the envelope
wiring removed on 2026-09-09 was worth.

Each variant re-solves the golden full-year fixture with one balance or bound
replaced, and reports the change in annual heating demand against the current
model. Nothing here is imported by the kit; it exists so the numbers in
`model-deviations.md` can be checked rather than believed.

The three ``removed:`` rows rebuild the pre-2026-09-09 wiring, where windows
and ventilation hung off ``T_m`` instead of ``T_s`` and ``T_air``. They should
reproduce the 198.111 kWh/m2a that the golden fixture held before the
correction, i.e. -11.0 kWh/m2a or -5.3 % against today's 209.143.

Every variant runs twice, under two input conditions:

  `baseline`    the comfort floor as captured, binding all 8760 h.
  `heat_limit`  the floor released outside the heating period, i.e. the
                Heizgrenztemperatur - a 24 h centred running mean of T_e
                below 17 degC. Nothing then obliges the model to heat in
                summer, and heat costs money, so it stops.

The heating-limit logic is duplicated from
`orchestrator/helpers/heating_period.py` rather than imported: this script
needs `test/conftest.py` and the golden fixtures, which live only inside
this repository, and esmkit must not grow a dependency on the orchestrator.
Keep the three constants below in step with that module.

    SOLVER=highs uv run python docs/measure_deviations.py
"""

import os
import sys

import numpy as np
import pandas as pd
import pyomo.environ as po

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test"))

from conftest import ZONE_SERIES, zone_fixture  # noqa: E402

from esmkit import SystemSpec, build_system, solve  # noqa: E402
from esmkit.components.zone5r1c import (  # noqa: E402
    MASS_ELEMENTS,
    SERIES_PARAMETERS,
    ThermalZone5R1C,
    ThermalZone5R1CBlock,
)
from esmkit.core import registry  # noqa: E402
from esmkit.core.registry import factory  # noqa: E402

#: reference area of the golden building [m2], from golden_meta.json
A_REF = 173.25

#: TABULA/EPISCOPE published q_h_nd for this archetype [kWh/m2a]. Reported
#: for orientation only: the comparison is confounded by thermal bridging,
#: a temperature reduction factor and a longwave term that only one side
#: has. See the TABULA section of docs/model-deviations.md - do not read
#: the last column as a ranking.
Q_TABULA = 195.26

#: Heizgrenztemperatur [degC], running-mean length [h] and the released
#: floor [degC]. Mirrors `orchestrator/helpers/heating_period.py`.
HEATING_LIMIT = 17.0
WINDOW = 24
RELEASED_LB = -50.0


def heating_mask(T_e, limit=HEATING_LIMIT, window=WINDOW):
    """Hours that belong to the heating period."""
    running = pd.Series(np.asarray(T_e, dtype=float)).rolling(
        window, min_periods=1, center=True
    ).mean()
    return (running < limit).to_numpy()


def release_floor(inputs):
    """Drop the comfort floor outside the heating period."""
    inputs = dict(inputs)
    lower = np.asarray(inputs["comfort_lb"], dtype=float).copy()
    lower[~heating_mask(inputs["T_e"])] = RELEASED_LB
    inputs["comfort_lb"] = lower
    return inputs


def _Q_cool(block, zone, t, m):
    if zone.cool_bus is None:
        return block.Q_cool_internal[zone, t]
    return m.flow[zone.cool_bus, zone, t]


class WindowsOnMass(ThermalZone5R1CBlock):
    """Rebuilds the pre-2026-09-09 wiring: windows driven by T_m."""

    def _create(self, group=None):
        super()._create(group)
        if group is None:
            return
        m = self.parent_block()
        self.surface_node_balance.deactivate()
        self.surface_variant = po.Constraint(
            self.ZONES, m.TIMESTEPS,
            rule=lambda b, z, t: (
                z.H_ms * (self.T_s[z, t] - self.T_m[z, t])
                + z.H_is * (self.T_s[z, t] - self.T_air[z, t])
                + z.H["Windows"] * (self.T_m[z, t] - z.T_e[t])
                == z.gain_surface[t]
            ),
        )


class VentilationOnMass(ThermalZone5R1CBlock):
    """Rebuilds the pre-2026-09-09 wiring: ventilation driven by T_m."""

    def _create(self, group=None):
        super()._create(group)
        if group is None:
            return
        m = self.parent_block()
        self.air_node_balance.deactivate()
        self.air_variant = po.Constraint(
            self.ZONES, m.TIMESTEPS,
            rule=lambda b, z, t: (
                z.H["Ventilation"] * (self.T_m[z, t] - z.T_e[t])
                + z.H_is * (self.T_air[z, t] - self.T_s[z, t])
                == z.gain_surface[t]
                - _Q_cool(self, z, t, m)
                + m.flow[z.heat_bus, z, t]
            ),
        )


class BothOnMass(VentilationOnMass):
    """The removed wiring in full - what zone5r1c was until 2026-09-09."""

    def _create(self, group=None):
        super()._create(group)
        if group is None:
            return
        m = self.parent_block()
        self.surface_node_balance.deactivate()
        self.surface_variant = po.Constraint(
            self.ZONES, m.TIMESTEPS,
            rule=lambda b, z, t: (
                z.H_ms * (self.T_s[z, t] - self.T_m[z, t])
                + z.H_is * (self.T_s[z, t] - self.T_air[z, t])
                + z.H["Windows"] * (self.T_m[z, t] - z.T_e[t])
                == z.gain_surface[t]
            ),
        )


class FreeFloat(ThermalZone5R1CBlock):
    """Item 4: no comfort ceiling and no cooling, i.e. [S17]'s free float."""

    def _create(self, group=None):
        super()._create(group)
        if group is None:
            return
        m = self.parent_block()
        self.comfort_ub.deactivate()
        self.no_cooling = po.Constraint(
            self.ZONES, m.TIMESTEPS,
            rule=lambda b, z, t: _Q_cool(self, z, t, m) == 0,
        )


class OperativeBand(ThermalZone5R1CBlock):
    """Item 5: comfort band on 0.3*T_air + 0.7*T_s instead of on T_air."""

    def _create(self, group=None):
        super()._create(group)
        if group is None:
            return
        m = self.parent_block()
        self.comfort_lb.deactivate()
        self.comfort_ub.deactivate()

        def operative(z, t):
            return 0.3 * self.T_air[z, t] + 0.7 * self.T_s[z, t]

        self.operative_lb = po.Constraint(
            self.ZONES, m.TIMESTEPS,
            rule=lambda b, z, t: operative(z, t) >= z.comfort_lb[t],
        )
        self.operative_ub = po.Constraint(
            self.ZONES, m.TIMESTEPS,
            rule=lambda b, z, t: operative(z, t) <= z.comfort_ub[t],
        )


class ImplicitEuler(ThermalZone5R1CBlock):
    """Item 6: conductances and gains taken at t+1, i.e. a backward step."""

    def _create(self, group=None):
        super()._create(group)
        if group is None:
            return
        m = self.parent_block()
        n_steps = len(m.TIMESTEPS)
        self.mass_node_balance.deactivate()

        def rule(b, zone, t):
            nxt = t + 1 if t < n_steps - 1 else 0
            if t == n_steps - 1 and zone.initial_T_m is not None:
                return po.Constraint.Skip
            T_e = zone.T_e[nxt]
            return (
                zone.H_ms * (self.T_m[zone, nxt] - self.T_s[zone, nxt])
                + sum(zone.H[e] * (self.T_m[zone, nxt] - T_e) for e in MASS_ELEMENTS)
                + zone.H_door * (self.T_m[zone, nxt] - T_e)
                == zone.gain_mass[nxt]
                - zone.C_m
                * (self.T_m[zone, nxt] - self.T_m[zone, t])
                / m.timeincrement[t]
            )

        self.mass_variant = po.Constraint(self.ZONES, m.TIMESTEPS, rule=rule)


def register(block_class, type_name):
    """Registers a zone component type using `block_class` for its constraints."""
    node_class = type(
        "Zone_" + type_name,
        (ThermalZone5R1C,),
        {"constraint_group": lambda self: block_class},
    )

    @factory(type_name)
    def _build(name, params, buses, inputs, n_steps, step_size_h):
        heat_bus = registry.bus(buses, params, "heat_bus", name)
        cool_bus = None
        if params.get("cool_bus") is not None:
            cool_bus = registry.bus(buses, params, "cool_bus", name)
        else:
            params.pop("cool_bus", None)
        series = {
            key: registry.profile(params, key, inputs, n_steps, name)
            for key in SERIES_PARAMETERS
        }
        return [node_class(name, heat_bus=heat_bus, cool_bus=cool_bus, **series, **params)]


def solve_year(type_name, release=False):
    """Annual heating and cooling [kWh] of the golden building, per variant.

    ``release=True`` lifts the comfort floor outside the heating period.
    """
    params, inputs, index = zone_fixture()
    if release:
        inputs = release_floor(inputs)

    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")
    spec.add_component("heat_supply", "source", bus="heat", price=0.08)
    spec.add_component("cool_supply", "source", bus="cool", price=0.02)
    spec.add_component("thermalzone", type_name, heat_bus="heat", cool_bus="cool",
                       **{key: "@" + key for key in ZONE_SERIES}, **params)

    es, nodes = build_system(spec, inputs, index)
    model, _ = solve(es)

    zone = nodes["thermalzone"]
    steps = list(model.TIMESTEPS)
    heat = np.array([po.value(model.flow[zone.heat_bus, zone, t]) for t in steps])
    cool = np.array([po.value(model.flow[zone.cool_bus, zone, t]) for t in steps])

    block = [b for b in model.component_objects(po.Block)
             if hasattr(b, "T_air") and zone in getattr(b, "ZONES", [])][0]
    T_air = np.array([po.value(block.T_air[zone, t]) for t in steps])
    return heat, cool, T_air


VARIANTS = [
    ("as implemented", "zone5r1c", None),
    ("removed: windows on T_m", "windows_on_mass", WindowsOnMass),
    ("removed: ventilation on T_m", "ventilation_on_mass", VentilationOnMass),
    ("removed: both on T_m", "both_on_mass", BothOnMass),
    ("4: free float, no cooling", "free_float", FreeFloat),
    ("5: operative comfort band", "operative_band", OperativeBand),
    ("6: implicit Euler step", "implicit_euler", ImplicitEuler),
]

#: (column label, whether the comfort floor is released outside the heating
#: period). The baseline runs first so its control fixes 198.111.
CONDITIONS = [("baseline", False), ("heat_limit", True)]


def main():
    for _, type_name, block_class in VARIANTS:
        if block_class is not None:
            register(block_class, type_name)

    # One mask, taken from the captured weather, splits both conditions the
    # same way - so the summer column compares like with like.
    _, base_inputs, _ = zone_fixture()
    winter = heating_mask(base_inputs["T_e"])
    print("heating period: {} of {} hours ({:.1f}%)".format(
        int(winter.sum()), len(winter), 100 * winter.mean()))

    rows = {}
    series = {}
    for condition, release in CONDITIONS:
        for label, type_name, _ in VARIANTS:
            heat, cool, T_air = solve_year(type_name, release=release)
            series[(condition, label)] = heat
            rows[(condition, label)] = {
                "heat [kWh/m2a]": heat.sum() / A_REF,
                "winter [kWh/m2a]": heat[winter].sum() / A_REF,
                "summer [kWh/m2a]": heat[~winter].sum() / A_REF,
                "cool [kWh/m2a]": cool.sum() / A_REF,
                "T_air max [degC]": T_air.max(),
                "hours over 26 degC": float((T_air > 26.001).sum()),
            }

    table = pd.DataFrame(rows).T
    table.index.names = ["condition", "variant"]

    # Each percentage runs against its own condition's control, so any
    # effect of releasing the floor cancels out of the deviation column.
    control = {c: table.loc[(c, "as implemented"), "heat [kWh/m2a]"]
               for c, _ in CONDITIONS}
    table["heat change [%]"] = [
        100 * (row["heat [kWh/m2a]"] / control[condition] - 1)
        for (condition, _), row in table.iterrows()
    ]
    table["max hourly change [kW]"] = [
        np.abs(series[key] - series[(key[0], "as implemented")]).max()
        for key in table.index
    ]
    # The distance the study actually cares about: how far each variant sits
    # from TABULA's published figure, signed.
    table["vs TABULA [kWh/m2a]"] = table["heat [kWh/m2a]"] - Q_TABULA

    print()
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
