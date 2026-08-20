# -*- coding: utf-8 -*-
"""
5R1C thermal zone as an oemof-solph component.

This is the kit's only custom solph component: every other component maps
onto a stock solph object and is built by a `@factory` in `stock.py`.

Why it must be a component and not a precomputed load profile: the zone's
temperature states (T_air, T_s, T_m) are free decision variables inside a
comfort band, so the thermal mass of the building is a dispatchable
flexibility resource in the same optimization as storage, PV and prices.
Reducing the zone to a fixed heat demand time series and optimizing dispatch
afterwards would throw that flexibility away.

The heat (and optionally cooling) supply is not a variable of this
component: it is `m.flow[heat_bus, zone, t]`, the edge solph creates for
the zone's input flow. That edge is the entire coupling mechanism.

**The component takes explicit parameters, never a building configuration.**
Everything that is building physics - irradiance on tilted surfaces, window
areas and shading, U-values, thermal class, the comfort control logic - is
the caller's job and reaches the zone as five time series and ten numbers.
The gains in particular are constants of the optimization: they enter the
node balances as a right-hand side and depend on no decision variable, so
they are computed before the model is built.

Four deviations from the source papers are carried over from tsib's
pre-migration model on purpose - they are load-bearing for its validated
~197 kWh/m2/a result. See `docs/model-deviations.md` before "correcting"
any node balance or the comfort bound.
"""

import warnings

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof.solph import Flow

from ..core import registry
from ..core.base import EsmBlock, EsmComponent
from ..core.registry import factory

#: envelope elements the zone distinguishes. Walls, Roof and Floor drive the
#: mass node together with the door; Windows drives the surface node and
#: Ventilation the air node.
ENVELOPE_ELEMENTS = ["Walls", "Roof", "Floor", "Windows", "Ventilation"]

#: elements whose flow enters the thermal mass balance
MASS_ELEMENTS = ["Walls", "Roof", "Floor"]

#: time series the component requires, in the order they are validated
SERIES_PARAMETERS = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]


class ThermalZone5R1C(EsmComponent):
    """
    5R1C thermal zone (DIN EN ISO 13790 / Schuetz et al. 2017).

    Three temperature nodes - mass `T_m` (the only one with capacity `C_m`),
    surface `T_s`, air `T_air` - coupled by heat transfer coefficients, with
    heating and cooling entering at the air node.

    Parameters
    ----------
    label: str, required
        Node label, unique within the energy system.
    heat_bus: solph.Bus, required
        Bus the zone draws its heating from. The zone never prices its own
        supply: put a Source with `variable_costs` on the bus instead.
    H_ms, H_is: float, required
        Heat transfer coefficients [kW/K] mass<->surface and surface<->air.
    H_door: float, required
        Heat transfer coefficient [kW/K] of the door, which enters the mass
        balance alongside the opaque elements.
    C_m: float, required
        Thermal capacity of the mass node [kWh/K].
    H: dict, required
        Heat transfer coefficient [kW/K] per element of
        `ENVELOPE_ELEMENTS`.
    T_e: sequence, required
        Ambient temperature [degC].
    gain_mass, gain_surface: sequence, required
        Heat gains [kW] at the mass and the surface node. These are the
        aggregate of internal and solar gains after distribution over the
        nodes; how they are computed is the caller's business.
    comfort_lb, comfort_ub: sequence, required
        Effective comfort band of the air temperature [degC] per time step.
        Both bounds are hard - see deviation 4.
    max_load: float, required
        Maximal load of the heating/cooling system [kW], enforced as a soft
        constraint so that an undersized system warns instead of turning
        the model infeasible.
    cool_bus: solph.Bus, optional (default: None)
        Bus the zone draws cooling from. If None, cooling is an internal
        free variable (still needed because the comfort ceiling is a hard
        bound - see deviation 4).
    max_load_violation_penalty: float, optional (default: 100.)
        Penalty of the soft constraint [EUR/kW].
    initial_T_m: float, optional (default: None)
        Initial thermal mass temperature [degC]. Replaces the periodic wrap
        of the mass node when given.
    design_capacity: float, optional (default: max_load)
        Design heat load [kW] reported in the static results. Purely
        descriptive; it constrains nothing.
    """

    def __init__(
        self,
        label,
        heat_bus,
        H_ms,
        H_is,
        H_door,
        C_m,
        H,
        T_e,
        gain_mass,
        gain_surface,
        comfort_lb,
        comfort_ub,
        max_load,
        cool_bus=None,
        max_load_violation_penalty=100.0,
        initial_T_m=None,
        design_capacity=None,
    ):
        inputs = {heat_bus: Flow()}
        if cool_bus is not None:
            inputs[cool_bus] = Flow()
        super().__init__(label=label, inputs=inputs)

        self.heat_bus = heat_bus
        self.cool_bus = cool_bus

        self.H_ms = float(H_ms)
        self.H_is = float(H_is)
        self.H_door = float(H_door)
        self.C_m = float(C_m)
        self.H = {element: float(H[element]) for element in ENVELOPE_ELEMENTS}

        self.max_load = float(max_load)
        self.max_load_violation_penalty = float(max_load_violation_penalty)
        self.initial_T_m = initial_T_m
        self.design_capacity = (
            float(design_capacity) if design_capacity is not None else self.max_load
        )

        series = {
            "T_e": T_e,
            "gain_mass": gain_mass,
            "gain_surface": gain_surface,
            "comfort_lb": comfort_lb,
            "comfort_ub": comfort_ub,
        }
        lengths = set()
        for key in SERIES_PARAMETERS:
            values = _as_array(series[key], "{} of zone '{}'".format(key, label))
            setattr(self, key, values)
            lengths.add(len(values))
        if len(lengths) > 1:
            raise ValueError(
                "Zone '{}' got time series of differing lengths: {}".format(
                    label,
                    ", ".join(
                        "{}={}".format(k, len(getattr(self, k)))
                        for k in SERIES_PARAMETERS
                    ),
                )
            )
        self.n_steps = lengths.pop()

    def constraint_group(self):
        return ThermalZone5R1CBlock

    def results(self, model, index=None):
        """
        Solved loads, temperatures and static figures of the zone.

        Returns a dict with "timeseries" (a DataFrame of loads and node
        temperatures), "static" (capacities and costs) and
        "max_load_violation". `node_results` calls this instead of the
        generic flow extraction.
        """
        block = model.ThermalZone5R1CBlock
        steps = list(model.TIMESTEPS)

        violation = po.value(block.max_load_violation[self])
        if violation is not None and violation > 1e-9:
            warnings.warn(
                "Maximal heat load exceeded by " + str(round(violation, 3)) + " kW",
                UserWarning,
            )

        heat = np.array(
            [po.value(model.flow[self.heat_bus, self, t]) for t in steps]
        )
        if self.cool_bus is None:
            cool = np.array([po.value(block.Q_cool_internal[self, t]) for t in steps])
        else:
            cool = np.array(
                [po.value(model.flow[self.cool_bus, self, t]) for t in steps]
            )

        timeseries = pd.DataFrame(index=index)
        timeseries["Heating Load"] = heat
        timeseries["Cooling Load"] = cool
        timeseries["T_air"] = [po.value(block.T_air[self, t]) for t in steps]
        timeseries["T_s"] = [po.value(block.T_s[self, t]) for t in steps]
        timeseries["T_m"] = [po.value(block.T_m[self, t]) for t in steps]
        timeseries["T_e"] = self.T_e

        static = {
            "Capacity": self.design_capacity,
            "FixCost": 0,
            "CAPEX": 0,
            "OPEX fix": 0.0,
            "VarCost": 0.0,
            "OPEX": 0.0,
            "OPEX var": 0.0,
        }

        return {
            "timeseries": timeseries,
            "static": static,
            "max_load_violation": violation,
        }


def _as_array(value, name):
    """Coerces a sequence parameter into a 1-d float array."""
    if value is None:
        raise ValueError("{} is required".format(name))
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    values = np.asarray(value, dtype=float).ravel()
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("{} must be a non-empty sequence".format(name))
    return values


class ThermalZone5R1CBlock(EsmBlock):
    """
    Constraints of all ThermalZone5R1C nodes in an energy system.
    """

    def _create(self, group=None):
        if group is None:
            return
        m = self.parent_block()
        n_steps = len(m.TIMESTEPS)
        for zone in group:
            if zone.n_steps != n_steps:
                raise ValueError(
                    "The model time index ({} steps) does not match the time "
                    "series of zone '{}' ({} steps)".format(
                        n_steps, zone.label, zone.n_steps
                    )
                )

        self.ZONES = po.Set(initialize=list(group), ordered=True)

        # --- variables --------------------------------------------------
        # node temperatures [degC]; free reals, the comfort band bounds them
        self.T_air = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        self.T_s = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        self.T_m = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        # cooling when no cool bus is attached (the comfort ceiling is hard)
        self.Q_cool_internal = po.Var(
            self.ZONES, m.TIMESTEPS, within=po.NonNegativeReals
        )
        # soft violation of the maximal system load [kW]
        self.max_load_violation = po.Var(self.ZONES, within=po.NonNegativeReals)

        def Q_heat(zone, t):
            return m.flow[zone.heat_bus, zone, t]

        def Q_cool(zone, t):
            if zone.cool_bus is None:
                return self.Q_cool_internal[zone, t]
            return m.flow[zone.cool_bus, zone, t]

        self._Q_heat = Q_heat
        self._Q_cool = Q_cool

        # --- envelope heat flows ----------------------------------------
        # POTENTIAL BUG (kept for parity with tsib's pre-migration model):
        # only the opaque elements may use (T_m - T_e) per Schuetz et al.
        # 2017 - eq. 20/9. Windows should be driven by (T_s - T_e) (eq. 21)
        # and Ventilation by (T_air - T_e) (eq. 22).
        # See docs/model-deviations.md, items 1 and 2.
        def envelope_flow(zone, element, t):
            T_e = zone.T_e[t]
            return zone.H[element] * (self.T_m[zone, t] - T_e)

        # --- 1) thermal mass node, the difference equation --------------
        def mass_node_balance(zone, t, t_next):
            T_e = zone.T_e[t]
            return (
                zone.H_ms * (self.T_m[zone, t] - self.T_s[zone, t])
                + sum(envelope_flow(zone, e, t) for e in MASS_ELEMENTS)
                + zone.H_door * (self.T_m[zone, t] - T_e)
                == zone.gain_mass[t]
                - zone.C_m
                * (self.T_m[zone, t_next] - self.T_m[zone, t])
                / m.timeincrement[t]
            )

        last = n_steps - 1

        def mass_rule(b, zone, t):
            if t < last:
                return mass_node_balance(zone, t, t + 1)
            if zone.initial_T_m is None:
                # periodic wrap: the last step feeds the first one
                return mass_node_balance(zone, last, 0)
            return po.Constraint.Skip

        self.mass_node_balance = po.Constraint(self.ZONES, m.TIMESTEPS, rule=mass_rule)

        self.mass_node_initial = po.Constraint(
            self.ZONES,
            rule=lambda b, zone: (
                po.Constraint.Skip
                if zone.initial_T_m is None
                else self.T_m[zone, 0] == zone.initial_T_m
            ),
        )

        # --- 2) surface node --------------------------------------------
        self.surface_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                zone.H_ms * (self.T_s[zone, t] - self.T_m[zone, t])
                + zone.H_is * (self.T_s[zone, t] - self.T_air[zone, t])
                + envelope_flow(zone, "Windows", t)
                == zone.gain_surface[t]
            ),
        )

        # --- 3) air node, supplied by heating and cooling ---------------
        # POTENTIAL BUG (kept for parity): per Schuetz et al. 2017 - eq. 22
        # the air node receives Q_ia = 0.5 * Q_ig (eq. 14), not the surface
        # gain Q_st (eq. 19). See docs/model-deviations.md, item 3.
        self.air_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                envelope_flow(zone, "Ventilation", t)
                + zone.H_is * (self.T_air[zone, t] - self.T_s[zone, t])
                == zone.gain_surface[t] - Q_cool(zone, t) + Q_heat(zone, t)
            ),
        )

        # --- comfort band ------------------------------------------------
        # Schuetz et al. 2017 has no upper bound at all (eq. 26) and
        # free-floats instead; the hard ceiling is Kotzur 2018 - eq. 3.2.
        # See docs/model-deviations.md, item 4.
        self.comfort_ub = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: self.T_air[zone, t] <= zone.comfort_ub[t],
        )
        self.comfort_lb = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: self.T_air[zone, t] >= zone.comfort_lb[t],
        )

        # --- maximal system load as a soft constraint --------------------
        self.max_heating_load = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: Q_heat(zone, t) - self.max_load_violation[zone]
            <= zone.max_load,
        )
        self.max_cooling_load = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: Q_cool(zone, t) - self.max_load_violation[zone]
            <= zone.max_load,
        )

    def _objective_expression(self):
        """Only the penalty of the soft max-load constraint. Energy cost
        belongs to whatever supplies the heat bus - a component pricing
        energy it did not import would double-count."""
        return sum(
            self.max_load_violation[zone] * zone.max_load_violation_penalty
            for zone in self.ZONES
        )


@factory("zone5r1c")
def _zone5r1c(name, params, buses, inputs, n_steps, step_size_h):
    """The 5R1C thermal zone - the kit's one custom component."""
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
    return [
        ThermalZone5R1C(
            name, heat_bus=heat_bus, cool_bus=cool_bus, **series, **params
        )
    ]
