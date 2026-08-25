import warnings

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof.solph import Flow

from ..core import registry
from ..core.base import EsmBlock, EsmComponent
from ..core.registry import factory


ENVELOPE_ELEMENTS = ["Walls", "Roof", "Floor", "Windows", "Ventilation"]


MASS_ELEMENTS = ["Walls", "Roof", "Floor"]


SERIES_PARAMETERS = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]


class ThermalZone5R1C(EsmComponent):
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
    if value is None:
        raise ValueError("{} is required".format(name))
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    values = np.asarray(value, dtype=float).ravel()
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("{} must be a non-empty sequence".format(name))
    return values


class ThermalZone5R1CBlock(EsmBlock):
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

        self.T_air = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        self.T_s = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)
        self.T_m = po.Var(self.ZONES, m.TIMESTEPS, within=po.Reals)

        self.Q_cool_internal = po.Var(
            self.ZONES, m.TIMESTEPS, within=po.NonNegativeReals
        )

        self.max_load_violation = po.Var(self.ZONES, within=po.NonNegativeReals)

        def Q_heat(zone, t):
            return m.flow[zone.heat_bus, zone, t]

        def Q_cool(zone, t):
            if zone.cool_bus is None:
                return self.Q_cool_internal[zone, t]
            return m.flow[zone.cool_bus, zone, t]

        self._Q_heat = Q_heat
        self._Q_cool = Q_cool

        def envelope_flow(zone, element, t):
            T_e = zone.T_e[t]
            return zone.H[element] * (self.T_m[zone, t] - T_e)

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

        self.air_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                envelope_flow(zone, "Ventilation", t)
                + zone.H_is * (self.T_air[zone, t] - self.T_s[zone, t])
                == zone.gain_surface[t] - Q_cool(zone, t) + Q_heat(zone, t)
            ),
        )

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
        return sum(
            self.max_load_violation[zone] * zone.max_load_violation_penalty
            for zone in self.ZONES
        )


@factory("zone5r1c")
def _zone5r1c(name, params, buses, inputs, n_steps, step_size_h):
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
