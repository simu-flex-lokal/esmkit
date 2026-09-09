"""The 5R1C reduced-order thermal zone: one custom solph node plus the
pyomo block that carries its energy balances.

The envelope is wired as Schuetz et al. 2017 writes it: window transmission
is driven by ``T_s`` (Eq. 21) and ventilation by ``T_air`` (Eq. 22). Until
2026-09-09 both hung off ``T_m`` instead, which understates their driving
temperature difference - ``T_m`` is the coldest indoor node while heating -
and so understated the heat demand by about 5 % on the reference building.
That was carried for parity with the pre-migration tsib implementation and
is now removed; ``docs/model-deviations.md`` records what remains.

``gain_air`` is optional and off by default. Passing it puts a separate
convective gain on the air node, phi_ia of Schuetz Eq. (14); omitted, the
air node reuses ``gain_surface`` and the surface gain is still spent at two
nodes. That one is a deviation still standing, because phi_int does not
reach the zone as a series of its own.
"""

import warnings

import numpy as np
import pandas as pd
import pyomo.environ as po
from oemof.solph import Flow

from ..core import registry
from ..core.base import EsmBlock, EsmComponent
from ..core.registry import factory


ENVELOPE_ELEMENTS = ["Walls", "Roof", "Floor", "Windows", "Ventilation"]


# The opaque elements, lumped into the equivalent mass-to-environment
# conductance H_tr,em of Schuetz Eqs. (8)-(9). Only these may hang off the
# mass node; windows sit on the surface node and ventilation on the air
# node, which is the whole point of this variant.
MASS_ELEMENTS = ["Walls", "Roof", "Floor"]


SERIES_PARAMETERS = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]

#: Series that may be omitted. ``gain_air`` is the convective share of the
#: internal gains, phi_ia in Schuetz Eq. (14); without it the air node falls
#: back to ``gain_surface`` and keeps deviation 3.
OPTIONAL_SERIES = ["gain_air"]


class ThermalZone5R1C(EsmComponent):
    """A 5R1C reduced-order thermal zone as formulated by Schuetz et al. 2017.

    The formulation follows Schuetz et al. 2017 ("Optimal design of energy
    conversion units and envelopes for residential building retrofits using
    a comprehensive MILP model", Applied Energy 185, Eqs. 20-22) with the
    comfort band of Kotzur 2018 (dissertation, Sec. 3.2.2, Eqs. 3.1-3.2).
    Windows sit on the surface node and ventilation on the air node, as
    those equations specify. The deviations that remain are listed in
    docs/model-deviations.md.

    Three temperatures are free variables over the whole horizon: ``T_m``
    (the thermal mass, the one node carrying a capacitance ``C_m``), ``T_s``
    (the inner surface) and ``T_air`` (the indoor air). ``H_ms`` couples
    mass to surface, ``H_is`` surface to air, ``H_door`` and the per-element
    conductances in ``H`` (Walls, Roof, Floor, Windows, Ventilation) couple
    the zone to the outside.

    The zone has no port mechanism: its heat supply is read straight off
    ``m.flow[heat_bus, zone, t]``, the edge solph itself creates for the
    input flow, so the zone is wired like any other node in the system.

    Because the temperatures may float anywhere inside the comfort band,
    the thermal mass is a dispatchable flexibility resource inside the same
    LP as the storage, the PV and the prices: the model may pre-heat into
    cheap hours and coast through expensive ones.
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
        gain_air=None,
        can_cool=True,
        comfort_ub_penalty=None,
        max_load_violation_penalty=100.0,
        initial_T_m=None,
        design_capacity=None,
    ):
        """Build the zone and check that all its time series agree in length.

        Args:
            label: node label, unique within the energy system.
            heat_bus: bus the zone draws its heating from.
            H_ms: mass-to-surface conductance, kW/K.
            H_is: surface-to-air conductance, kW/K.
            H_door: door transmission conductance, kW/K.
            C_m: heat capacity of the thermal mass, kWh/K.
            H: per-element envelope conductances in kW/K, with one entry
                for each of Walls, Roof, Floor, Windows and Ventilation.
            T_e: ambient air temperature per time step, degC.
            gain_mass: solar and internal gains onto the mass node, kW.
            gain_surface: solar and internal gains onto the surface node, kW.
            comfort_lb: lower bound of the indoor air temperature, degC.
            comfort_ub: upper bound of the indoor air temperature, degC.
            max_load: heating and cooling power the supply can deliver, kW.
            can_cool: whether the zone may reject heat at all. False fixes
                the cooling to zero, which is what an uncooled dwelling
                actually is; pair it with ``comfort_ub_penalty``, or a
                summer with any solar gain becomes infeasible.
            comfort_ub_penalty: price of exceeding ``comfort_ub``, EUR per
                Kelvin-hour. Given, the ceiling becomes soft: the zone may
                drift above the band and pays for it. Left at None the
                ceiling stays hard, which is the historical behaviour the
                golden fixtures were captured under.
            cool_bus: optional bus supplying cooling. Without one, cooling
                is an internal variable that costs nothing.
            gain_air: optional convective gain on the air node, kW - phi_ia
                of Schuetz Eq. (14), usually 0.5 * the internal gains. Given,
                it replaces ``gain_surface`` there and removes deviation 3;
                omitted, the air node keeps ``gain_surface`` and the surface
                gain is still spent at two nodes.
            max_load_violation_penalty: price of the slack, EUR per kW.
            initial_T_m: mass temperature at the first step, degC. Given, it
                replaces the cyclic closure of the mass balance.
            design_capacity: capacity reported in the results, kW. Defaults
                to ``max_load``.
        """
        inputs = {heat_bus: Flow()}
        if cool_bus is not None:
            inputs[cool_bus] = Flow()
        super().__init__(label=label, inputs=inputs)

        self.heat_bus = heat_bus
        self.cool_bus = cool_bus
        self.can_cool = bool(can_cool)
        self.comfort_ub_penalty = (
            None if comfort_ub_penalty is None else float(comfort_ub_penalty)
        )
        if cool_bus is not None and not self.can_cool:
            raise ValueError(
                "Zone '{}' has a cool_bus but can_cool=False - a zone either "
                "rejects heat or it does not".format(label)
            )

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
            "gain_air": gain_air,
        }
        lengths = set()
        for key in SERIES_PARAMETERS:
            values = _as_array(series[key], "{} of zone '{}'".format(key, label))
            setattr(self, key, values)
            lengths.add(len(values))
        # An omitted optional series stays None, and the balance below reads
        # that as "reuse gain_surface" rather than "no gain at all".
        for key in OPTIONAL_SERIES:
            if series[key] is None:
                setattr(self, key, None)
                continue
            values = _as_array(series[key], "{} of zone '{}'".format(key, label))
            setattr(self, key, values)
            lengths.add(len(values))
        if len(lengths) > 1:
            raise ValueError(
                "Zone '{}' got time series of differing lengths: {}".format(
                    label,
                    ", ".join(
                        "{}={}".format(k, len(getattr(self, k)))
                        for k in SERIES_PARAMETERS + OPTIONAL_SERIES
                        if getattr(self, k) is not None
                    ),
                )
            )
        self.n_steps = lengths.pop()

    def constraint_group(self):
        """Return the block class that constrains this zone."""
        return ThermalZone5R1CBlock

    def results(self, model, index=None):
        """Read the solved zone out of the model.

        Warns with a ``UserWarning`` if the max-load slack came out
        non-zero, since the supply then had to be oversized to stay
        feasible.

        Args:
            model: the solved solph model.
            index: time index for the returned frame.

        Returns:
            A dict with ``"timeseries"`` (heating and cooling load in kW
            plus the four temperatures in degC), ``"static"`` (the reported
            capacity and the zero cost entries, so the zone lines up with
            the other components) and ``"max_load_violation"`` (kW by which
            ``max_load`` had to be exceeded, 0 when it held).
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
        timeseries["Comfort Violation"] = [
            po.value(block.comfort_violation[self, t]) for t in steps
        ]

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
            # Kelvin-hours above the band over the whole horizon; the unit
            # EN 15251 and DIN 4108-2 state summer comfort in.
            "comfort_violation_Kh": float(
                (timeseries["Comfort Violation"] * model.timeincrement[0]).sum()
            ),
        }


def _as_array(value, name):
    """Coerce a time series parameter into a 1-D float array."""
    if value is None:
        raise ValueError("{} is required".format(name))
    if isinstance(value, (pd.Series, pd.DataFrame)):
        value = value.values
    values = np.asarray(value, dtype=float).ravel()
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("{} must be a non-empty sequence".format(name))
    return values


class ThermalZone5R1CBlock(EsmBlock):
    """The variables and energy balances of every zone in the model."""

    def _create(self, group=None):
        """Add the three node balances, the comfort band and the load limits."""
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

        # Cooling with no bus behind it: free and unpriced, which is what
        # relieves summer overheating at zero cost when no cool_bus is
        # attached. See docs/model-deviations.md.
        self.Q_cool_internal = po.Var(
            self.ZONES, m.TIMESTEPS, within=po.NonNegativeReals
        )

        # A zone that cannot cool has the variable pinned to zero rather
        # than the constraint dropped, so every expression below still
        # reads the same whether or not the zone rejects heat.
        for zone in self.ZONES:
            if not getattr(zone, "can_cool", True):
                for t in m.TIMESTEPS:
                    self.Q_cool_internal[zone, t].fix(0.0)

        # Slack above comfort_ub, priced per Kelvin-hour. Zero-fixed where
        # no penalty was given, which leaves the ceiling hard.
        self.comfort_violation = po.Var(
            self.ZONES, m.TIMESTEPS, within=po.NonNegativeReals
        )
        for zone in self.ZONES:
            if getattr(zone, "comfort_ub_penalty", None) is None:
                for t in m.TIMESTEPS:
                    self.comfort_violation[zone, t].fix(0.0)

        # One scalar slack per zone, priced in _objective_expression: an
        # undersized heat supply then yields a feasible model that reports
        # the breach, instead of an infeasible one nobody can debug.
        self.max_load_violation = po.Var(self.ZONES, within=po.NonNegativeReals)

        def Q_heat(zone, t):
            return m.flow[zone.heat_bus, zone, t]

        def Q_cool(zone, t):
            if zone.cool_bus is None:
                return self.Q_cool_internal[zone, t]
            return m.flow[zone.cool_bus, zone, t]

        self._Q_heat = Q_heat
        self._Q_cool = Q_cool

        # Only the opaque elements are referenced to the mass temperature.
        # Windows and ventilation are referenced to their own nodes below;
        # passing anything but a MASS_ELEMENT here would reintroduce
        # deviations 1 and 2, so the call is guarded.
        def envelope_flow(zone, element, t):
            if element not in MASS_ELEMENTS:
                raise ValueError(
                    "'{}' does not belong on the mass node in the Schuetz "
                    "formulation; use the surface or air balance".format(element)
                )
            return zone.H[element] * (self.T_m[zone, t] - zone.T_e[t])

        # Mass node, Eq. (20) of Schuetz et al. 2017.
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

        # The last step balances against step 0, so T_m closes cyclically
        # over the horizon; with an initial_T_m the first step is pinned
        # instead and the wrap-around constraint is dropped.
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

        # Surface node, Eq. (21) of Schuetz et al. 2017:
        # window transmission is H_tr,w * (theta_s - theta_e), driven by the
        # surface temperature, so it responds to a warming room at once
        # instead of being damped by the thermal mass.
        self.surface_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                zone.H_ms * (self.T_s[zone, t] - self.T_m[zone, t])
                + zone.H_is * (self.T_s[zone, t] - self.T_air[zone, t])
                + zone.H["Windows"] * (self.T_s[zone, t] - zone.T_e[t])
                == zone.gain_surface[t]
            ),
        )

        # phi_ia of Schuetz Eq. (14) where it was supplied, otherwise the
        # phi_st, i.e. the surface gain spent at two nodes.
        def air_gain(zone, t):
            if zone.gain_air is None:
                return zone.gain_surface[t]
            return zone.gain_air[t]

        # Air node, Eq. (22) of Schuetz et al. 2017: the node where the
        # heating and cooling flows enter the zone.
        # ventilation is H_ve * (theta_air - theta_e). This is the change
        # that matters most for flexibility - it reopens the direct path
        # from the injection point to ambient, so part of any pre-heating
        # leaves the zone immediately instead of all of it reaching C_m.
        self.air_node_balance = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: (
                zone.H["Ventilation"] * (self.T_air[zone, t] - zone.T_e[t])
                + zone.H_is * (self.T_air[zone, t] - self.T_s[zone, t])
                == air_gain(zone, t) - Q_cool(zone, t) + Q_heat(zone, t)
            ),
        )

        # Comfort band on the air temperature, Kotzur 2018 Eqs. (3.1)-(3.2).
        # Together with C_m this is what makes the zone dispatchable.
        self.comfort_ub = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: self.T_air[zone, t]
            <= zone.comfort_ub[t] + self.comfort_violation[zone, t],
        )
        self.comfort_lb = po.Constraint(
            self.ZONES,
            m.TIMESTEPS,
            rule=lambda b, zone, t: self.T_air[zone, t] >= zone.comfort_lb[t],
        )

        # Heating and cooling share the one slack, so the reported violation
        # is the largest single breach rather than a sum of both.
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
        """Price the max-load slack.

        solph discovers this method by name on every constraint group and
        adds what it returns to the objective.
        """
        model = self.parent_block()
        expression = sum(
            self.max_load_violation[zone] * zone.max_load_violation_penalty
            for zone in self.ZONES
        )
        # Kelvin-hours, so the penalty is independent of the step length.
        for zone in self.ZONES:
            penalty = getattr(zone, "comfort_ub_penalty", None)
            if penalty is None:
                continue
            expression += sum(
                self.comfort_violation[zone, t] * penalty * model.timeincrement[t]
                for t in model.TIMESTEPS
            )
        return expression


@factory("zone5r1c")
def _zone5r1c(name, params, buses, inputs, n_steps, step_size_h):
    """Thermal zone: the custom ``ThermalZone5R1C`` node.

    Consumes ``heat_bus``, the optional ``cool_bus`` and the time series
    ``T_e``, ``gain_mass``, ``gain_surface``, ``comfort_lb`` and
    ``comfort_ub``, plus the optional ``gain_air``; whatever is left in
    ``params`` (``H_ms``, ``H_is``, ``H_door``, ``C_m``, ``H``,
    ``max_load``, ...) goes to the constructor.
    """
    heat_bus = registry.bus(buses, params, "heat_bus", name)
    cool_bus = None
    if params.get("cool_bus") is not None:
        cool_bus = registry.bus(buses, params, "cool_bus", name)
    else:
        # Drop the explicit None, or **params would pass cool_bus twice.
        params.pop("cool_bus", None)

    series = {
        key: registry.profile(params, key, inputs, n_steps, name)
        for key in SERIES_PARAMETERS
    }
    for key in OPTIONAL_SERIES:
        if params.get(key) is not None:
            series[key] = registry.profile(params, key, inputs, n_steps, name)
        else:
            # Drop the explicit None, or **params would pass it twice.
            params.pop(key, None)
    return [
        ThermalZone5R1C(
            name, heat_bus=heat_bus, cool_bus=cool_bus, **series, **params
        )
    ]
