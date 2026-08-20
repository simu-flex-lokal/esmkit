# -*- coding: utf-8 -*-
"""
The 5R1C thermal zone as an oemof-solph component.

Parity against the golden fixtures captured from tsib's pre-migration
`tsib.energysystem` stack. Those fixtures are the acceptance test of the
whole extraction: the zone here takes explicit parameters instead of a
building configuration, and must still reproduce the same trajectory.

The parameters and the inputs are pinned alongside the expected results, so
these tests depend on neither tsib nor its stochastic occupancy model.
"""

import json

import numpy as np
import pandas as pd
import pyomo.environ as po
import pytest

from esmkit import build_system, node_results, objective_value, solve

from conftest import COOL_COST, HEAT_COST, golden, zone_fixture, zone_spec


def _solve_zone(n_steps=None, **zone_kwargs):
    params, inputs, index = zone_fixture(n_steps=n_steps)
    spec = zone_spec(params, **zone_kwargs)
    es, nodes = build_system(spec, inputs, index)
    model, _ = solve(es)
    return model, nodes, node_results(model, nodes, index=index)


# --- migration parity ------------------------------------------------


def test_parity_short_horizon():
    """Every state and flow of the 168 h horizon reproduces the
    pre-migration model, not just the aggregates."""
    _, _, results = _solve_zone(n_steps=168)

    expected = pd.read_csv(golden("zone_168h.csv"), index_col=0)
    actual = results["thermalzone"]["timeseries"]
    expected.index = actual.index

    for column in ["Heating Load", "Cooling Load", "T_air", "T_s", "T_m", "T_e"]:
        deviation = np.abs(actual[column].values - expected[column].values).max()
        assert deviation < 1e-5, "{} deviates by {:.2e}".format(column, deviation)


def test_parity_full_year_aggregates():
    """Annual and monthly heat demand reproduce the pre-migration model.

    Per-time-step equality is deliberately not asserted here: with the
    comfort band open, at a constant heat price the T_m trajectory is
    degenerate and a different solver path may pick another optimum of equal
    cost. The aggregates are what the results are used for.
    """
    _, _, results = _solve_zone()
    expected = json.load(open(golden("zone_year_aggregates.json")))
    zone = results["thermalzone"]

    heat = zone["timeseries"]["Heating Load"]
    assert heat.sum() == pytest.approx(expected["annual_heat_kWh"], rel=1e-3)
    assert zone["timeseries"]["Cooling Load"].sum() == pytest.approx(
        expected["annual_cool_kWh"], rel=1e-3
    )
    assert zone["static"]["Capacity"] == pytest.approx(
        expected["design_heat_load_kW"], rel=1e-9
    )

    monthly = heat.groupby(heat.index.month).sum()
    for month, value in expected["monthly_heat_kWh"].items():
        assert monthly[int(month)] == pytest.approx(value, rel=1e-2)


def test_parity_full_year_series():
    """The full-year heating load series itself, at the 6 significant
    digits the golden fixture stores."""
    _, _, results = _solve_zone()

    expected = pd.read_csv(golden("zone_year.csv.gz"), index_col=0)["Heating Load"]
    actual = results["thermalzone"]["timeseries"]["Heating Load"]
    assert len(actual) == len(expected) == 8760

    deviation = np.abs(actual.values - expected.values)
    assert deviation.max() < 1e-3, "max deviation {:.2e} kW".format(deviation.max())


# --- structure -------------------------------------------------------


def test_zone_is_an_lp():
    """The zone creates no binary variables: the heat load simulation
    stays a pure LP."""
    model, _, _ = _solve_zone(n_steps=168)

    for var in model.component_data_objects(po.Var):
        assert not var.is_binary(), "unexpected binary variable {}".format(var.name)


def test_comfort_band_respected():
    """The comfort band bounds the air temperature in both directions."""
    params, inputs, _ = zone_fixture(n_steps=168)
    _, _, results = _solve_zone(n_steps=168)
    T_air = results["thermalzone"]["timeseries"]["T_air"]

    assert (T_air.values <= inputs["comfort_ub"] + 1e-4).all()
    assert (T_air.values >= inputs["comfort_lb"] - 1e-4).all()


def test_heat_flows_through_the_bus():
    """The zone's heating is the solph edge from the heat bus, so the bus
    balance ties it to whatever supplies it."""
    model, nodes, results = _solve_zone(n_steps=48)

    zone = nodes["thermalzone"]
    supply = nodes["heat_supply"]
    heat_bus = zone.heat_bus

    from_supply = np.array(
        [po.value(model.flow[supply, heat_bus, t]) for t in model.TIMESTEPS]
    )
    into_zone = results["thermalzone"]["timeseries"]["Heating Load"].values
    assert np.abs(from_supply - into_zone).max() < 1e-6


def test_objective_is_energy_cost():
    """With no load violation the objective is exactly the priced energy
    of the two supply sources."""
    model, _, results = _solve_zone(n_steps=168)
    zone = results["thermalzone"]

    heat = zone["timeseries"]["Heating Load"].sum()
    cool = zone["timeseries"]["Cooling Load"].sum()
    violation = zone["max_load_violation"] or 0.0

    expected = heat * HEAT_COST + cool * COOL_COST + violation * 100.0
    assert objective_value(model) == pytest.approx(expected, rel=1e-6)


# --- the explicit parameter contract ---------------------------------


def test_series_length_must_match_the_time_index():
    """A zone whose series are shorter than the model horizon is rejected
    before the solver sees it, not silently truncated - and every offending
    key is named at once."""
    params, inputs, index = zone_fixture(n_steps=168)
    short = {key: values[:100] for key, values in inputs.items()}

    with pytest.raises(ValueError, match="wrong number of steps") as excinfo:
        build_system(zone_spec(params), short, index)
    for key in inputs:
        assert "{} (100 of 168)".format(key) in str(excinfo.value)


def test_series_must_agree_with_each_other():
    """Inconsistent series lengths are caught in the component itself."""
    from esmkit import ThermalZone5R1C
    from oemof import solph

    params, inputs, _ = zone_fixture(n_steps=168)
    values = dict(inputs)
    values["gain_mass"] = values["gain_mass"][:100]

    with pytest.raises(ValueError, match="differing lengths"):
        ThermalZone5R1C(
            "thermalzone", heat_bus=solph.Bus(label="heat"), **params, **values
        )


def test_design_capacity_defaults_to_max_load():
    """A zone reports its max_load as capacity unless told otherwise."""
    from esmkit import ThermalZone5R1C
    from oemof import solph

    params, inputs, _ = zone_fixture(n_steps=168)
    params = dict(params)
    params.pop("design_capacity")

    zone = ThermalZone5R1C(
        "thermalzone", heat_bus=solph.Bus(label="heat"), **params, **inputs
    )
    assert zone.design_capacity == zone.max_load
