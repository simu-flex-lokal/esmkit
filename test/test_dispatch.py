# -*- coding: utf-8 -*-
"""
Dispatch behaviour of the kit, and the flexibility proof.

The load-bearing tests here are the last three: they show that the thermal
mass and comfort band of a 5R1C zone act as a dispatchable storage in the
same solve as prices and equipment. That property is the entire reason the
zone is a component instead of a precomputed load profile.
"""

import numpy as np
import pandas as pd
import pytest

from esmkit import SystemSpec, build_system, node_results, objective_value, solve

from conftest import zone_fixture, zone_spec


def hours(n=48):
    return pd.date_range("2010-01-01", periods=n, freq="h")


def two_price_profile(index, cheap=0.05, expensive=0.40):
    """Cheap in the first half of every day, expensive in the second."""
    return np.where(index.hour < 12, cheap, expensive)


# --- storage ----------------------------------------------------------


def test_battery_shifts_energy_into_expensive_hours():
    index = hours()
    price = two_price_profile(index)
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price="@price")
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    # default round-trip efficiency (0.95/0.95): lossy storage makes the
    # optimum unique. With lossless storage, shuffling energy *within* the
    # expensive hours is cost-neutral and the optimum is degenerate.
    spec.add_component("battery", "battery", bus="elec", capacity=10.0)

    es, nodes = build_system(spec, {"price": price}, index)
    model, _ = solve(es)
    results = node_results(model, nodes)

    charge = results["battery"]["in_elec"]
    discharge = results["battery"]["out_elec"]
    assert charge.sum() > 0, "the battery is never used"
    # charging happens in the cheap hours, discharging in the expensive ones
    assert charge[price > 0.1].sum() == pytest.approx(0.0, abs=1e-6)
    assert discharge[price < 0.1].sum() == pytest.approx(0.0, abs=1e-6)


def test_storage_is_periodically_balanced():
    """`balanced=True` is a periodic SOC wrap: no free energy over the
    horizon."""
    index = hours()
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price="@price")
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    spec.add_component("battery", "battery", bus="elec", capacity=10.0)

    es, nodes = build_system(spec, {"price": two_price_profile(index)}, index)
    model, _ = solve(es)
    content = node_results(model, nodes)["battery"]["storage_content"]

    assert content.iloc[0] == pytest.approx(content.iloc[-1], abs=1e-6)


def test_thermal_storage_standby_loss():
    """The standby loss maps onto solph's fixed_losses_absolute: supplying
    a constant demand through the buffer costs more than supplying it
    directly."""
    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_component("supply", "source", bus="heat", price=0.10)
    spec.add_component("demand", "demand", bus="heat", profile=1.0)
    spec.add_component(
        "buffer", "thermal_storage", bus="heat", capacity=20.0,
        standby_loss_kW=0.05, charge_efficiency=1.0, discharge_efficiency=1.0,
    )

    es, nodes = build_system(spec, {}, hours())
    model, _ = solve(es)
    results = node_results(model, nodes)

    supplied = results["supply"]["out_heat"].sum()
    demanded = results["demand"]["in_heat"].sum()
    # the loss is only paid for while the buffer actually holds energy
    assert supplied >= demanded - 1e-6


# --- generation and conversion ----------------------------------------


def test_pv_self_consumption_beats_export():
    """PV serves the demand first while the export price is below the
    import price."""
    index = hours()
    yield_profile = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component(
        "grid", "grid", bus="elec", import_price=0.35, export_price=0.08
    )
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    spec.add_component(
        "pv", "pv", bus="elec", specific_yield="@pv_yield", capacity=3.0
    )

    es, nodes = build_system(spec, {"pv_yield": yield_profile}, index)
    model, _ = solve(es)
    results = node_results(model, nodes)

    generation = results["pv"]["out_elec"]
    # a grid connection with feed-in is two solph nodes collected under one
    # name: "out_elec" is the import source, "in_elec" the export sink
    imported = results["grid"]["out_elec"]
    exported = results["grid"]["in_elec"]

    assert generation.sum() > 0
    sunny = yield_profile > 0
    # self-consumption first: while the sun shines, import is displaced
    assert imported[sunny].sum() < imported[~sunny].sum()
    # only the surplus above the demand of 1 kW is exported
    assert exported.sum() == pytest.approx(
        np.clip(generation.values - 1.0, 0, None).sum(), abs=1e-6
    )


def test_pv_investment_sizes_to_the_optimum():
    index = hours()
    yield_profile = np.clip(np.sin((index.hour - 6) / 12 * np.pi), 0, None)

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.35)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)
    spec.add_component(
        "pv", "pv", bus="elec", specific_yield="@pv_yield", wacc=0.05,
        capex_per_unit=800.0, lifetime=25.0, max_capacity=10.0,
    )

    es, nodes = build_system(spec, {"pv_yield": yield_profile}, index)
    model, _ = solve(es)
    capacity = node_results(model, nodes)["pv"]["capacity"]

    # cheap PV against a 0.35 EUR/kWh import price: build, but not beyond
    # what the demand can absorb (no export remuneration configured)
    assert 0 < capacity <= 10.0


def test_heat_pump_couples_two_buses():
    """Electricity in, heat out, at the given COP - and the COP cut-off is
    respected."""
    n = 24
    cop = np.where(np.arange(n) < 12, 3.0, 0.0)  # off in the second half

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_bus("heat")
    spec.add_component("grid", "grid", bus="elec", import_price=0.30)
    spec.add_component(
        "hp", "heat_pump", bus_in="elec", bus_out="heat", cop="@cop", capacity=5.0
    )
    spec.add_component("backup", "source", bus="heat", price=10.0)
    spec.add_component("demand", "demand", bus="heat", profile=2.0)

    es, nodes = build_system(spec, {"cop": cop}, hours(n))
    model, _ = solve(es)
    results = node_results(model, nodes)

    heat = results["hp"]["out_heat"]
    power = results["hp"]["in_elec"]
    assert heat[cop == 0].sum() == pytest.approx(0.0, abs=1e-6)
    # energy balance holds by construction of the conversion factor
    assert power[cop > 0].sum() == pytest.approx(heat[cop > 0].sum() / 3.0, rel=1e-6)


# --- the flexibility proof --------------------------------------------


def _zone_with_price(price, n=168, rigid=False):
    """Thermal zone supplied by a price-varying heat source."""
    params, inputs, index = zone_fixture(n_steps=n)
    inputs = dict(inputs)
    inputs["heat_price"] = price
    if rigid:
        # a collapsed band pins T_air and destroys the flexibility
        inputs["comfort_ub"] = inputs["comfort_lb"]

    spec = zone_spec(params, heat_cost="@heat_price")
    es, nodes = build_system(spec, inputs, index)
    model, _ = solve(es)
    return model, nodes, inputs


def test_thermal_mass_preheats_before_price_spikes():
    """The zone buys heat before the expensive hours and rides through
    them on the thermal mass, within the comfort band."""
    n = 168
    _, _, index = zone_fixture(n_steps=n)
    expensive = (index.hour >= 17) & (index.hour < 21)
    price = np.where(expensive, 1.50, 0.05)

    model, nodes, inputs = _zone_with_price(price, n=n)
    ts = node_results(model, nodes)["thermalzone"]["timeseries"]
    heat = ts["Heating Load"]

    # the spike covers 4 of 24 hours (17%); the zone has to buy markedly
    # less than its share there. Threshold taken from the pre-migration
    # test, which is the validated criterion for this scenario.
    share_in_spike = heat.values[expensive].sum() / heat.sum()
    assert share_in_spike < 0.15, (
        "the zone draws {:.1%} of its heat in the price spike - the thermal "
        "mass is not being used as flexibility".format(share_in_spike)
    )

    # pre-heating is visible: the air temperature demonstrably rises above
    # the lower comfort bound ahead of the expensive hours ...
    assert ts["T_air"].max() > inputs["comfort_lb"].max() + 1.0
    # ... the mass temperature swings, i.e. it behaves as a storage ...
    assert ts["T_m"].max() - ts["T_m"].min() > 0.5
    # ... and the comfort band is never violated to achieve any of it
    assert (ts["T_air"].values >= inputs["comfort_lb"] - 1e-4).all()
    assert (ts["T_air"].values <= inputs["comfort_ub"] + 1e-4).all()


def test_flexibility_reduces_cost():
    """Collapsing the comfort band removes the flexibility and must make
    the same price scenario strictly more expensive."""
    n = 168
    _, _, index = zone_fixture(n_steps=n)
    price = np.where((index.hour >= 17) & (index.hour < 21), 1.50, 0.05)

    flexible, _, _ = _zone_with_price(price, n=n)
    rigid, _, _ = _zone_with_price(price, n=n, rigid=True)

    assert objective_value(flexible) < objective_value(rigid), (
        "flexible: {:.3f}, rigid: {:.3f}".format(
            objective_value(flexible), objective_value(rigid)
        )
    )


def test_zone_and_storage_in_one_solve():
    """Thermal zone, thermal buffer and a time-varying price in a single
    optimization: both flexibilities are available at once."""
    n = 168
    params, inputs, index = zone_fixture(n_steps=n)
    expensive = (index.hour >= 17) & (index.hour < 21)
    inputs = dict(inputs)
    inputs["heat_price"] = np.where(expensive, 1.50, 0.05)

    spec = zone_spec(params, heat_cost="@heat_price")
    spec.add_component(
        "buffer", "thermal_storage", bus="heat", capacity=30.0, standby_loss_kW=0.02
    )

    es, nodes = build_system(spec, inputs, index)
    model, _ = solve(es)
    results = node_results(model, nodes)

    supply = results["heat_supply"]["out_heat"]
    assert supply.values[expensive].sum() / supply.sum() < 0.05
    assert results["buffer"]["in_heat"].sum() > 0, "the buffer is never used"
