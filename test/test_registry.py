"""The component registry and the base classes: what the kit registers, how a
third party registers its own, and the guard against unwired pyomo blocks."""


import numpy as np
import pandas as pd
import pytest
from oemof import solph
from oemof.network.network import Node
from pyomo.core.base.block import ScalarBlock

from esmkit import (
    COMPONENT_FACTORIES,
    EsmBlock,
    EsmComponent,
    SystemSpec,
    annuity_factor,
    build_system,
    factory,
    node_results,
    objective_value,
    solve,
)


def hours(n=24):
    return pd.date_range("2010-01-01", periods=n, freq="h")


def test_registry_covers_the_kit():
    expected = {
        "demand", "source", "grid", "meter", "pv",
        "heat_pump", "battery", "thermal_storage", "zone5r1c",
    }
    assert expected <= set(COMPONENT_FACTORIES)


def test_duplicate_factory_is_rejected():
    with pytest.raises(ValueError, match="Duplicate component factory 'grid'"):
        factory("grid")(lambda *args: [])


def test_a_factory_registers_from_outside_the_package():
    from esmkit import bus, capacity

    @factory("chp")
    def _chp(name, params, buses, inputs, n_steps, step_size_h):
        fuel = bus(buses, params, "bus_fuel", name)
        elec = bus(buses, params, "bus_elec", name)
        nominal = capacity(params, n_steps * step_size_h, params.pop("wacc", 0.0))
        return [
            solph.components.Converter(
                label=name,
                inputs={fuel: solph.Flow(nominal_capacity=nominal)},
                outputs={elec: solph.Flow()},
                conversion_factors={elec: params.pop("electrical_efficiency", 0.35)},
            )
        ]

    spec = SystemSpec()
    spec.add_bus("gas")
    spec.add_bus("elec")
    spec.add_component("gas_grid", "source", bus="gas", price=0.08)
    spec.add_component("chp", "chp", bus_fuel="gas", bus_elec="elec", capacity=12.0)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    try:
        es, nodes = build_system(spec, {}, hours())
        model, _ = solve(es)
        results = node_results(model, nodes)

        assert results["chp"]["out_elec"].sum() == pytest.approx(
            results["chp"]["in_gas"].sum() * 0.35
        )
    finally:
        COMPONENT_FACTORIES.pop("chp")


def test_bases_are_complete_or_uninstantiable():
    class NodeWithoutGroup(EsmComponent):
        pass

    class BlockWithoutCreate(EsmBlock):
        pass

    with pytest.raises(TypeError, match="constraint_group"):
        NodeWithoutGroup(label="zone")
    with pytest.raises(TypeError, match="_create"):
        BlockWithoutCreate()

    class Block(EsmBlock):
        def _create(self, group=None):
            pass

    assert hasattr(Block(), "CONSTRAINT_GROUP")


def test_block_without_constraint_group_is_rejected():
    class OrphanBlock(ScalarBlock):
        def _create(self, group=None):
            pass

    class OrphanComponent(Node):
        def constraint_group(self):
            return OrphanBlock

    @factory("orphan")
    def _orphan(name, params, buses, inputs, n_steps, step_size_h):
        return [
            OrphanComponent(label=name, inputs={buses[params["bus"]]: solph.Flow()})
        ]

    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("orphan", "orphan", bus="elec")
    try:
        with pytest.raises(TypeError, match="CONSTRAINT_GROUP"):
            build_system(spec, {}, hours())
    finally:
        COMPONENT_FACTORIES.pop("orphan")


def test_grid_expands_into_two_nodes():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3, export_price=0.08)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    _, nodes = build_system(spec, {}, hours())
    assert isinstance(nodes["grid"], list) and len(nodes["grid"]) == 2
    assert {str(n.label) for n in nodes["grid"]} == {"grid_import", "grid_export"}


def test_grid_without_export_is_one_node():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    _, nodes = build_system(spec, {}, hours())
    assert not isinstance(nodes["grid"], list)


def test_timeindex_covers_every_step():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    es, _ = build_system(spec, {}, hours(8760))
    assert len(es.timeincrement) == 8760


def test_annuity_factor():
    assert annuity_factor(20, 0.0) == pytest.approx(1.0 / 20)
    assert annuity_factor(20, 0.05) == pytest.approx(0.080242, abs=1e-6)


def test_investment_scaling_is_resolution_independent():
    def ep_costs(freq, periods):
        index = pd.date_range("2010-01-01", periods=periods, freq=freq)
        spec = SystemSpec()
        spec.add_bus("elec")
        spec.add_component(
            "pv", "pv", bus="elec", specific_yield=0.5, wacc=0.05,
            capex_per_unit=1000.0, lifetime=20.0,
        )
        spec.add_component("demand", "demand", bus="elec", profile=1.0)
        _, nodes = build_system(spec, {}, index)
        bus = next(iter(nodes["pv"].outputs))

        return nodes["pv"].outputs[bus].investment.ep_costs

    hourly = ep_costs("h", 72)
    quarterly = ep_costs("15min", 288)
    assert hourly[0] == pytest.approx(quarterly[0], rel=1e-12)


def test_bus_balance_and_cost():
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price=0.30)
    spec.add_component("demand", "demand", bus="elec", profile=2.0)

    es, nodes = build_system(spec, {}, hours())
    model, _ = solve(es)

    results = node_results(model, nodes)
    assert results["grid"]["out_elec"].sum() == pytest.approx(48.0)
    assert results["demand"]["in_elec"].sum() == pytest.approx(48.0)
    assert objective_value(model) == pytest.approx(48.0 * 0.30)


def test_meter_is_a_lossless_transfer():
    spec = SystemSpec()
    spec.add_bus("grid_bus")
    spec.add_bus("steuerbar")
    spec.add_component("supply", "source", bus="grid_bus", price=0.0)
    spec.add_component(
        "meter", "meter", bus_up="grid_bus", bus_down="steuerbar", price=0.10
    )
    spec.add_component("demand", "demand", bus="steuerbar", profile=3.0)

    es, nodes = build_system(spec, {}, hours())
    model, _ = solve(es)

    results = node_results(model, nodes)
    assert results["meter"]["in_grid_bus"].sum() == pytest.approx(72.0)
    assert results["meter"]["out_steuerbar"].sum() == pytest.approx(72.0)

    assert objective_value(model) == pytest.approx(72.0 * 0.10)


def test_node_results_shape():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.5)

    es, nodes = build_system(spec, {}, hours(12))
    model, _ = solve(es)
    results = node_results(model, nodes)

    assert set(results) == {"grid", "demand"}
    assert results["demand"]["in_elec"].sum() == pytest.approx(18.0)
    assert results["grid"]["labels"] == ["grid_import"]


def test_a_component_reports_its_own_results():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", profile=1.0)

    es, nodes = build_system(spec, {}, hours(6))
    model, _ = solve(es)

    nodes["demand"].results = lambda model, index: {"custom": 42}
    assert node_results(model, nodes)["demand"] == {"custom": 42}
