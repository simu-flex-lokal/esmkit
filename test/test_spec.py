# -*- coding: utf-8 -*-
"""
The spec: what a system description is, how profile references resolve, and
what is checked before anything is built.

Deliberately without a thermal zone, so these stay fast and isolate the
description from any physics.
"""

import numpy as np
import pandas as pd
import pytest

from esmkit import (
    SystemSpec,
    build_system,
    capacity_params,
    check_inputs,
    required_inputs,
    resolve_profile,
    validate,
)
from esmkit.core.spec import SPEC_VERSION


def elec_spec(**demand):
    spec = SystemSpec()
    spec.add_bus("elec", carrier="electricity")
    spec.add_component("grid", "grid", bus="elec", import_price=0.3)
    spec.add_component("demand", "demand", bus="elec", **(demand or {"profile": 1.0}))
    return spec


def hours(n=24):
    return pd.date_range("2010-01-01", periods=n, freq="h")


# --- the spec is data -------------------------------------------------


def test_spec_round_trip():
    """A spec is plain data: it survives a JSON round trip unchanged."""
    spec = elec_spec(profile="@elecLoad")
    restored = SystemSpec.from_json(spec.to_json())
    assert restored.to_dict() == spec.to_dict()


def test_spec_carries_its_schema_version():
    """The exchange format is versioned, so a reader can refuse a layout it
    does not know instead of misinterpreting it."""
    assert SystemSpec().to_dict()["version"] == SPEC_VERSION

    with pytest.raises(ValueError, match="version '99'"):
        SystemSpec.from_dict({"version": "99", "buses": {}, "components": {}})


def test_spec_rejects_duplicates():
    spec = SystemSpec()
    spec.add_bus("elec")
    with pytest.raises(ValueError, match="Duplicate bus"):
        spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec")
    with pytest.raises(ValueError, match="Duplicate component"):
        spec.add_component("grid", "grid", bus="elec")


def test_spec_copy_is_deep():
    spec = elec_spec()
    clone = spec.copy()
    clone.components["grid"]["import_price"] = 9.9
    assert spec.components["grid"]["import_price"] == 0.3


def test_capacity_params_offers_two_forms():
    """A number is a fixed capacity, a dict an investment decision."""
    assert capacity_params(8.0) == {"capacity": 8.0}
    invest = {"capex_per_unit": 1200.0, "lifetime": 25.0}
    assert capacity_params(invest) == invest
    assert capacity_params(invest) is not invest


# --- profile resolution ------------------------------------------------


def test_profile_reference_resolution():
    inputs = {"load": pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])}
    values = resolve_profile("@load", inputs, 5, "test")
    assert isinstance(values, np.ndarray)
    assert values.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_scalars_pass_through():
    """Scalars stay scalars so solph broadcasts them itself."""
    assert resolve_profile(0.35, {}, 24, "test") == 0.35


def test_profile_errors_are_specific():
    inputs = {"load": pd.Series([1.0, 2.0])}
    with pytest.raises(KeyError, match="missing"):
        resolve_profile("@missing", inputs, 5, "test")
    with pytest.raises(ValueError, match="has 2 values, expected 5"):
        resolve_profile("@load", inputs, 5, "test")
    with pytest.raises(ValueError, match="must be an input reference"):
        resolve_profile("elecLoad", inputs, 5, "test")


def test_required_inputs_is_the_contract():
    """What a spec will look up, inspectable before anything is built."""
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", bus="elec", import_price="@price")
    spec.add_component("demand", "demand", bus="elec", profile="@load")
    spec.add_component("pv", "pv", bus="elec", specific_yield=0.4, capacity=5.0)

    assert required_inputs(spec) == ["load", "price"]
    assert required_inputs(SystemSpec()) == []


# --- validation before building ---------------------------------------


def test_validate_reports_every_problem_at_once():
    """A generated spec should be fixable in one pass, not one error at a
    time."""
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("thing", "wind_turbine", bus="elec")
    spec.add_component("grid", "grid", bus="heat", import_price=0.3)

    with pytest.raises(ValueError) as excinfo:
        validate(spec)
    message = str(excinfo.value)
    assert "unknown type 'wind_turbine'" in message
    assert "undeclared bus 'heat'" in message


def test_validate_finds_suffixed_bus_parameters():
    """Bus parameters are named `bus`, `bus_in` or `heat_bus` - all three
    have to be checked, or a zone's typo slips through."""
    spec = SystemSpec()
    spec.add_bus("heat")
    spec.add_component("hp", "heat_pump", bus_in="elec", bus_out="heat", cop=3.0)

    with pytest.raises(ValueError, match="undeclared bus 'elec'"):
        validate(spec)


def test_unknown_component_type_lists_alternatives():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("thing", "wind_turbine", bus="elec")
    with pytest.raises(ValueError, match="available: battery"):
        build_system(spec, {}, hours())


def test_missing_bus_parameter_is_rejected():
    spec = SystemSpec()
    spec.add_bus("elec")
    spec.add_component("grid", "grid", import_price=0.3)
    with pytest.raises(ValueError, match="needs a 'bus'"):
        build_system(spec, {}, hours())


def test_check_inputs_names_what_is_missing():
    """Reported up front rather than as a KeyError halfway through the
    build."""
    spec = elec_spec(profile="@load")

    with pytest.raises(KeyError, match="load"):
        check_inputs(spec, {}, 24)
    with pytest.raises(KeyError, match="load"):
        check_inputs(spec, {"load": None}, 24)
    with pytest.raises(ValueError, match=r"load \(10 of 24\)"):
        check_inputs(spec, {"load": np.ones(10)}, 24)

    check_inputs(spec, {"load": np.ones(24)}, 24)
    check_inputs(spec, {"load": 1.5}, 24)


def test_build_system_checks_inputs_before_building():
    spec = elec_spec(profile="@load")
    with pytest.raises(KeyError, match="load"):
        build_system(spec, {}, hours())
