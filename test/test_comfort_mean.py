"""The window-mean comfort setpoint of the 5R1C zone: off by default, the
same as the constraint written by hand, energy neutral over a cyclic
horizon, and strict about its inputs."""


import numpy as np
import pyomo.environ as po
import pytest
from oemof import solph

from esmkit import SystemSpec, ThermalZone5R1C, build_system, node_results, objective_value, solve

from conftest import ZONE_SERIES, zone_fixture


N = 168


def _price(n):
    """A daily price with a cheap night and an expensive evening, EUR/kWh."""
    hour = np.arange(n) % 24
    return 0.08 + 0.03 * np.sin(2 * np.pi * (hour - 9) / 24) + 0.02 * (hour == 18)


def _build(lb, ub, price, setpoint=None, window=None, penalty=10.0):
    params, inputs, index = zone_fixture(n_steps=N)
    inputs = dict(inputs)
    inputs.update(comfort_lb=np.full(N, float(lb)), comfort_ub=np.full(N, float(ub)), price=price)
    zone = dict(params)
    if setpoint is not None:
        inputs["comfort_setpoint"] = setpoint
        inputs["comfort_window"] = window
        zone.update(comfort_setpoint="@comfort_setpoint", comfort_window="@comfort_window")

    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_component("heat_supply", "source", bus="heat", price="@price")
    spec.add_component(
        "thermalzone", "zone5r1c", heat_bus="heat", can_cool=False,
        comfort_ub_penalty=penalty, **{k: "@" + k for k in ZONE_SERIES}, **zone,
    )
    es, nodes = build_system(spec, inputs, index)
    return solph.Model(es), nodes, index


def _solve(model, nodes, index):
    model, _ = solve(model)
    return model, node_results(model, nodes, index=index)["thermalzone"]


def _days():
    return (np.arange(N) // 24).astype(float)


def test_off_by_default():
    model, _, _ = _build(19, 21, _price(N))
    assert len(model.ThermalZone5R1CBlock.comfort_mean) == 0


def test_nan_setpoint_counts_nothing():
    plain, nodes_p, index = _build(19, 21, _price(N))
    nan, nodes_n, _ = _build(19, 21, _price(N), setpoint=np.full(N, np.nan), window=_days())
    assert len(nan.ThermalZone5R1CBlock.comfort_mean) == 0

    plain, _ = _solve(plain, nodes_p, index)
    nan, _ = _solve(nan, nodes_n, index)
    assert objective_value(nan) == pytest.approx(objective_value(plain), rel=1e-9)


def test_matches_the_constraint_written_by_hand():
    """The component must give exactly what the prototype added to the
    pyomo model after building it."""
    price = _price(N)
    built, nodes_b, index = _build(19, 21, price, setpoint=np.full(N, 20.0), window=_days())
    built, _ = _solve(built, nodes_b, index)

    by_hand, nodes_h, _ = _build(19, 21, price)
    block = by_hand.ThermalZone5R1CBlock
    zone = list(block.ZONES)[0]
    days = [list(range(d * 24, (d + 1) * 24)) for d in range(N // 24)]
    by_hand.window_mean = po.Constraint(
        range(len(days)),
        rule=lambda m, d: sum(block.T_air[zone, t] - block.comfort_violation[zone, t] for t in days[d])
        >= 20.0 * len(days[d]),
    )
    by_hand, _ = _solve(by_hand, nodes_h, index)

    assert objective_value(built) == pytest.approx(objective_value(by_hand), rel=1e-8)


def test_window_mean_is_met_and_band_is_used_both_ways():
    model, nodes, index = _build(19, 21, _price(N), setpoint=np.full(N, 20.0), window=_days())
    model, zone = _solve(model, nodes, index)
    ts = zone["timeseries"]
    credit = (ts["T_air"] - ts["Comfort Violation"]).values.reshape(-1, 24).mean(axis=1)

    assert (credit >= 20.0 - 1e-6).all()
    assert ts["T_air"].min() < 19.5 and ts["T_air"].max() > 20.5
    assert (ts["Comfort Setpoint"] == 20.0).all()


def test_shifting_around_the_setpoint_costs_no_heat():
    """Over a cyclic horizon the total heat depends only on the mean air
    temperature, so one window over the whole horizon must need exactly the
    heat of holding the setpoint."""
    price = _price(N)
    fixed, nodes_f, index = _build(20, 20, price)
    fixed, fixed_zone = _solve(fixed, nodes_f, index)
    assert fixed_zone["comfort_violation_Kh"] < 1e-6

    mean, nodes_m, _ = _build(19, 21, price, setpoint=np.full(N, 20.0), window=np.zeros(N))
    mean, mean_zone = _solve(mean, nodes_m, index)

    heat_fixed = fixed_zone["timeseries"]["Heating Load"].sum()
    heat_mean = mean_zone["timeseries"]["Heating Load"].sum()
    assert heat_mean == pytest.approx(heat_fixed, rel=1e-6)
    assert objective_value(mean) < objective_value(fixed) - 1.0


def test_window_ids_need_not_be_contiguous():
    hour = np.arange(N) % 24
    awake = np.where((hour >= 6) & (hour < 22), _days(), np.nan)
    model, _, _ = _build(19, 21, _price(N), setpoint=np.full(N, 20.0), window=awake)
    zone = list(model.ThermalZone5R1CBlock.ZONES)[0]

    assert len(zone.comfort_windows) == N // 24
    assert all(len(steps) == 16 for steps in zone.comfort_windows)
    assert all(6 <= t % 24 < 22 for steps in zone.comfort_windows for t in steps)


def _zone_kwargs(**series):
    params, inputs, _ = zone_fixture(n_steps=N)
    values = dict(inputs)
    values.update(series)
    return dict(params, **values)


def test_setpoint_and_window_go_together():
    with pytest.raises(ValueError, match="together"):
        ThermalZone5R1C("z", heat_bus=solph.Bus(label="heat"), **_zone_kwargs(comfort_setpoint=np.full(N, 20.0)))
    with pytest.raises(ValueError, match="together"):
        ThermalZone5R1C("z", heat_bus=solph.Bus(label="heat"), **_zone_kwargs(comfort_window=_days()))


def test_setpoint_outside_the_band_is_rejected():
    kwargs = _zone_kwargs(
        comfort_lb=np.full(N, 19.0), comfort_ub=np.full(N, 21.0),
        comfort_setpoint=np.full(N, 21.5), comfort_window=_days(),
    )
    with pytest.raises(ValueError, match="outside"):
        ThermalZone5R1C("z", heat_bus=solph.Bus(label="heat"), **kwargs)


def test_setpoint_outside_the_band_is_fine_where_it_does_not_count():
    setpoint = np.full(N, 20.0)
    setpoint[:24] = np.nan
    lb = np.full(N, 19.0)
    lb[:24] = -50.0
    zone = ThermalZone5R1C(
        "z", heat_bus=solph.Bus(label="heat"),
        **_zone_kwargs(comfort_lb=lb, comfort_ub=np.full(N, 21.0), comfort_setpoint=setpoint, comfort_window=_days()),
    )
    assert len(zone.comfort_windows) == N // 24 - 1


def test_setpoint_length_must_agree():
    with pytest.raises(ValueError, match="differing lengths"):
        ThermalZone5R1C(
            "z", heat_bus=solph.Bus(label="heat"),
            **_zone_kwargs(comfort_setpoint=np.full(N - 1, 20.0), comfort_window=_days()[:-1]),
        )
