import pytest

from esmkit import build_system, node_results, solve
from esmkit.core import solverutils

from conftest import zone_fixture, zone_spec


DOOMED = {"time_limit": 1e-6}


def _zone_system():
    params, inputs, index = zone_fixture(n_steps=168)
    return build_system(zone_spec(params), inputs, index)


def test_highs_retries_with_other_settings(monkeypatch):
    monkeypatch.setattr(solverutils, "HIGHS_FALLBACKS", (DOOMED, {}))

    es, nodes = _zone_system()
    model, results = solve(es, solver="highs")

    assert results.termination_condition.name == "optimal"

    heat = node_results(model, nodes)["thermalzone"]["timeseries"]["Heating Load"]
    assert heat.sum() > 0


def test_highs_reports_every_attempt_when_all_fail(monkeypatch):
    monkeypatch.setattr(solverutils, "HIGHS_FALLBACKS", (DOOMED,))

    es, _ = _zone_system()
    with pytest.raises(RuntimeError, match="no optimal solution"):
        solve(es, solver="highs")


def test_explicit_options_are_used_as_given(monkeypatch):
    monkeypatch.setattr(solverutils, "HIGHS_FALLBACKS", ({},))

    es, _ = _zone_system()
    with pytest.raises(RuntimeError):
        solve(es, solver="highs", solverOpts=DOOMED)


def test_glpk_is_refused_with_a_reason():
    with pytest.raises(ValueError, match="glpk"):
        solverutils.solve_model(None, solver="glpk")
