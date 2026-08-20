# -*- coding: utf-8 -*-
"""
Shared fixtures/helpers for the esmkit test suite.

The zone fixture in `data/golden/` was captured once from tsib (see
`data/golden/_capture.py`); nothing here imports tsib, and the suite runs
without any building data.
"""

import json
import os

import numpy as np
import pandas as pd

from esmkit import SystemSpec

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden")

#: energy prices [EUR/kWh] the golden zone results were produced with. Their
#: absolute level is irrelevant to the resulting load as long as both are
#: positive; they exist so the objective has something to minimize.
HEAT_COST = 0.08
COOL_COST = 0.02

#: time series the zone reads out of the inputs mapping
ZONE_SERIES = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]


def golden(filename):
    """Path of a golden reference fixture."""
    return os.path.join(GOLDEN_DIR, filename)


def capture_meta():
    """Provenance of the fixture: which tsib commit it was captured from."""
    with open(golden("capture_meta.json")) as handle:
        return json.load(handle)


def _timezone():
    return capture_meta()["timezone"]


def zone_fixture(n_steps=None):
    """
    The captured 5R1C parameters, inputs and time index.

    Parameters
    ----------
    n_steps: int, optional
        Truncate to the first n_steps hours. Defaults to the full year.

    Returns
    -------
    (params, inputs, timeindex) - params is the scalar dict the zone
    component takes, inputs the mapping its "@" references resolve against.
    """
    horizon = "168h" if n_steps == 168 else "year"
    source = "zone_inputs_168h.csv" if horizon == "168h" else "zone_inputs_year.csv.gz"

    with open(golden("zone_params_{}.json".format(horizon))) as handle:
        params = json.load(handle)

    frame = pd.read_csv(golden(source), index_col=0)
    # stamps are stored in UTC because a year of local time carries two
    # offsets; converting back is what makes month boundaries local again
    frame.index = pd.to_datetime(frame.index, utc=True).tz_convert(_timezone())
    if n_steps is not None:
        if n_steps > len(frame):
            raise ValueError(
                "Fixture covers {} steps, {} requested".format(len(frame), n_steps)
            )
        frame = frame.iloc[:n_steps]

    inputs = {key: frame[key].to_numpy(dtype=float) for key in ZONE_SERIES}
    return params, inputs, frame.index


def zone_spec(params, heat_cost=HEAT_COST, cool_cost=COOL_COST, **zone_kwargs):
    """
    Bare heat load system: a thermal zone supplied by priced heat and
    cooling sources. The smallest system a zone can sit in, and the shape
    the golden results were produced from.

    `zone_kwargs` overrides zone parameters (max_load, initial_T_m, ...).
    """
    values = dict(params)
    values.update(zone_kwargs)

    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")
    spec.add_component("heat_supply", "source", bus="heat", price=heat_cost)
    spec.add_component("cool_supply", "source", bus="cool", price=cool_cost)
    spec.add_component(
        "thermalzone",
        "zone5r1c",
        heat_bus="heat",
        cool_bus="cool",
        **{key: "@" + key for key in ZONE_SERIES},
        **values,
    )
    return spec


def flat_inputs(n, **series):
    """An inputs mapping of constant or given arrays, for systems without a zone."""
    return {
        key: np.full(n, value) if np.isscalar(value) else np.asarray(value, dtype=float)
        for key, value in series.items()
    }
