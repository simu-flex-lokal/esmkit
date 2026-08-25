"""Shared fixtures: paths into the golden data set, the captured 5R1C zone
parameters, and the small spec every zone test is built on."""


import json
import os

import numpy as np
import pandas as pd

from esmkit import SystemSpec

GOLDEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "golden")


HEAT_COST = 0.08
COOL_COST = 0.02


ZONE_SERIES = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]


def golden(filename):
    """Return the absolute path of a file in the golden data directory."""
    return os.path.join(GOLDEN_DIR, filename)


def capture_meta():
    """Return the provenance metadata written by ``data/golden/_capture.py``."""
    with open(golden("capture_meta.json")) as handle:
        return json.load(handle)


def _timezone():
    return capture_meta()["timezone"]


def zone_fixture(n_steps=None):
    """Load the captured zone parameters, input series and time index.

    ``n_steps=168`` uses the short fixture, anything else the full year; a
    smaller ``n_steps`` truncates the year to that many leading steps.

    Returns:
        ``(params, inputs, index)`` ready to hand to ``build_system``.
    """
    horizon = "168h" if n_steps == 168 else "year"
    source = "zone_inputs_168h.csv" if horizon == "168h" else "zone_inputs_year.csv.gz"

    with open(golden("zone_params_{}.json".format(horizon))) as handle:
        params = json.load(handle)

    frame = pd.read_csv(golden(source), index_col=0)

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
    """Build the minimal spec around one zone: a heat and a cool supply.

    Both supplies are priced sources, so the objective is pure energy cost and
    the zone alone decides when to draw heat. ``zone_kwargs`` override the
    captured parameters.
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
    """Expand scalars into constant ``n``-step arrays, passing arrays through."""
    return {
        key: np.full(n, value) if np.isscalar(value) else np.asarray(value, dtype=float)
        for key, value in series.items()
    }
