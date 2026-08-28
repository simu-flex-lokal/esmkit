"""Re-capture the golden fixture that pins the 5R1C zone to its origin.

The zone in this kit is a port: the same physics used to live inside ``tsib``
(``tsib.optimization.zone5r1c``), built there from a full building
configuration. To prove the port did not change any number, the inputs and the
expected results of one concrete building were frozen and are replayed by
``test/test_zone.py``.

The building is the EPISCOPE single-family archetype
``DE.N.SFH.06.Gen.ReEx.001.001`` (173.25 m2 reference area, comfort band
20-26 degC) under DWD test reference year weather region TRY_4, on a 168 h and
a full 8760 h horizon, solved with HiGHS. Provenance of the two halves is
recorded in ``capture_meta.json`` (this side: the tsib commit the inputs were
read from) and ``golden_meta.json`` (the other side: the pre-migration
``tsib.energysystem`` stack the expected results came from).

**It no longer runs against current tsib.** The 5R1C component it imports was removed
from tsib on 2026-08-28 (phase 2 of the split); the equivalent numbers now come from
``tsib.envelope.zone_parameters``, which tsib pins against *this* fixture in its own
``test/test_envelope_contract.py``. Re-capturing from today's tsib would therefore
re-pin parity rather than check it, which is why this script is kept as it was: it
documents where the frozen data came from, and it runs only against a tsib checkout at
the commit recorded in ``capture_meta.json``.

This script writes the input half - it instantiates the old ``tsib`` zone,
reads its resolved scalars and node-gain series back out, and stores them as
``zone_params_*.json`` and ``zone_inputs_*.csv`` - then copies the expected
results across unchanged. It therefore only runs in a checkout that has
``tsib`` as a sibling directory, and is not part of the test suite: the
fixture is deliberately static, and re-running this is a decision to re-pin
parity to a new tsib commit.
"""


import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TSIB = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "tsib"))
sys.path.insert(0, os.path.join(TSIB, "test"))

from conftest import golden_zone_cfg

from tsib.optimization import ThermalZone5R1C
from tsib.optimization.zone5r1c import ENVELOPE_ELEMENTS


EXPECTED_FILES = [
    "golden_meta.json",
    "zone_168h.csv",
    "zone_year.csv.gz",
    "zone_year_aggregates.json",
]

SERIES = ["T_e", "gain_mass", "gain_surface", "comfort_lb", "comfort_ub"]


def zone_parameters(cfg):
    from oemof import solph

    from tsib.optimization.zone5r1c import comfort_bounds

    n = len(cfg["weather"].index)

    zone = ThermalZone5R1C("capture", cfg, heat_bus=solph.Bus(label="heat"))
    zone.prepare(n)

    lower, upper = comfort_bounds(cfg, n)

    return {
        "scalars": {
            "H_ms": float(zone.config.H_ms),
            "H_is": float(zone.config.H_is),
            "H_door": float(zone.config.H_door),
            "C_m": float(zone.config.C_m),
            "H": {e: float(zone.H_element(e)) for e in ENVELOPE_ELEMENTS},
            "max_load": float(zone.max_load),
            "design_capacity": float(zone.design_heat_load_value()),
        },
        "series": {
            "T_e": np.asarray(zone._profiles["T_e"], dtype=float),
            "gain_mass": np.array([zone.gain_mass_node(t) for t in range(n)]),
            "gain_surface": np.array([zone.gain_surface_node(t) for t in range(n)]),
            "comfort_lb": np.asarray(lower, dtype=float),
            "comfort_ub": np.asarray(upper, dtype=float),
        },
    }


def _write(horizon, n_steps, compress):
    cfg = golden_zone_cfg(n_steps=n_steps)
    captured = zone_parameters(cfg)

    params_path = os.path.join(HERE, "zone_params_{}.json".format(horizon))
    with open(params_path, "w") as handle:
        json.dump(captured["scalars"], handle, indent=2, sort_keys=True)
        handle.write("\n")

    index = cfg["weather"].index
    frame = pd.DataFrame(
        {key: captured["series"][key] for key in SERIES},

        index=index.tz_convert("UTC"),
    )
    name = "zone_inputs_{}.csv{}".format(horizon, ".gz" if compress else "")
    frame.to_csv(os.path.join(HERE, name), float_format="%.10g")

    print("{}: {} steps -> {}, {}".format(
        horizon, len(frame), os.path.basename(params_path), name))
    return cfg


def main():
    cfg_short = _write("168h", 168, compress=False)
    cfg_year = _write("year", None, compress=True)

    source = os.path.join(TSIB, "test", "data", "golden")
    for filename in EXPECTED_FILES:
        shutil.copy(os.path.join(source, filename), os.path.join(HERE, filename))
    print("copied expectations:", ", ".join(EXPECTED_FILES))

    commit = subprocess.check_output(
        ["git", "-C", TSIB, "rev-parse", "HEAD"], text=True
    ).strip()
    meta = {
        "captured_by": "test/data/golden/_capture.py",
        "captured_from": "tsib",
        "tsib_commit": commit,
        "n_steps_short": len(cfg_short["weather"].index),
        "n_steps_year": len(cfg_year["weather"].index),
        "timezone": str(cfg_year["weather"].index.tz),
    }
    with open(os.path.join(HERE, "capture_meta.json"), "w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("tsib commit:", commit)


if __name__ == "__main__":
    main()
