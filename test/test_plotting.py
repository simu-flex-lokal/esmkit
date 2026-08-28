"""The result-side helpers: the derived series (``bus_flows``,
``net_grid_exchange``) and the plots built on them.

Nothing here solves a model. The plot functions and the analysis helpers
consume a results dict, so a hand-built dict of pandas Series exercises them
completely and in milliseconds.

Assertions are on the returned Axes rather than on pixels: what matters is
that the right number of artists carry the right labels, that ``window``
actually narrows the data, and that no function draws to a global figure or
calls ``plt.show()``.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from esmkit import SystemSpec, bus_flows, net_grid_exchange  # noqa: E402
from esmkit.plotting import (  # noqa: E402
    plot_dispatch,
    plot_grid_exchange,
    plot_price_response,
    plot_storage,
    use_style,
)

N = 48
INDEX = pd.date_range("2010-01-01", periods=N, freq="h")


def series(values):
    return pd.Series(np.asarray(values, dtype=float), index=INDEX)


@pytest.fixture
def results():
    """A building with PV, a battery, a heat pump and a zone, by hand."""
    hours = INDEX.hour.values
    sun = np.clip(np.sin((hours - 6) / 12 * np.pi), 0, None)
    return {
        "grid": {"out_elec": series(np.full(N, 2.0)),
                 "in_elec": series(sun * 0.5)},
        "pv": {"out_elec": series(sun * 3.0)},
        "battery": {"in_elec": series(np.where(hours < 12, 1.0, 0.0)),
                    "out_elec": series(np.where(hours >= 18, 1.0, 0.0)),
                    "storage_content": pd.Series(np.linspace(0, 5, N + 1))},
        "household_load": {"in_elec": series(np.full(N, 0.8))},
        "heat_pump": {"in_elec": series(np.full(N, 1.5)),
                      "out_heat": series(np.full(N, 4.5))},
        "dhw_load": {"in_heat": series(np.full(N, 0.5))},
        "thermalzone": {
            "timeseries": pd.DataFrame({
                "Heating Load": np.full(N, 4.0),
                "Cooling Load": np.zeros(N),
                "T_air": np.full(N, 21.5),
                "T_m": np.full(N, 21.0),
                "T_e": np.full(N, 2.0),
            }, index=INDEX),
            "static": {},
        },
    }


@pytest.fixture
def price():
    return np.where((INDEX.hour >= 17) & (INDEX.hour < 21), 0.55, 0.15)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


# --- derived series ----------------------------------------------------


def test_bus_flows_signs_supply_positive_and_demand_negative(results):
    flows = bus_flows(results, "elec")

    assert flows["pv"].max() > 0, "generation must be positive"
    assert flows["household_load"].max() <= 0, "a load must be negative"
    assert flows["heat_pump"].max() <= 0, "the heat pump consumes electricity"
    # the battery nets charge against discharge into one column
    assert flows["battery"].min() < 0 < flows["battery"].max()


def test_bus_flows_includes_the_zone_which_reports_no_flows(results):
    """The zone is the biggest heat sink but reports a frame, not flows."""
    heat = bus_flows(results, "heat")

    assert "thermalzone" in heat
    assert heat["thermalzone"].max() <= 0
    assert heat["heat_pump"].min() >= 0


def test_bus_flows_resolves_the_zone_buses_from_a_spec(results):
    spec = SystemSpec()
    spec.add_bus("heat", carrier="heat")
    spec.add_bus("cool", carrier="cool")
    spec.add_component("thermalzone", "zone5r1c", heat_bus="heat",
                       cool_bus="cool")

    assert "thermalzone" in bus_flows(results, "heat", spec=spec)
    assert "thermalzone" not in bus_flows(results, "elec", spec=spec)


def test_bus_flows_is_empty_for_an_unknown_bus(results):
    assert bus_flows(results, "gas").empty


def test_net_grid_exchange_is_import_minus_export(results):
    net = net_grid_exchange(results)

    expected = results["grid"]["out_elec"] - results["grid"]["in_elec"]
    pd.testing.assert_series_equal(net, expected)
    assert net.max() > 0


def test_net_grid_exchange_reports_a_missing_component(results):
    with pytest.raises(KeyError, match="No component"):
        net_grid_exchange(results, name="substation")


# --- the plots ---------------------------------------------------------


def test_every_plot_returns_its_axes(results, price):
    fig, ax = plt.subplots()
    assert plot_dispatch(results, "elec", ax=ax) is ax
    assert plot_grid_exchange(results, price=price, ax=ax) is ax
    assert plot_storage(results, ax=ax) is ax
    assert plot_price_response(results, price, ax=ax) is ax


def test_plots_create_their_own_axes_when_given_none(results):
    before = len(plt.get_fignums())
    ax = plot_dispatch(results, "elec")

    assert len(plt.get_fignums()) == before + 1
    assert ax.figure in [plt.figure(n) for n in plt.get_fignums()]


def test_dispatch_draws_one_band_per_active_component(results):
    ax = plot_dispatch(results, "elec")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]

    assert "pv" in labels and "grid" in labels
    assert "net" in labels


def test_dispatch_labels_pure_consumers_too(results):
    """The heat pump only consumes electricity, so it lives entirely below
    zero - and still has to reach the legend."""
    ax = plot_dispatch(results, "elec")
    labels = [t.get_text() for t in ax.get_legend().get_texts()]

    assert "heat_pump" in labels
    assert "household_load" in labels
    assert len(labels) == len(set(labels)), "a component was labelled twice"


def test_dispatch_rejects_a_bus_nothing_touches(results):
    with pytest.raises(ValueError, match="bus 'gas'"):
        plot_dispatch(results, "gas")


def test_window_narrows_the_data(results):
    full = plot_dispatch(results, "elec")
    windowed = plot_dispatch(results, "elec", window=12)

    assert windowed.get_xlim()[1] - windowed.get_xlim()[0] < (
        full.get_xlim()[1] - full.get_xlim()[0]
    )


def test_window_accepts_a_date_range(results):
    ax = plot_grid_exchange(results, window=("2010-01-01", "2010-01-01 05:00"))
    line = ax.get_lines()[0]

    assert len(line.get_xdata()) == 6


def test_storage_content_is_trimmed_to_the_time_index(results):
    """solph reports one more value than there are time steps."""
    ax = plot_storage(results)
    drawn = ax.get_lines()[0].get_ydata()

    assert len(results["battery"]["storage_content"]) == N + 1
    assert len(drawn) == N


def test_storage_reports_when_there_is_none(results):
    del results["battery"]
    with pytest.raises(ValueError, match="storage content"):
        plot_storage(results)


def test_grid_exchange_shades_the_expensive_hours(results, price):
    shaded = plot_grid_exchange(results, price=price)
    plain = plot_grid_exchange(results)

    assert len(shaded.collections) > len(plain.collections)
    labels = [t.get_text() for t in shaded.get_legend().get_texts()]
    assert any("price >=" in label for label in labels)


def test_grid_exchange_keeps_price_aligned_under_a_window(results, price):
    """A window must narrow price and exchange together, not shift them."""
    ax = plot_grid_exchange(results, price=price, window=("2010-01-01 17:00",
                                                          "2010-01-01 20:00"))
    # those four hours are all expensive, so the shading spans the window
    collection = ax.collections[0]
    assert collection.get_paths(), "the expensive hours were not shaded"


def test_price_response_bins_by_price(results, price):
    ax = plot_price_response(results, price, bins=4)

    assert len(ax.patches) == 4
    assert ax.get_ylabel().startswith("mean import")


def test_use_style_does_not_leak():
    before = plt.rcParams["axes.grid"]
    with use_style():
        pass

    assert plt.rcParams["axes.grid"] == before
