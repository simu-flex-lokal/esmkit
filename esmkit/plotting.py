"""Plots of a solved energy system: dispatch on a bus, storage content, the
exchange at the grid connection and the load's response to a price signal.

Every function takes an optional ``ax`` and returns the axes it drew on, so
one function draws one thing and the caller composes the figure. Nothing
calls ``plt.show()`` or touches global state outside ``use_style()``.

Needs matplotlib, which the kit itself does not require: install the extra,
``pip install esmkit[plots]``.

A year is 8760 points and does not read as a line plot - pass ``window=`` to
zoom in on a few days.
"""

import contextlib
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core.analysis import bus_flows, net_grid_exchange


#: same colour across figures. Taken from examples/energysystem/.
COLORS = {
    "heat": "#c1440e",
    "pv": "#e0a800",
    "grid": "#4a4a4a",
    "battery": "#2e6f9e",
    "secondary": "#8c8c8c",
    "cool": "#4a90a4",
}

#: component name -> colour, for the component names a building model
#: typically uses. Anything else falls back to the colour cycle.
ROLE_COLORS = {
    "grid": COLORS["grid"],
    "pv": COLORS["pv"],
    "battery": COLORS["battery"],
    "heat_pump": COLORS["heat"],
    "hp": COLORS["heat"],
    "buffer": "#e08a4a",
    "heat_supply": COLORS["heat"],
    "cool_supply": COLORS["cool"],
    "thermalzone": COLORS["secondary"],
    "dhw_load": "#7fa8c4",
    "household_load": COLORS["secondary"],
}

#: fallback for components the palette does not know
_FALLBACK_COLORS = ["#6a51a3", "#31a354", "#d95f0e", "#756bb1", "#636363"]

RC_PARAMS = {
    "figure.figsize": (11, 3.2),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
}

@contextlib.contextmanager
def use_style(**overrides):
    """
    Applies the kit's figure defaults for the duration of the block.

    The plot functions colour their own artists, so this only affects the
    figure furniture - size, grid, spines.

        with esmkit.plotting.use_style():
            fig, ax = plt.subplots()
            esmkit.plotting.plot_grid_exchange(results, ax=ax)
    """
    params = dict(RC_PARAMS)
    params.update(overrides)
    with plt.rc_context(params):
        yield

def _color(name, cycle=None):
    if name in ROLE_COLORS:
        return ROLE_COLORS[name]
    return next(cycle) if cycle is not None else COLORS["secondary"]

def _legend(ax, ncol=1):
    """Adds the legend with enough headroom that it clears the data."""
    ax.margins(y=0.20)
    ax.legend(loc="upper right", ncol=ncol)
    return ax

def _ax(ax, **subplot_kwargs):
    if ax is not None:
        return ax
    _, ax = plt.subplots(**subplot_kwargs)
    return ax

def _window(obj, window):
    """
    Narrows a Series/DataFrame to `window`.

    Accepts an int (the first n steps), a slice of positions, a
    `(start, end)` pair of dates, or a single partial date string such as
    "2010-01".
    """
    if window is None:
        return obj
    if isinstance(window, int):
        return obj.iloc[:window]
    if isinstance(window, slice):
        if isinstance(window.start, (int, type(None))) and isinstance(
            window.stop, (int, type(None))
        ):
            return obj.iloc[window]
        return obj.loc[window]
    if isinstance(window, tuple) and len(window) == 2:
        return obj.loc[window[0]:window[1]]
    return obj.loc[window]

def _resample(obj, rule, how="mean"):
    if rule is None:
        return obj
    return getattr(obj.resample(rule), how)()

def _time_index(results):
    """Any datetime index found in a results dict, for series that lack one."""
    for res in results.values():
        if not isinstance(res, dict):
            continue
        frame = res.get("timeseries")
        if frame is not None and isinstance(frame.index, pd.DatetimeIndex):
            return frame.index
        for value in res.values():
            if isinstance(value, pd.Series) and isinstance(
                value.index, pd.DatetimeIndex
            ):
                return value.index
    return None

def plot_dispatch(results, bus, ax=None, window=None, spec=None, net=True,
                  title=None):
    """
    Every flow on one bus, stacked: supply above zero, demand below.

    Parameters
    ----------
    results: dict, required
    bus: str, required
    ax: matplotlib.axes.Axes, optional
    window: optional - see the module docstring.
    spec: SystemSpec, optional - resolves the thermal zone's buses.
    net: bool, optional (default: True)
        Also draw the net balance, which should sit on zero.
    """
    ax = _ax(ax)
    flows = _window(bus_flows(results, bus, spec=spec), window)
    if flows.empty:
        raise ValueError("No component in the results touches bus '{}'".format(bus))

    index = flows.index
    cycle = itertools.cycle(_FALLBACK_COLORS)

    # a component is labelled once, on whichever side it first appears -
    # otherwise a pure consumer like the heat pump never reaches the legend
    labelled = set()
    colors = {name: _color(name, cycle) for name in flows.columns}

    for sign in (1, -1):
        base = np.zeros(len(index))
        for name in flows.columns:
            values = flows[name].values
            part = np.clip(values, 0, None) if sign > 0 else np.clip(values, None, 0)
            if not np.any(np.abs(part) > 1e-9):
                continue
            ax.fill_between(
                index, base, base + part, step="post", alpha=0.85,
                color=colors[name],
                label=None if name in labelled else name,
                linewidth=0,
            )
            labelled.add(name)
            base = base + part

    if net:
        ax.step(index, flows.sum(axis=1).values, where="post", color="black",
                lw=0.8, ls=":", label="net")

    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kW")
    ax.set_title(title if title is not None else "Dispatch on '{}'".format(bus))
    return _legend(ax, ncol=3)

def plot_storage(results, ax=None, window=None, index=None, title=None):
    """
    Storage content of every storage in the system.

    solph reports the content on time *points*, one more than there are time
    steps, and without a time index - so the last value is dropped and the
    model's index reattached.
    """
    ax = _ax(ax)
    index = index if index is not None else _time_index(results)
    cycle = itertools.cycle(_FALLBACK_COLORS)
    drawn = 0

    for name, res in results.items():
        if not isinstance(res, dict) or "storage_content" not in res:
            continue
        content = res["storage_content"]
        if index is not None:
            content = pd.Series(content.values[:len(index)], index=index)
        content = _window(content, window)
        ax.plot(content.index, content.values, lw=1.3, color=_color(name, cycle),
                label=name)
        drawn += 1

    if not drawn:
        raise ValueError("No component in the results has a storage content")

    ax.set_ylabel("storage content [kWh]")
    ax.set_title(title if title is not None else "Storage")
    return _legend(ax, ncol=2)

def plot_grid_exchange(results, price=None, ax=None, window=None, name="grid",
                       bus="elec", price_quantile=0.75, show_price=True,
                       title=None):
    """
    Net exchange at the grid connection, with the tariff behind it.

    Import is positive, export negative. Hours at or above
    `price_quantile` of the tariff are shaded, which is what makes a
    load shift visible at a glance.

    Parameters
    ----------
    price: array-like or pandas.Series, optional
        The tariff the building was optimized against.
    price_quantile: float, optional (default: 0.75)
        Threshold for shading the expensive hours.
    show_price: bool, optional (default: True)
        Draw the tariff itself on a secondary axis.
    """
    ax = _ax(ax)
    full = net_grid_exchange(results, name=name, bus=bus)

    # the tariff is aligned to the full horizon before either is narrowed, so
    # a window cannot silently shift the price against the exchange
    if price is not None:
        price = pd.Series(np.asarray(price, dtype=float)[:len(full)],
                          index=full.index)
        threshold = price.quantile(price_quantile)
        price = _window(price, window)
    net = _window(full, window)

    if price is not None:
        expensive = (price >= threshold).values
        span = max(abs(net.min()), abs(net.max())) or 1.0
        ax.fill_between(net.index, -span, span, where=expensive, step="post",
                        color=COLORS["heat"], alpha=0.10, linewidth=0,
                        label="price >= {:.2f}".format(threshold))

    ax.step(net.index, net.values, where="post", color=COLORS["grid"], lw=1.2,
            label="net exchange")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("kW (import > 0)")
    ax.set_title(title if title is not None else "Grid exchange")
    _legend(ax, ncol=2)

    if price is not None and show_price:
        twin = ax.twinx()
        twin.step(net.index, price.values, where="post", color=COLORS["pv"],
                  lw=0.9, alpha=0.8)
        # keep the tariff in the lower band of the axes so it reads as
        # context rather than competing with the exchange itself
        span = (price.max() - price.min()) or 1.0
        twin.set_ylim(price.min() - 0.1 * span, price.max() + 1.5 * span)
        twin.set_ylabel("price [EUR/kWh]")
        twin.grid(False)
    return ax

def plot_price_response(results, price, ax=None, bins=5, name="grid",
                        bus="elec", title=None):
    """
    Mean import power per price bin - did the tariff move the load?

    Mean power rather than energy, because bins hold different numbers of
    hours: a flat bar chart means the tariff changed nothing, a downward
    slope means the building buys when it is cheap. The hour count per bin
    is annotated so a bin resting on few hours is not over-read.
    """
    ax = _ax(ax)
    imported = np.clip(net_grid_exchange(results, name=name, bus=bus).values, 0, None)
    price = np.asarray(price, dtype=float)[:len(imported)]

    edges = np.linspace(price.min(), price.max(), bins + 1)
    # the top edge is inclusive, otherwise the most expensive hour drops out
    which = np.clip(np.digitize(price, edges) - 1, 0, bins - 1)

    means, labels, counts = [], [], []
    for b in range(bins):
        mask = which == b
        means.append(imported[mask].mean() if mask.any() else 0.0)
        counts.append(int(mask.sum()))
        labels.append("{:.2f}\n{:.2f}".format(edges[b], edges[b + 1]))

    positions = np.arange(bins)
    ax.bar(positions, means, color=COLORS["grid"], alpha=0.85, width=0.7)
    for pos, mean, count in zip(positions, means, counts):
        ax.annotate("{} h".format(count), (pos, mean), ha="center",
                    va="bottom", fontsize=7, color=COLORS["secondary"])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_xlabel("price bin [EUR/kWh]")
    ax.set_ylabel("mean import [kW]")
    ax.set_title(title if title is not None else "Price response")
    return ax
