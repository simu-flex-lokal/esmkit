"""Derived series read off a results dict: what flows on a bus, and what
crosses the grid connection.

Both are plain pandas and useful without drawing anything - the net grid
exchange in particular is the one series per building a grid powerflow
computation consumes.
"""

import pandas as pd

def bus_flows(results, bus, spec=None):
    """
    Every flow on one bus, as a signed frame.

    Positive is energy *into* the bus (generation, import, storage
    discharge), negative is energy *out* of it (demand, export, charging).
    A component doing both nets to one column, so a battery reads positive
    while discharging and negative while charging.

    Components are discovered from the `in_<bus>` / `out_<bus>` keys
    `node_results` produces - nothing here hardcodes a component name. The thermal zone is the exception: it reports a
    "timeseries" frame rather than flows, so its heating and cooling loads
    are added from there.

    Parameters
    ----------
    results: dict, required
        As returned by `node_results`.
    bus: str, required
        Bus name, e.g. "elec" or "heat".
    spec: SystemSpec or dict, optional
        Used to find out which buses the thermal zone is attached to.
        Without it the template defaults ("heat"/"cool") are assumed.

    Returns
    -------
    pandas.DataFrame - one column per component, in kW.
    """
    columns = {}
    for name, res in results.items():
        if not isinstance(res, dict):
            continue

        if "timeseries" in res:
            columns.update(_zone_flows(name, res, bus, spec))
            continue

        into = res.get("out_" + bus)
        out_of = res.get("in_" + bus)
        if into is None and out_of is None:
            continue
        series = 0.0
        if into is not None:
            series = series + into
        if out_of is not None:
            series = series - out_of
        columns[name] = series

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns)


def _zone_flows(name, res, bus, spec):
    """The thermal zone's withdrawals, which are not reported as flows."""
    heat_bus, cool_bus = "heat", "cool"
    # a spec reaches this either as a SystemSpec or in its dict form, which
    # is how it arrives from a model that only ever emitted the dict
    components = getattr(spec, "components", None)
    if components is None and isinstance(spec, dict):
        components = spec.get("components", {})
    if components and name in components:
        params = components[name]
        heat_bus = params.get("heat_bus", heat_bus)
        cool_bus = params.get("cool_bus", cool_bus)

    frame = res["timeseries"]
    if bus == heat_bus and "Heating Load" in frame:
        return {name: -frame["Heating Load"]}
    if bus == cool_bus and "Cooling Load" in frame:
        return {name: -frame["Cooling Load"]}
    return {}


def net_grid_exchange(results, name="grid", bus="elec"):
    """
    Net exchange at the grid connection [kW], import positive.

    This is the one series a grid powerflow computation needs per building.

    Parameters
    ----------
    results: dict, required
    name: str, optional (default: "grid")
        Name of the grid component in the spec.
    bus: str, optional (default: "elec")

    Returns
    -------
    pandas.Series [kW], positive while drawing from the grid.
    """
    if name not in results:
        raise KeyError(
            "No component '{}' in the results - available: {}".format(
                name, sorted(results)
            )
        )
    grid = results[name]
    # the grid connection is a Source plus a Sink merged under one name:
    # "out_<bus>" is the import, "in_<bus>" the export
    imported = grid.get("out_" + bus)
    exported = grid.get("in_" + bus)
    if imported is None and exported is None:
        raise KeyError(
            "Component '{}' has no flows on bus '{}'".format(name, bus)
        )
    if imported is None:
        return -exported
    if exported is None:
        return imported
    return imported - exported
