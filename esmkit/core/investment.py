# -*- coding: utf-8 -*-
"""
Investment costing.

solph expects an already annualized cost per unit (`ep_costs`), so the
annuity and the horizon scaling stay on the kit's side. `registry.investment`
turns capex/lifetime spec entries into a `solph.Investment` using the factor
below.
"""


def annuity_factor(lifetime, wacc):
    """
    Annuity factor to distribute an investment over its lifetime with
    interest rate `wacc`. Falls back to straight-line depreciation for
    wacc == 0.
    """
    if wacc > 0:
        return ((1 + wacc) ** lifetime) * wacc / (((1 + wacc) ** lifetime) - 1)
    else:
        return 1.0 / lifetime
