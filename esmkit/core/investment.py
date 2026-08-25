"""Annualisation of investment costs."""


def annuity_factor(lifetime, wacc):
    """Return the capital recovery factor for a lifetime in years and a wacc."""
    if wacc > 0:
        return ((1 + wacc) ** lifetime) * wacc / (((1 + wacc) ** lifetime) - 1)
    else:
        # Without a cost of capital the annuity is plain straight-line repayment.
        return 1.0 / lifetime
