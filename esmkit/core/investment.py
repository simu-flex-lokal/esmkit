def annuity_factor(lifetime, wacc):
    if wacc > 0:
        return ((1 + wacc) ** lifetime) * wacc / (((1 + wacc) ** lifetime) - 1)
    else:
        return 1.0 / lifetime
