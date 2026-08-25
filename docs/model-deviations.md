# Known deviations of the 5R1C zone

`esmkit/components/zone5r1c.py` implements the 5R1C reduced-order zone of
DIN EN ISO 13790 as formulated by Schuetz et al. 2017 (*Optimal design of energy
conversion units and envelopes for residential building retrofits using a
comprehensive MILP model*, Applied Energy 185, Eqs. 20-22) and extended by
Kotzur 2018 (dissertation, Sec. 3.2.2, Eqs. 3.1-3.2).

Several node balances do **not** match those references. They are deviations on
purpose, carried over unchanged from the pre-migration implementation so that
the numbers stay comparable with everything published from it. The golden
fixtures in `test/data/golden/` were captured from that implementation, and
`test/test_zone.py` pins the hourly results to 1e-5 against them. Every item
below is therefore load-bearing: "fixing" any of them will break parity.

## 1. Ventilation hangs off the mass node

*Code*: `air_node_balance` uses `envelope_flow(zone, "Ventilation", t)`, which is
`H["Ventilation"] * (T_m - T_e)`.
*Reference*: Schuetz Eq. (22) writes ventilation as `H_ve * (theta_air - theta_e)`,
and ISO 13790 likewise puts the ventilation conductance on the air node.
*Consequence*: ventilation losses follow the slow, well-damped mass temperature
instead of the air temperature, and the air node keeps no direct conductance to
the outside at all - ambient reaches the air only through the mass and surface.

## 2. Windows hang off the mass node

*Code*: `surface_node_balance` uses `envelope_flow(zone, "Windows", t)`, again
referenced to `T_m`.
*Reference*: Schuetz Eq. (21) writes window transmission as
`H_tr,w * (theta_s - theta_e)`, i.e. driven by the surface temperature.
*Consequence*: together with (1), all five envelope conductances are referenced
to the same node, which is why `envelope_flow` can take `T_m` unconditionally.
Window losses are damped by the thermal mass rather than tracking the surface.

## 3. The surface gain is spent twice, and there is no air-node gain

*Code*: the right-hand side of `air_node_balance` is
`gain_surface[t] - Q_cool + Q_heat` - the very same series that already appears
on the right-hand side of `surface_node_balance`.
*Reference*: Schuetz Eq. (22) has `phi_ia + phi_HC`, where `phi_ia = 0.5 * phi_int`
(Eq. 14) is a separate quantity from the surface gain `phi_st` (Eq. 19).
*Consequence*: the surface gain is credited at both nodes, and the convective
share of the internal gains is never modelled in its own right. The air node is
over-supplied with gains, which lowers the computed heating load.

## 4. The comfort band bounds the air temperature

*Code*: `comfort_lb` / `comfort_ub` constrain `T_air`.
*Reference*: this one follows the papers - Schuetz Eq. (26) and Kotzur
Eqs. (3.1)-(3.2) bound `theta_air` too. It is ISO 13790 itself that defines the
set point on the *operative* temperature, roughly `0.3 * T_air + 0.7 * T_s`.
*Consequence*: the radiant half of perceived comfort is ignored. In winter the
surfaces sit below the air, so an air-temperature band is the looser constraint
and the heating load comes out slightly lower than an operative band would give.

## 5. Cooling is free when no `cool_bus` is attached

*Code*: without a `cool_bus`, cooling is `Q_cool_internal`, a non-negative
variable that appears in no cost term anywhere.
*Reference*: Schuetz models no cooling system at all and deliberately omits the
upper temperature bound (Sec. 2.3.5). Kotzur adds the upper bound (Eq. 3.2),
reads the resulting cooling load off it, but never prices it either - and notes
that the 5R1C model overestimates cooling demand against VDI 6007 (p. 33).
*Consequence*: summer overheating is relieved at zero cost, so the reported
cooling load is an unpriced upper bound and the zone never trades summer comfort
against anything else in the system. Attach a `cool_bus` whenever cooling should
compete for money or capacity - the parity fixture does exactly that.

## 6. The mass balance steps forward, not backward

*Code*: `mass_node_balance` evaluates every conductance and gain term at step `t`
and sets them against `C_m * (T_m[t+1] - T_m[t]) / dt`.
*Reference*: Schuetz Sec. 2.3.5 approximates `C_m * d(theta_m)/dt` as
`C_m * (theta_m,t - theta_m,t-1) / dt`, a backward difference taken at the same
instant the balance is written.
*Consequence*: an explicit rather than an implicit Euler step. Annual sums are
barely affected; hourly traces are shifted by one step against a backward-Euler
implementation of the same equations.

## Boundary condition on `T_m`

Not a deviation, but easy to trip over: at the last step `mass_rule` balances
back around to step 0, so `T_m` closes cyclically over the horizon. Passing
`initial_T_m` drops that wrap-around and pins the first step instead, turning
the horizon open. Neither paper specifies this; a short horizon solved cyclically
will disagree with the same horizon solved from a fixed start.

## Before changing any of this

Do not "correct" a balance, a bound or the discretisation without re-capturing
the golden fixtures in `test/data/golden/` and re-validating the heat loads
against the source the fixtures came from. Silently fixing one of these six will
turn `test/test_zone.py` red, and fixing them all will produce a different -
possibly better, but no longer validated - model.
