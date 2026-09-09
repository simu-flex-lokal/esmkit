# Known deviations of the 5R1C zone

`esmkit/components/zone5r1c.py` implements the 5R1C reduced-order zone of
DIN EN ISO 13790 as formulated by Schuetz et al. 2017 (*Optimal design of energy
conversion units and envelopes for residential building retrofits using a
comprehensive MILP model*, Applied Energy 185, Eqs. 20-22) and extended by
Kotzur 2018 (dissertation, Sec. 3.2.2, Eqs. 3.1-3.2).

Some node balances still do **not** match those references. What is listed below
is what remains after the envelope wiring was corrected on 2026-09-09.

## Removed on 2026-09-09: windows and ventilation on the mass node

Until then, `air_node_balance` drove ventilation from `T_m` instead of `T_air`
(against Schuetz Eq. 22) and `surface_node_balance` drove window transmission
from `T_m` instead of `T_s` (against Eq. 21). Both referenced the loss to the
coldest indoor node while heating, so both understated the driving temperature
difference and with it the heat demand - by 11.0 kWh/m2a on the
reference building, 5.6 % of the old figure (198.111 to 209.143 kWh/m2a).

They had been carried deliberately, for parity with the pre-migration tsib
implementation, and were kept because the lower figure sat closer to the TABULA
benchmark. That argument does not hold: the TABULA comparison is confounded by
larger, uncontrolled differences - TABULA adds a thermal-bridging surcharge of
0.1 W/m2K that tsib omits (worth about 21 kWh/m2a), applies a temperature
reduction factor of 0.8385 that the LP does not (about 40 kWh/m2a), and has no
counterpart to the constant longwave radiation term in `tsib.envelope.gains`
(about 19 kWh/m2a over the heating season). A benchmark that is uncalibrated at
three larger points cannot arbitrate an 11 kWh/m2a question.

The golden fixtures were regenerated from the corrected model; they no longer
pin parity with the pre-migration stack. See `test/data/golden/golden_meta.json`.

## 1. The surface gain is spent twice, and there is no air-node gain

*Code*: without `gain_air`, the right-hand side of `air_node_balance` is
`gain_surface[t] - Q_cool + Q_heat` - the very same series that already appears
on the right-hand side of `surface_node_balance`.
*Reference*: Schuetz Eq. (22) has `phi_ia + phi_HC`, where `phi_ia = 0.5 * phi_int`
(Eq. 14) is a separate quantity from the surface gain `phi_st` (Eq. 19).
*Consequence*: the surface gain is credited at both nodes, and the convective
share of the internal gains is not modelled in its own right unless `gain_air`
is supplied. The air node is over-supplied with gains, which lowers the computed
heating load.

## 2. The comfort band bounds the air temperature

*Code*: `comfort_lb` / `comfort_ub` constrain `T_air`.
*Reference*: this one follows the papers - Schuetz Eq. (26) and Kotzur
Eqs. (3.1)-(3.2) bound `theta_air` too. It is ISO 13790 itself that defines the
set point on the *operative* temperature, roughly `0.3 * T_air + 0.7 * T_s`.
*Consequence*: the radiant half of perceived comfort is ignored. In winter the
surfaces sit below the air, so an air-temperature band is the looser constraint
and the heating load comes out slightly lower than an operative band would give.

## 3. Cooling is free when no `cool_bus` is attached

*Code*: without a `cool_bus`, cooling is `Q_cool_internal`, a non-negative
variable that appears in no cost term anywhere. `can_cool=False` fixes it to
zero, and `comfort_ub_penalty` prices the band breach that then follows.
*Reference*: Schuetz models no cooling system at all and deliberately omits the
upper temperature bound (Sec. 2.3.5). Kotzur adds the upper bound (Eq. 3.2),
reads the resulting cooling load off it, but never prices it either - and notes
that the 5R1C model overestimates cooling demand against VDI 6007 (p. 33).
*Consequence*: summer overheating is relieved at zero cost, so the reported
cooling load is an unpriced upper bound and the zone never trades summer comfort
against anything else in the system. Attach a `cool_bus` whenever cooling should
compete for money or capacity - the parity fixture does exactly that.

## 4. The mass balance steps forward, not backward

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

Each of these four changes the numbers, and `test/test_zone.py` will go red the
moment one moves. That is the point: the fixtures catch drift, they do not tell
you which formulation is right. If you change one deliberately, regenerate the
expected half of the fixture and record what changed and why in
`test/data/golden/golden_meta.json`, as the 2026-09-09 entry does.

Note what the fixtures cannot do: they cannot arbitrate between formulations,
and neither can the TABULA figure - see the section above. Deciding whether a
balance is right needs a reference with the same node structure, which neither
TABULA nor the golden fixture is.
