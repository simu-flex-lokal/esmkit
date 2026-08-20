# Known deviations of the 5R1C zone from its sources

**Nothing here is fixed.** The implementation deliberately reproduces tsib's original
`thermal/model5R1C.py` bit-for-bit, so the results stay comparable. That parity survived three
rewrites — the component refactor, the migration to oemof-solph, and the extraction into
esmkit — and is pinned by the golden fixtures in `test/data/golden/`. This file records what looks wrong and why it was left
alone, so it gets investigated on purpose rather than rediscovered by accident — or "corrected"
without realising the reference numbers move.

**Read this before changing any equation in `esmkit/components/zone5r1c.py`.**

## Sources

- **[S17]** Schütz, T., Schiffer, L., Harb, H., Fuchs, M., Müller, D. (2017): *Optimal design of
  energy conversion units and envelopes for residential building retrofits using a comprehensive
  MILP model.* Applied Energy 185(1), 1–15.
  [doi:10.1016/j.apenergy.2016.10.049](https://doi.org/10.1016/j.apenergy.2016.10.049).
  The origin of the MILP 5R1C formulation. **Unqualified equation numbers below are [S17]'s.**
- **[K18]** Kotzur, L. (2018): *Future Grid Load of the Residential Building Sector.*
  Dissertation, RWTH Aachen / FZJ Energie & Umwelt 442, ISBN 978-3-95806-370-9, §3.2.2. The work
  tsib implements.

**Triage rule.** [K18] defers all model equations to [S17] — *"The detailed description of the
mathematical model itself is done by Schütz et al. [2017a]. Hence, this section only describes
its extensions"* — and states only two comfort bounds of its own (Eqs. 3.1/3.2). Therefore:
**node balances are [S17]'s to define, so a mismatch there is sanctioned by neither source
(items 1–3). The comfort band is [K18]'s (item 4).**

## Summary

| # | Deviation | Verdict | Measured impact |
|---|---|---|---|
| 1 | Windows driven by mass node, not surface node | likely bug | unmeasured |
| 2 | Ventilation driven by mass node, not air node | likely bug | unmeasured |
| 3 | Air node gets surface gain, not internal gain | likely bug | unmeasured |
| 4 | Fictitious cooling against a hard comfort ceiling | deliberate, but limiting | 8.2 kWh/m²a of cooling that cannot exist; +0.3 % heating |

---

## 1. Windows are driven by the mass node instead of the surface node

**Likely bug. Unmeasured.** `esmkit/components/zone5r1c.py`, `envelope_flow` (driver) and
`surface_node_balance` (use)

Every envelope element shares one driver — `envelope_flow` returns
`H_element(element) * (T_m[zone, t] - T_e[t])` for all elements, including `Windows`.
Eq. (21) couples windows to the **surface** node:

$$H_{tr,ms}(\theta_s - \theta_m) + H_{tr,is}(\theta_s - \theta_{air}) + \underbrace{H_{tr,w}(\theta_s - \theta_e)}_{\text{code uses } (T_m - T_e)} = \phi_{st}$$

Only the *opaque* elements may legitimately use $(\theta_m - \theta_e)$ — via Eq. (20) with the
$H_{tr,em} \approx H_{tr,op}$ approximation of Eq. (9). The mass node is correct on that count.

**Why it plausibly matters:** $T_m$ is the slow, inertia-damped node while $T_s$ tracks the air
closely. Driving a low-inertia, high-$U$ element from the slow node misplaces both phase and
amplitude of the window losses.

**To verify:** give `Windows` the driver `T_s[zone, t] - T_e[t]`; re-run
`test/test_zone.py` and, on the tsib side, the IWU/TABULA reference comparison.

## 2. Ventilation is driven by the mass node instead of the air node

**Likely bug. Unmeasured.** `esmkit/components/zone5r1c.py`, `envelope_flow` (driver) and
`air_node_balance` (use)

Same shared driver. Eq. (22) couples ventilation to the **air** node:

$$\underbrace{H_{ve}(\theta_{air} - \theta_e)}_{\text{code uses } (T_m - T_e)} + H_{tr,is}(\theta_{air} - \theta_s) = \phi_{ia} + \phi_{HC}$$

Physically the most clear-cut of the three: ventilation exchanges *air* with the environment, so
its driving temperature difference is the air node's by definition. Eq. (12) defines
$H_{ve} = \kappa_{air}\,\rho_{air}\,q_{ve,avg}$ — a pure air heat-capacity flow.

**To verify:** as item 1, with driver `T_air[zone, t] - T_e[t]`.

## 3. The air node receives the surface gain instead of the internal gain

**Likely bug. Unmeasured.** `esmkit/components/zone5r1c.py`, `air_node_balance`

The air balance uses `zone.gain_surface_node(t)` — that is $\phi_{st}$, the window/solar
weighted surface gain of Eq. (19). Per Eq. (22) the right-hand side is $\phi_{ia} + \phi_{HC}$,
and Eq. (14) defines simply

$$\phi_{ia} = 0.5 \cdot \phi_{int}$$

— half the internal gains, with no solar term at all. $\phi_{st}$ belongs to the *surface*
balance (Eq. 21), where the code does use it correctly.

The original model computed a `Q_ia` term and never used it; the component refactor dropped it
as dead code. That dead variable suggests an old slip rather than a decision.

**Why it plausibly matters:** a magnitude error rather than a phase error — solar-weighted gains
are injected into the air node where the standard puts only half the internal gains.

**To verify:** the air node would need half the internal gains, which esmkit's zone no longer
receives separately - `gain_surface` arrives already aggregated. Testing this means passing
`Q_ig` in as a fourth series, or making the substitution on the caller's side; either way
re-run `test/test_zone.py`.

## 4. Fictitious cooling against a hard comfort ceiling

**Deliberate in [K18], but a real modelling limitation.** `esmkit/components/zone5r1c.py`
(`Q_cool_internal`, the `comfort_ub` constraint)

The `comfort_ub` series enters as a hard constraint, and `Q_cool` is an unbounded non-negative
variable priced by whatever supplies the cool bus (0.02 EUR/kWh in the reference case). Gains above the band therefore *force* cooling — including in
buildings with no cooling device, which is the norm for the German residential stock this model
targets. There is no parameter to declare that a building cannot cool.

**This is intentional.** [K18] p. 36, Eq. (3.2): *"The upper bound of the temperature
$T_{air,ub}^{com}$ limits the heating operation and determines the cooling load in the summer
period […] The general upper bound for the temperature is set to 26 °C."* It is an addition, not
inherited: [S17] has no upper bound at all (its Eq. 26 is $\theta_{air} \ge \theta_{set}^{lb}$)
and free-floats instead.

**But [K18] §3.2.1 also concedes the output is weak**, of the very model it adopts: *"Only the
cooling demands are overestimated, which is not relevant for this work since they are not
considered in the optimization"* — which sits in tension with `coolCost` appearing in the
objective. Reasonable reading: an acknowledged-inaccurate byproduct the author never used (his
thesis concerns *electrical* grid load, which is heating-dominated). tsib is now applied to
things [K18] was not.

**Measured impact.** Reference building (TABULA `index[24]`, TRY 4, band 20–26 °C, full year),
current versus $Q_{cool}$ fixed to 0 with `comfort_ub` deactivated:

| | heating | cooling | $T_{air}$ max | h > 26 °C |
|---|---|---|---|---|
| current | 197.32 kWh/m²a | **8.20 kWh/m²a** | 26.00 °C | 0 |
| free-float | 196.72 kWh/m²a | 0 | 39.59 °C | 752 |

1420 kWh/a of a service that cannot physically exist; the knock-on error on *heating* is only
0.6 kWh/m²a (**0.3 %**), because the seasons barely overlap. [K18] Fig. 3.8 reports RMSE
6.79–6.82 kWh/m²a against IWU, so the heating side-effect sits an order of magnitude inside the
existing validation noise — **fixing the cooling will not threaten the reference test.**

**Do not simply delete the bound.** It does two jobs, as [K18]'s own sentence says: it *"limits
the heating operation"* **and** *"determines the cooling load"*. Only the second is wrong.
[S17]'s justification for having no bound — the optimiser will not overheat because heat costs
money — holds only under *constant* prices; with time-varying prices, pre-heating pays and an
unbounded zone would be driven to 40 °C for arbitrage. Separating the two jobs exactly requires a
disjunction and one binary per time step, which would destroy the full-year LP.

→ **This item is written up in full, self-contained, for outside readers in a separate note
held with the tsib project**: the
physics, the proposed penalised-slack formulation, the missing window-opening model, the Kh/a
comfort metrics (DIN 4108-2, EN 15251/16798-1), and five open research questions. Start there
before touching this.

---

## Note on [K18] Eq. (3.2): the code corrects a typo

Do **not** "fix" the comfort ceiling to match the printed equation. As printed, Eq. (3.2)
subtracts $(T_{air,lb}^{com} - T_{air,ub}^{vac})$, which yields a 36 °C vacancy bound and
contradicts the surrounding text. The code uses $(T_{ub}^{com} - 30)$, matching the stated
$T_{air,ub}^{vac} = 30\ °C$. The code is right and the thesis has a typo.

tsib additionally gates the band width on the installed comfort control equipment, which
Eq. (3.2) does not show — without it both bounds collapse to $T_{lb}$ and the zone is pinned,
which is [K18]'s "5R1C fix" validation case.

**Where the band is built.** esmkit's zone receives `comfort_lb` / `comfort_ub` as two arrays
and only enforces them. Deriving those arrays from a nominal band, control equipment and
occupancy is the caller's job — in tsib, `comfort_bounds()`. The deviation recorded here is
therefore split: the *hard bound and the fictitious cooling it forces* belong to this component,
the *band construction* to whoever supplies it.
