# The 5R1C thermal zone

The kit's one custom solph component, and its reference implementation of a component whose
physics is not expressible as flows between buses.

Three temperature nodes — mass $T_m$ (the only one with capacity $C_m$), surface $T_s$, air
$T_{air}$ — per DIN EN ISO 13790 / Schütz et al. 2017.

**The design rule:** the zone is a *component*, not a precomputed load profile. Its temperature
states are free decision variables inside a comfort band, so thermal mass is a dispatchable
flexibility resource in the same solve as storage, PV and prices. Reducing the zone to a fixed
heat demand series and optimizing dispatch afterwards would throw that flexibility away.

**The heat supply is not a variable of the component.** It is `m.flow[heat_bus, zone, t]`, the
edge solph creates for the zone's input flow. That edge *is* the coupling mechanism, and the bus
balance ties it to whatever supplies it.

---

## The parameter contract

The zone takes **explicit parameters, never a building configuration.** Ten numbers and five time
series, none of which mentions a building:

```python
from esmkit import ThermalZone5R1C

zone = ThermalZone5R1C(
    "thermalzone", heat_bus=heat, cool_bus=cool,
    H_ms=3.9414, H_is=2.6897, H_door=0.0060, C_m=8.4219,
    H={"Walls": 0.17755, "Roof": 0.09157, "Floor": 0.06712,
       "Windows": 0.09579, "Ventilation": 0.08714},
    T_e=..., gain_mass=..., gain_surface=...,
    comfort_lb=..., comfort_ub=...,
    max_load=20.17,
)
```

| parameter | unit | what it is |
|---|---|---|
| `H_ms` | kW/K | mass ↔ surface node coupling |
| `H_is` | kW/K | surface ↔ air node coupling |
| `H_door` | kW/K | door, entering the mass balance with the opaque elements |
| `C_m` | kWh/K | thermal capacity of the mass node |
| `H` | kW/K | per element: `Walls`, `Roof`, `Floor`, `Windows`, `Ventilation` |
| `T_e` | °C | ambient temperature |
| `gain_mass` | kW | heat gain at the mass node |
| `gain_surface` | kW | heat gain at the surface node |
| `comfort_lb` / `comfort_ub` | °C | effective comfort band per time step |
| `max_load` | kW | soft cap on heating and cooling power |

Optional: `cool_bus`, `max_load_violation_penalty` (default 100 EUR/kW), `initial_T_m`, and
`design_capacity` (reported in the static results; it constrains nothing).

In a spec the scalars are written inline and the series as `"@key"` references:

```python
spec.add_component("thermalzone", "zone5r1c", heat_bus="heat", cool_bus="cool",
                   H_ms=3.9414, H_is=2.6897, H_door=0.0060, C_m=8.4219, H={...},
                   T_e="@T_e", gain_mass="@gain_mass", gain_surface="@gain_surface",
                   comfort_lb="@comfort_lb", comfort_ub="@comfort_ub",
                   max_load=20.17)
```

### Why the gains arrive precomputed

`gain_mass` and `gain_surface` are the aggregate of internal and solar gains after distribution
over the nodes (Schütz et al. 2017, eq. 15/16):

$$\Phi^m_t = \tfrac{1}{2}\tfrac{A_m}{A_{tot}} Q_{ig,t} + \tfrac{A_f}{A_{tot}} Q_{sol,t}$$

$$\Phi^s_t = \Big(1 - \tfrac{U_{win}}{h_{ms} A_{tot}}\Big)\tfrac{Q_{ig,t}}{2}
- \tfrac{\text{cross}_t}{h_{ms} A_{tot}} + Q_{sol,t} - \Phi^m_t$$

Both enter the node balances as a **constant right-hand side** — they depend on no decision
variable. So everything behind them (irradiance on tilted surfaces, window areas and shading
factors, frame and non-perpendicular corrections, $g_{gl}$, thermal class, U-values and areas) can
be, and is, computed before the model is built. That is what keeps the component free of building
physics: it is the caller who owns the envelope, the weather and the occupants.

The same applies to the comfort band. `comfort_lb` / `comfort_ub` are the *effective* band per
time step. Deriving them from a nominal band, the installed control equipment and where the
occupants are is the caller's job.

---

## The equations

**Mass node** (the difference equation, hence the thermal storage behaviour):

$$H_{ms}(T_{m,t} - T_{s,t}) + \!\!\sum_{e \in \{W,R,F\}}\!\! H_e (T_{m,t} - T_{e,t})
+ H_{door}(T_{m,t} - T_{e,t})
= \Phi^m_t - C_m \frac{T_{m,t+1} - T_{m,t}}{\Delta t}$$

**Surface node:**

$$H_{ms}(T_{s,t} - T_{m,t}) + H_{is}(T_{s,t} - T_{air,t}) + H_{Win}(T_{m,t} - T_{e,t}) = \Phi^s_t$$

**Air node**, where heating and cooling enter:

$$H_{Vent}(T_{m,t} - T_{e,t}) + H_{is}(T_{air,t} - T_{s,t}) = \Phi^s_t - Q_{cool,t} + Q_{heat,t}$$

**Comfort band** — this is the flexibility:

$$\text{comfort\_lb}_t \le T_{air,t} \le \text{comfort\_ub}_t$$

A collapsed band pins $T_{air}$ and there is no flexibility left, which is exactly the reference
case the flexibility tests compare against.

**Max load** is a *soft* constraint (a violation variable penalized at 100 EUR/kW) so an
undersized system warns instead of going infeasible. It is the component's only objective term;
energy cost belongs to whatever supplies the bus.

**The mass node wraps periodically.** With `initial_T_m=None` (the default) the balance at the
last step uses $T_{m,0}$ as its next value, so $T_m$ is periodic over the horizon. Passing
`initial_T_m` replaces that with a fixed initial condition.

> **Four inherited deviations from the source papers, all kept deliberately:** envelope flows are
> driven by $(T_m - T_e)$ even for windows and ventilation, which touch other nodes; the air-node
> balance uses the surface gain $\Phi^s$ rather than half the internal gains; and the hard comfort
> ceiling forces fictitious cooling in buildings that have no cooling device. All are load-bearing
> for tsib's validated ~197 kWh/m²/a result. **Evidence and measured impact:
> [`model-deviations.md`](model-deviations.md).**

---

## Results

The zone implements the `results(model, index)` hook, so `node_results` reports it as:

- `timeseries` — a DataFrame with `Heating Load`, `Cooling Load`, `T_air`, `T_s`, `T_m`, `T_e`
- `static` — `Capacity` (the `design_capacity`) plus zeroed cost fields
- `max_load_violation` — the soft constraint's slack [kW]; a violation above 1e-9 also warns

"Maximal heat load exceeded" warnings are the soft constraint doing its job — the design system is
slightly undersized in some hours.

---

## Parity

The implementation reproduces tsib's pre-migration `thermal/model5R1C.py` bit-for-bit. That parity
is pinned by the golden fixtures in [`../test/data/golden/`](../test/data/golden/), captured from
tsib by `_capture.py`: the zone parameters are read off tsib's own component, the expected results
are tsib's own golden values, and `test/test_zone.py` reproduces them to 1e-5 over 168 h and
1e-3 kW over the full year.
