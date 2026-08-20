# esmkit documentation

| document | what it is | read it when |
|---|---|---|
| [`kit.md`](kit.md) | The kit: minimal example, what a spec is and what is checked before building, the component table, results, a tutorial on adding a component, and gotchas. | You are using or extending esmkit. |
| [`zone5r1c.md`](zone5r1c.md) | The 5R1C thermal zone: its explicit parameter contract, the node balances, why the gains arrive precomputed, and how parity is pinned. | You are putting a thermal zone in a system, or supplying its parameters from a building model. |
| [`model-deviations.md`](model-deviations.md) | Register of four known deviations of the 5R1C zone from its source publications. All deliberately **unfixed** to preserve parity with the original model. | **Before changing any equation in `esmkit/components/zone5r1c.py`**, or if a result looks physically wrong. |

Runnable demonstrations live in [`../examples/`](../examples/):
`district_chp.py` (a system with no building in it, and the factory from the tutorial).

The test suite doubles as executable specification — `test/test_spec.py` for the description
format, `test/test_registry.py` for the extension API, `test/test_dispatch.py` for the
flexibility proof, and `test/test_zone.py` for the parity fixtures.
