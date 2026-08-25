"""Solver selection and invocation: which MILP solver to use, with which
options, and how to retry HiGHS when it does not converge."""

import logging
import os

import pyomo.opt as opt
from pyomo.contrib import appsi
from pyomo.contrib.appsi.base import TerminationCondition


def manageSolverOpts(solver, solverOpts):
    """Fill ``solverOpts`` with the defaults for ``solver`` and return it.

    Given options always win; only keys that are absent are added.
    """
    defaultOpts = {}

    defaultOpts_gurobi = {
        "Threads": 3,
        "OptimalityTol": 1e-8,
        "Method": 2,
        "Crossover": 0,
        "Cuts": 0,
        "NodeMethod": 2,
        "IntFeasTol": 1e-9,
    }

    defaultOpts_scip = {}

    defaultOpts_cbc = {"primalT": 1e-3}

    defaultOpts_glpk = {}

    defaultOpts_cplex = {
        "threads": 3,
        "lp_method": 4,
        "barrier_crossover_algorithm": -1,
    }

    defaultOpts_highs = {}

    if solver == "gurobi":
        defaultOpts.update(defaultOpts_gurobi)
    elif solver == "scip":
        defaultOpts.update(defaultOpts_scip)
    elif solver == "cplex":
        defaultOpts.update(defaultOpts_cplex)
    elif solver == "cbc":
        defaultOpts.update(defaultOpts_cbc)
    elif solver == "glpk":
        defaultOpts.update(defaultOpts_glpk)
        if "Threads" in solverOpts:
            solverOpts.pop("Threads")
        if "LogFile" in solverOpts:
            solverOpts.pop("LogFile")
    elif solver == "highs":
        defaultOpts.update(defaultOpts_highs)
    else:
        raise ValueError(
            'Solver name unknown. Please use one of "gurobi", "scip", "cbc", "glpk", "cplex" or "highs".'
        )

    for option in defaultOpts:
        if not option in solverOpts:
            solverOpts[option] = defaultOpts[option]

    return solverOpts


def detect_solver():
    """Return the solver to use: ``$SOLVER`` if set, else the first available.

    Auto-detection tries gurobi, cplex, scip, cbc, highs - commercial and
    faster solvers first, the free open-source HiGHS last.
    """
    try:
        return os.environ["SOLVER"]
    except KeyError:
        DEFAULT_SOLVERS = ["gurobi", "cplex", "scip", "cbc", "highs"]
        for potential_solver in DEFAULT_SOLVERS:
            if opt.SolverFactory(potential_solver).available():
                return potential_solver
    raise LookupError(
        "No MILP solver found. Recommended: install the free, open-source "
        "HiGHS solver via `pip install esmkit[highs]` (or `uv sync --extra highs` "
        "in this repo) - no license required. Alternatively, install a "
        "commercial solver (`pip install esmkit[gurobi]`/`uv sync --extra gurobi`, "
        "which needs a Gurobi license) or install cplex/scip/cbc separately "
        "(e.g. `apt install coinor-cbc`), and declare it with the environment "
        "variable 'SOLVER' if it isn't auto-detected."
    )


# Interior point without crossover or scaling: the setting that copes best
# with the badly conditioned models this kit produces.
HIGHS_OPTIONS = {
    "solver": "ipm",
    "simplex_scale_strategy": "off",
    "run_crossover": "off",
}


# Tried in order, each merged onto HIGHS_OPTIONS: the aggressive interior
# point setting first, then progressively tamer ones.
HIGHS_FALLBACKS = (
    {},
    {"simplex_scale_strategy": "choose"},
    {"presolve": "off"},
    {"solver": "pdlp"},
)


def solve_highs(pyomo_model, tee=False, solverOpts=None):
    """Solve with HiGHS, walking the fallback ladder until a run is optimal.

    Passing explicit ``solverOpts`` disables the ladder - only that single
    setting is attempted. Raises ``RuntimeError`` listing every setting tried
    if none reaches optimality.
    """
    if solverOpts:
        attempts = [dict(HIGHS_OPTIONS, **solverOpts)]
    else:
        attempts = [dict(HIGHS_OPTIONS, **fallback) for fallback in HIGHS_FALLBACKS]

    failures = []
    for options in attempts:
        highs = appsi.solvers.Highs()
        highs.config.stream_solver = tee
        # Do not load a non-optimal solution: only a successful attempt
        # writes its values back into the model.
        highs.config.load_solution = False
        highs.highs_options = dict(options)
        results = highs.solve(pyomo_model)
        if results.termination_condition == TerminationCondition.optimal:
            results.solution_loader.load_vars()
            return results
        failures.append((options, results.termination_condition))
        if options is not attempts[-1]:
            logging.warning(
                "HiGHS terminated %s with %s, retrying with other settings.",
                results.termination_condition,
                options,
            )

    raise RuntimeError(
        "HiGHS found no optimal solution. Settings tried:\n"
        + "\n".join(
            "  {} -> {}".format(options, condition) for options, condition in failures
        )
    )


def solve_model(pyomo_model, solver=None, tee=False, solverOpts=None):
    """Solve a pyomo model with the given or auto-detected solver."""
    if solver is None:
        solver = detect_solver()

    # glpk is a MILP solver, but it does not survive these models - reject it
    # up front rather than let it return a wrong or infeasible answer.
    if solver == "glpk":
        raise ValueError(
            "Solver 'glpk' fails on badly conditioned energy system models such "
            "as the 5R1C thermal zone, although it is a MILP solver"
        )

    opts = manageSolverOpts(solver, dict(solverOpts) if solverOpts else {"Threads": 1, "LogFile": ""})

    if solver == "highs":
        results = solve_highs(pyomo_model, tee=tee, solverOpts=solverOpts)
    else:
        optprob = opt.SolverFactory(solver)
        optprob.options = opts
        results = optprob.solve(pyomo_model, tee=tee)

    return results
