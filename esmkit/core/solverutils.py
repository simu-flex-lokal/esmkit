# -*- coding: utf-8 -*-
"""
Solver handling for the energy system MILP: option management per solver
and auto-detection of an available solver.
"""

import logging
import os

import pyomo.opt as opt
from pyomo.contrib import appsi
from pyomo.contrib.appsi.base import TerminationCondition


def manageSolverOpts(solver, solverOpts):
    """
    Adds solver specific options.

    Parameters
    ----------
    solver: str, required
        Solver name used by pyomo
    solverOpts: dict, required
        Other solveroptions

    Returns
    -------
    solverOpts: dict
    """

    defaultOpts = {}

    defaultOpts_gurobi = {
        "Threads": 3,
        "OptimalityTol": 1e-8,
        "Method": 2,  # interior point/barrier
        "Crossover": 0,  # skip crossover: it's numerically unstable on this model
        "Cuts": 0,  # no precut of solution spce
        "NodeMethod": 2,  # interior points
        "IntFeasTol": 1e-9,  # small values in order to avoid errors with BigM
    }

    # note: scip needs a param file, which is located in the dir of he model - so, no direct options can be given
    defaultOpts_scip = {}

    defaultOpts_cbc = {"primalT": 1e-3}

    defaultOpts_glpk = {}

    defaultOpts_cplex = {
        "threads": 3,
        "lp_method": 4,  # -> "lp method 4": force barrier
        "barrier_crossover_algorithm": -1,  # -> "barrier crossover algorithm -1": skip crossover, same numerical-stability fix as gurobi/highs
    }

    defaultOpts_highs = {}

    # append default options
    if solver == "gurobi":
        defaultOpts.update(defaultOpts_gurobi)
    elif solver == "scip":
        # note: scip needs a param file, which is located in the dir of he model - so, no direct options can be given
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
        # highs is solved separately via appsi.solvers.Highs(); these
        # options end up unused, but the solver name still needs to
        # pass validation here
        defaultOpts.update(defaultOpts_highs)
    else:
        raise ValueError(
            'Solver name unknown. Please use one of "gurobi", "scip", "cbc", "glpk", "cplex" or "highs".'
        )

    # just add default options if not defined in solverOpts
    for option in defaultOpts:
        if not option in solverOpts:
            solverOpts[option] = defaultOpts[option]

    return solverOpts


def detect_solver():
    """
    Determines the MILP solver to use: first the $SOLVER environment
    variable, otherwise the first available solver in performance
    priorization, with HiGHS last as the free, open-source fallback.

    Returns
    -------
    solver name as str
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


#: base HiGHS settings for models carrying free state variables: the
#: interior point method without crossover, on an unscaled problem
HIGHS_OPTIONS = {
    "solver": "ipm",
    "simplex_scale_strategy": "off",
    "run_crossover": "off",
}

#: settings tried in order until one terminates optimal, each merged into
#: HIGHS_OPTIONS. A component with free state variables - the 5R1C thermal
#: zone is the kit's example - gives the LP primal values wide enough that
#: HiGHS breaks on some inputs: IPX in its
#: basis construction, simplex and crossover in postsolve ("excessive primal
#: values"). Which inputs those are is not predictable, and the breakdown is
#: not even reproducible for a fixed one, so the fallbacks buy robustness
#: with runtime instead of touching the model formulation.
HIGHS_FALLBACKS = (
    {},
    {"simplex_scale_strategy": "choose"},
    {"presolve": "off"},
    {"solver": "pdlp"},
)


def solve_highs(pyomo_model, tee=False, solverOpts=None):
    """
    Solves a pyomo model with HiGHS, retrying with more robust settings while
    HiGHS reports anything but an optimal solution.

    Parameters
    ----------
    pyomo_model: pyomo.ConcreteModel, required
    tee: bool, optional (default: False)
        Stream the solver log.
    solverOpts: dict, optional
        HiGHS options overriding `HIGHS_OPTIONS`. Given explicitly, they are
        taken as deliberate and used without the fallbacks.

    Returns
    -------
    The appsi results object, with the solution loaded into the model.
    """
    if solverOpts:
        attempts = [dict(HIGHS_OPTIONS, **solverOpts)]
    else:
        attempts = [dict(HIGHS_OPTIONS, **fallback) for fallback in HIGHS_FALLBACKS]

    failures = []
    for options in attempts:
        highs = appsi.solvers.Highs()
        highs.config.stream_solver = tee
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
    """
    Solves a pyomo model with the given or an auto-detected solver.

    Parameters
    ----------
    pyomo_model: pyomo.ConcreteModel, required
    solver: str, optional (default: $SOLVER or auto-detected)
    tee: bool, optional (default: False)
        Stream the solver log.
    solverOpts: dict, optional
        Additional solver options.

    Returns
    -------
    The solver results object.
    """
    if solver is None:
        solver = detect_solver()

    if solver == "glpk":
        raise ValueError(
            "Solver 'glpk' fails on badly conditioned energy system models such "
            "as the 5R1C thermal zone, although it is a MILP solver"
        )

    opts = manageSolverOpts(solver, dict(solverOpts) if solverOpts else {"Threads": 1, "LogFile": ""})

    if solver == "highs":
        # only explicitly passed options reach HiGHS (the generic
        # Threads/LogFile defaults are not valid HiGHS options)
        results = solve_highs(pyomo_model, tee=tee, solverOpts=solverOpts)
    else:
        optprob = opt.SolverFactory(solver)
        optprob.options = opts
        results = optprob.solve(pyomo_model, tee=tee)

    return results
