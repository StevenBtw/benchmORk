"""Solver adapters for various optimization libraries."""

from solvers.base import BaseSolver, SolverResult, SolverStatus
from solvers.scipy_solver import ScipyHighsSolver, ScipySimplexSolver, ScipySolver
from solvers.pulp_solver import PulpCbcSolver, PulpGlpkSolver, PulpSolver
from solvers.solvor_solver import SolvorCPSATSolver, SolvorSolver
from solvers.solvor_metaheuristics import (
    SolvorAnnealSolver,
    SolvorTabuSolver,
    SolvorGeneticSolver,
)
from solvers.ortools_solver import (
    OrtoolsGlopSolver,
    OrtoolsCpsatSolver,
    OrtoolsScipSolver,
    OrtoolsSolver,
)
from solvers.pyomo_solver import (
    PyomoGlpkSolver,
    PyomoCbcSolver,
    PyomoIpoptSolver,
    PyomoSolver,
)

__all__ = [
    "BaseSolver",
    "SolverResult",
    "SolverStatus",
    # SciPy solvers
    "ScipyHighsSolver",
    "ScipySimplexSolver",
    "ScipySolver",  # Legacy alias
    # PuLP solvers
    "PulpCbcSolver",
    "PulpGlpkSolver",
    "PulpSolver",  # Legacy alias
    # OR-Tools solvers
    "OrtoolsGlopSolver",
    "OrtoolsCpsatSolver",
    "OrtoolsScipSolver",
    "OrtoolsSolver",  # Legacy alias
    # Pyomo solvers
    "PyomoGlpkSolver",
    "PyomoCbcSolver",
    "PyomoIpoptSolver",
    "PyomoSolver",  # Legacy alias
    # solvOR solvers
    "SolvorCPSATSolver",
    "SolvorSolver",  # Legacy alias
    # solvOR metaheuristics
    "SolvorAnnealSolver",
    "SolvorTabuSolver",
    "SolvorGeneticSolver",
]

# Registry mapping solver names to classes
SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {
    # SciPy
    "scipy_highs": ScipyHighsSolver,
    "scipy_simplex": ScipySimplexSolver,
    # PuLP
    "pulp_cbc": PulpCbcSolver,
    "pulp_glpk": PulpGlpkSolver,
    # OR-Tools
    "ortools_glop": OrtoolsGlopSolver,
    "ortools_cpsat": OrtoolsCpsatSolver,
    "ortools_scip": OrtoolsScipSolver,
    # Pyomo
    "pyomo_glpk": PyomoGlpkSolver,
    "pyomo_cbc": PyomoCbcSolver,
    "pyomo_ipopt": PyomoIpoptSolver,
    # solvOR
    "solvor_cpsat": SolvorCPSATSolver,
    # solvOR metaheuristics
    "solvor_anneal": SolvorAnnealSolver,
    "solvor_tabu": SolvorTabuSolver,
    "solvor_genetic": SolvorGeneticSolver,
}


def get_solver_by_name(name: str) -> BaseSolver | None:
    """Get a solver instance by its name."""
    solver_class = SOLVER_REGISTRY.get(name)
    if solver_class:
        return solver_class()
    return None


def get_available_solvers(problem_type: str | None = None) -> list[BaseSolver]:
    """Return list of all available solver instances.

    Args:
        problem_type: Optional problem type to filter by (e.g., "linear", "integer")
    """
    from benchmork.problems.base import ProblemType

    all_solvers = [cls() for cls in SOLVER_REGISTRY.values()]
    available = [s for s in all_solvers if s.is_available()]

    if problem_type:
        try:
            pt = ProblemType(problem_type)
            available = [s for s in available if pt in s.supported_problem_types]
        except ValueError:
            pass

    return available
