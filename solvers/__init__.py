"""Solver adapters for various optimization libraries."""

from solvers.base import BaseSolver, SolverResult, SolverStatus
from solvers.scipy_solver import ScipyHighsSolver, ScipySimplexSolver, ScipySolver
from solvers.pulp_solver import PulpCbcSolver, PulpGlpkSolver, PulpSolver
from solvers.solvor_solver import SolvorCPSATSolver, SolvorSolver

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
    # solvOR solvers
    "SolvorCPSATSolver",
    "SolvorSolver",  # Legacy alias
]

# Registry mapping solver names to classes
SOLVER_REGISTRY: dict[str, type[BaseSolver]] = {
    "scipy_highs": ScipyHighsSolver,
    "scipy_simplex": ScipySimplexSolver,
    "pulp_cbc": PulpCbcSolver,
    "pulp_glpk": PulpGlpkSolver,
    "solvor_cpsat": SolvorCPSATSolver,
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
