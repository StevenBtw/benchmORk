"""Base solver interface for optimization solvers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class SolverStatus(Enum):
    """Status of a solver run."""

    OPTIMAL = "optimal"
    FEASIBLE = "feasible"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    TIME_LIMIT = "time_limit"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SolverResult:
    """Result from running a solver on a problem.

    Attributes:
        status: The termination status of the solver.
        objective_value: The objective function value (if found).
        solution: Dictionary of variable names to values.
        solve_time: Time spent solving (seconds).
        setup_time: Time spent setting up the problem (seconds).
        gap: Optimality gap for MIP solvers (if applicable).
        iterations: Number of iterations performed.
        nodes: Number of branch-and-bound nodes explored (for MIP).
        metadata: Additional solver-specific information.
    """

    status: SolverStatus
    objective_value: float | None = None
    solution: dict[str, Any] = field(default_factory=dict)
    solve_time: float = 0.0
    setup_time: float = 0.0
    gap: float | None = None
    iterations: int | None = None
    nodes: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Return total time (setup + solve)."""
        return self.setup_time + self.solve_time

    @property
    def is_optimal(self) -> bool:
        """Return True if solver found an optimal solution."""
        return self.status == SolverStatus.OPTIMAL

    @property
    def is_feasible(self) -> bool:
        """Return True if solver found any feasible solution."""
        return self.status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE)


class BaseSolver(ABC):
    """Abstract base class for solver adapters.

    All solver implementations should inherit from this class and implement
    the required methods.

    Attributes:
        name: Identifier for this solver.
        supported_problem_types: Set of problem types this solver can handle.
        options: Solver-specific configuration options.
    """

    def __init__(self, **options: Any) -> None:
        """Initialize the solver with optional configuration.

        Args:
            **options: Solver-specific options (time_limit, threads, etc.)
        """
        self.options = options
        self._time_limit = options.get("time_limit", None)
        self._threads = options.get("threads", None)

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the solver name."""
        ...

    @property
    @abstractmethod
    def supported_problem_types(self) -> set[ProblemType]:
        """Return the set of problem types this solver supports."""
        ...

    @abstractmethod
    def solve(self, problem: BaseProblem) -> SolverResult:
        """Solve the given problem.

        Args:
            problem: The problem instance to solve.

        Returns:
            SolverResult containing the solution and metrics.
        """
        ...

    @abstractmethod
    def _solve_impl(self, problem_data: ProblemData) -> SolverResult:
        """Internal implementation of the solve method.

        This method should be implemented by each solver adapter to
        translate the problem data into the solver's native format
        and execute the solver.

        Args:
            problem_data: Generated problem data.

        Returns:
            SolverResult containing the solution and metrics.
        """
        ...

    def is_available(self) -> bool:
        """Check if the solver library is installed and available.

        Returns:
            True if the solver can be used, False otherwise.
        """
        return True

    def supports(self, problem: BaseProblem) -> bool:
        """Check if this solver supports the given problem type.

        Args:
            problem: The problem to check.

        Returns:
            True if the solver can handle this problem type.
        """
        return problem.problem_type in self.supported_problem_types

    def get_version(self) -> str | None:
        """Return the version of the underlying solver library.

        Returns:
            Version string, or None if not available.
        """
        return None

    def __str__(self) -> str:
        """Return a string representation of the solver."""
        return f"{self.name}"

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        return f"{self.__class__.__name__}(options={self.options})"
