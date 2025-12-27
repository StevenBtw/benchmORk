"""Unified types for benchmarking optimization solvers.

This module provides the core type definitions used across the benchmark suite,
inspired by solvOR's proven patterns but extended for benchmarking needs.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import IntEnum, auto
from os import environ
from typing import Any

__all__ = [
    "Status",
    "SolverResult",
    "Progress",
    "ProgressCallback",
    "Evaluator",
]

_DEBUG = bool(environ.get("DEBUG"))


class Status(IntEnum):
    """Solver termination status."""

    OPTIMAL = auto()
    FEASIBLE = auto()
    INFEASIBLE = auto()
    UNBOUNDED = auto()
    TIME_LIMIT = auto()
    MAX_ITER = auto()
    ERROR = auto()
    UNKNOWN = auto()


@dataclass(frozen=True, slots=True)
class SolverResult[T]:
    """Result from running a solver on a problem.

    Generic over solution type T. Immutable for safe sharing and comparison.

    Attributes:
        solution: The solution found (variable assignments, path, etc.)
        objective: Objective function value (if applicable)
        status: Termination status
        solve_time: Time spent solving (seconds)
        setup_time: Time spent setting up the problem (seconds)
        iterations: Number of iterations/nodes explored
        evaluations: Number of objective function evaluations
        gap: Optimality gap for MIP solvers (if applicable)
        peak_memory_mb: Peak memory usage during solve (if tracked)
        metadata: Additional solver-specific information
    """

    solution: T
    objective: float
    status: Status = Status.OPTIMAL
    solve_time: float = 0.0
    setup_time: float = 0.0
    iterations: int = 0
    evaluations: int = 0
    gap: float | None = None
    peak_memory_mb: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_time(self) -> float:
        """Return total time (setup + solve)."""
        return self.setup_time + self.solve_time

    @property
    def ok(self) -> bool:
        """Return True if solver found a feasible solution."""
        return self.status in (Status.OPTIMAL, Status.FEASIBLE)

    @property
    def is_optimal(self) -> bool:
        """Return True if solver proved optimality."""
        return self.status == Status.OPTIMAL

    def log(self, prefix: str = "") -> "SolverResult[T]":
        """Print debug info when DEBUG=1. Returns self for chaining."""
        if _DEBUG:
            msg = f"{prefix}{self.status.name}: obj={self.objective:.6g}"
            msg += f" solve={self.solve_time:.3f}s"
            if self.gap is not None:
                msg += f" gap={self.gap:.2%}"
            print(msg)
        return self


@dataclass(frozen=True, slots=True)
class Progress:
    """Solver progress info passed to callbacks.

    Attributes:
        iteration: Current iteration number
        objective: Current objective value
        best: Best objective found so far (None if same as objective)
        evaluations: Number of objective function evaluations
        elapsed: Elapsed time in seconds (if tracked)
    """

    iteration: int
    objective: float
    best: float | None = None
    evaluations: int = 0
    elapsed: float | None = None


ProgressCallback = Callable[[Progress], bool | None]
"""Callback for solver progress updates.

Return True to request early termination, None/False to continue.
"""


class Evaluator[T]:
    """Wraps objective function to track evaluations and handle minimize/maximize.

    Internally works with minimization semantics (smaller is better).
    Use `to_user()` to convert back to user-facing values.

    Example:
        evaluator = Evaluator(objective_fn, minimize=True)
        obj = evaluator(solution)  # Internal (signed) value
        user_obj = evaluator.to_user(obj)  # User-facing value
        print(f"Evaluations: {evaluator.evals}")
    """

    __slots__ = ("objective_fn", "sign", "evals")

    def __init__(self, objective_fn: Callable[[T], float], *, minimize: bool = True):
        self.objective_fn = objective_fn
        self.sign = 1 if minimize else -1
        self.evals = 0

    def __call__(self, sol: T) -> float:
        """Evaluate solution, returning internal (signed) value."""
        self.evals += 1
        return self.sign * self.objective_fn(sol)

    def to_user(self, internal_obj: float) -> float:
        """Convert internal objective to user-facing value."""
        return internal_obj * self.sign

    def reset(self) -> None:
        """Reset evaluation counter."""
        self.evals = 0
