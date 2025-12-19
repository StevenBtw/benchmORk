"""Base problem interface for benchmark problems."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ProblemType(Enum):
    """Types of optimization problems."""

    LINEAR = "linear"
    INTEGER = "integer"
    NONLINEAR = "nonlinear"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"


@dataclass
class ProblemData:
    """Container for generated problem data."""

    objective: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    bounds: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_variables(self) -> int:
        """Return the number of decision variables."""
        return self.metadata.get("n_variables", 0)

    @property
    def n_constraints(self) -> int:
        """Return the number of constraints."""
        return self.metadata.get("n_constraints", 0)


@dataclass
class BaseProblem(ABC):
    """Abstract base class for optimization problems.

    All benchmark problems should inherit from this class and implement
    the required methods.

    Attributes:
        name: Human-readable name of the problem.
        problem_type: Type of optimization problem (LP, MIP, NLP, etc.).
        seed: Random seed for reproducible problem generation.
    """

    seed: int = 42

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the problem name."""
        ...

    @property
    @abstractmethod
    def problem_type(self) -> ProblemType:
        """Return the type of optimization problem."""
        ...

    @abstractmethod
    def generate(self) -> ProblemData:
        """Generate the problem data.

        Returns:
            ProblemData containing objective, constraints, bounds, and metadata.
        """
        ...

    @abstractmethod
    def get_scaling_params(self) -> dict[str, Any]:
        """Return parameters that control problem size.

        These parameters can be varied to test solver performance
        at different scales.

        Returns:
            Dictionary of parameter names to their current values.
        """
        ...

    def validate_solution(self, solution: dict[str, Any]) -> bool:
        """Validate that a solution satisfies all constraints.

        Args:
            solution: Dictionary containing the solution values.

        Returns:
            True if the solution is valid, False otherwise.
        """
        # Default implementation - subclasses can override for specific validation
        return True

    def get_known_optimum(self) -> float | None:
        """Return the known optimal objective value, if available.

        Returns:
            The optimal value, or None if unknown.
        """
        return None

    def __str__(self) -> str:
        """Return a string representation of the problem."""
        params = self.get_scaling_params()
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        return f"{self.name}({param_str})"
