"""Problem definitions for benchmarking."""

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType
from benchmork.problems.linear import DietProblem, TransportationProblem

__all__ = [
    "BaseProblem",
    "ProblemData",
    "ProblemType",
    "TransportationProblem",
    "DietProblem",
]
