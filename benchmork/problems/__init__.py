"""Problem definitions for benchmarking."""

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType
from benchmork.problems.linear import DietProblem, TransportationProblem
from benchmork.problems.timetabling import SchoolTimetablingProblem
from benchmork.problems.combinatorial import (
    KnapsackProblem,
    AssignmentProblem,
    TSPProblem,
    BinPackingProblem,
)

__all__ = [
    "BaseProblem",
    "ProblemData",
    "ProblemType",
    # Linear problems
    "TransportationProblem",
    "DietProblem",
    # Constraint satisfaction
    "SchoolTimetablingProblem",
    # Combinatorial problems
    "KnapsackProblem",
    "AssignmentProblem",
    "TSPProblem",
    "BinPackingProblem",
]
