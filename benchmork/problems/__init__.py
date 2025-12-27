"""Problem definitions for benchmarking."""

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType
from benchmork.problems.linear import DietProblem, TransportationProblem
from benchmork.problems.timetabling import SchoolTimetablingProblem

__all__ = [
    "BaseProblem",
    "ProblemData",
    "ProblemType",
    "TransportationProblem",
    "DietProblem",
    "SchoolTimetablingProblem",
]
