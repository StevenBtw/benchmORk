"""benchmORk - A benchmarking framework for Python optimization solvers."""

from benchmork.types import Status, SolverResult, Progress, ProgressCallback, Evaluator
from benchmork.progress import (
    default_progress,
    timed_progress,
    report_progress,
    ConvergenceTracker,
)
from benchmork.validate import (
    check_matrix_dims,
    check_sequence_lengths,
    check_bounds,
    check_positive,
    check_non_negative,
    check_in_range,
)

__version__ = "0.2.0"

__all__ = [
    # Types
    "Status",
    "SolverResult",
    "Progress",
    "ProgressCallback",
    "Evaluator",
    # Progress utilities
    "default_progress",
    "timed_progress",
    "report_progress",
    "ConvergenceTracker",
    # Validation utilities
    "check_matrix_dims",
    "check_sequence_lengths",
    "check_bounds",
    "check_positive",
    "check_non_negative",
    "check_in_range",
]
