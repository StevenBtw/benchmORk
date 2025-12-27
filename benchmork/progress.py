"""Progress tracking utilities for benchmark solvers.

Provides callbacks and helpers for real-time solver monitoring, time limits,
and convergence tracking during benchmarks.
"""

from collections.abc import Callable
from time import perf_counter

from benchmork.types import Progress, ProgressCallback

__all__ = [
    "timed_progress",
    "default_progress",
    "report_progress",
    "ConvergenceTracker",
]


def timed_progress(
    callback: Callable[[Progress, float], bool | None],
) -> ProgressCallback:
    """Wrap a callback to receive elapsed time as second argument.

    Example:
        def my_callback(progress, elapsed):
            print(f"iter {progress.iteration}, time {elapsed:.2f}s")
            return elapsed > 60  # Stop after 60 seconds

        result = solver(func, on_progress=timed_progress(my_callback))
    """
    start = perf_counter()

    def wrapper(progress: Progress) -> bool | None:
        elapsed = perf_counter() - start
        return callback(progress, elapsed)

    return wrapper


def default_progress(
    name: str = "",
    *,
    interval: int = 100,
    time_limit: float | None = None,
) -> ProgressCallback:
    """Create a default progress callback with formatted output.

    Args:
        name: Solver name prefix for output (optional)
        interval: Print every N iterations (default 100)
        time_limit: Stop after this many seconds (optional)

    Example:
        result = solver(func, on_progress=default_progress("HiGHS", time_limit=30))
        # Output: HiGHS iter=100 obj=1.234 best=0.567 time=0.42s
    """
    start = perf_counter()
    prefix = f"{name} " if name else ""

    def callback(progress: Progress) -> bool | None:
        elapsed = perf_counter() - start
        if progress.iteration % interval == 0:
            best = progress.best if progress.best is not None else progress.objective
            print(
                f"{prefix}iter={progress.iteration} "
                f"obj={progress.objective:.6g} best={best:.6g} "
                f"time={elapsed:.2f}s"
            )
        if time_limit is not None and elapsed > time_limit:
            return True
        return None

    return callback


def report_progress(
    on_progress: ProgressCallback | None,
    progress_interval: int,
    iteration: int,
    current_obj: float,
    best_obj: float,
    evals: int,
    *,
    elapsed: float | None = None,
) -> bool:
    """Report progress if interval reached. Returns True if callback requested stop.

    Args:
        on_progress: Progress callback or None
        progress_interval: Report every N iterations (0 = disabled)
        iteration: Current iteration number
        current_obj: Current objective value (user-facing)
        best_obj: Best objective found so far (user-facing)
        evals: Number of objective evaluations
        elapsed: Elapsed time in seconds (optional)

    Returns:
        True if callback returned True (stop requested), False otherwise.
    """
    if not (on_progress and progress_interval > 0 and iteration % progress_interval == 0):
        return False

    progress = Progress(
        iteration=iteration,
        objective=current_obj,
        best=best_obj if best_obj != current_obj else None,
        evaluations=evals,
        elapsed=elapsed,
    )
    return on_progress(progress) is True


class ConvergenceTracker:
    """Track solver convergence over time for benchmarking.

    Records objective values at regular intervals for plotting convergence curves.

    Example:
        tracker = ConvergenceTracker(interval=10)
        for i in range(1000):
            obj = solve_iteration()
            tracker.record(i, obj)
        print(tracker.history)  # [(0, 100.0), (10, 95.2), (20, 88.1), ...]
    """

    __slots__ = ("interval", "history", "_start_time", "_best")

    def __init__(self, *, interval: int = 1):
        self.interval = interval
        self.history: list[tuple[int, float, float]] = []  # (iter, obj, elapsed)
        self._start_time = perf_counter()
        self._best: float | None = None

    def record(self, iteration: int, objective: float) -> None:
        """Record objective value at given iteration."""
        if iteration % self.interval == 0:
            elapsed = perf_counter() - self._start_time
            self.history.append((iteration, objective, elapsed))

        if self._best is None or objective < self._best:
            self._best = objective

    @property
    def best(self) -> float | None:
        """Return best objective seen."""
        return self._best

    def reset(self) -> None:
        """Clear history and reset timer."""
        self.history.clear()
        self._start_time = perf_counter()
        self._best = None

    def to_callback(self) -> ProgressCallback:
        """Convert tracker to a progress callback."""

        def callback(progress: Progress) -> None:
            self.record(progress.iteration, progress.objective)

        return callback
