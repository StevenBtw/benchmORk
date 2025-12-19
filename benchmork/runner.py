"""Benchmark execution engine."""

from dataclasses import dataclass, field
from statistics import mean, stdev
from typing import Any
from benchmork.problems.base import BaseProblem
from solvers.base import BaseSolver, SolverResult


@dataclass
class BenchmarkRun:
    """Single benchmark run result."""

    solver: str
    problem: str
    result: SolverResult
    run_index: int = 0


@dataclass
class BenchmarkSummary:
    """Summary statistics for a solver-problem pair."""

    solver: str
    problem: str
    n_runs: int
    status: str
    objective_value: float | None
    solve_time_mean: float
    solve_time_std: float
    setup_time_mean: float
    total_time_mean: float
    all_runs: list[BenchmarkRun] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "solver": self.solver,
            "problem": self.problem,
            "n_runs": self.n_runs,
            "status": self.status,
            "objective_value": self.objective_value,
            "solve_time_mean": self.solve_time_mean,
            "solve_time_std": self.solve_time_std,
            "setup_time_mean": self.setup_time_mean,
            "total_time_mean": self.total_time_mean,
        }


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    summaries: list[BenchmarkSummary] = field(default_factory=list)
    runs: list[BenchmarkRun] = field(default_factory=list)

    def to_dataframe(self):
        """Convert summaries to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([s.to_dict() for s in self.summaries])

    def get_fastest(self, problem: str | None = None) -> BenchmarkSummary | None:
        """Get the fastest solver summary."""
        summaries = self.summaries
        if problem:
            summaries = [s for s in summaries if s.problem == problem]
        if not summaries:
            return None
        return min(summaries, key=lambda s: s.solve_time_mean)


class BenchmarkRunner:
    """Runs benchmarks across solvers and problems."""

    def __init__(
        self,
        solvers: list[BaseSolver],
        problems: list[BaseProblem],
        *,
        repeats: int = 3,
        warmup: bool = True,
    ):
        self.solvers = solvers
        self.problems = problems
        self.repeats = repeats
        self.warmup = warmup

    def run(self) -> BenchmarkResults:
        """Execute all benchmarks."""
        all_runs: list[BenchmarkRun] = []
        summaries: list[BenchmarkSummary] = []

        for problem in self.problems:
            for solver in self.solvers:
                if not solver.is_available():
                    continue
                if not solver.supports(problem):
                    continue

                runs = self._run_solver_problem(solver, problem)
                all_runs.extend(runs)

                summary = self._summarize_runs(runs, solver.name, str(problem))
                summaries.append(summary)

        return BenchmarkResults(summaries=summaries, runs=all_runs)

    def _run_solver_problem(
        self, solver: BaseSolver, problem: BaseProblem
    ) -> list[BenchmarkRun]:
        """Run a solver on a problem multiple times."""
        runs = []

        # Warmup run (not counted)
        if self.warmup:
            _ = solver.solve(problem)

        # Timed runs
        for i in range(self.repeats):
            result = solver.solve(problem)
            run = BenchmarkRun(
                solver=solver.name,
                problem=str(problem),
                result=result,
                run_index=i,
            )
            runs.append(run)

        return runs

    def _summarize_runs(
        self, runs: list[BenchmarkRun], solver: str, problem: str
    ) -> BenchmarkSummary:
        """Compute summary statistics from runs."""
        solve_times = [r.result.solve_time for r in runs]
        setup_times = [r.result.setup_time for r in runs]
        total_times = [r.result.total_time for r in runs]

        # Use first run's status and objective (should be consistent)
        first = runs[0].result
        status = first.status.value
        objective = first.objective_value

        return BenchmarkSummary(
            solver=solver,
            problem=problem,
            n_runs=len(runs),
            status=status,
            objective_value=objective,
            solve_time_mean=mean(solve_times),
            solve_time_std=stdev(solve_times) if len(solve_times) > 1 else 0.0,
            setup_time_mean=mean(setup_times),
            total_time_mean=mean(total_times),
            all_runs=runs,
        )


def run_benchmark(
    solvers: list[BaseSolver],
    problems: list[BaseProblem],
    *,
    repeats: int = 3,
    warmup: bool = True,
) -> BenchmarkResults:
    """Convenience function to run benchmarks."""
    runner = BenchmarkRunner(solvers, problems, repeats=repeats, warmup=warmup)
    return runner.run()


def main():
    """CLI entry point."""
    from benchmork.problems import TransportationProblem
    from solvers import get_available_solvers

    print("benchmORk - Optimization Solver Benchmarking")
    print("=" * 50)

    solvers = get_available_solvers()
    print(f"Available solvers: {[s.name for s in solvers]}")

    problems = [
        TransportationProblem(n_sources=10, n_destinations=10),
        TransportationProblem(n_sources=50, n_destinations=50),
    ]

    results = run_benchmark(solvers, problems, repeats=3)

    print("\nResults:")
    print("-" * 50)
    for summary in results.summaries:
        print(f"{summary.solver} on {summary.problem}:")
        print(f"  Status: {summary.status}")
        print(f"  Objective: {summary.objective_value:.2f}" if summary.objective_value else "  Objective: N/A")
        print(f"  Solve time: {summary.solve_time_mean:.4f} ± {summary.solve_time_std:.4f} sec")
        print()


if __name__ == "__main__":
    main()
