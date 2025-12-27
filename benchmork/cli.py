"""Command-line interface for benchmORk."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from benchmork.problems import (
    AssignmentProblem,
    BinPackingProblem,
    DietProblem,
    KnapsackProblem,
    MaxFlowProblem,
    MinCostFlowProblem,
    SchoolTimetablingProblem,
    ShortestPathProblem,
    TransportationProblem,
    TSPProblem,
)
from benchmork.problems.base import BaseProblem
from benchmork.runner import run_benchmark
from solvers import SOLVER_REGISTRY, get_available_solvers
from solvers.base import BaseSolver


PROBLEM_REGISTRY: dict[str, type[BaseProblem]] = {
    "transportation": TransportationProblem,
    "diet": DietProblem,
    "timetabling": SchoolTimetablingProblem,
    "knapsack": KnapsackProblem,
    "assignment": AssignmentProblem,
    "tsp": TSPProblem,
    "binpacking": BinPackingProblem,
    "shortest_path": ShortestPathProblem,
    "max_flow": MaxFlowProblem,
    "min_cost_flow": MinCostFlowProblem,
}


def create_problem(name: str, size: int | None = None, seed: int = 42) -> BaseProblem:
    """Create a problem instance by name with optional size scaling."""
    problem_class = PROBLEM_REGISTRY.get(name)
    if not problem_class:
        raise ValueError(f"Unknown problem: {name}. Available: {list(PROBLEM_REGISTRY.keys())}")

    # Apply size parameter based on problem type
    if name == "transportation":
        n = size or 20
        return TransportationProblem(n_sources=n, n_destinations=n, seed=seed)
    elif name == "diet":
        n = size or 10
        return DietProblem(n_foods=n, n_nutrients=max(3, n // 2), seed=seed)
    elif name == "timetabling":
        n = size or 5
        return SchoolTimetablingProblem(n_teachers=n, n_rooms=n, seed=seed)
    elif name == "knapsack":
        n = size or 20
        return KnapsackProblem(n_items=n, seed=seed)
    elif name == "assignment":
        n = size or 10
        return AssignmentProblem(n_agents=n, seed=seed)
    elif name == "tsp":
        n = size or 10
        return TSPProblem(n_cities=n, seed=seed)
    elif name == "binpacking":
        n = size or 20
        return BinPackingProblem(n_items=n, seed=seed)
    elif name == "shortest_path":
        n = size or 100
        return ShortestPathProblem(n_nodes=n, seed=seed)
    elif name == "max_flow":
        n = size or 50
        return MaxFlowProblem(n_nodes=n, seed=seed)
    elif name == "min_cost_flow":
        n = size or 30
        return MinCostFlowProblem(n_nodes=n, seed=seed)
    else:
        return problem_class()


def cmd_run(args: argparse.Namespace) -> int:
    """Run benchmarks."""
    # Parse solvers
    if args.solvers:
        solver_names = [s.strip() for s in args.solvers.split(",")]
        solvers: list[BaseSolver] = []
        for name in solver_names:
            solver_class = SOLVER_REGISTRY.get(name)
            if not solver_class:
                print(f"Error: Unknown solver '{name}'", file=sys.stderr)
                print(f"Available: {list(SOLVER_REGISTRY.keys())}", file=sys.stderr)
                return 1
            solver = solver_class()
            if not solver.is_available():
                print(f"Warning: Solver '{name}' is not available (missing dependencies)")
                continue
            solvers.append(solver)
    else:
        solvers = get_available_solvers()

    if not solvers:
        print("Error: No solvers available", file=sys.stderr)
        return 1

    # Parse problems
    if args.problems:
        problem_names = [p.strip() for p in args.problems.split(",")]
        problems: list[BaseProblem] = []
        for name in problem_names:
            try:
                problem = create_problem(name, args.size, args.seed)
                problems.append(problem)
            except ValueError as e:
                print(f"Error: {e}", file=sys.stderr)
                return 1
    else:
        # Default problems
        problems = [
            TransportationProblem(n_sources=args.size or 20, n_destinations=args.size or 20),
        ]

    print(f"benchmORk - Running {len(problems)} problem(s) with {len(solvers)} solver(s)")
    print("=" * 60)
    print(f"Solvers: {[s.name for s in solvers]}")
    print(f"Problems: {[str(p) for p in problems]}")
    print(f"Repeats: {args.repeats}, Memory tracking: {args.memory}")
    print()

    results = run_benchmark(
        solvers,
        problems,
        repeats=args.repeats,
        warmup=not args.no_warmup,
        track_memory=args.memory,
    )

    # Print results
    print("Results:")
    print("-" * 60)
    for summary in results.summaries:
        print(f"\n{summary.solver} on {summary.problem}:")
        print(f"  Status: {summary.status}")
        if summary.objective_value is not None:
            print(f"  Objective: {summary.objective_value:.6g}")
        mean_ms = summary.solve_time_mean * 1000
        std_ms = summary.solve_time_std * 1000
        print(f"  Solve time: {mean_ms:.3f} +/- {std_ms:.3f} ms")
        print(f"  Setup time: {summary.setup_time_mean*1000:.3f} ms")
        if summary.peak_memory_mean_mb is not None:
            mem_mean = summary.peak_memory_mean_mb
            mem_max = summary.peak_memory_max_mb
            print(f"  Peak memory: {mem_mean:.2f} MB (max: {mem_max:.2f} MB)")

    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "summaries": [s.to_dict() for s in results.summaries],
            "config": {
                "repeats": args.repeats,
                "memory_tracking": args.memory,
                "seed": args.seed,
            },
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return 0


def cmd_list_solvers(args: argparse.Namespace) -> int:
    """List available solvers."""
    print("benchmORk - Solver Registry")
    print("=" * 60)

    if args.available:
        print("\nAvailable Solvers:")
        available = get_available_solvers()
        if not available:
            print("  No solvers available")
        for solver in available:
            version = solver.get_version() or "unknown"
            types = ", ".join(t.value for t in solver.supported_problem_types)
            print(f"  {solver.name:<20} v{version:<10} [{types}]")
    else:
        print("\nAll Registered Solvers:")
        for name, solver_class in sorted(SOLVER_REGISTRY.items()):
            solver = solver_class()
            status = "+" if solver.is_available() else "-"
            types = ", ".join(t.value for t in solver.supported_problem_types)
            print(f"  [{status}] {name:<20} [{types}]")

    return 0


def cmd_list_problems(_args: argparse.Namespace) -> int:
    """List available problems."""
    print("benchmORk - Problem Registry")
    print("=" * 60)
    print()

    for name, problem_class in sorted(PROBLEM_REGISTRY.items()):
        # Create instance to get type
        problem = problem_class()
        ptype = problem.problem_type.value
        print(f"  {name:<20} [{ptype}]")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare benchmark results."""
    files = args.files

    if len(files) < 2:
        print("Error: Need at least 2 result files to compare", file=sys.stderr)
        return 1

    results: list[dict[str, Any]] = []
    for f in files:
        path = Path(f)
        if not path.exists():
            print(f"Error: File not found: {f}", file=sys.stderr)
            return 1
        with open(path, encoding="utf-8") as fp:
            results.append(json.load(fp))

    print("benchmORk - Result Comparison")
    print("=" * 60)

    # Build comparison table
    all_solvers: set[str] = set()
    all_problems: set[str] = set()
    for result in results:
        for s in result.get("summaries", []):
            all_solvers.add(s["solver"])
            all_problems.add(s["problem"])

    for problem in sorted(all_problems):
        print(f"\n{problem}:")
        print("-" * 40)

        # Find best time for this problem across all files
        best_time = float("inf")
        data_by_solver: dict[str, list[dict[str, Any] | None]] = {}

        for i, result in enumerate(results):
            for s in result.get("summaries", []):
                if s["problem"] == problem:
                    solver = s["solver"]
                    if solver not in data_by_solver:
                        data_by_solver[solver] = [None] * len(results)
                    data_by_solver[solver][i] = s
                    if s["solve_time_mean"] < best_time:
                        best_time = s["solve_time_mean"]

        for solver in sorted(data_by_solver.keys()):
            entries = data_by_solver[solver]
            line = f"  {solver:<20}"
            for i, entry in enumerate(entries):
                if entry:
                    t = entry["solve_time_mean"]
                    ratio = t / best_time if best_time > 0 else 0
                    marker = " *" if ratio == 1.0 else ""
                    line += f"  {t*1000:>8.2f} ms ({ratio:.2f}x){marker}"
                else:
                    line += f"  {'N/A':>8}"
            print(line)

    print("\n* = fastest")
    return 0


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="benchmork",
        description="benchmORk - Optimization Solver Benchmarking",
    )
    parser.add_argument("--version", action="version", version="benchmORk 0.2.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run benchmarks")
    run_parser.add_argument(
        "--solvers", "-s",
        help="Comma-separated list of solvers (default: all available)"
    )
    run_parser.add_argument(
        "--problems", "-p",
        help="Comma-separated list of problems (default: transportation)"
    )
    run_parser.add_argument(
        "--size", "-n",
        type=int,
        help="Problem size parameter"
    )
    run_parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=3,
        help="Number of benchmark repeats (default: 3)"
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    run_parser.add_argument(
        "--memory", "-m",
        action="store_true",
        help="Track memory usage"
    )
    run_parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable warmup run"
    )
    run_parser.add_argument(
        "--output", "-o",
        help="Output file for results (JSON)"
    )

    # list-solvers command
    list_solvers_parser = subparsers.add_parser("list-solvers", help="List solvers")
    list_solvers_parser.add_argument(
        "--available", "-a",
        action="store_true",
        help="Only show available solvers"
    )

    # list-problems command
    subparsers.add_parser("list-problems", help="List problems")

    # compare command
    compare_parser = subparsers.add_parser("compare", help="Compare benchmark results")
    compare_parser.add_argument(
        "files",
        nargs="+",
        help="Result files to compare (JSON)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "run":
        return cmd_run(args)
    elif args.command == "list-solvers":
        return cmd_list_solvers(args)
    elif args.command == "list-problems":
        return cmd_list_problems(args)
    elif args.command == "compare":
        return cmd_compare(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
