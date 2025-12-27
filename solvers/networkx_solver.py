"""NetworkX graph algorithm solvers for benchmORk."""

from __future__ import annotations

import time
from typing import Any

from solvers.base import BaseSolver, SolverResult, SolverStatus
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class NetworkxShortestPathSolver(BaseSolver):
    """Shortest path solver using NetworkX (Dijkstra's algorithm)."""

    def __init__(self, **options: Any):
        super().__init__(**options)

    @property
    def name(self) -> str:
        return "networkx_dijkstra"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.COMBINATORIAL}

    def is_available(self) -> bool:
        try:
            import networkx as nx
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import networkx as nx
            return nx.__version__
        except ImportError:
            return None

    def solve(self, problem: BaseProblem) -> SolverResult:
        if not self.supports(problem):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Problem type {problem.problem_type} not supported"},
            )

        start_setup = time.perf_counter()
        problem_data = problem.generate()
        setup_time = time.perf_counter() - start_setup

        result = self._solve_impl(problem_data)

        return SolverResult(
            status=result.status,
            objective_value=result.objective_value,
            solution=result.solution,
            solve_time=result.solve_time,
            setup_time=setup_time,
            iterations=result.iterations,
            metadata=result.metadata,
        )

    def _solve_impl(self, problem_data: ProblemData) -> SolverResult:
        try:
            import networkx as nx
        except ImportError:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "NetworkX not installed"},
            )

        problem_class = problem_data.metadata.get("problem_class", "")

        if problem_class != "shortest_path":
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Unsupported problem class: {problem_class}"},
            )

        edges = problem_data.metadata.get("edges", [])
        source = problem_data.metadata.get("source", 0)
        n_nodes = problem_data.metadata.get("n_nodes", 0)

        # Build NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for u, v, w in edges:
            G.add_edge(u, v, weight=w)

        start_time = time.perf_counter()

        try:
            # Compute shortest paths from source
            distances, paths = nx.single_source_dijkstra(G, source)
            solve_time = time.perf_counter() - start_time

            # Total distance to all reachable nodes
            total_distance = sum(distances.values())

            return SolverResult(
                status=SolverStatus.OPTIMAL,
                objective_value=total_distance,
                solution={"distances": distances, "paths": paths},
                solve_time=solve_time,
                metadata={"solver": "networkx_dijkstra", "reachable_nodes": len(distances)},
            )
        except nx.NetworkXNoPath:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                metadata={"error": "No path exists"},
            )


class NetworkxMaxFlowSolver(BaseSolver):
    """Maximum flow solver using NetworkX (Edmonds-Karp algorithm)."""

    def __init__(self, **options: Any):
        super().__init__(**options)

    @property
    def name(self) -> str:
        return "networkx_maxflow"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.COMBINATORIAL}

    def is_available(self) -> bool:
        try:
            import networkx as nx
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import networkx as nx
            return nx.__version__
        except ImportError:
            return None

    def solve(self, problem: BaseProblem) -> SolverResult:
        if not self.supports(problem):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Problem type {problem.problem_type} not supported"},
            )

        start_setup = time.perf_counter()
        problem_data = problem.generate()
        setup_time = time.perf_counter() - start_setup

        result = self._solve_impl(problem_data)

        return SolverResult(
            status=result.status,
            objective_value=result.objective_value,
            solution=result.solution,
            solve_time=result.solve_time,
            setup_time=setup_time,
            iterations=result.iterations,
            metadata=result.metadata,
        )

    def _solve_impl(self, problem_data: ProblemData) -> SolverResult:
        try:
            import networkx as nx
        except ImportError:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "NetworkX not installed"},
            )

        problem_class = problem_data.metadata.get("problem_class", "")

        if problem_class != "max_flow":
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Unsupported problem class: {problem_class}"},
            )

        edges = problem_data.metadata.get("edges", [])
        source = problem_data.metadata.get("source", 0)
        sink = problem_data.metadata.get("sink", 0)
        n_nodes = problem_data.metadata.get("n_nodes", 0)

        # Build NetworkX graph
        G = nx.DiGraph()
        G.add_nodes_from(range(n_nodes))
        for u, v, cap in edges:
            G.add_edge(u, v, capacity=cap)

        start_time = time.perf_counter()

        try:
            flow_value, flow_dict = nx.maximum_flow(G, source, sink)
            solve_time = time.perf_counter() - start_time

            return SolverResult(
                status=SolverStatus.OPTIMAL,
                objective_value=flow_value,
                solution={"flow_value": flow_value, "flow_dict": flow_dict},
                solve_time=solve_time,
                metadata={"solver": "networkx_maxflow"},
            )
        except nx.NetworkXError as e:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": str(e)},
            )


class NetworkxMinCostFlowSolver(BaseSolver):
    """Minimum cost flow solver using NetworkX."""

    def __init__(self, **options: Any):
        super().__init__(**options)

    @property
    def name(self) -> str:
        return "networkx_mincost"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.COMBINATORIAL}

    def is_available(self) -> bool:
        try:
            import networkx as nx
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import networkx as nx
            return nx.__version__
        except ImportError:
            return None

    def solve(self, problem: BaseProblem) -> SolverResult:
        if not self.supports(problem):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Problem type {problem.problem_type} not supported"},
            )

        start_setup = time.perf_counter()
        problem_data = problem.generate()
        setup_time = time.perf_counter() - start_setup

        result = self._solve_impl(problem_data)

        return SolverResult(
            status=result.status,
            objective_value=result.objective_value,
            solution=result.solution,
            solve_time=result.solve_time,
            setup_time=setup_time,
            iterations=result.iterations,
            metadata=result.metadata,
        )

    def _solve_impl(self, problem_data: ProblemData) -> SolverResult:
        try:
            import networkx as nx
        except ImportError:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "NetworkX not installed"},
            )

        problem_class = problem_data.metadata.get("problem_class", "")

        if problem_class != "min_cost_flow":
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Unsupported problem class: {problem_class}"},
            )

        edges = problem_data.metadata.get("edges", [])
        supplies = problem_data.metadata.get("supplies", {})
        n_nodes = problem_data.metadata.get("n_nodes", 0)

        # Build NetworkX graph
        G = nx.DiGraph()
        for i in range(n_nodes):
            demand = -supplies.get(i, 0)  # NetworkX uses demand (opposite of supply)
            G.add_node(i, demand=demand)

        for u, v, cap, cost in edges:
            G.add_edge(u, v, capacity=cap, weight=cost)

        start_time = time.perf_counter()

        try:
            flow_cost, flow_dict = nx.network_simplex(G)
            solve_time = time.perf_counter() - start_time

            return SolverResult(
                status=SolverStatus.OPTIMAL,
                objective_value=flow_cost,
                solution={"flow_cost": flow_cost, "flow_dict": flow_dict},
                solve_time=solve_time,
                metadata={"solver": "networkx_mincost"},
            )
        except nx.NetworkXUnfeasible:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                metadata={"error": "No feasible flow exists"},
            )
        except nx.NetworkXError as e:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": str(e)},
            )
