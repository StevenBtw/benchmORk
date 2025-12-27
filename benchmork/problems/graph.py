"""Graph optimization problems for benchmarking."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


@dataclass
class ShortestPathProblem(BaseProblem):
    """Single-source shortest path problem.

    Find the shortest path from a source node to all other nodes
    in a weighted directed graph.

    Attributes:
        n_nodes: Number of nodes in the graph
        density: Edge density (0.0 to 1.0)
        max_weight: Maximum edge weight
        seed: Random seed for reproducibility
    """

    n_nodes: int = 100
    density: float = 0.3
    max_weight: int = 100
    seed: int = 42

    @property
    def name(self) -> str:
        return "shortest_path"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.COMBINATORIAL

    def generate(self) -> ProblemData:
        rng = random.Random(self.seed)

        # Generate random directed graph
        edges: list[tuple[int, int, float]] = []
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and rng.random() < self.density:
                    weight = rng.randint(1, self.max_weight)
                    edges.append((i, j, weight))

        # Build adjacency list
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(self.n_nodes)}
        for u, v, w in edges:
            adj[u].append((v, w))

        # Source is node 0
        source = 0

        return ProblemData(
            objective={},
            constraints={},
            bounds={},
            metadata={
                "n_variables": self.n_nodes,
                "n_constraints": len(edges),
                "problem_class": "shortest_path",
                "n_nodes": self.n_nodes,
                "n_edges": len(edges),
                "edges": edges,
                "adjacency": adj,
                "source": source,
                "density": self.density,
            },
        )

    def get_scaling_params(self) -> dict[str, Any]:
        return {"n_nodes": self.n_nodes, "density": self.density}


@dataclass
class MaxFlowProblem(BaseProblem):
    """Maximum flow problem.

    Find the maximum flow from source to sink in a capacitated network.

    Attributes:
        n_nodes: Number of nodes in the network
        density: Edge density (0.0 to 1.0)
        max_capacity: Maximum edge capacity
        seed: Random seed for reproducibility
    """

    n_nodes: int = 50
    density: float = 0.4
    max_capacity: int = 100
    seed: int = 42

    @property
    def name(self) -> str:
        return "max_flow"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.COMBINATORIAL

    def generate(self) -> ProblemData:
        rng = random.Random(self.seed)

        # Source is 0, sink is n_nodes-1
        source = 0
        sink = self.n_nodes - 1

        # Generate random capacitated network
        # Ensure path from source to sink exists
        edges: list[tuple[int, int, int]] = []
        capacities: dict[tuple[int, int], int] = {}

        # First ensure a path exists
        path = list(range(self.n_nodes))
        rng.shuffle(path[1:-1])  # Keep source at start, sink at end
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            cap = rng.randint(1, self.max_capacity)
            edges.append((u, v, cap))
            capacities[(u, v)] = cap

        # Add random edges
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and (i, j) not in capacities:
                    if rng.random() < self.density:
                        cap = rng.randint(1, self.max_capacity)
                        edges.append((i, j, cap))
                        capacities[(i, j)] = cap

        # Build adjacency with capacities
        adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(self.n_nodes)}
        for u, v, c in edges:
            adj[u].append((v, c))

        return ProblemData(
            objective={},
            constraints={},
            bounds={},
            metadata={
                "n_variables": self.n_nodes,
                "n_constraints": len(edges),
                "problem_class": "max_flow",
                "n_nodes": self.n_nodes,
                "n_edges": len(edges),
                "edges": edges,
                "capacities": capacities,
                "adjacency": adj,
                "source": source,
                "sink": sink,
            },
        )

    def get_scaling_params(self) -> dict[str, Any]:
        return {"n_nodes": self.n_nodes, "density": self.density}


@dataclass
class MinCostFlowProblem(BaseProblem):
    """Minimum cost flow problem.

    Find the minimum cost way to send flow from sources to sinks
    while respecting capacity constraints.

    Attributes:
        n_nodes: Number of nodes in the network
        n_sources: Number of source nodes
        n_sinks: Number of sink nodes
        density: Edge density
        max_capacity: Maximum edge capacity
        max_cost: Maximum edge cost
        seed: Random seed for reproducibility
    """

    n_nodes: int = 30
    n_sources: int = 3
    n_sinks: int = 3
    density: float = 0.5
    max_capacity: int = 50
    max_cost: int = 20
    seed: int = 42

    @property
    def name(self) -> str:
        return "min_cost_flow"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.COMBINATORIAL

    def generate(self) -> ProblemData:
        rng = random.Random(self.seed)

        # Designate sources and sinks
        nodes = list(range(self.n_nodes))
        sources = nodes[: self.n_sources]
        sinks = nodes[-self.n_sinks :]
        intermediate = nodes[self.n_sources : -self.n_sinks] if self.n_sinks > 0 else nodes[self.n_sources:]

        # Generate supplies (positive) and demands (negative)
        total_supply = rng.randint(50, 200)
        supplies: dict[int, int] = {}

        # Distribute supply among sources
        remaining = total_supply
        for i, s in enumerate(sources[:-1]):
            supply = rng.randint(1, remaining - (len(sources) - i - 1))
            supplies[s] = supply
            remaining -= supply
        supplies[sources[-1]] = remaining

        # Distribute demand among sinks (negative supply)
        remaining = total_supply
        for i, s in enumerate(sinks[:-1]):
            demand = rng.randint(1, remaining - (len(sinks) - i - 1))
            supplies[s] = -demand
            remaining -= demand
        supplies[sinks[-1]] = -remaining

        # Intermediate nodes have zero supply
        for n in intermediate:
            supplies[n] = 0

        # Generate edges
        edges: list[tuple[int, int, int, int]] = []  # (from, to, capacity, cost)

        # Ensure connectivity from sources to sinks
        for src in sources:
            if intermediate:
                target = rng.choice(intermediate)
            else:
                target = rng.choice(sinks)
            cap = rng.randint(1, self.max_capacity)
            cost = rng.randint(1, self.max_cost)
            edges.append((src, target, cap, cost))

        for sink in sinks:
            if intermediate:
                src = rng.choice(intermediate)
            else:
                src = rng.choice(sources)
            cap = rng.randint(1, self.max_capacity)
            cost = rng.randint(1, self.max_cost)
            edges.append((src, sink, cap, cost))

        # Add random edges
        edge_set = {(e[0], e[1]) for e in edges}
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if i != j and (i, j) not in edge_set:
                    if rng.random() < self.density:
                        cap = rng.randint(1, self.max_capacity)
                        cost = rng.randint(1, self.max_cost)
                        edges.append((i, j, cap, cost))
                        edge_set.add((i, j))

        return ProblemData(
            objective={},
            constraints={},
            bounds={},
            metadata={
                "n_variables": self.n_nodes,
                "n_constraints": len(edges),
                "problem_class": "min_cost_flow",
                "n_nodes": self.n_nodes,
                "n_edges": len(edges),
                "edges": edges,
                "supplies": supplies,
                "sources": sources,
                "sinks": sinks,
            },
        )

    def get_scaling_params(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_sources": self.n_sources,
            "n_sinks": self.n_sinks,
        }
