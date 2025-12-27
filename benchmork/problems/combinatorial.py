"""Combinatorial optimization benchmark problems.

Classic NP-hard problems for benchmarking exact and heuristic solvers.
"""

from dataclasses import dataclass, field
from random import Random
from typing import Any

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


@dataclass
class KnapsackProblem(BaseProblem):
    """0/1 Knapsack problem benchmark.

    Select items to maximize total value without exceeding capacity.
    Each item is either fully selected or not (binary decision).

    Can be solved by:
    - MIP solvers (exact)
    - DP algorithms (exact for integer weights)
    - Metaheuristics (heuristic)
    """

    n_items: int = 20
    capacity_ratio: float = 0.5
    seed: int = 42

    @property
    def name(self) -> str:
        return "knapsack"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.INTEGER

    def get_scaling_params(self) -> dict[str, Any]:
        return {"n_items": self.n_items, "capacity_ratio": self.capacity_ratio}

    def generate(self) -> ProblemData:
        rng = Random(self.seed)
        n = self.n_items

        values = [rng.randint(1, 100) for _ in range(n)]
        weights = [rng.randint(1, 50) for _ in range(n)]
        capacity = int(sum(weights) * self.capacity_ratio)

        c = [-v for v in values]
        A_ub = [weights]
        b_ub = [capacity]
        bounds = [(0, 1) for _ in range(n)]

        return ProblemData(
            objective={"c": c, "minimize": True},
            constraints={"A_ub": A_ub, "b_ub": b_ub},
            bounds={"bounds": bounds},
            metadata={
                "n_variables": n,
                "n_constraints": 1,
                "integers": list(range(n)),
                "values": values,
                "weights": weights,
                "capacity": capacity,
                "problem_class": "knapsack",
            },
        )

    def get_known_optimum(self) -> float | None:
        return None


@dataclass
class AssignmentProblem(BaseProblem):
    """Assignment problem benchmark.

    Assign n workers to n tasks minimizing total cost.
    Each worker assigned to exactly one task, each task to one worker.

    Can be solved by:
    - Hungarian algorithm (exact, O(n^3))
    - MIP solvers (exact)
    - Network flow (exact)
    """

    n_agents: int = 10
    seed: int = 42

    @property
    def name(self) -> str:
        return "assignment"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.INTEGER

    def get_scaling_params(self) -> dict[str, Any]:
        return {"n_agents": self.n_agents}

    def generate(self) -> ProblemData:
        rng = Random(self.seed)
        n = self.n_agents

        cost_matrix = [[rng.randint(1, 100) for _ in range(n)] for _ in range(n)]

        c = [cost_matrix[i][j] for i in range(n) for j in range(n)]
        n_vars = n * n

        A_eq = []
        b_eq = []

        for i in range(n):
            row = [0] * n_vars
            for j in range(n):
                row[i * n + j] = 1
            A_eq.append(row)
            b_eq.append(1)

        for j in range(n):
            row = [0] * n_vars
            for i in range(n):
                row[i * n + j] = 1
            A_eq.append(row)
            b_eq.append(1)

        bounds = [(0, 1) for _ in range(n_vars)]

        return ProblemData(
            objective={"c": c, "minimize": True},
            constraints={"A_eq": A_eq, "b_eq": b_eq},
            bounds={"bounds": bounds},
            metadata={
                "n_variables": n_vars,
                "n_constraints": 2 * n,
                "integers": list(range(n_vars)),
                "n_agents": n,
                "cost_matrix": cost_matrix,
                "problem_class": "assignment",
            },
        )


@dataclass
class TSPProblem(BaseProblem):
    """Traveling Salesman Problem benchmark.

    Find shortest tour visiting all cities exactly once and returning to start.

    Can be solved by:
    - MIP solvers with subtour elimination (exact, slow)
    - Branch and bound (exact)
    - Metaheuristics: simulated annealing, genetic, tabu search (heuristic)
    """

    n_cities: int = 10
    grid_size: int = 100
    seed: int = 42
    _coordinates: list[tuple[int, int]] = field(default_factory=list, repr=False)
    _distance_matrix: list[list[float]] = field(default_factory=list, repr=False)

    @property
    def name(self) -> str:
        return "tsp"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.INTEGER

    def get_scaling_params(self) -> dict[str, Any]:
        return {"n_cities": self.n_cities}

    def _generate_instance(self) -> None:
        """Generate city coordinates and distance matrix."""
        if self._coordinates:
            return

        rng = Random(self.seed)
        n = self.n_cities

        self._coordinates = [
            (rng.randint(0, self.grid_size), rng.randint(0, self.grid_size))
            for _ in range(n)
        ]

        self._distance_matrix = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                x1, y1 = self._coordinates[i]
                x2, y2 = self._coordinates[j]
                dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                self._distance_matrix[i][j] = dist
                self._distance_matrix[j][i] = dist

    def generate(self) -> ProblemData:
        self._generate_instance()
        n = self.n_cities

        c = []
        for i in range(n):
            for j in range(n):
                c.append(self._distance_matrix[i][j])

        n_vars = n * n
        A_eq = []
        b_eq = []

        for i in range(n):
            row = [0] * n_vars
            for j in range(n):
                row[i * n + j] = 1
            A_eq.append(row)
            b_eq.append(1)

        for j in range(n):
            row = [0] * n_vars
            for i in range(n):
                row[i * n + j] = 1
            A_eq.append(row)
            b_eq.append(1)

        bounds = [(0, 1) for _ in range(n_vars)]

        return ProblemData(
            objective={"c": c, "minimize": True},
            constraints={"A_eq": A_eq, "b_eq": b_eq},
            bounds={"bounds": bounds},
            metadata={
                "n_variables": n_vars,
                "n_constraints": 2 * n,
                "integers": list(range(n_vars)),
                "n_cities": n,
                "coordinates": self._coordinates,
                "distance_matrix": self._distance_matrix,
                "problem_class": "tsp",
            },
        )

    def get_distance_matrix(self) -> list[list[float]]:
        """Return the distance matrix for direct solver use."""
        self._generate_instance()
        return self._distance_matrix

    def tour_length(self, tour: list[int]) -> float:
        """Calculate the total length of a tour."""
        self._generate_instance()
        total = 0.0
        for i in range(len(tour)):
            total += self._distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return total


@dataclass
class BinPackingProblem(BaseProblem):
    """Bin Packing problem benchmark.

    Pack items of various sizes into minimum number of bins of fixed capacity.

    Can be solved by:
    - MIP solvers (exact)
    - First-fit decreasing heuristic
    - Metaheuristics
    """

    n_items: int = 20
    bin_capacity: int = 100
    seed: int = 42

    @property
    def name(self) -> str:
        return "bin_packing"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.INTEGER

    def get_scaling_params(self) -> dict[str, Any]:
        return {"n_items": self.n_items, "bin_capacity": self.bin_capacity}

    def generate(self) -> ProblemData:
        rng = Random(self.seed)
        n = self.n_items
        cap = self.bin_capacity

        sizes = [rng.randint(1, cap // 2) for _ in range(n)]

        max_bins = n
        n_vars = n * max_bins + max_bins

        c = [0] * (n * max_bins) + [1] * max_bins

        A_eq = []
        b_eq = []

        for i in range(n):
            row = [0] * n_vars
            for b in range(max_bins):
                row[i * max_bins + b] = 1
            A_eq.append(row)
            b_eq.append(1)

        A_ub = []
        b_ub = []

        for b in range(max_bins):
            row = [0] * n_vars
            for i in range(n):
                row[i * max_bins + b] = sizes[i]
            row[n * max_bins + b] = -cap
            A_ub.append(row)
            b_ub.append(0)

        bounds = [(0, 1) for _ in range(n_vars)]

        return ProblemData(
            objective={"c": c, "minimize": True},
            constraints={"A_eq": A_eq, "b_eq": b_eq, "A_ub": A_ub, "b_ub": b_ub},
            bounds={"bounds": bounds},
            metadata={
                "n_variables": n_vars,
                "n_constraints": n + max_bins,
                "integers": list(range(n_vars)),
                "n_items": n,
                "sizes": sizes,
                "bin_capacity": cap,
                "max_bins": max_bins,
                "problem_class": "bin_packing",
            },
        )
