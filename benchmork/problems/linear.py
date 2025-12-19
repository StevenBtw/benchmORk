"""Linear programming problems for benchmarking."""

from dataclasses import dataclass
import numpy as np
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


@dataclass
class TransportationProblem(BaseProblem):
    """Classic transportation problem.

    Minimize total shipping cost from sources to destinations.

    min  sum(c[i,j] * x[i,j])
    s.t. sum_j(x[i,j]) <= supply[i]    for all sources i
         sum_i(x[i,j]) >= demand[j]    for all destinations j
         x[i,j] >= 0
    """

    n_sources: int = 10
    n_destinations: int = 10
    seed: int = 42

    @property
    def name(self) -> str:
        return "transportation"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.LINEAR

    def generate(self) -> ProblemData:
        rng = np.random.default_rng(self.seed)

        m, n = self.n_sources, self.n_destinations

        # Cost matrix (random costs between 1 and 100)
        costs = rng.integers(1, 100, size=(m, n)).astype(float)

        # Supply at each source
        supply = rng.integers(50, 150, size=m).astype(float)

        # Demand at each destination (scaled to be feasible)
        total_supply = supply.sum()
        demand_raw = rng.integers(50, 150, size=n).astype(float)
        demand = demand_raw * (total_supply * 0.9 / demand_raw.sum())

        # Flatten for standard LP form: min c @ x
        c = costs.flatten()

        # Inequality constraints: Ax <= b
        # Supply constraints: sum_j(x[i,j]) <= supply[i]
        n_vars = m * n
        A_supply = np.zeros((m, n_vars))
        for i in range(m):
            A_supply[i, i * n : (i + 1) * n] = 1.0

        # Demand constraints: -sum_i(x[i,j]) <= -demand[j] (flipped for <=)
        A_demand = np.zeros((n, n_vars))
        for j in range(n):
            for i in range(m):
                A_demand[j, i * n + j] = -1.0

        A_ub = np.vstack([A_supply, A_demand])
        b_ub = np.concatenate([supply, -demand])

        # Variable bounds: x >= 0
        bounds = [(0.0, None) for _ in range(n_vars)]

        return ProblemData(
            objective={"c": c, "minimize": True},
            constraints={
                "A_ub": A_ub,
                "b_ub": b_ub,
                "A_eq": None,
                "b_eq": None,
            },
            bounds={"bounds": bounds},
            metadata={
                "n_variables": n_vars,
                "n_constraints": m + n,
                "n_sources": m,
                "n_destinations": n,
                "costs": costs,
                "supply": supply,
                "demand": demand,
            },
        )

    def get_scaling_params(self) -> dict[str, int]:
        return {
            "n_sources": self.n_sources,
            "n_destinations": self.n_destinations,
        }


@dataclass
class DietProblem(BaseProblem):
    """Classic diet problem.

    Minimize cost of food while meeting nutritional requirements.

    min  sum(cost[j] * x[j])
    s.t. sum_j(nutrient[i,j] * x[j]) >= requirement[i]  for all nutrients i
         x[j] >= 0
    """

    n_foods: int = 20
    n_nutrients: int = 8
    seed: int = 42

    @property
    def name(self) -> str:
        return "diet"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.LINEAR

    def generate(self) -> ProblemData:
        rng = np.random.default_rng(self.seed)

        n_foods = self.n_foods
        n_nutrients = self.n_nutrients

        # Cost per unit of each food
        costs = rng.uniform(1, 10, size=n_foods)

        # Nutrient content: nutrients[i, j] = amount of nutrient i in food j
        nutrients = rng.uniform(0, 10, size=(n_nutrients, n_foods))

        # Minimum requirements for each nutrient
        requirements = rng.uniform(10, 50, size=n_nutrients)

        # LP form: min c @ x, A_ub @ x <= b_ub
        c = costs

        # Constraints: -nutrients @ x <= -requirements (nutrient >= requirement)
        A_ub = -nutrients
        b_ub = -requirements

        # Bounds: x >= 0
        bounds = [(0.0, None) for _ in range(n_foods)]

        return ProblemData(
            objective={"c": c, "minimize": True},
            constraints={
                "A_ub": A_ub,
                "b_ub": b_ub,
                "A_eq": None,
                "b_eq": None,
            },
            bounds={"bounds": bounds},
            metadata={
                "n_variables": n_foods,
                "n_constraints": n_nutrients,
                "n_foods": n_foods,
                "n_nutrients": n_nutrients,
                "costs": costs,
                "nutrients": nutrients,
                "requirements": requirements,
            },
        )

    def get_scaling_params(self) -> dict[str, int]:
        return {
            "n_foods": self.n_foods,
            "n_nutrients": self.n_nutrients,
        }
