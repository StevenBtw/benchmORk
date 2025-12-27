"""solvOR metaheuristic solver adapters for benchmORk.

Wraps solvOR's metaheuristic solvers (anneal, tabu_search, evolve) for
benchmarking against exact optimization methods.
"""

from __future__ import annotations

import random
import time
from typing import TYPE_CHECKING, Any

from solvers.base import BaseSolver, ProblemData, ProblemType

if TYPE_CHECKING:
    from benchmork.types import SolverResult


class SolvorAnnealSolver(BaseSolver):
    """Simulated annealing solver via solvOR.

    Best suited for combinatorial optimization problems like TSP,
    where the solution space can be explored through local moves.
    """

    name = "solvor_anneal"
    supported_problem_types = [ProblemType.COMBINATORIAL]

    def __init__(
        self,
        temperature: float = 1000.0,
        cooling: float = 0.9995,
        max_iter: int = 100_000,
        seed: int | None = None,
    ):
        self.temperature = temperature
        self.cooling = cooling
        self.max_iter = max_iter
        self.seed = seed

    def solve(self, problem_data: ProblemData) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        try:
            from solvor import anneal
        except ImportError:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "solvOR not installed"},
            )

        problem_type = problem_data.metadata.get("problem_type", "")

        if problem_type == "tsp":
            return self._solve_tsp(problem_data, anneal)
        elif problem_type == "knapsack":
            return self._solve_knapsack(problem_data, anneal)
        else:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": f"Unsupported problem type: {problem_type}"},
            )

    def _solve_tsp(self, problem_data: ProblemData, anneal: Any) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        distance_matrix = problem_data.metadata.get("distance_matrix")
        n_cities = problem_data.metadata.get("n_cities")

        if distance_matrix is None or n_cities is None:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "Missing distance_matrix or n_cities in metadata"},
            )

        # Set seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)

        # Initial solution: random tour
        initial = list(range(n_cities))
        random.shuffle(initial)

        # Objective: total tour distance
        def objective(tour: list[int]) -> float:
            total = 0.0
            for i in range(len(tour)):
                total += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
            return total

        # Neighbors: 2-opt swaps
        def neighbors(tour: list[int]) -> list[list[int]]:
            result = []
            n = len(tour)
            for i in range(n):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue  # Skip if it would just reverse the tour
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    result.append(new_tour)
            return result

        start_time = time.perf_counter()

        result = anneal(
            initial=initial,
            objective_fn=objective,
            neighbors=neighbors,
            minimize=True,
            temperature=self.temperature,
            cooling=self.cooling,
            max_iter=self.max_iter,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            solution=result.solution,
            objective=result.objective,
            status=Status.OPTIMAL if result.objective < float("inf") else Status.FEASIBLE,
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_anneal", "problem_type": "tsp"},
        )

    def _solve_knapsack(self, problem_data: ProblemData, anneal: Any) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        weights = problem_data.metadata.get("weights")
        values = problem_data.metadata.get("values")
        capacity = problem_data.metadata.get("capacity")
        n_items = problem_data.metadata.get("n_items")

        if any(x is None for x in [weights, values, capacity, n_items]):
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "Missing knapsack parameters in metadata"},
            )

        if self.seed is not None:
            random.seed(self.seed)

        # Initial: empty knapsack
        initial = [0] * n_items

        # Objective: negative value (we minimize, want max value)
        def objective(selection: list[int]) -> float:
            total_weight = sum(w * s for w, s in zip(weights, selection))
            if total_weight > capacity:
                return float("inf")  # Infeasible
            return -sum(v * s for v, s in zip(values, selection))

        # Neighbors: flip one item
        def neighbors(selection: list[int]) -> list[list[int]]:
            result = []
            for i in range(len(selection)):
                new_sel = selection.copy()
                new_sel[i] = 1 - new_sel[i]
                result.append(new_sel)
            return result

        start_time = time.perf_counter()

        result = anneal(
            initial=initial,
            objective_fn=objective,
            neighbors=neighbors,
            minimize=True,
            temperature=self.temperature,
            cooling=self.cooling,
            max_iter=self.max_iter,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            solution=result.solution,
            objective=-result.objective,  # Convert back to positive value
            status=Status.OPTIMAL if result.objective < float("inf") else Status.FEASIBLE,
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_anneal", "problem_type": "knapsack"},
        )


class SolvorTabuSolver(BaseSolver):
    """Tabu search solver via solvOR.

    Uses short-term memory to avoid cycling and explore diverse solutions.
    Good for scheduling and assignment problems.
    """

    name = "solvor_tabu"
    supported_problem_types = [ProblemType.COMBINATORIAL]

    def __init__(
        self,
        cooldown: int = 10,
        max_iter: int = 1000,
        max_no_improve: int = 100,
        seed: int | None = None,
    ):
        self.cooldown = cooldown
        self.max_iter = max_iter
        self.max_no_improve = max_no_improve
        self.seed = seed

    def solve(self, problem_data: ProblemData) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        try:
            from solvor import tabu_search
        except ImportError:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "solvOR not installed"},
            )

        problem_type = problem_data.metadata.get("problem_type", "")

        if problem_type == "tsp":
            return self._solve_tsp(problem_data, tabu_search)
        elif problem_type == "assignment":
            return self._solve_assignment(problem_data, tabu_search)
        else:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": f"Unsupported problem type: {problem_type}"},
            )

    def _solve_tsp(self, problem_data: ProblemData, tabu_search: Any) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        distance_matrix = problem_data.metadata.get("distance_matrix")
        n_cities = problem_data.metadata.get("n_cities")

        if distance_matrix is None or n_cities is None:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "Missing distance_matrix or n_cities"},
            )

        if self.seed is not None:
            random.seed(self.seed)

        initial = list(range(n_cities))
        random.shuffle(initial)

        def objective(tour: list[int]) -> float:
            total = 0.0
            for i in range(len(tour)):
                total += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
            return total

        def neighbors(tour: list[int]) -> list[list[int]]:
            result = []
            n = len(tour)
            for i in range(n):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    result.append(new_tour)
            return result

        start_time = time.perf_counter()

        result = tabu_search(
            initial=initial,
            objective_fn=objective,
            neighbors=neighbors,
            minimize=True,
            cooldown=self.cooldown,
            max_iter=self.max_iter,
            max_no_improve=self.max_no_improve,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            solution=result.solution,
            objective=result.objective,
            status=Status.OPTIMAL if result.objective < float("inf") else Status.FEASIBLE,
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_tabu", "problem_type": "tsp"},
        )

    def _solve_assignment(self, problem_data: ProblemData, tabu_search: Any) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        cost_matrix = problem_data.metadata.get("cost_matrix")
        n_agents = problem_data.metadata.get("n_agents")

        if cost_matrix is None or n_agents is None:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "Missing cost_matrix or n_agents"},
            )

        if self.seed is not None:
            random.seed(self.seed)

        # Initial: random assignment (permutation)
        initial = list(range(n_agents))
        random.shuffle(initial)

        def objective(assignment: list[int]) -> float:
            return sum(cost_matrix[i][assignment[i]] for i in range(n_agents))

        # Neighbors: swap two assignments
        def neighbors(assignment: list[int]) -> list[list[int]]:
            result = []
            n = len(assignment)
            for i in range(n):
                for j in range(i + 1, n):
                    new_assign = assignment.copy()
                    new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
                    result.append(new_assign)
            return result

        start_time = time.perf_counter()

        result = tabu_search(
            initial=initial,
            objective_fn=objective,
            neighbors=neighbors,
            minimize=True,
            cooldown=self.cooldown,
            max_iter=self.max_iter,
            max_no_improve=self.max_no_improve,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            solution=result.solution,
            objective=result.objective,
            status=Status.OPTIMAL if result.objective < float("inf") else Status.FEASIBLE,
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_tabu", "problem_type": "assignment"},
        )


class SolvorGeneticSolver(BaseSolver):
    """Genetic algorithm solver via solvOR.

    Uses evolutionary operators (selection, crossover, mutation) to
    explore the solution space. Good for problems with complex landscapes.
    """

    name = "solvor_genetic"
    supported_problem_types = [ProblemType.COMBINATORIAL]

    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        max_generations: int = 100,
        seed: int | None = None,
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.seed = seed

    def solve(self, problem_data: ProblemData) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        try:
            from solvor import evolve
        except ImportError:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "solvOR not installed"},
            )

        problem_type = problem_data.metadata.get("problem_type", "")

        if problem_type == "tsp":
            return self._solve_tsp(problem_data, evolve)
        elif problem_type == "knapsack":
            return self._solve_knapsack(problem_data, evolve)
        else:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": f"Unsupported problem type: {problem_type}"},
            )

    def _solve_tsp(self, problem_data: ProblemData, evolve: Any) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        distance_matrix = problem_data.metadata.get("distance_matrix")
        n_cities = problem_data.metadata.get("n_cities")

        if distance_matrix is None or n_cities is None:
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "Missing distance_matrix or n_cities"},
            )

        if self.seed is not None:
            random.seed(self.seed)

        # Generate initial population
        population = []
        for _ in range(self.population_size):
            tour = list(range(n_cities))
            random.shuffle(tour)
            population.append(tour)

        def objective(tour: list[int]) -> float:
            total = 0.0
            for i in range(len(tour)):
                total += distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
            return total

        # Order crossover (OX)
        def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
            n = len(parent1)
            start, end = sorted(random.sample(range(n), 2))
            child = [-1] * n
            child[start:end] = parent1[start:end]
            remaining = [x for x in parent2 if x not in child]
            idx = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = remaining[idx]
                    idx += 1
            return child

        # Swap mutation
        def mutate(tour: list[int]) -> list[int]:
            new_tour = tour.copy()
            i, j = random.sample(range(len(tour)), 2)
            new_tour[i], new_tour[j] = new_tour[j], new_tour[i]
            return new_tour

        start_time = time.perf_counter()

        result = evolve(
            objective_fn=objective,
            population=population,
            crossover=crossover,
            mutate=mutate,
            minimize=True,
            elite_size=self.elite_size,
            mutation_rate=self.mutation_rate,
            max_generations=self.max_generations,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            solution=result.solution,
            objective=result.objective,
            status=Status.OPTIMAL if result.objective < float("inf") else Status.FEASIBLE,
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_genetic", "problem_type": "tsp"},
        )

    def _solve_knapsack(self, problem_data: ProblemData, evolve: Any) -> SolverResult[Any]:
        from benchmork.types import SolverResult, Status

        weights = problem_data.metadata.get("weights")
        values = problem_data.metadata.get("values")
        capacity = problem_data.metadata.get("capacity")
        n_items = problem_data.metadata.get("n_items")

        if any(x is None for x in [weights, values, capacity, n_items]):
            return SolverResult(
                solution=None,
                objective=float("inf"),
                status=Status.ERROR,
                solve_time=0.0,
                iterations=0,
                metadata={"error": "Missing knapsack parameters"},
            )

        if self.seed is not None:
            random.seed(self.seed)

        # Generate initial population
        population = []
        for _ in range(self.population_size):
            selection = [random.randint(0, 1) for _ in range(n_items)]
            population.append(selection)

        def objective(selection: list[int]) -> float:
            total_weight = sum(w * s for w, s in zip(weights, selection))
            if total_weight > capacity:
                return float("inf")
            return -sum(v * s for v, s in zip(values, selection))

        # Single-point crossover
        def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
            point = random.randint(1, len(parent1) - 1)
            return parent1[:point] + parent2[point:]

        # Bit-flip mutation
        def mutate(selection: list[int]) -> list[int]:
            new_sel = selection.copy()
            i = random.randint(0, len(selection) - 1)
            new_sel[i] = 1 - new_sel[i]
            return new_sel

        start_time = time.perf_counter()

        result = evolve(
            objective_fn=objective,
            population=population,
            crossover=crossover,
            mutate=mutate,
            minimize=True,
            elite_size=self.elite_size,
            mutation_rate=self.mutation_rate,
            max_generations=self.max_generations,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            solution=result.solution,
            objective=-result.objective,  # Convert back to positive
            status=Status.OPTIMAL if result.objective < float("inf") else Status.FEASIBLE,
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_genetic", "problem_type": "knapsack"},
        )
