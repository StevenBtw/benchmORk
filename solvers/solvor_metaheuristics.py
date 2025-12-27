"""solvOR metaheuristic solver adapters for benchmORk.

Wraps solvOR's metaheuristic solvers (anneal, tabu_search, evolve) for
benchmarking against exact optimization methods.
"""

from __future__ import annotations

import random
import time
from typing import Any

from solvers.base import BaseSolver, SolverResult, SolverStatus
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class SolvorAnnealSolver(BaseSolver):
    """Simulated annealing solver via solvOR.

    Best suited for combinatorial optimization problems like TSP,
    where the solution space can be explored through local moves.
    """

    def __init__(
        self,
        temperature: float = 1000.0,
        cooling: float = 0.9995,
        max_iter: int = 100_000,
        seed: int | None = None,
        **options: Any,
    ):
        super().__init__(**options)
        self.temperature = temperature
        self.cooling = cooling
        self.max_iter = max_iter
        self.seed = seed

    @property
    def name(self) -> str:
        return "solvor_anneal"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.COMBINATORIAL, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            from solvor import anneal
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import solvor
            return getattr(solvor, "__version__", "unknown")
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
            from solvor import anneal
        except ImportError:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "solvOR not installed"},
            )

        # Check problem_class (used by combinatorial problems) or problem_type
        problem_class = problem_data.metadata.get("problem_class", "")

        if problem_class == "tsp":
            return self._solve_tsp(problem_data, anneal)
        elif problem_class == "knapsack":
            return self._solve_knapsack(problem_data, anneal)
        else:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Unsupported problem class: {problem_class}"},
            )

    def _solve_tsp(self, problem_data: ProblemData, anneal: Any) -> SolverResult:
        distance_matrix = problem_data.metadata.get("distance_matrix")
        n_cities = problem_data.metadata.get("n_cities")

        if distance_matrix is None or n_cities is None:
            return SolverResult(
                status=SolverStatus.ERROR,
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

        # Neighbor: single 2-opt swap (anneal expects one neighbor, not a list)
        def neighbor(tour: list[int]) -> list[int]:
            n = len(tour)
            # Pick random i, j for 2-opt
            i = random.randint(0, n - 2)
            j = random.randint(i + 2, n - 1) if i + 2 < n else i + 2
            if j >= n:
                j = n - 1
            return tour[:i] + tour[i:j+1][::-1] + tour[j+1:]

        start_time = time.perf_counter()

        result = anneal(
            initial=initial,
            objective_fn=objective,
            neighbors=neighbor,
            minimize=True,
            temperature=self.temperature,
            cooling=self.cooling,
            max_iter=self.max_iter,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            status=SolverStatus.OPTIMAL if result.objective < float("inf") else SolverStatus.FEASIBLE,
            objective_value=result.objective,
            solution={"tour": result.solution},
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_anneal", "problem_type": "tsp"},
        )

    def _solve_knapsack(self, problem_data: ProblemData, anneal: Any) -> SolverResult:
        weights = problem_data.metadata.get("weights")
        values = problem_data.metadata.get("values")
        capacity = problem_data.metadata.get("capacity")

        if any(x is None for x in [weights, values, capacity]):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "Missing knapsack parameters in metadata"},
            )

        n_items = len(weights)

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

        # Neighbor: flip one random item (anneal expects one neighbor)
        def neighbor(selection: list[int]) -> list[int]:
            new_sel = selection.copy()
            i = random.randint(0, len(selection) - 1)
            new_sel[i] = 1 - new_sel[i]
            return new_sel

        start_time = time.perf_counter()

        result = anneal(
            initial=initial,
            objective_fn=objective,
            neighbors=neighbor,
            minimize=True,
            temperature=self.temperature,
            cooling=self.cooling,
            max_iter=self.max_iter,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            status=SolverStatus.OPTIMAL if result.objective < float("inf") else SolverStatus.FEASIBLE,
            objective_value=-result.objective,  # Convert back to positive value
            solution={"selection": result.solution},
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_anneal", "problem_type": "knapsack"},
        )


class SolvorTabuSolver(BaseSolver):
    """Tabu search solver via solvOR.

    Uses short-term memory to avoid cycling and explore diverse solutions.
    Good for scheduling and assignment problems.
    """

    def __init__(
        self,
        cooldown: int = 10,
        max_iter: int = 1000,
        max_no_improve: int = 100,
        seed: int | None = None,
        **options: Any,
    ):
        super().__init__(**options)
        self.cooldown = cooldown
        self.max_iter = max_iter
        self.max_no_improve = max_no_improve
        self.seed = seed

    @property
    def name(self) -> str:
        return "solvor_tabu"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.COMBINATORIAL, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            from solvor import tabu_search
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import solvor
            return getattr(solvor, "__version__", "unknown")
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
            from solvor import tabu_search
        except ImportError:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "solvOR not installed"},
            )

        problem_class = problem_data.metadata.get("problem_class", "")

        if problem_class == "tsp":
            return self._solve_tsp(problem_data, tabu_search)
        elif problem_class == "assignment":
            return self._solve_assignment(problem_data, tabu_search)
        else:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Unsupported problem class: {problem_class}"},
            )

    def _solve_tsp(self, problem_data: ProblemData, tabu_search: Any) -> SolverResult:
        distance_matrix = problem_data.metadata.get("distance_matrix")
        n_cities = problem_data.metadata.get("n_cities")

        if distance_matrix is None or n_cities is None:
            return SolverResult(
                status=SolverStatus.ERROR,
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

        # Neighbors: return (move, solution) pairs where move is hashable
        def neighbors(tour: list[int]) -> list[tuple[tuple[int, int], list[int]]]:
            result = []
            n = len(tour)
            for i in range(n):
                for j in range(i + 2, n):
                    if j == n - 1 and i == 0:
                        continue
                    new_tour = tour[:i] + tour[i:j+1][::-1] + tour[j+1:]
                    result.append(((i, j), new_tour))  # (move, solution)
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
            status=SolverStatus.OPTIMAL if result.objective < float("inf") else SolverStatus.FEASIBLE,
            objective_value=result.objective,
            solution={"tour": result.solution},
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_tabu", "problem_type": "tsp"},
        )

    def _solve_assignment(self, problem_data: ProblemData, tabu_search: Any) -> SolverResult:
        cost_matrix = problem_data.metadata.get("cost_matrix")
        n_agents = problem_data.metadata.get("n_agents")

        if cost_matrix is None or n_agents is None:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "Missing cost_matrix or n_agents"},
            )

        if self.seed is not None:
            random.seed(self.seed)

        # Initial: random assignment (permutation)
        initial = list(range(n_agents))
        random.shuffle(initial)

        def objective(assignment: list[int]) -> float:
            return sum(cost_matrix[i][assignment[i]] for i in range(n_agents))

        # Neighbors: return (move, solution) pairs where move is hashable
        def neighbors(assignment: list[int]) -> list[tuple[tuple[int, int], list[int]]]:
            result = []
            n = len(assignment)
            for i in range(n):
                for j in range(i + 1, n):
                    new_assign = assignment.copy()
                    new_assign[i], new_assign[j] = new_assign[j], new_assign[i]
                    result.append(((i, j), new_assign))  # (move, solution)
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
            status=SolverStatus.OPTIMAL if result.objective < float("inf") else SolverStatus.FEASIBLE,
            objective_value=result.objective,
            solution={"assignment": result.solution},
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_tabu", "problem_type": "assignment"},
        )


class SolvorGeneticSolver(BaseSolver):
    """Genetic algorithm solver via solvOR.

    Uses evolutionary operators (selection, crossover, mutation) to
    explore the solution space. Good for problems with complex landscapes.
    """

    def __init__(
        self,
        population_size: int = 50,
        elite_size: int = 2,
        mutation_rate: float = 0.1,
        max_generations: int = 100,
        seed: int | None = None,
        **options: Any,
    ):
        super().__init__(**options)
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.seed = seed

    @property
    def name(self) -> str:
        return "solvor_genetic"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.COMBINATORIAL, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            from solvor import evolve
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import solvor
            return getattr(solvor, "__version__", "unknown")
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
            from solvor import evolve
        except ImportError:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "solvOR not installed"},
            )

        problem_class = problem_data.metadata.get("problem_class", "")

        if problem_class == "tsp":
            return self._solve_tsp(problem_data, evolve)
        elif problem_class == "knapsack":
            return self._solve_knapsack(problem_data, evolve)
        else:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Unsupported problem class: {problem_class}"},
            )

    def _solve_tsp(self, problem_data: ProblemData, evolve: Any) -> SolverResult:
        distance_matrix = problem_data.metadata.get("distance_matrix")
        n_cities = problem_data.metadata.get("n_cities")

        if distance_matrix is None or n_cities is None:
            return SolverResult(
                status=SolverStatus.ERROR,
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
            max_gen=self.max_generations,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            status=SolverStatus.OPTIMAL if result.objective < float("inf") else SolverStatus.FEASIBLE,
            objective_value=result.objective,
            solution={"tour": result.solution},
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_genetic", "problem_type": "tsp"},
        )

    def _solve_knapsack(self, problem_data: ProblemData, evolve: Any) -> SolverResult:
        weights = problem_data.metadata.get("weights")
        values = problem_data.metadata.get("values")
        capacity = problem_data.metadata.get("capacity")

        if any(x is None for x in [weights, values, capacity]):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "Missing knapsack parameters"},
            )

        n_items = len(weights)

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
            max_gen=self.max_generations,
        )

        solve_time = time.perf_counter() - start_time

        return SolverResult(
            status=SolverStatus.OPTIMAL if result.objective < float("inf") else SolverStatus.FEASIBLE,
            objective_value=-result.objective,  # Convert back to positive
            solution={"selection": result.solution},
            solve_time=solve_time,
            iterations=result.iterations,
            metadata={"solver": "solvor_genetic", "problem_type": "knapsack"},
        )
