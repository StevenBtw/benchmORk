"""Google OR-Tools optimization solver adapters."""

from time import perf_counter
from typing import Any

from solvers.base import BaseSolver, SolverResult, SolverStatus
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class OrtoolsGlopSolver(BaseSolver):
    """OR-Tools GLOP linear programming solver adapter.

    GLOP is Google's open-source LP solver, optimized for large sparse problems.
    """

    @property
    def name(self) -> str:
        return "ortools_glop"

    @property
    def display_name(self) -> str:
        return "OR-Tools GLOP"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR}

    def is_available(self) -> bool:
        try:
            from ortools.linear_solver import pywraplp
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            from ortools import __version__
            return __version__
        except (ImportError, AttributeError):
            return None

    def solve(self, problem: BaseProblem) -> SolverResult:
        if not self.supports(problem):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Problem type {problem.problem_type} not supported"},
            )

        start_setup = perf_counter()
        problem_data = problem.generate()
        setup_time = perf_counter() - start_setup

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
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("GLOP")
        if not solver:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "Failed to create GLOP solver"},
            )

        c = problem_data.objective["c"]
        n_vars = len(c)
        minimize = problem_data.objective.get("minimize", True)

        bounds = problem_data.bounds.get("bounds", [(0, float("inf"))] * n_vars)
        variables = []
        for i in range(n_vars):
            lb, ub = bounds[i] if i < len(bounds) else (0, float("inf"))
            lb = lb if lb is not None else -solver.infinity()
            ub = ub if ub is not None else solver.infinity()
            variables.append(solver.NumVar(lb, ub, f"x_{i}"))

        objective = solver.Objective()
        for i, coef in enumerate(c):
            objective.SetCoefficient(variables[i], coef)
        if minimize:
            objective.SetMinimization()
        else:
            objective.SetMaximization()

        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        if A_ub is not None and b_ub is not None:
            for i, (row, rhs) in enumerate(zip(A_ub, b_ub)):
                constraint = solver.Constraint(-solver.infinity(), rhs, f"ub_{i}")
                for j, coef in enumerate(row):
                    constraint.SetCoefficient(variables[j], coef)

        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        if A_eq is not None and b_eq is not None:
            for i, (row, rhs) in enumerate(zip(A_eq, b_eq)):
                constraint = solver.Constraint(rhs, rhs, f"eq_{i}")
                for j, coef in enumerate(row):
                    constraint.SetCoefficient(variables[j], coef)

        if self._time_limit:
            solver.SetTimeLimit(int(self._time_limit * 1000))

        start_solve = perf_counter()
        status_code = solver.Solve()
        solve_time = perf_counter() - start_solve

        status_map = {
            pywraplp.Solver.OPTIMAL: SolverStatus.OPTIMAL,
            pywraplp.Solver.FEASIBLE: SolverStatus.FEASIBLE,
            pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
            pywraplp.Solver.UNBOUNDED: SolverStatus.UNBOUNDED,
            pywraplp.Solver.NOT_SOLVED: SolverStatus.UNKNOWN,
        }
        status = status_map.get(status_code, SolverStatus.ERROR)

        solution = {}
        obj_value = None
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {f"x_{i}": var.solution_value() for i, var in enumerate(variables)}
            obj_value = objective.Value()

        return SolverResult(
            status=status,
            objective_value=obj_value,
            solution=solution,
            solve_time=solve_time,
            iterations=solver.iterations() if hasattr(solver, "iterations") else None,
            metadata={"method": "GLOP", "wall_time": solver.wall_time() / 1000},
        )


class OrtoolsCpsatSolver(BaseSolver):
    """OR-Tools CP-SAT constraint programming solver adapter.

    CP-SAT is a powerful constraint programming solver that handles integer
    variables and complex constraints efficiently.
    """

    @property
    def name(self) -> str:
        return "ortools_cpsat"

    @property
    def display_name(self) -> str:
        return "OR-Tools CP-SAT"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.INTEGER, ProblemType.CONSTRAINT_SATISFACTION}

    def is_available(self) -> bool:
        try:
            from ortools.sat.python import cp_model
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            from ortools import __version__
            return __version__
        except (ImportError, AttributeError):
            return None

    def solve(self, problem: BaseProblem) -> SolverResult:
        if not self.supports(problem):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Problem type {problem.problem_type} not supported"},
            )

        start_setup = perf_counter()
        problem_data = problem.generate()
        setup_time = perf_counter() - start_setup

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
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()

        c = problem_data.objective.get("c", [])
        n_vars = len(c) if c else problem_data.metadata.get("n_variables", 0)
        minimize = problem_data.objective.get("minimize", True)

        bounds = problem_data.bounds.get("bounds", [])
        int_min = cp_model.INT_MIN // 2
        int_max = cp_model.INT_MAX // 2

        variables = []
        for i in range(n_vars):
            if i < len(bounds):
                lb, ub = bounds[i]
                lb = int(lb) if lb is not None and lb > int_min else int_min
                ub = int(ub) if ub is not None and ub < int_max else int_max
            else:
                lb, ub = 0, int_max
            variables.append(model.NewIntVar(lb, ub, f"x_{i}"))

        if c:
            if minimize:
                model.Minimize(sum(int(c[i]) * variables[i] for i in range(n_vars)))
            else:
                model.Maximize(sum(int(c[i]) * variables[i] for i in range(n_vars)))

        A_ub = problem_data.constraints.get("A_ub") or []
        b_ub = problem_data.constraints.get("b_ub") or []
        for row, rhs in zip(A_ub, b_ub):
            model.Add(sum(int(row[j]) * variables[j] for j in range(len(row))) <= int(rhs))

        A_eq = problem_data.constraints.get("A_eq") or []
        b_eq = problem_data.constraints.get("b_eq") or []
        for row, rhs in zip(A_eq, b_eq):
            model.Add(sum(int(row[j]) * variables[j] for j in range(len(row))) == int(rhs))

        solver = cp_model.CpSolver()
        if self._time_limit:
            solver.parameters.max_time_in_seconds = self._time_limit
        if self._threads:
            solver.parameters.num_search_workers = self._threads

        start_solve = perf_counter()
        status_code = solver.Solve(model)
        solve_time = perf_counter() - start_solve

        status_map = {
            cp_model.OPTIMAL: SolverStatus.OPTIMAL,
            cp_model.FEASIBLE: SolverStatus.FEASIBLE,
            cp_model.INFEASIBLE: SolverStatus.INFEASIBLE,
            cp_model.MODEL_INVALID: SolverStatus.ERROR,
            cp_model.UNKNOWN: SolverStatus.UNKNOWN,
        }
        status = status_map.get(status_code, SolverStatus.ERROR)

        solution = {}
        obj_value = None
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {f"x_{i}": solver.Value(var) for i, var in enumerate(variables)}
            obj_value = solver.ObjectiveValue() if c else None

        return SolverResult(
            status=status,
            objective_value=obj_value,
            solution=solution,
            solve_time=solve_time,
            iterations=None,
            nodes=solver.NumBranches() if hasattr(solver, "NumBranches") else None,
            gap=solver.BestObjectiveBound() if status == SolverStatus.FEASIBLE else None,
            metadata={
                "method": "CP-SAT",
                "wall_time": solver.WallTime(),
                "conflicts": solver.NumConflicts(),
            },
        )


class OrtoolsScipSolver(BaseSolver):
    """OR-Tools SCIP mixed-integer programming solver adapter.

    SCIP is one of the fastest non-commercial MIP solvers available.
    """

    @property
    def name(self) -> str:
        return "ortools_scip"

    @property
    def display_name(self) -> str:
        return "OR-Tools SCIP"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            from ortools.linear_solver import pywraplp
            solver = pywraplp.Solver.CreateSolver("SCIP")
            return solver is not None
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            from ortools import __version__
            return __version__
        except (ImportError, AttributeError):
            return None

    def solve(self, problem: BaseProblem) -> SolverResult:
        if not self.supports(problem):
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": f"Problem type {problem.problem_type} not supported"},
            )

        start_setup = perf_counter()
        problem_data = problem.generate()
        setup_time = perf_counter() - start_setup

        result = self._solve_impl(problem_data, problem.problem_type)

        return SolverResult(
            status=result.status,
            objective_value=result.objective_value,
            solution=result.solution,
            solve_time=result.solve_time,
            setup_time=setup_time,
            iterations=result.iterations,
            nodes=result.nodes,
            gap=result.gap,
            metadata=result.metadata,
        )

    def _solve_impl(
        self, problem_data: ProblemData, problem_type: ProblemType
    ) -> SolverResult:
        from ortools.linear_solver import pywraplp

        solver = pywraplp.Solver.CreateSolver("SCIP")
        if not solver:
            return SolverResult(
                status=SolverStatus.ERROR,
                metadata={"error": "Failed to create SCIP solver"},
            )

        c = problem_data.objective["c"]
        n_vars = len(c)
        minimize = problem_data.objective.get("minimize", True)
        is_integer = problem_type == ProblemType.INTEGER
        integers = problem_data.metadata.get("integers", list(range(n_vars)) if is_integer else [])

        bounds = problem_data.bounds.get("bounds", [(0, float("inf"))] * n_vars)
        variables = []
        for i in range(n_vars):
            lb, ub = bounds[i] if i < len(bounds) else (0, float("inf"))
            lb = lb if lb is not None else -solver.infinity()
            ub = ub if ub is not None else solver.infinity()
            if i in integers:
                variables.append(solver.IntVar(int(lb), int(ub), f"x_{i}"))
            else:
                variables.append(solver.NumVar(lb, ub, f"x_{i}"))

        objective = solver.Objective()
        for i, coef in enumerate(c):
            objective.SetCoefficient(variables[i], coef)
        if minimize:
            objective.SetMinimization()
        else:
            objective.SetMaximization()

        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        if A_ub is not None and b_ub is not None:
            for i, (row, rhs) in enumerate(zip(A_ub, b_ub)):
                constraint = solver.Constraint(-solver.infinity(), rhs, f"ub_{i}")
                for j, coef in enumerate(row):
                    constraint.SetCoefficient(variables[j], coef)

        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        if A_eq is not None and b_eq is not None:
            for i, (row, rhs) in enumerate(zip(A_eq, b_eq)):
                constraint = solver.Constraint(rhs, rhs, f"eq_{i}")
                for j, coef in enumerate(row):
                    constraint.SetCoefficient(variables[j], coef)

        if self._time_limit:
            solver.SetTimeLimit(int(self._time_limit * 1000))

        start_solve = perf_counter()
        status_code = solver.Solve()
        solve_time = perf_counter() - start_solve

        status_map = {
            pywraplp.Solver.OPTIMAL: SolverStatus.OPTIMAL,
            pywraplp.Solver.FEASIBLE: SolverStatus.FEASIBLE,
            pywraplp.Solver.INFEASIBLE: SolverStatus.INFEASIBLE,
            pywraplp.Solver.UNBOUNDED: SolverStatus.UNBOUNDED,
            pywraplp.Solver.NOT_SOLVED: SolverStatus.UNKNOWN,
        }
        status = status_map.get(status_code, SolverStatus.ERROR)

        solution = {}
        obj_value = None
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {f"x_{i}": var.solution_value() for i, var in enumerate(variables)}
            obj_value = objective.Value()

        return SolverResult(
            status=status,
            objective_value=obj_value,
            solution=solution,
            solve_time=solve_time,
            nodes=solver.nodes() if hasattr(solver, "nodes") else None,
            metadata={"method": "SCIP", "wall_time": solver.wall_time() / 1000},
        )


# Legacy aliases
OrtoolsSolver = OrtoolsGlopSolver
