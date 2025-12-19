"""SciPy optimization solver adapters."""

from time import perf_counter
from typing import Any
from solvers.base import BaseSolver, SolverResult, SolverStatus
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class ScipyHighsSolver(BaseSolver):
    """SciPy HiGHS LP solver adapter."""

    @property
    def name(self) -> str:
        return "scipy_highs"

    @property
    def display_name(self) -> str:
        return "SciPy HiGHS"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR}

    def is_available(self) -> bool:
        try:
            from scipy.optimize import linprog
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import scipy
            return scipy.__version__
        except ImportError:
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
        from scipy.optimize import linprog

        c = problem_data.objective["c"]
        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        bounds = problem_data.bounds.get("bounds")

        options: dict[str, Any] = {}
        if self._time_limit:
            options["time_limit"] = self._time_limit

        start_solve = perf_counter()
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
            options=options,
        )
        solve_time = perf_counter() - start_solve

        status_map = {
            0: SolverStatus.OPTIMAL,
            1: SolverStatus.TIME_LIMIT,
            2: SolverStatus.INFEASIBLE,
            3: SolverStatus.UNBOUNDED,
        }
        status = status_map.get(result.status, SolverStatus.ERROR)

        solution = {}
        if result.x is not None:
            solution = {f"x_{i}": float(v) for i, v in enumerate(result.x)}

        return SolverResult(
            status=status,
            objective_value=float(result.fun) if result.success else None,
            solution=solution,
            solve_time=solve_time,
            iterations=result.nit if hasattr(result, "nit") else None,
            metadata={"message": result.message, "method": "highs"},
        )


class ScipySimplexSolver(BaseSolver):
    """SciPy Simplex LP solver adapter."""

    @property
    def name(self) -> str:
        return "scipy_simplex"

    @property
    def display_name(self) -> str:
        return "SciPy Simplex"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR}

    def is_available(self) -> bool:
        try:
            from scipy.optimize import linprog
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import scipy
            return scipy.__version__
        except ImportError:
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
        from scipy.optimize import linprog

        c = problem_data.objective["c"]
        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        bounds = problem_data.bounds.get("bounds")

        options: dict[str, Any] = {}
        if self._time_limit:
            options["time_limit"] = self._time_limit

        start_solve = perf_counter()
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs-ds",  # Dual simplex via HiGHS
            options=options,
        )
        solve_time = perf_counter() - start_solve

        status_map = {
            0: SolverStatus.OPTIMAL,
            1: SolverStatus.TIME_LIMIT,
            2: SolverStatus.INFEASIBLE,
            3: SolverStatus.UNBOUNDED,
        }
        status = status_map.get(result.status, SolverStatus.ERROR)

        solution = {}
        if result.x is not None:
            solution = {f"x_{i}": float(v) for i, v in enumerate(result.x)}

        return SolverResult(
            status=status,
            objective_value=float(result.fun) if result.success else None,
            solution=solution,
            solve_time=solve_time,
            iterations=result.nit if hasattr(result, "nit") else None,
            metadata={"message": result.message, "method": "highs-ds"},
        )


# Legacy alias for backwards compatibility
ScipySolver = ScipyHighsSolver
