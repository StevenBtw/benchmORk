"""PuLP optimization solver adapters."""

from time import perf_counter
from solvers.base import BaseSolver, SolverResult, SolverStatus
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class PulpCbcSolver(BaseSolver):
    """PuLP CBC (COIN-OR Branch and Cut) solver adapter."""

    @property
    def name(self) -> str:
        return "pulp_cbc"

    @property
    def display_name(self) -> str:
        return "PuLP CBC"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            import pulp
            return True
        except ImportError:
            return False

    def get_version(self) -> str | None:
        try:
            import pulp
            return pulp.__version__
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
        import pulp

        c = problem_data.objective["c"]
        minimize = problem_data.objective.get("minimize", True)
        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        bounds = problem_data.bounds.get("bounds", [])

        n_vars = len(c)

        sense = pulp.LpMinimize if minimize else pulp.LpMaximize
        prob = pulp.LpProblem("benchmark", sense)

        variables = []
        for i in range(n_vars):
            lb, ub = bounds[i] if i < len(bounds) else (0, None)
            var = pulp.LpVariable(f"x_{i}", lowBound=lb, upBound=ub)
            variables.append(var)

        prob += pulp.lpSum(c[i] * variables[i] for i in range(n_vars))

        if A_ub is not None and b_ub is not None:
            for j, row in enumerate(A_ub):
                prob += pulp.lpSum(row[i] * variables[i] for i in range(n_vars)) <= b_ub[j]

        if A_eq is not None and b_eq is not None:
            for j, row in enumerate(A_eq):
                prob += pulp.lpSum(row[i] * variables[i] for i in range(n_vars)) == b_eq[j]

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=self._time_limit)

        start_solve = perf_counter()
        prob.solve(solver)
        solve_time = perf_counter() - start_solve

        status_map = {
            pulp.LpStatusOptimal: SolverStatus.OPTIMAL,
            pulp.LpStatusNotSolved: SolverStatus.UNKNOWN,
            pulp.LpStatusInfeasible: SolverStatus.INFEASIBLE,
            pulp.LpStatusUnbounded: SolverStatus.UNBOUNDED,
            pulp.LpStatusUndefined: SolverStatus.ERROR,
        }
        status = status_map.get(prob.status, SolverStatus.ERROR)

        solution = {}
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {var.name: var.varValue for var in variables if var.varValue is not None}

        objective_value = None
        if prob.status == pulp.LpStatusOptimal:
            objective_value = pulp.value(prob.objective)

        return SolverResult(
            status=status,
            objective_value=objective_value,
            solution=solution,
            solve_time=solve_time,
            metadata={"pulp_status": pulp.LpStatus[prob.status], "solver": "CBC"},
        )


class PulpGlpkSolver(BaseSolver):
    """PuLP GLPK solver adapter."""

    @property
    def name(self) -> str:
        return "pulp_glpk"

    @property
    def display_name(self) -> str:
        return "PuLP GLPK"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            import pulp
            # Check if GLPK is actually available
            solver = pulp.GLPK_CMD(msg=0)
            return solver.available()
        except Exception:
            return False

    def get_version(self) -> str | None:
        try:
            import pulp
            return pulp.__version__
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
        import pulp

        c = problem_data.objective["c"]
        minimize = problem_data.objective.get("minimize", True)
        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        bounds = problem_data.bounds.get("bounds", [])

        n_vars = len(c)

        sense = pulp.LpMinimize if minimize else pulp.LpMaximize
        prob = pulp.LpProblem("benchmark", sense)

        variables = []
        for i in range(n_vars):
            lb, ub = bounds[i] if i < len(bounds) else (0, None)
            var = pulp.LpVariable(f"x_{i}", lowBound=lb, upBound=ub)
            variables.append(var)

        prob += pulp.lpSum(c[i] * variables[i] for i in range(n_vars))

        if A_ub is not None and b_ub is not None:
            for j, row in enumerate(A_ub):
                prob += pulp.lpSum(row[i] * variables[i] for i in range(n_vars)) <= b_ub[j]

        if A_eq is not None and b_eq is not None:
            for j, row in enumerate(A_eq):
                prob += pulp.lpSum(row[i] * variables[i] for i in range(n_vars)) == b_eq[j]

        solver = pulp.GLPK_CMD(msg=0, timeLimit=self._time_limit)

        start_solve = perf_counter()
        prob.solve(solver)
        solve_time = perf_counter() - start_solve

        status_map = {
            pulp.LpStatusOptimal: SolverStatus.OPTIMAL,
            pulp.LpStatusNotSolved: SolverStatus.UNKNOWN,
            pulp.LpStatusInfeasible: SolverStatus.INFEASIBLE,
            pulp.LpStatusUnbounded: SolverStatus.UNBOUNDED,
            pulp.LpStatusUndefined: SolverStatus.ERROR,
        }
        status = status_map.get(prob.status, SolverStatus.ERROR)

        solution = {}
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {var.name: var.varValue for var in variables if var.varValue is not None}

        objective_value = None
        if prob.status == pulp.LpStatusOptimal:
            objective_value = pulp.value(prob.objective)

        return SolverResult(
            status=status,
            objective_value=objective_value,
            solution=solution,
            solve_time=solve_time,
            metadata={"pulp_status": pulp.LpStatus[prob.status], "solver": "GLPK"},
        )


# Legacy alias for backwards compatibility
PulpSolver = PulpCbcSolver
