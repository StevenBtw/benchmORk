"""Pyomo optimization solver adapters."""

from time import perf_counter
from typing import Any

from solvers.base import BaseSolver, SolverResult, SolverStatus
from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


class PyomoGlpkSolver(BaseSolver):
    """Pyomo adapter for GLPK (GNU Linear Programming Kit).

    GLPK is a free, open-source LP/MIP solver.
    Requires GLPK to be installed on the system.
    """

    @property
    def name(self) -> str:
        return "pyomo_glpk"

    @property
    def display_name(self) -> str:
        return "Pyomo GLPK"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            import pyomo.environ as pyo
            from pyomo.opt import SolverFactory
            solver = SolverFactory("glpk")
            return solver.available()
        except (ImportError, Exception):
            return False

    def get_version(self) -> str | None:
        try:
            import pyomo
            return pyomo.__version__
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
            gap=result.gap,
            metadata=result.metadata,
        )

    def _solve_impl(
        self, problem_data: ProblemData, problem_type: ProblemType
    ) -> SolverResult:
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory, TerminationCondition

        model = pyo.ConcreteModel()

        c = problem_data.objective["c"]
        n_vars = len(c)
        minimize = problem_data.objective.get("minimize", True)
        is_integer = problem_type == ProblemType.INTEGER
        integers = set(problem_data.metadata.get("integers", range(n_vars) if is_integer else []))

        bounds = problem_data.bounds.get("bounds", [(0, None)] * n_vars)

        def var_bounds(m, i):
            if i < len(bounds):
                return bounds[i]
            return (0, None)

        if is_integer and len(integers) == n_vars:
            model.x = pyo.Var(range(n_vars), domain=pyo.NonNegativeIntegers, bounds=var_bounds)
        elif integers:
            model.x = pyo.Var(range(n_vars), bounds=var_bounds)
            for i in integers:
                model.x[i].domain = pyo.Integers
        else:
            model.x = pyo.Var(range(n_vars), domain=pyo.NonNegativeReals, bounds=var_bounds)

        def objective_rule(m):
            return sum(c[i] * m.x[i] for i in range(n_vars))

        if minimize:
            model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        else:
            model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        model.ub_constraints = pyo.ConstraintList()
        if A_ub is not None and b_ub is not None:
            for row, rhs in zip(A_ub, b_ub):
                model.ub_constraints.add(sum(row[j] * model.x[j] for j in range(len(row))) <= rhs)

        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        model.eq_constraints = pyo.ConstraintList()
        if A_eq is not None and b_eq is not None:
            for row, rhs in zip(A_eq, b_eq):
                model.eq_constraints.add(sum(row[j] * model.x[j] for j in range(len(row))) == rhs)

        solver = SolverFactory("glpk")
        options: dict[str, Any] = {}
        if self._time_limit:
            options["tmlim"] = int(self._time_limit)

        start_solve = perf_counter()
        try:
            results = solver.solve(model, options=options, tee=False)
            solve_time = perf_counter() - start_solve
        except Exception as e:
            return SolverResult(
                status=SolverStatus.ERROR,
                solve_time=perf_counter() - start_solve,
                metadata={"error": str(e), "method": "GLPK"},
            )

        tc = results.solver.termination_condition
        status_map = {
            TerminationCondition.optimal: SolverStatus.OPTIMAL,
            TerminationCondition.feasible: SolverStatus.FEASIBLE,
            TerminationCondition.infeasible: SolverStatus.INFEASIBLE,
            TerminationCondition.unbounded: SolverStatus.UNBOUNDED,
            TerminationCondition.maxTimeLimit: SolverStatus.TIME_LIMIT,
            TerminationCondition.maxIterations: SolverStatus.TIME_LIMIT,
        }
        status = status_map.get(tc, SolverStatus.UNKNOWN)

        solution = {}
        obj_value = None
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {f"x_{i}": pyo.value(model.x[i]) for i in range(n_vars)}
            obj_value = pyo.value(model.obj)

        return SolverResult(
            status=status,
            objective_value=obj_value,
            solution=solution,
            solve_time=solve_time,
            metadata={"method": "GLPK", "termination": str(tc)},
        )


class PyomoCbcSolver(BaseSolver):
    """Pyomo adapter for COIN-OR CBC (Branch and Cut).

    CBC is a powerful open-source MIP solver.
    Requires CBC to be installed on the system.
    """

    @property
    def name(self) -> str:
        return "pyomo_cbc"

    @property
    def display_name(self) -> str:
        return "Pyomo CBC"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR, ProblemType.INTEGER}

    def is_available(self) -> bool:
        try:
            import pyomo.environ as pyo
            from pyomo.opt import SolverFactory
            solver = SolverFactory("cbc")
            return solver.available()
        except (ImportError, Exception):
            return False

    def get_version(self) -> str | None:
        try:
            import pyomo
            return pyomo.__version__
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
            gap=result.gap,
            metadata=result.metadata,
        )

    def _solve_impl(
        self, problem_data: ProblemData, problem_type: ProblemType
    ) -> SolverResult:
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory, TerminationCondition

        model = pyo.ConcreteModel()

        c = problem_data.objective["c"]
        n_vars = len(c)
        minimize = problem_data.objective.get("minimize", True)
        is_integer = problem_type == ProblemType.INTEGER
        integers = set(problem_data.metadata.get("integers", range(n_vars) if is_integer else []))

        bounds = problem_data.bounds.get("bounds", [(0, None)] * n_vars)

        def var_bounds(m, i):
            if i < len(bounds):
                return bounds[i]
            return (0, None)

        if is_integer and len(integers) == n_vars:
            model.x = pyo.Var(range(n_vars), domain=pyo.NonNegativeIntegers, bounds=var_bounds)
        elif integers:
            model.x = pyo.Var(range(n_vars), bounds=var_bounds)
            for i in integers:
                model.x[i].domain = pyo.Integers
        else:
            model.x = pyo.Var(range(n_vars), domain=pyo.NonNegativeReals, bounds=var_bounds)

        def objective_rule(m):
            return sum(c[i] * m.x[i] for i in range(n_vars))

        if minimize:
            model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        else:
            model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        model.ub_constraints = pyo.ConstraintList()
        if A_ub is not None and b_ub is not None:
            for row, rhs in zip(A_ub, b_ub):
                model.ub_constraints.add(sum(row[j] * model.x[j] for j in range(len(row))) <= rhs)

        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        model.eq_constraints = pyo.ConstraintList()
        if A_eq is not None and b_eq is not None:
            for row, rhs in zip(A_eq, b_eq):
                model.eq_constraints.add(sum(row[j] * model.x[j] for j in range(len(row))) == rhs)

        solver = SolverFactory("cbc")
        options: dict[str, Any] = {}
        if self._time_limit:
            options["sec"] = int(self._time_limit)
        if self._threads:
            options["threads"] = self._threads

        start_solve = perf_counter()
        try:
            results = solver.solve(model, options=options, tee=False)
            solve_time = perf_counter() - start_solve
        except Exception as e:
            return SolverResult(
                status=SolverStatus.ERROR,
                solve_time=perf_counter() - start_solve,
                metadata={"error": str(e), "method": "CBC"},
            )

        tc = results.solver.termination_condition
        status_map = {
            TerminationCondition.optimal: SolverStatus.OPTIMAL,
            TerminationCondition.feasible: SolverStatus.FEASIBLE,
            TerminationCondition.infeasible: SolverStatus.INFEASIBLE,
            TerminationCondition.unbounded: SolverStatus.UNBOUNDED,
            TerminationCondition.maxTimeLimit: SolverStatus.TIME_LIMIT,
            TerminationCondition.maxIterations: SolverStatus.TIME_LIMIT,
        }
        status = status_map.get(tc, SolverStatus.UNKNOWN)

        solution = {}
        obj_value = None
        gap = None
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {f"x_{i}": pyo.value(model.x[i]) for i in range(n_vars)}
            obj_value = pyo.value(model.obj)
            if hasattr(results.problem, "lower_bound") and hasattr(results.problem, "upper_bound"):
                lb = results.problem.lower_bound
                ub = results.problem.upper_bound
                if lb is not None and ub is not None and ub != 0:
                    gap = abs(ub - lb) / abs(ub)

        return SolverResult(
            status=status,
            objective_value=obj_value,
            solution=solution,
            solve_time=solve_time,
            gap=gap,
            metadata={"method": "CBC", "termination": str(tc)},
        )


class PyomoIpoptSolver(BaseSolver):
    """Pyomo adapter for IPOPT (Interior Point Optimizer).

    IPOPT is a powerful open-source solver for large-scale nonlinear optimization.
    Requires IPOPT to be installed on the system.
    """

    @property
    def name(self) -> str:
        return "pyomo_ipopt"

    @property
    def display_name(self) -> str:
        return "Pyomo IPOPT"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.LINEAR, ProblemType.NONLINEAR}

    def is_available(self) -> bool:
        try:
            import pyomo.environ as pyo
            from pyomo.opt import SolverFactory
            solver = SolverFactory("ipopt")
            return solver.available()
        except (ImportError, Exception):
            return False

    def get_version(self) -> str | None:
        try:
            import pyomo
            return pyomo.__version__
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
        import pyomo.environ as pyo
        from pyomo.opt import SolverFactory, TerminationCondition

        model = pyo.ConcreteModel()

        c = problem_data.objective["c"]
        n_vars = len(c)
        minimize = problem_data.objective.get("minimize", True)

        bounds = problem_data.bounds.get("bounds", [(0, None)] * n_vars)

        def var_bounds(m, i):
            if i < len(bounds):
                return bounds[i]
            return (0, None)

        model.x = pyo.Var(range(n_vars), domain=pyo.NonNegativeReals, bounds=var_bounds)

        def objective_rule(m):
            return sum(c[i] * m.x[i] for i in range(n_vars))

        if minimize:
            model.obj = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        else:
            model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        A_ub = problem_data.constraints.get("A_ub")
        b_ub = problem_data.constraints.get("b_ub")
        model.ub_constraints = pyo.ConstraintList()
        if A_ub is not None and b_ub is not None:
            for row, rhs in zip(A_ub, b_ub):
                model.ub_constraints.add(sum(row[j] * model.x[j] for j in range(len(row))) <= rhs)

        A_eq = problem_data.constraints.get("A_eq")
        b_eq = problem_data.constraints.get("b_eq")
        model.eq_constraints = pyo.ConstraintList()
        if A_eq is not None and b_eq is not None:
            for row, rhs in zip(A_eq, b_eq):
                model.eq_constraints.add(sum(row[j] * model.x[j] for j in range(len(row))) == rhs)

        solver = SolverFactory("ipopt")
        options: dict[str, Any] = {"print_level": 0}
        if self._time_limit:
            options["max_cpu_time"] = self._time_limit

        start_solve = perf_counter()
        try:
            results = solver.solve(model, options=options, tee=False)
            solve_time = perf_counter() - start_solve
        except Exception as e:
            return SolverResult(
                status=SolverStatus.ERROR,
                solve_time=perf_counter() - start_solve,
                metadata={"error": str(e), "method": "IPOPT"},
            )

        tc = results.solver.termination_condition
        status_map = {
            TerminationCondition.optimal: SolverStatus.OPTIMAL,
            TerminationCondition.feasible: SolverStatus.FEASIBLE,
            TerminationCondition.locallyOptimal: SolverStatus.OPTIMAL,
            TerminationCondition.infeasible: SolverStatus.INFEASIBLE,
            TerminationCondition.unbounded: SolverStatus.UNBOUNDED,
            TerminationCondition.maxTimeLimit: SolverStatus.TIME_LIMIT,
            TerminationCondition.maxIterations: SolverStatus.TIME_LIMIT,
        }
        status = status_map.get(tc, SolverStatus.UNKNOWN)

        solution = {}
        obj_value = None
        iterations = None
        if status in (SolverStatus.OPTIMAL, SolverStatus.FEASIBLE):
            solution = {f"x_{i}": pyo.value(model.x[i]) for i in range(n_vars)}
            obj_value = pyo.value(model.obj)

        return SolverResult(
            status=status,
            objective_value=obj_value,
            solution=solution,
            solve_time=solve_time,
            iterations=iterations,
            metadata={"method": "IPOPT", "termination": str(tc)},
        )


# Legacy aliases
PyomoSolver = PyomoCbcSolver
