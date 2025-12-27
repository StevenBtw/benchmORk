"""solvOR optimization solver adapters."""

from time import perf_counter
from typing import Any

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType
from solvers.base import BaseSolver, SolverResult, SolverStatus


class SolvorCPSATSolver(BaseSolver):
    """solvOR CP-SAT constraint programming solver adapter.

    Uses solvOR's Model class for constraint satisfaction problems
    like school timetabling.
    """

    @property
    def name(self) -> str:
        return "solvor_cpsat"

    @property
    def display_name(self) -> str:
        return "solvOR CP-SAT"

    @property
    def supported_problem_types(self) -> set[ProblemType]:
        return {ProblemType.CONSTRAINT_SATISFACTION}

    def is_available(self) -> bool:
        try:
            from solvor import Model
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
        from solvor import Model

        metadata = problem_data.metadata
        n_lessons = metadata["n_lessons"]
        n_timeslots = metadata["n_timeslots"]
        n_rooms = metadata["n_rooms"]
        lessons = metadata["lessons"]
        teachers = {t["id"]: t for t in metadata["teachers"]}

        m = Model()

        # Create decision variables
        # For each lesson: which timeslot and which room
        timeslot_vars = []
        room_vars = []

        for lesson in lessons:
            ts_var = m.int_var(0, n_timeslots - 1, f"ts_{lesson['id']}")
            rm_var = m.int_var(0, n_rooms - 1, f"rm_{lesson['id']}")
            timeslot_vars.append(ts_var)
            room_vars.append(rm_var)

        # Hard constraints using conflict detection
        # Group lessons by teacher, room, and student group for conflict constraints
        lessons_by_teacher: dict[int, list[int]] = {}
        lessons_by_group: dict[int, list[int]] = {}

        for i, lesson in enumerate(lessons):
            teacher_id = lesson["teacher_id"]
            group_id = lesson["student_group_id"]

            if teacher_id not in lessons_by_teacher:
                lessons_by_teacher[teacher_id] = []
            lessons_by_teacher[teacher_id].append(i)

            if group_id not in lessons_by_group:
                lessons_by_group[group_id] = []
            lessons_by_group[group_id].append(i)

        # Constraint: No teacher teaches two lessons at the same time
        # For lessons taught by same teacher, their timeslots must differ
        for teacher_id, lesson_indices in lessons_by_teacher.items():
            if len(lesson_indices) > 1:
                teacher_timeslots = [timeslot_vars[i] for i in lesson_indices]
                m.add(m.all_different(teacher_timeslots))

        # Constraint: No student group has two lessons at the same time
        for group_id, lesson_indices in lessons_by_group.items():
            if len(lesson_indices) > 1:
                group_timeslots = [timeslot_vars[i] for i in lesson_indices]
                m.add(m.all_different(group_timeslots))

        # Constraint: No room hosts two lessons at same time
        # This is trickier - we need (timeslot, room) pairs to be unique
        # We encode this as: timeslot * n_rooms + room must be all different
        combined_vars = []
        for i in range(n_lessons):
            combined = m.int_var(0, n_timeslots * n_rooms - 1, f"combined_{i}")
            # combined = timeslot * n_rooms + room
            m.add(combined == timeslot_vars[i] * n_rooms + room_vars[i])
            combined_vars.append(combined)

        m.add(m.all_different(combined_vars))

        # Solve
        start_solve = perf_counter()
        result = m.solve()
        solve_time = perf_counter() - start_solve

        # Process results
        if result is None or not hasattr(result, "solution") or result.solution is None:
            return SolverResult(
                status=SolverStatus.INFEASIBLE,
                solve_time=solve_time,
                metadata={"message": "No feasible solution found"},
            )

        # Extract solution
        assignments = []
        soft_score = 0.0

        for i, lesson in enumerate(lessons):
            ts_val = result.solution.get(f"ts_{lesson['id']}", 0)
            rm_val = result.solution.get(f"rm_{lesson['id']}", 0)

            # Ensure values are integers
            ts_val = int(ts_val) if ts_val is not None else 0
            rm_val = int(rm_val) if rm_val is not None else 0

            assignments.append({
                "lesson_id": lesson["id"],
                "teacher_id": lesson["teacher_id"],
                "student_group_id": lesson["student_group_id"],
                "subject": lesson["subject"],
                "timeslot_id": ts_val,
                "room_id": rm_val,
            })

            # Calculate soft score (teacher preferences)
            teacher = teachers[lesson["teacher_id"]]
            if ts_val < len(teacher["preferences"]):
                soft_score += teacher["preferences"][ts_val]

        solution = {
            "assignments": assignments,
            "soft_score": soft_score,
        }

        return SolverResult(
            status=SolverStatus.OPTIMAL,
            objective_value=soft_score,
            solution=solution,
            solve_time=solve_time,
            metadata={
                "message": "Solution found",
                "n_lessons": n_lessons,
                "soft_score": soft_score,
            },
        )


# Convenience alias
SolvorSolver = SolvorCPSATSolver
