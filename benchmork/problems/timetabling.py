"""School timetabling problem for benchmarking constraint solvers."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from benchmork.problems.base import BaseProblem, ProblemData, ProblemType


@dataclass
class SchoolTimetablingProblem(BaseProblem):
    """School timetabling problem.

    Assign lessons to timeslots and rooms such that:
    - Hard: No teacher teaches two lessons at the same time
    - Hard: No room hosts two lessons at the same time
    - Hard: No student group has two lessons at the same time
    - Soft: Minimize gaps in student schedules
    - Soft: Respect teacher preferences for timeslots

    This is a classic constraint satisfaction problem used by Timefold
    as their introductory quickstart example.
    """

    n_teachers: int = 5
    n_rooms: int = 3
    n_student_groups: int = 4
    n_timeslots: int = 20  # e.g., 4 periods/day * 5 days
    lessons_per_group: int = 15
    seed: int = 42

    @property
    def name(self) -> str:
        return "school_timetabling"

    @property
    def problem_type(self) -> ProblemType:
        return ProblemType.CONSTRAINT_SATISFACTION

    def generate(self) -> ProblemData:
        rng = np.random.default_rng(self.seed)

        # Generate teachers with subject specializations
        subjects = ["Math", "English", "Science", "History", "Art", "PE", "Music", "IT"]
        teachers = []
        for i in range(self.n_teachers):
            # Each teacher can teach 1-3 subjects
            n_subjects = rng.integers(1, min(4, len(subjects) + 1))
            teacher_subjects = list(rng.choice(subjects, size=n_subjects, replace=False))
            teachers.append({
                "id": i,
                "name": f"Teacher_{i}",
                "subjects": teacher_subjects,
                # Preferred timeslots (0-1 preference score for each slot)
                "preferences": rng.uniform(0.3, 1.0, size=self.n_timeslots).tolist(),
            })

        # Generate rooms with capacities
        rooms = []
        for i in range(self.n_rooms):
            rooms.append({
                "id": i,
                "name": f"Room_{i}",
                "capacity": int(rng.integers(20, 40)),
            })

        # Generate student groups
        student_groups = []
        for i in range(self.n_student_groups):
            student_groups.append({
                "id": i,
                "name": f"Group_{i}",
                "size": int(rng.integers(15, 30)),
            })

        # Generate timeslots
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        periods_per_day = max(1, self.n_timeslots // len(days))
        timeslots = []
        slot_id = 0
        for day_idx, day in enumerate(days):
            for period in range(periods_per_day):
                if slot_id >= self.n_timeslots:
                    break
                timeslots.append({
                    "id": slot_id,
                    "day": day,
                    "day_index": day_idx,
                    "period": period,
                    "label": f"{day}_P{period + 1}",
                })
                slot_id += 1

        # Generate lessons (what needs to be scheduled)
        lessons = []
        lesson_id = 0
        for group in student_groups:
            # Each group needs lessons_per_group lessons across various subjects
            group_subjects = list(rng.choice(subjects, size=self.lessons_per_group, replace=True))
            for subject in group_subjects:
                # Find a teacher who can teach this subject
                eligible_teachers = [t for t in teachers if subject in t["subjects"]]
                if not eligible_teachers:
                    # Assign to any teacher if no specialist
                    teacher = rng.choice(teachers)
                else:
                    teacher = rng.choice(eligible_teachers)

                lessons.append({
                    "id": lesson_id,
                    "subject": subject,
                    "teacher_id": teacher["id"],
                    "student_group_id": group["id"],
                    # To be assigned by solver:
                    "timeslot_id": None,
                    "room_id": None,
                })
                lesson_id += 1

        n_lessons = len(lessons)

        return ProblemData(
            objective={
                "type": "constraint_satisfaction",
                "minimize_gaps": True,
                "respect_preferences": True,
            },
            constraints={
                "hard": [
                    "no_teacher_conflict",  # Teacher can't teach 2 lessons at same time
                    "no_room_conflict",  # Room can't host 2 lessons at same time
                    "no_group_conflict",  # Student group can't have 2 lessons at same time
                ],
                "soft": [
                    "minimize_gaps",  # Minimize gaps in student schedules
                    "teacher_preferences",  # Respect teacher timeslot preferences
                    "room_capacity",  # Prefer rooms with adequate capacity
                ],
            },
            bounds={
                "timeslot_range": (0, self.n_timeslots - 1),
                "room_range": (0, self.n_rooms - 1),
            },
            metadata={
                "n_variables": n_lessons * 2,  # timeslot + room per lesson
                "n_constraints": n_lessons * 3,  # approx hard constraints
                "n_teachers": self.n_teachers,
                "n_rooms": self.n_rooms,
                "n_student_groups": self.n_student_groups,
                "n_timeslots": self.n_timeslots,
                "n_lessons": n_lessons,
                "teachers": teachers,
                "rooms": rooms,
                "student_groups": student_groups,
                "timeslots": timeslots,
                "lessons": lessons,
            },
        )

    def get_scaling_params(self) -> dict[str, int]:
        return {
            "n_teachers": self.n_teachers,
            "n_rooms": self.n_rooms,
            "n_student_groups": self.n_student_groups,
            "n_timeslots": self.n_timeslots,
        }

    def validate_solution(self, solution: dict[str, Any]) -> bool:
        """Validate that a solution has no hard constraint violations."""
        assignments = solution.get("assignments", [])
        if not assignments:
            return False

        # Check for conflicts
        timeslot_teacher: dict[tuple[int, int], int] = {}
        timeslot_room: dict[tuple[int, int], int] = {}
        timeslot_group: dict[tuple[int, int], int] = {}

        for assignment in assignments:
            lesson_id = assignment["lesson_id"]
            timeslot = assignment["timeslot_id"]
            room = assignment["room_id"]
            teacher = assignment["teacher_id"]
            group = assignment["student_group_id"]

            # Check teacher conflict
            key = (timeslot, teacher)
            if key in timeslot_teacher:
                return False
            timeslot_teacher[key] = lesson_id

            # Check room conflict
            key = (timeslot, room)
            if key in timeslot_room:
                return False
            timeslot_room[key] = lesson_id

            # Check student group conflict
            key = (timeslot, group)
            if key in timeslot_group:
                return False
            timeslot_group[key] = lesson_id

        return True

    def score_solution(self, solution: dict[str, Any], problem_data: ProblemData) -> dict[str, Any]:
        """Score a solution including soft constraint violations."""
        assignments = solution.get("assignments", [])
        metadata = problem_data.metadata
        teachers = {t["id"]: t for t in metadata["teachers"]}

        hard_violations = 0
        soft_score = 0.0

        # Validate hard constraints
        if not self.validate_solution(solution):
            hard_violations = -1  # At least one violation

        # Score soft constraints
        for assignment in assignments:
            teacher = teachers[assignment["teacher_id"]]
            timeslot = assignment["timeslot_id"]
            # Add preference score
            soft_score += teacher["preferences"][timeslot]

        return {
            "hard_score": hard_violations,
            "soft_score": soft_score,
            "feasible": hard_violations == 0,
        }
