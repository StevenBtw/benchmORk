"""Configuration loading for benchmORk."""

from pathlib import Path
from dataclasses import dataclass, field
import yaml


@dataclass
class SolverConfig:
    """Configuration for a solver."""

    id: str
    tool: str
    solver: str
    display_name: str
    problem_types: list[str]
    description: str = ""
    planned: bool = False

    @property
    def is_available(self) -> bool:
        """Check if this solver is implemented and available."""
        if self.planned:
            return False
        # Check if the tool is importable
        try:
            if self.tool == "scipy":
                import scipy
                return True
            elif self.tool == "pulp":
                import pulp
                return True
            elif self.tool == "ortools":
                import ortools
                return True
            elif self.tool == "pyomo":
                import pyomo
                return True
            elif self.tool == "solvor":
                import solvor
                return True
        except ImportError:
            return False
        return False


@dataclass
class ProblemTypeConfig:
    """Configuration for a problem type."""

    id: str
    display_name: str
    description: str = ""


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""

    solvers: dict[str, SolverConfig] = field(default_factory=dict)
    problem_types: dict[str, ProblemTypeConfig] = field(default_factory=dict)

    def get_solvers_for_problem(self, problem_type: str) -> list[SolverConfig]:
        """Get all solvers that support a given problem type."""
        return [
            s for s in self.solvers.values()
            if problem_type in s.problem_types
        ]

    def get_available_solvers(self, problem_type: str | None = None) -> list[SolverConfig]:
        """Get all available (non-planned) solvers, optionally filtered by problem type."""
        solvers = self.solvers.values()
        if problem_type:
            solvers = [s for s in solvers if problem_type in s.problem_types]
        return [s for s in solvers if s.is_available]


def load_config(config_path: Path | str | None = None) -> BenchmarkConfig:
    """Load benchmark configuration from YAML file."""
    if config_path is None:
        # Default to configs/solvers.yaml relative to project root
        config_path = Path(__file__).parent.parent / "configs" / "solvers.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        return BenchmarkConfig()

    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Parse solvers
    solvers = {}
    for solver_id, solver_data in data.get("solvers", {}).items():
        solvers[solver_id] = SolverConfig(
            id=solver_id,
            tool=solver_data.get("tool", ""),
            solver=solver_data.get("solver", ""),
            display_name=solver_data.get("display_name", solver_id),
            problem_types=solver_data.get("problem_types", []),
            description=solver_data.get("description", ""),
            planned=solver_data.get("planned", False),
        )

    # Parse problem types
    problem_types = {}
    for type_id, type_data in data.get("problem_types", {}).items():
        problem_types[type_id] = ProblemTypeConfig(
            id=type_id,
            display_name=type_data.get("display_name", type_id),
            description=type_data.get("description", ""),
        )

    return BenchmarkConfig(solvers=solvers, problem_types=problem_types)


# Global config instance
_config: BenchmarkConfig | None = None


def get_config() -> BenchmarkConfig:
    """Get the global configuration, loading if necessary."""
    global _config
    if _config is None:
        _config = load_config()
    return _config
