"""Instance management for benchmark problems.

Provides utilities for:
- Loading/saving problem instances in JSON/YAML format
- Generating reproducible synthetic instances
- Loading standard instance libraries (TSPLIB, MIPLIB)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

__all__ = [
    "Instance",
    "InstanceGenerator",
    "InstanceLibrary",
    "load_instance",
    "save_instance",
    "load_tsplib",
]


@dataclass
class Instance:
    """A problem instance with metadata for reproducibility.

    Attributes:
        name: Unique identifier for this instance
        problem_type: Type of problem (tsp, knapsack, linear, etc.)
        data: Problem-specific data
        metadata: Additional info (source, optimal value, etc.)
    """

    name: str
    problem_type: str
    data: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checksum(self) -> str:
        """Return MD5 hash of instance data for integrity verification."""
        content = json.dumps(self.data, sort_keys=True).encode()
        return hashlib.md5(content).hexdigest()[:8]

    @property
    def known_optimum(self) -> float | None:
        """Return known optimal value if available."""
        return self.metadata.get("optimal_value")

    def to_dict(self) -> dict[str, Any]:
        """Convert instance to dictionary for serialization."""
        return {
            "name": self.name,
            "problem_type": self.problem_type,
            "data": self.data,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Instance:
        """Create instance from dictionary."""
        return cls(
            name=d["name"],
            problem_type=d["problem_type"],
            data=d["data"],
            metadata=d.get("metadata", {}),
        )


def load_instance(path: str | Path) -> Instance:
    """Load an instance from JSON or YAML file.

    Args:
        path: Path to the instance file

    Returns:
        Loaded Instance object
    """
    path = Path(path)
    content = path.read_text()

    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(content)
    else:
        data = json.loads(content)

    return Instance.from_dict(data)


def _convert_tuples(obj: Any) -> Any:
    """Recursively convert tuples to lists for YAML/JSON compatibility."""
    if isinstance(obj, tuple):
        return [_convert_tuples(x) for x in obj]
    elif isinstance(obj, list):
        return [_convert_tuples(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _convert_tuples(v) for k, v in obj.items()}
    return obj


def save_instance(instance: Instance, path: str | Path, format: str = "json") -> None:
    """Save an instance to JSON or YAML file.

    Args:
        instance: Instance to save
        path: Output file path
        format: Output format ('json' or 'yaml')
    """
    path = Path(path)
    data = _convert_tuples(instance.to_dict())

    if format == "yaml" or path.suffix in (".yaml", ".yml"):
        content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    else:
        content = json.dumps(data, indent=2)

    path.write_text(content)


class InstanceGenerator:
    """Generate reproducible synthetic problem instances.

    Examples:
        gen = InstanceGenerator(seed=42)

        # Generate TSP instance
        tsp = gen.tsp(n_cities=50, grid_size=100)

        # Generate knapsack instance
        knapsack = gen.knapsack(n_items=100, capacity_ratio=0.5)

        # Generate random LP
        lp = gen.linear_program(n_vars=10, n_constraints=5)
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._rng_state = seed

    def _next_seed(self) -> int:
        """Get next deterministic seed."""
        import random

        rng = random.Random(self._rng_state)
        self._rng_state = rng.randint(0, 2**31 - 1)
        return self._rng_state

    def tsp(
        self,
        n_cities: int,
        grid_size: int = 100,
        name: str | None = None,
    ) -> Instance:
        """Generate a random TSP instance.

        Args:
            n_cities: Number of cities
            grid_size: Size of the coordinate grid
            name: Instance name (auto-generated if None)

        Returns:
            Instance with TSP data
        """
        import random

        seed = self._next_seed()
        rng = random.Random(seed)

        coordinates = [(rng.randint(0, grid_size), rng.randint(0, grid_size))
                       for _ in range(n_cities)]

        # Compute distance matrix
        import math

        distance_matrix = []
        for i in range(n_cities):
            row = []
            for j in range(n_cities):
                if i == j:
                    row.append(0.0)
                else:
                    dx = coordinates[i][0] - coordinates[j][0]
                    dy = coordinates[i][1] - coordinates[j][1]
                    row.append(math.sqrt(dx * dx + dy * dy))
            distance_matrix.append(row)

        return Instance(
            name=name or f"tsp_{n_cities}_{seed}",
            problem_type="tsp",
            data={
                "n_cities": n_cities,
                "coordinates": coordinates,
                "distance_matrix": distance_matrix,
            },
            metadata={
                "generator": "InstanceGenerator.tsp",
                "seed": seed,
                "grid_size": grid_size,
            },
        )

    def knapsack(
        self,
        n_items: int,
        capacity_ratio: float = 0.5,
        max_weight: int = 100,
        max_value: int = 100,
        name: str | None = None,
    ) -> Instance:
        """Generate a random 0-1 knapsack instance.

        Args:
            n_items: Number of items
            capacity_ratio: Capacity as fraction of total weight
            max_weight: Maximum item weight
            max_value: Maximum item value
            name: Instance name (auto-generated if None)

        Returns:
            Instance with knapsack data
        """
        import random

        seed = self._next_seed()
        rng = random.Random(seed)

        weights = [rng.randint(1, max_weight) for _ in range(n_items)]
        values = [rng.randint(1, max_value) for _ in range(n_items)]
        capacity = int(sum(weights) * capacity_ratio)

        return Instance(
            name=name or f"knapsack_{n_items}_{seed}",
            problem_type="knapsack",
            data={
                "n_items": n_items,
                "weights": weights,
                "values": values,
                "capacity": capacity,
            },
            metadata={
                "generator": "InstanceGenerator.knapsack",
                "seed": seed,
                "capacity_ratio": capacity_ratio,
            },
        )

    def assignment(
        self,
        n_agents: int,
        max_cost: int = 100,
        name: str | None = None,
    ) -> Instance:
        """Generate a random assignment problem instance.

        Args:
            n_agents: Number of agents/tasks
            max_cost: Maximum cost value
            name: Instance name (auto-generated if None)

        Returns:
            Instance with assignment data
        """
        import random

        seed = self._next_seed()
        rng = random.Random(seed)

        cost_matrix = [
            [rng.randint(1, max_cost) for _ in range(n_agents)]
            for _ in range(n_agents)
        ]

        return Instance(
            name=name or f"assignment_{n_agents}_{seed}",
            problem_type="assignment",
            data={
                "n_agents": n_agents,
                "cost_matrix": cost_matrix,
            },
            metadata={
                "generator": "InstanceGenerator.assignment",
                "seed": seed,
            },
        )

    def linear_program(
        self,
        n_vars: int,
        n_constraints: int,
        density: float = 1.0,
        name: str | None = None,
    ) -> Instance:
        """Generate a random LP instance.

        Args:
            n_vars: Number of decision variables
            n_constraints: Number of inequality constraints
            density: Fraction of non-zero coefficients
            name: Instance name (auto-generated if None)

        Returns:
            Instance with LP data
        """
        import random

        seed = self._next_seed()
        rng = random.Random(seed)

        # Objective coefficients
        c = [rng.uniform(-10, 10) for _ in range(n_vars)]

        # Constraint matrix and RHS
        A_ub = []
        b_ub = []
        for _ in range(n_constraints):
            row = []
            for _ in range(n_vars):
                if rng.random() < density:
                    row.append(rng.uniform(-10, 10))
                else:
                    row.append(0.0)
            A_ub.append(row)
            b_ub.append(rng.uniform(0, 100))

        # Variable bounds (non-negative)
        bounds = [(0.0, None) for _ in range(n_vars)]

        return Instance(
            name=name or f"lp_{n_vars}x{n_constraints}_{seed}",
            problem_type="linear",
            data={
                "c": c,
                "A_ub": A_ub,
                "b_ub": b_ub,
                "bounds": bounds,
            },
            metadata={
                "generator": "InstanceGenerator.linear_program",
                "seed": seed,
                "density": density,
            },
        )


def load_tsplib(path: str | Path) -> Instance:
    """Load a TSP instance from TSPLIB format (.tsp file).

    Supports:
    - EUC_2D: 2D Euclidean distance
    - EXPLICIT: Explicit distance matrix
    - GEO: Geographic coordinates

    Args:
        path: Path to .tsp file

    Returns:
        Instance with TSP data

    Example:
        inst = load_tsplib("berlin52.tsp")
        print(inst.data["n_cities"])  # 52
    """
    import math
    import re

    path = Path(path)
    content = path.read_text()
    lines = content.strip().split("\n")

    # Parse header
    name = path.stem
    dimension = 0
    edge_weight_type = "EUC_2D"
    edge_weight_format = None
    best_known = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("NAME"):
            name = line.split(":")[1].strip() if ":" in line else line.split()[1]
        elif line.startswith("DIMENSION"):
            dimension = int(re.search(r"\d+", line).group())
        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":")[1].strip() if ":" in line else line.split()[1]
        elif line.startswith("EDGE_WEIGHT_FORMAT"):
            edge_weight_format = line.split(":")[1].strip() if ":" in line else line.split()[1]
        elif line.startswith("BEST_KNOWN"):
            best_known = float(re.search(r"[\d.]+", line).group())
        elif line.startswith("NODE_COORD_SECTION") or line.startswith("EDGE_WEIGHT_SECTION"):
            i += 1
            break
        i += 1

    coordinates = []
    distance_matrix = [[0.0] * dimension for _ in range(dimension)]

    if edge_weight_type in ("EUC_2D", "GEO", "ATT", "CEIL_2D"):
        # Read coordinates
        while i < len(lines) and not lines[i].strip().startswith("EOF"):
            parts = lines[i].strip().split()
            if len(parts) >= 3 and parts[0].isdigit():
                x, y = float(parts[1]), float(parts[2])
                coordinates.append((x, y))
            i += 1

        # Compute distance matrix
        n = len(coordinates)
        for i_coord in range(n):
            for j_coord in range(n):
                if i_coord != j_coord:
                    x1, y1 = coordinates[i_coord]
                    x2, y2 = coordinates[j_coord]

                    if edge_weight_type == "EUC_2D":
                        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    elif edge_weight_type == "CEIL_2D":
                        dist = math.ceil(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
                    elif edge_weight_type == "ATT":
                        # Pseudo-Euclidean for ATT instances
                        dx = x1 - x2
                        dy = y1 - y2
                        r = math.sqrt((dx * dx + dy * dy) / 10.0)
                        t = int(r + 0.5)
                        dist = t + 1 if t < r else t
                    elif edge_weight_type == "GEO":
                        # Geographic distance
                        def to_radians(deg: float) -> float:
                            d = int(deg)
                            m = deg - d
                            return math.pi * (d + 5.0 * m / 3.0) / 180.0

                        lat1, lon1 = to_radians(x1), to_radians(y1)
                        lat2, lon2 = to_radians(x2), to_radians(y2)
                        RRR = 6378.388
                        q1 = math.cos(lon1 - lon2)
                        q2 = math.cos(lat1 - lat2)
                        q3 = math.cos(lat1 + lat2)
                        dist = int(RRR * math.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)
                    else:
                        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                    distance_matrix[i_coord][j_coord] = dist

    elif edge_weight_type == "EXPLICIT":
        # Read explicit distance matrix
        values = []
        while i < len(lines) and not lines[i].strip().startswith("EOF"):
            parts = lines[i].strip().split()
            for p in parts:
                try:
                    values.append(float(p))
                except ValueError:
                    pass
            i += 1

        # Parse based on format
        idx = 0
        if edge_weight_format in ("FULL_MATRIX",):
            for row in range(dimension):
                for col in range(dimension):
                    distance_matrix[row][col] = values[idx]
                    idx += 1
        elif edge_weight_format in ("UPPER_ROW", "UPPER_DIAG_ROW"):
            for row in range(dimension):
                start = row if edge_weight_format == "UPPER_DIAG_ROW" else row + 1
                for col in range(start, dimension):
                    distance_matrix[row][col] = values[idx]
                    distance_matrix[col][row] = values[idx]
                    idx += 1
        elif edge_weight_format in ("LOWER_ROW", "LOWER_DIAG_ROW"):
            for row in range(dimension):
                end = row + 1 if edge_weight_format == "LOWER_DIAG_ROW" else row
                for col in range(end):
                    distance_matrix[row][col] = values[idx]
                    distance_matrix[col][row] = values[idx]
                    idx += 1

    return Instance(
        name=name,
        problem_type="tsp",
        data={
            "n_cities": dimension,
            "coordinates": coordinates if coordinates else None,
            "distance_matrix": distance_matrix,
        },
        metadata={
            "source": "TSPLIB",
            "file": str(path),
            "edge_weight_type": edge_weight_type,
            "optimal_value": best_known,
        },
    )


class InstanceLibrary:
    """Manage collections of problem instances.

    Provides access to:
    - Built-in benchmark instances
    - User-defined instance directories
    - Standard libraries (TSPLIB, MIPLIB) when available
    """

    def __init__(self, paths: list[str | Path] | None = None):
        """Initialize the library.

        Args:
            paths: List of directories to search for instances
        """
        self.paths: list[Path] = []
        if paths:
            self.paths = [Path(p) for p in paths]

        self._cache: dict[str, Instance] = {}

    def add_path(self, path: str | Path) -> None:
        """Add a directory to the search path."""
        self.paths.append(Path(path))

    def list_instances(self, problem_type: str | None = None) -> list[str]:
        """List all available instance names.

        Args:
            problem_type: Filter by problem type if specified

        Returns:
            List of instance names
        """
        instances = []

        for path in self.paths:
            if not path.exists():
                continue

            for ext in ("json", "yaml", "yml"):
                for file in path.glob(f"**/*.{ext}"):
                    try:
                        inst = load_instance(file)
                        if problem_type is None or inst.problem_type == problem_type:
                            instances.append(inst.name)
                    except Exception:
                        continue  # Skip invalid files

        return sorted(set(instances))

    def get(self, name: str) -> Instance | None:
        """Get an instance by name.

        Args:
            name: Instance name

        Returns:
            Instance if found, None otherwise
        """
        if name in self._cache:
            return self._cache[name]

        for path in self.paths:
            if not path.exists():
                continue

            for ext in ("json", "yaml", "yml"):
                for file in path.glob(f"**/*.{ext}"):
                    try:
                        inst = load_instance(file)
                        if inst.name == name:
                            self._cache[name] = inst
                            return inst
                    except Exception:
                        continue

        return None

    def __getitem__(self, name: str) -> Instance:
        """Get instance by name, raising KeyError if not found."""
        inst = self.get(name)
        if inst is None:
            raise KeyError(f"Instance not found: {name}")
        return inst
