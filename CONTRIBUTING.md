# Contributing to benchmORk

Thank you for your interest in contributing to benchmORk! This document provides guidelines and information for contributors.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Working with Marimo](#working-with-marimo)
- [Adding a New Solver](#adding-a-new-solver)
- [Adding a New Problem](#adding-a-new-problem)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)

## Development Setup

### Prerequisites

- Python 3.13 or higher
- Git
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/benchmORk.git
cd benchmORk

# Install in development mode with all dependencies
uv sync --all-extras

# Install pre-commit hooks (optional)
pre-commit install
```

### Running the Marimo App

```bash
uv run marimo run app/main.py
```

## Project Structure

```
benchmORk/
├── README.md                   # Project overview and usage
├── CONTRIBUTING.md             # This file
├── LICENSE                     # Apache-2.0 License
├── pyproject.toml              # Project configuration and dependencies
│
├── benchmork/                  # Core benchmark engine
│   ├── __init__.py
│   ├── problems/               # Problem definitions (solver-agnostic)
│   │   ├── __init__.py
│   │   ├── base.py             # Abstract problem interface
│   │   ├── linear.py           # LP: transportation, diet, blending
│   │   ├── integer.py          # MIP: knapsack, assignment, TSP
│   │   └── nonlinear.py        # NLP problems
│   │
│   ├── runner.py               # Benchmark execution engine
│   ├── metrics.py              # Timing, memory, solution quality
│   ├── scaling.py              # Problem size generators
│   │
│   └── reporting/              # Results & visualization
│       ├── __init__.py
│       ├── export.py           # CSV/JSON export
│       └── plots.py            # Comparison charts (for Marimo)
│
├── solvers/                    # Solver adapters
│   ├── __init__.py
│   ├── base.py                 # Abstract solver interface
│   ├── ortools_solver.py       # Google OR-Tools adapter
│   ├── pyomo_solver.py         # Pyomo adapter
│   ├── scipy_solver.py         # SciPy optimize adapter
│   ├── solvor_solver.py        # solvOR adapter
│   └── pulp_solver.py          # PuLP adapter
│
├── app/                        # Marimo interactive UI
│   ├── __init__.py
│   ├── main.py                 # Main Marimo app entry point
│   ├── components/             # Reusable UI components
│   │   ├── __init__.py
│   │   ├── solver_selector.py  # Multi-select solver picker
│   │   ├── problem_config.py   # Problem type & parameters
│   │   ├── scaling_controls.py # Vars, constraints, data size sliders
│   │   └── results_view.py     # Tables & charts display
│   │
│   └── pages/                  # Marimo notebook pages
│       ├── benchmark.py        # Main benchmark runner page
│       ├── compare.py          # Side-by-side comparison
│       ├── history.py          # Past runs & trends
│       └── explorer.py         # Deep-dive single solver
│
├── tests/                      # Test suite
│   ├── __init__.py
│   ├── test_problems.py        # Problem definition tests
│   ├── test_solvers.py         # Solver adapter tests
│   └── test_benchmarks.py      # Benchmark runner tests
│
├── configs/                    # Preset benchmark configurations
│   ├── quick.yaml              # Fast smoke test
│   ├── standard.yaml           # Default benchmark suite
│   └── full.yaml               # Comprehensive (slow)
│
└── results/                    # Output directory (gitignored)
    └── .gitkeep
```

## Working with Marimo

The interactive dashboard is built with [Marimo](https://marimo.io/), a reactive Python notebook. See the [official Marimo documentation](https://docs.marimo.io/) for comprehensive guides.

### Development Mode

Use `edit` mode for development with hot-reloading:

```bash
uv run marimo edit app/main.py
```

This opens the notebook in edit mode where you can:

- Modify cells and see changes instantly
- Add new UI components
- Debug with the built-in variable explorer

### Run Mode

Use `run` mode for the production dashboard:

```bash
uv run marimo run app/main.py
```

### Key Marimo Patterns

When working with Marimo cells, keep these patterns in mind:

1. **Unique variable names**: Each variable can only be defined in ONE cell. Use unique names to avoid conflicts.

2. **Early exit with `mo.stop()`**: Instead of `return` for early exit, use:

   ```python
   mo.stop(condition, mo.md("Message to show"))
   ```

3. **Reactive updates**: When a UI element's value changes, all cells that depend on it automatically re-run.

4. **Cell dependencies**: Marimo automatically tracks which cells depend on which variables. Structure your cells so data flows clearly.

5. **UI elements must be returned**: For a UI element to be displayed, it must be returned from a cell (either directly or wrapped in `mo.vstack`/`mo.hstack`).

### Adding New Dashboard Features

1. Create UI elements in a cell and return them for display
2. Use those elements in downstream cells via their `.value` property
3. Keep visualization logic separate from data processing
4. Use `mo.stop()` to conditionally show content

Example:

```python
@app.cell
def my_control(mo):
    slider = mo.ui.slider(start=1, stop=100, value=50, label="Size")
    return slider,

@app.cell
def use_control(mo, slider):
    # slider.value contains the current value
    return mo.md(f"Selected size: {slider.value}")
```

## Adding a New Solver

To add support for a new optimization solver:

1. Create a new file in `solvers/` (e.g., `solvers/newsolver_solver.py`)

2. Implement the `BaseSolver` interface:

```python
from solvers.base import BaseSolver, SolverResult
from benchmork.problems.base import BaseProblem

class NewSolver(BaseSolver):
    """Adapter for NewSolver optimization library."""

    name = "newsolver"
    supported_problem_types = ["linear", "integer"]

    def __init__(self, **options):
        super().__init__(**options)
        # Initialize solver-specific settings

    def solve(self, problem: BaseProblem) -> SolverResult:
        # Translate problem to solver's format
        # Execute solver
        # Return standardized result
        pass

    def is_available(self) -> bool:
        # Check if solver library is installed
        try:
            import newsolver
            return True
        except ImportError:
            return False
```

3. Add the solver dependency to `pyproject.toml`:

```toml
[project.optional-dependencies]
newsolver = ["newsolver>=1.0"]
```

4. Register the solver in `solvers/__init__.py`
5. Add tests in `tests/test_solvers.py`
6. Update the README solver table

## Adding a New Problem

To add a new benchmark problem:

1. Identify the problem category (`linear.py`, `integer.py`, or `nonlinear.py`)
2. Implement the `BaseProblem` interface:

```python
from benchmork.problems.base import BaseProblem
from dataclasses import dataclass
import numpy as np

@dataclass
class NewProblem(BaseProblem):
    """Description of the optimization problem."""

    name = "new_problem"
    problem_type = "linear"  # or "integer", "nonlinear"

    # Problem parameters
    size: int = 100
    seed: int = 42

    def generate(self) -> dict:
        """Generate problem data."""
        rng = np.random.default_rng(self.seed)
        # Generate coefficients, constraints, etc.
        return {
            "c": ...,  # Objective coefficients
            "A": ...,  # Constraint matrix
            "b": ...,  # Constraint bounds
        }

    def get_scaling_params(self) -> dict:
        """Return parameters that affect problem size."""
        return {"size": self.size}
```

3. Add tests in `tests/test_problems.py`
4. Update the README problem list

## Code Style

We use the following tools to maintain code quality:

- **Ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

### Formatting

```bash
# Format code
ruff format .

# Check linting
ruff check .

# Type checking
mypy benchmork solvers
```

### Guidelines

- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Keep functions focused and under 50 lines when possible
- Use descriptive variable names

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=benchmork --cov=solvers

# Run specific test file
pytest tests/test_solvers.py

# Run specific test
pytest tests/test_solvers.py::test_ortools_knapsack
```

### Test Requirements

- All new solvers must have integration tests
- All new problems must have validation tests
- Aim for >80% code coverage on new code

## Pull Request Process

1. **Fork** the repository and create a feature branch
2. **Write tests** for your changes
3. **Ensure all tests pass** locally
4. **Update documentation** if needed (README, docstrings)
5. **Submit a PR** with a clear description of changes

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`ruff format .`)
- [ ] Linting passes (`ruff check .`)
- [ ] Type checking passes (`mypy benchmork solvers`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG updated for notable changes

## Questions?

Feel free to open an issue for questions or discussions about the project.
