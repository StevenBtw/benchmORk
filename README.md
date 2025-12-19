# benchmORk

A benchmarking framework for Python optimization and operations research solvers. Compare performance across OR-Tools, Pyomo, SciPy, solvOR, and PuLP with an interactive Marimo dashboard.

## Features

- **Multi-solver support**: Unified interface for OR-Tools, Pyomo, SciPy, solvOR, and PuLP
- **Problem library**: Pre-built LP, MIP, and NLP problem definitions
- **Scalable benchmarks**: Test solvers across varying problem sizes (variables, constraints, data)
- **Interactive UI**: Marimo-powered dashboard for visual benchmarking
- **Metrics collection**: Solve time, setup time, memory usage, solution quality
- **Export & reporting**: Results in CSV/JSON with comparison charts

## Supported Solvers

| Solver                                                          | Type        | Status  |
| --------------------------------------------------------------- | ----------- | ------- |
| [OR-Tools](https://developers.google.com/optimization)          | LP, MIP, CP | Planned |
| [Pyomo](http://www.pyomo.org/)                                  | LP, MIP, NLP| Planned |
| [SciPy](https://docs.scipy.org/doc/scipy/reference/optimize.html)| LP, NLP    | Planned |
| [solvOR](https://github.com/yourrepo/solvor)                    | LP, MIP, CP | Planned |
| [PuLP](https://coin-or.github.io/pulp/)                         | LP, MIP     | Planned |

## Installation

Requires [uv](https://docs.astral.sh/uv/) for package management.

```bash
# Clone the repository
git clone https://github.com/yourusername/benchmORk.git
cd benchmORk

# Install with all dependencies
uv sync --all-extras
```

## Quick Start

### Command Line

```bash
# Run a quick benchmark
python -m benchmork.runner --config configs/quick.yaml

# Run with specific solvers
python -m benchmork.runner --solvers ortools pulp --problem knapsack
```

### Interactive Dashboard (Marimo)

The benchmark dashboard is built with [Marimo](https://marimo.io/), a reactive Python notebook.

```bash
# Launch the interactive dashboard
uv run marimo run app/main.py
```

This opens a browser-based dashboard where you can:

- **Select problem type** (LP, MIP) - filters available solvers automatically
- **Choose solvers** to compare (multi-select from available solvers)
- **Adjust problem size** with sliders
- **Run benchmarks** and view results with timing charts
- **Compare performance** across different solver/problem combinations

For development mode with hot-reloading:

```bash
uv run marimo edit app/main.py
```

### Python API

```python
from benchmork.problems import Knapsack
from benchmork.runner import BenchmarkRunner
from solvers import ORToolsSolver, PuLPSolver

# Define a problem
problem = Knapsack(n_items=100, capacity=500)

# Run benchmark
runner = BenchmarkRunner(
    solvers=[ORToolsSolver(), PuLPSolver()],
    problems=[problem]
)
results = runner.run()

# View results
print(results.summary())
```

## Problem Types

### Linear Programming (LP)

- Transportation problem
- Diet problem
- Blending problem

### Mixed Integer Programming (MIP)

- Knapsack problem
- Assignment problem
- Traveling Salesman (TSP)

### Nonlinear Programming (NLP)

- Rosenbrock function
- Portfolio optimization

## Configuration

Benchmark configurations are defined in YAML:

```yaml
# configs/standard.yaml
solvers:
  - ortools
  - pulp
  - pyomo

problems:
  - type: knapsack
    sizes: [100, 500, 1000, 5000]
  - type: transportation
    sizes: [10x10, 50x50, 100x100]

metrics:
  - solve_time
  - setup_time
  - memory_peak
  - solution_value
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
