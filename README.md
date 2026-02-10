
# Gold Collection Problem Solver

A computational intelligence approach to solve the Gold Collection Problem, a variant of the Vehicle Routing Problem (VRP) with load-dependent edge costs.

**Authors:** Alberto Migliorato, Andrea Di Felice
**Course:** Computational Intelligence 2026

## Problem Description

Minimize the total cost of collecting gold from all cities and returning it to the depot (node 0).

**Cost Function:**
```
C = d(u,v) + (α × d(u,v) × w)^β
```

Where:
- `d(u,v)` = shortest path distance between nodes
- `α` = load penalty scale parameter
- `β` = load penalty exponent
- `w` = current gold being carried

**Key Insight (β interpretation):**
- `β < 1`: Sub-linear costs → combine pickups in fewer, longer routes
- `β = 1`: Linear costs → classic VRP behavior
- `β > 1`: Super-linear costs → many small trips preferred

## Project Structure

```
gold-collection-solver/
├── src/
│   └── gold_collection/           # Main package
│       ├── core/                  # Problem, Solution, Trip, Evaluator
│       ├── solvers/               # Solving algorithms
│       │   ├── adaptive_solver.py # Main regime-adaptive solver
│       │   ├── heuristics/        # Constructive & local search
│       │   └── genetic/           # GA-based alternative solver
│       └── distance/              # Distance oracle
├── experiments/                   # Experiment framework
│   ├── configs.py                 # Instance configurations
│   ├── benchmark.py               # Benchmarking utilities
│   └── run_evaluation.py          # Main evaluation script
├── tests/                         # Unit tests
└── report/                        # LaTeX documentation
```

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

**Dependencies:** numpy, networkx, scikit-learn, matplotlib

## Quick Start

### Basic Usage

```python
from gold_collection import Problem, solve

# Create a problem instance
problem = Problem(
    num_cities=100,
    alpha=1.0,
    beta=0.5,
    density=0.5,
    seed=42
)

# Compute baseline (naive solution)
baseline_cost = problem.baseline()

# Compute optimized solution
optimized_cost = solve(problem, time_budget_s=60.0)

# Calculate improvement
improvement = 100 * (baseline_cost - optimized_cost) / baseline_cost
print(f"Improvement: {improvement:.1f}%")
```

### Run Benchmarks

```bash
# Basic evaluation (5 seeds, 120s per run)
python -m experiments.run_evaluation

# Custom settings
python -m experiments.run_evaluation --seeds 10 --time 60

# Include stress tests and GA solver
python -m experiments.run_evaluation --hard --genetic
```

## Solving Approach

The solver automatically selects one of three **regimes** based on β:

### Regime T (β < 1) - Tour-based
1. Build giant tour via nearest insertion + 2-opt
2. Split into trips using dynamic programming
3. Improve with LNS (destroy/repair)

### Regime L (β = 1) - Star-based
1. Start with individual trips per city
2. Merge using Clarke-Wright savings
3. Re-sequence with NN + 2-opt
4. Multi-restart LNS with simulated annealing

### Regime S (β > 1) - Capacity-based
1. Compute soft capacity Q from problem characteristics
2. Build routes by demand chunking
3. Light LNS refinement

### Key Components

- **Distance Oracle:** Landmark-based approximation for fast cost estimates
- **LNS:** Destroy-repair with regret-k insertion
- **Local Search:** 2-opt, relocate, swap operators

## Typical Results

| Regime | β | Improvement vs Baseline |
|--------|---|------------------------|
| T (sub-linear) | < 1 | 15-25% |
| L (linear) | = 1 | 0-2% |
| S (super-linear) | > 1 | 80-99% |

## API Reference

### Core Classes

```python
from gold_collection import Problem, Trip, Solution, Evaluator

# Problem: defines the instance
problem = Problem(n, alpha=1.0, beta=1.0, density=0.5, seed=42)

# Trip: sequence of stops with pickups
trip = Trip(stops=[0, 1, 2, 0], pickups=[0, gold_1, gold_2, 0])

# Solution: collection of trips
solution = Solution(trips=[trip1, trip2])

# Evaluator: compute costs
evaluator = Evaluator(problem, oracle)
cost = evaluator.solution_cost(solution)
```

### Solver Functions

```python
from gold_collection import solve, get_solution_cost

# Full solve with custom time budget
cost = solve(problem, time_budget_s=120.0, rng_seed=42)

# Quick convenience function
cost = get_solution_cost(problem)
```

### Genetic Algorithm Solver

```python
from gold_collection.solvers.genetic import solve_with_genetic_algorithm

cost = solve_with_genetic_algorithm(
    problem,
    multi_start=2,
    use_refinement=True,
    use_ils=True
)
```

## Testing

```bash
pytest tests/ -v
```

## Documentation

- Full documentation: [`DOCUMENTAZIONE_PROGETTO.md`](DOCUMENTAZIONE_PROGETTO.md)
- LaTeX report: `report/report.tex`

Build report:
```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

## License

MIT License - see LICENSE for details.