# Gold collection — Computational Intelligence project

Minimize total cost to collect all gold from cities and bring it to depot 0. Edge cost: `d + (alpha * d * w)^beta`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

**Evaluate** (baseline vs strategy on instance configs):

```bash
python evaluate.py
```

**Beta regimes** (quick comparison for beta &lt; 1, = 1, &gt; 1):

```bash
python run_beta_regimes.py [n_cities] [n_seeds]
```

Example: `python run_beta_regimes.py 30 2` uses 30 cities and 2 seeds per regime.

## Results (typical)

Strategy vs baseline (improvement %): **beta &lt; 1** ~+24%, **beta = 1** ~0%, **beta &gt; 1** ~+99%. Beta &gt; 1 uses bulk removal + GA on remainder; beta ≤ 1 uses GA on full graph with Prins split and refinement.

## Entry point

`strategy.solution(problem)` returns the total cost. `evaluate.py` builds problems from `instance_gen` and reports baseline and strategy costs.
