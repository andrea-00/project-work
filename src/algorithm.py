"""
Entry point for the submission format: runs the adaptive solver and returns
the solution in the format required by the guidelines: a list of tuples
(node_id, gold_amount) for each stop in the solution (path and item choices).
"""

from src.gold_collection.solvers.adaptive_solver import solve_return_solution


def run(problem_instance, *, time_budget_s: float = 1200.0, rng_seed=None):
    """
    Run the adaptive solver and return the solution in the guidelines format:
    list of tuples (node_id, gold_amount), one per stop, in visit order
    (all trips concatenated).

    Args:
        problem_instance: Problem instance (same interface as gold_collection.Problem)
        time_budget_s: Time budget in seconds
        rng_seed: Random seed for reproducibility

    Returns:
        List of (node_id, gold_amount) tuples representing the path and item choices.
    """
    sol = solve_return_solution(
        problem_instance,
        time_budget_s=time_budget_s,
        rng_seed=rng_seed,
    )

    if not sol.trips:
        return []

    path = []
    for trip in sol.trips:
        for node, gold in zip(trip.stops, trip.pickups):
            path.append((node, gold))
    return path
