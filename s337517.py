# Folder: project-work/
# |- src/
# |   |- gold_collection/   (auxiliary code)
# |   |- algorithm.py
# |- s337517.py

from src.algorithm import run


def solution(problem_instance):
    """
    Solve the Gold Collection Problem for the given instance.

    Args:
        problem_instance: Problem instance (graph, alpha, beta, etc.)

    Returns:
        List of tuples (node_id, gold_amount) representing the path and
        item choices (one tuple per stop, in visit order).
    """
    path = run(problem_instance)
    return path
