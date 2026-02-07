"""
Adapter del Problem (gold collection) per il solver gerarchico.

Espone:
- get_cost_metric(): matrice N×N di distanze (per clustering).
- get_fitness_function(): callable(permutation) -> fitness (da massimizzare: -costo).
- create_sub_problem(): sottoproblema su un sottoinsieme di nodi (per cluster).
"""

from typing import List, Dict, Callable, Optional
import numpy as np
import networkx as nx

from problem_base import Problem
from .simulator import simulate_tour, path_cost_with_load

# Tipo genotipo per routing (permutazione di indici)
RoutingGenotype = List[int]


def _build_distance_matrix(problem: Problem) -> np.ndarray:
    """
    Matrice N×N di distanze geodesiche sul grafo (weight='dist').
    Per coppie non connesse non dovrebbe capitare (grafo connesso).
    """
    G = problem.graph
    n = G.number_of_nodes()
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="dist"))
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i, j] = 0.0
            else:
                matrix[i, j] = lengths.get(i, {}).get(j, float("inf"))
    return matrix


class GoldCollectionAdapter:
    """
    Wrapper di Problem che espone l'interfaccia usata dal solver gerarchico:
    - cost metric (matrice distanze) per partizionamento e factory
    - fitness (permutazione -> valore da massimizzare) per EA
    - create_sub_problem per sottoinsiemi di nodi (cluster)
    """

    def __init__(
        self,
        problem: Problem,
        *,
        entry_load: float = 0.0,
        entry_node: Optional[int] = None,
    ):
        """
        Args:
            problem: istanza Problem (grafo, cost, gold).
            entry_load: (solo per sub-problem) carico all'ingresso nel cluster.
            entry_node: (solo per sub-problem) nodo di ingresso (indice originale).
        """
        self._problem = problem
        self._entry_load = entry_load
        self._entry_node = entry_node
        self._distance_matrix: Optional[np.ndarray] = None
        # Per sub-problem: sottoinsieme di indici originali e mappa local -> original
        self._element_indices: Optional[List[int]] = None
        self._local_to_original: Optional[Dict[int, int]] = None

    @property
    def num_elements(self) -> int:
        """Numero di nodi (root) o di nodi nel sottoinsieme (sub)."""
        if self._element_indices is not None:
            return len(self._element_indices)
        return self._problem.graph.number_of_nodes()

    def get_cost_metric(self) -> np.ndarray:
        """
        Matrice delle distanze per clustering e factory.
        Root: N×N geodesiche. Sub: sottomatrice sugli element_indices.
        """
        if self._element_indices is not None:
            full = self._get_full_distance_matrix()
            ix = self._element_indices
            return full[np.ix_(ix, ix)].copy()
        return self._get_full_distance_matrix()

    def _get_full_distance_matrix(self) -> np.ndarray:
        if self._distance_matrix is None:
            self._distance_matrix = _build_distance_matrix(self._problem)
        return self._distance_matrix

    def get_fitness_function(self) -> Callable[[RoutingGenotype], float]:
        """
        Restituisce una funzione fitness(permutation) -> float.
        Maggiore è meglio (fitness = -costo del tour simulato).
        """
        if self._element_indices is not None:
            return self._sub_fitness
        return self._root_fitness

    def _root_fitness(self, genotype: RoutingGenotype) -> float:
        """Fitness problema completo: -simulate_tour."""
        cost = simulate_tour(self._problem, genotype, start_at_base=True, end_at_base=True)
        return -cost

    def _sub_fitness(self, genotype: RoutingGenotype) -> float:
        """
        Fitness sottoproblema (intra-cluster): costo del sotto-tour con entry_load/entry_node.
        genotype = permutazione di indici locali (0 .. num_elements-1).
        Simuliamo: da entry_node con entry_load, visitiamo i nodi in ordine (genotype
        tradotto in original), e ritorniamo il costo di quel segmento (senza tornare a 0).
        """
        if self._element_indices is None or self._local_to_original is None:
            return -float("inf")  # fallback
        G = self._problem.graph
        # Traduci genotipo locale -> original
        orig_order = [self._local_to_original[i] for i in genotype]
        entry = self._entry_node if self._entry_node is not None else self._element_indices[0]
        load = self._entry_load
        current = entry
        total_cost = 0.0
        for next_node in orig_order:
            if next_node == current:
                continue
            c, _ = path_cost_with_load(self._problem, current, next_node, load)
            total_cost += c
            load += G.nodes[next_node]["gold"]
            current = next_node
        return -total_cost

    def create_sub_problem(
        self,
        element_indices: List[int],
        index_mapping: Dict[int, int],
        *,
        entry_load: float = 0.0,
        entry_node: Optional[int] = None,
    ) -> "GoldCollectionAdapter":
        """
        Crea un adapter per il sottoproblema (es. un cluster).

        Args:
            element_indices: indici (originali) dei nodi nel cluster.
            index_mapping: mappa indice_locale -> indice_originale (local_to_original).
            entry_load: carico all'ingresso nel cluster (per fitness intra).
            entry_node: nodo di ingresso nel cluster (indice originale).
        """
        sub = GoldCollectionAdapter(
            self._problem,
            entry_load=entry_load,
            entry_node=entry_node,
        )
        sub._element_indices = element_indices
        sub._local_to_original = index_mapping
        # La matrice distanze del root è condivisa (riferimento)
        sub._distance_matrix = self._get_full_distance_matrix()
        return sub

    def get_initializer_function(self) -> Callable[[], RoutingGenotype]:
        """
        Per compatibilità con pipeline EA: restituisce una funzione che
        genera una permutazione casuale (root: 0..N-1; sub: 0..len-1).
        """
        n = self.num_elements
        rng = __import__("random")

        def init() -> RoutingGenotype:
            if self._element_indices is not None:
                return rng.sample(range(n), n)
            # Root: permutazione che inizia con 0 (base) poi 1..N-1 mescolati
            rest = list(range(1, n))
            rng.shuffle(rest)
            return [0] + rest

        return init
