"""
Partizionamento gerarchico: clustering con nodo 0 come cluster 0.

Cluster 0 = {0} (base). I nodi 1..N-1 vengono partizionati in k-1 cluster
con Agglomerative Clustering sulla matrice delle distanze.
"""

from typing import Dict, List
import numpy as np

try:
    from sklearn.cluster import AgglomerativeClustering
except ImportError:
    AgglomerativeClustering = None

from .gold_problem_adapter import GoldCollectionAdapter


def partition(
    adapter: GoldCollectionAdapter,
    k: int,
) -> Dict[int, List[int]]:
    """
    Partiziona i nodi in k cluster. Cluster 0 = {0}. Resto in k-1 cluster.

    Args:
        adapter: GoldCollectionAdapter (get_cost_metric).
        k: numero totale di cluster (incluso il cluster base).

    Returns:
        cluster_map: {cluster_id: [indici nodi originali]}.
        cluster_id 0 è sempre [0]. cluster_id 1..k-1 sono le partizioni su 1..N-1.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    metric = adapter.get_cost_metric()
    n = metric.shape[0]
    if k > n:
        raise ValueError(f"k ({k}) cannot exceed number of nodes ({n})")

    cluster_map: Dict[int, List[int]] = {0: [0]}

    if k == 1:
        # Tutti i nodi in un unico cluster (oltre lo 0)
        cluster_map[0] = list(range(n))
        return cluster_map

    # Nodi da clusterizzare: 1, 2, ..., n-1
    non_base = list(range(1, n))
    n_non = len(non_base)
    n_clusters_rest = k - 1  # k-1 cluster oltre lo 0

    if n_clusters_rest >= n_non:
        # Un nodo per cluster (eccetto 0)
        for i, node in enumerate(non_base):
            cluster_map[i + 1] = [node]
        # Eventuali cluster vuoti (k-1 > n_non) non li usiamo
        return cluster_map

    if AgglomerativeClustering is None:
        raise ImportError("partition requires sklearn: pip install scikit-learn")

    # Sottomatrice distanze tra nodi non-base
    ix = non_base
    sub_metric = metric[np.ix_(ix, ix)]
    # sklearn richiede matrice simmetrica; sostituire inf con valore grande
    big = np.nanmax(sub_metric[np.isfinite(sub_metric)]) * 2 + 1 if np.any(np.isfinite(sub_metric)) else 1.0
    sub_metric = np.where(np.isfinite(sub_metric), sub_metric, big)

    model = AgglomerativeClustering(
        n_clusters=n_clusters_rest,
        metric="precomputed",
        linkage="average",
    )
    model.fit(sub_metric)
    labels = model.labels_

    for i in range(n_clusters_rest):
        cluster_map[i + 1] = []
    for idx, node in enumerate(non_base):
        cid = int(labels[idx])
        cluster_map[cid + 1].append(node)

    return cluster_map
