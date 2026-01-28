"""Task ontology analysis: clustering and similarity."""

import numpy as np
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import torch


def compute_similarity_matrix(
    task_vectors: dict[str, torch.Tensor],
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise cosine similarity between task vectors.

    Returns:
        (similarity_matrix, task_names)
    """
    names = sorted(task_vectors.keys())
    vecs = np.stack([task_vectors[n].numpy() for n in names])

    # Cosine similarity = 1 - cosine distance
    dists = pdist(vecs, metric="cosine")
    sim_matrix = 1.0 - squareform(dists)

    return sim_matrix, names


def hierarchical_clustering(
    task_vectors: dict[str, torch.Tensor],
    method: str = "ward",
) -> tuple[np.ndarray, list[str]]:
    """Perform hierarchical clustering on task vectors.

    Returns:
        (linkage_matrix, task_names)
    """
    names = sorted(task_vectors.keys())
    vecs = np.stack([task_vectors[n].numpy() for n in names])

    if method == "ward":
        Z = linkage(vecs, method="ward")
    else:
        dists = pdist(vecs, metric="cosine")
        Z = linkage(dists, method=method)

    return Z, names


def compute_regime_clustering_score(
    task_vectors: dict[str, torch.Tensor],
    task_regimes: dict[str, str],
) -> dict:
    """Test whether task vectors cluster by regime.

    Returns:
        silhouette_score and permutation test p-value.
    """
    names = sorted(task_vectors.keys())
    vecs = np.stack([task_vectors[n].numpy() for n in names])
    labels = [task_regimes[n] for n in names]

    # Need at least 2 clusters with at least 2 members each
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        return {"silhouette_score": 0.0, "p_value": 1.0}

    label_ints = [sorted(unique_labels).index(l) for l in labels]

    sil = silhouette_score(vecs, label_ints, metric="cosine")

    # Permutation test
    rng = np.random.RandomState(42)
    n_perms = 1000
    perm_scores = []
    for _ in range(n_perms):
        perm_labels = rng.permutation(label_ints)
        try:
            perm_sil = silhouette_score(vecs, perm_labels, metric="cosine")
            perm_scores.append(perm_sil)
        except ValueError:
            continue

    p_value = float(np.mean([s >= sil for s in perm_scores])) if perm_scores else 1.0

    return {
        "silhouette_score": float(sil),
        "p_value": p_value,
        "n_permutations": len(perm_scores),
    }


def pca_embedding(
    task_vectors: dict[str, torch.Tensor],
    n_components: int = 2,
) -> tuple[np.ndarray, list[str]]:
    """PCA projection of task vectors.

    Returns:
        (embedding of shape (n_tasks, n_components), task_names)
    """
    names = sorted(task_vectors.keys())
    vecs = np.stack([task_vectors[n].numpy() for n in names])

    pca = PCA(n_components=n_components)
    embedding = pca.fit_transform(vecs)

    return embedding, names
