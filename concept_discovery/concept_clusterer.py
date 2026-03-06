"""
Concept Clusterer
=================
After extracting embeddings with SimCLR, this module clusters them
to discover latent concepts: shapes, patterns, objects, textures.
"""

import numpy as np
import torch
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


CONCEPT_LABELS = {
    "auto": "auto-discovered",
    "shape": ["circle", "square", "triangle", "rectangle", "polygon", "ellipse"],
    "pattern": ["stripes", "dots", "grid", "texture", "gradient", "noise"],
    "object": ["organic", "geometric", "complex", "simple", "structured", "random"],
}


class ConceptClusterer:
    """
    Discovers visual concepts from embeddings using unsupervised clustering.

    Pipeline:
        1. Extract embeddings from SimCLR encoder
        2. Optionally reduce dimensions (PCA / UMAP)
        3. Cluster embeddings (KMeans / DBSCAN / Agglomerative)
        4. Assign human-readable concept labels
        5. Score cluster quality
    """

    def __init__(
        self,
        n_concepts: int = 10,
        method: str = "kmeans",
        normalize_embeddings: bool = True,
        random_state: int = 42,
    ):
        """
        Args:
            n_concepts:            Number of concepts to discover
            method:                Clustering algorithm: 'kmeans', 'dbscan', 'agglomerative'
            normalize_embeddings:  L2-normalize before clustering
            random_state:          Random seed for reproducibility
        """
        self.n_concepts = n_concepts
        self.method = method
        self.normalize = normalize_embeddings
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.concept_names_ = []
        self.metrics_ = {}

    def fit(self, embeddings: np.ndarray) -> "ConceptClusterer":
        """
        Fit the clusterer on embeddings.

        Args:
            embeddings: Array of shape (N, D) - extracted feature vectors

        Returns:
            self
        """
        X = embeddings.copy()
        if self.normalize:
            X = normalize(X, norm="l2")

        if self.method == "kmeans":
            clusterer = KMeans(
                n_clusters=self.n_concepts,
                random_state=self.random_state,
                n_init=10,
                max_iter=300,
            )
        elif self.method == "agglomerative":
            clusterer = AgglomerativeClustering(n_clusters=self.n_concepts, linkage="ward")
        elif self.method == "dbscan":
            clusterer = DBSCAN(eps=0.5, min_samples=5, metric="cosine")
        else:
            raise ValueError(f"Unknown method: {self.method}. Choose kmeans/agglomerative/dbscan")

        self.labels_ = clusterer.fit_predict(X)

        if hasattr(clusterer, "cluster_centers_"):
            self.cluster_centers_ = clusterer.cluster_centers_

        # Compute quality metrics
        unique_labels = np.unique(self.labels_[self.labels_ >= 0])
        if len(unique_labels) > 1 and len(X) > len(unique_labels):
            self.metrics_["silhouette"] = round(float(silhouette_score(X, self.labels_)), 4)
            self.metrics_["davies_bouldin"] = round(float(davies_bouldin_score(X, self.labels_)), 4)

        self.metrics_["n_clusters"] = len(unique_labels)
        self.metrics_["n_noise"] = int(np.sum(self.labels_ == -1))

        # Auto-assign concept names
        self.concept_names_ = [f"concept_{i:02d}" for i in range(len(unique_labels))]

        return self

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """Assign cluster labels to new embeddings."""
        X = normalize(embeddings, norm="l2") if self.normalize else embeddings.copy()
        if self.cluster_centers_ is not None:
            # KMeans-style: nearest center
            dists = np.linalg.norm(X[:, None, :] - self.cluster_centers_[None, :, :], axis=2)
            return np.argmin(dists, axis=1)
        raise RuntimeError("Clusterer not fitted or method does not support predict()")

    def get_cluster_samples(
        self, embeddings: np.ndarray, images: Optional[List] = None, top_k: int = 5
    ) -> Dict:
        """
        Return top-k representative samples per concept cluster.

        Args:
            embeddings: (N, D) feature array
            images:     Optional list of N images corresponding to embeddings
            top_k:      Number of samples per cluster to return

        Returns:
            dict mapping cluster_id -> {"indices": [...], "images": [...]}
        """
        if self.labels_ is None:
            raise RuntimeError("Call fit() first")

        cluster_dict = {}
        X = normalize(embeddings, norm="l2") if self.normalize else embeddings.copy()

        for cluster_id in np.unique(self.labels_):
            if cluster_id == -1:
                continue
            mask = self.labels_ == cluster_id
            indices = np.where(mask)[0]

            # Sort by distance to center if available
            if self.cluster_centers_ is not None:
                center = self.cluster_centers_[cluster_id]
                dists = np.linalg.norm(X[indices] - center, axis=1)
                sorted_idx = indices[np.argsort(dists)][:top_k]
            else:
                sorted_idx = indices[:top_k]

            cluster_dict[cluster_id] = {
                "name": self.concept_names_[cluster_id] if cluster_id < len(self.concept_names_) else f"concept_{cluster_id}",
                "size": int(mask.sum()),
                "indices": sorted_idx.tolist(),
                "images": [images[i] for i in sorted_idx] if images else [],
            }
        return cluster_dict

    def find_optimal_k(
        self, embeddings: np.ndarray, k_range: range = range(3, 20)
    ) -> Tuple[int, List[float]]:
        """
        Find optimal number of clusters using elbow method (inertia) + silhouette.

        Args:
            embeddings: (N, D) feature array
            k_range:    Range of k values to try

        Returns:
            best_k, list of silhouette scores
        """
        X = normalize(embeddings, norm="l2") if self.normalize else embeddings.copy()
        scores = []
        inertias = []

        for k in k_range:
            if k >= len(X):
                break
            km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X)
            if len(np.unique(labels)) > 1:
                s = silhouette_score(X, labels)
                scores.append(s)
                inertias.append(km.inertia_)
            else:
                scores.append(-1)
                inertias.append(float("inf"))

        best_k = list(k_range)[int(np.argmax(scores))]
        return best_k, scores

    def summary(self) -> str:
        """Return a text summary of discovered concepts."""
        lines = ["=== Concept Discovery Summary ==="]
        lines.append(f"Method:     {self.method}")
        lines.append(f"Concepts:   {self.metrics_.get('n_clusters', 0)}")
        if "silhouette" in self.metrics_:
            lines.append(f"Silhouette: {self.metrics_['silhouette']} (higher is better, max=1)")
        if "davies_bouldin" in self.metrics_:
            lines.append(f"Davies-Bouldin: {self.metrics_['davies_bouldin']} (lower is better)")
        if self.metrics_.get("n_noise", 0) > 0:
            lines.append(f"Noise pts:  {self.metrics_['n_noise']} (DBSCAN outliers)")
        return "\n".join(lines)


if __name__ == "__main__":
    print("=== ConceptClusterer Test ===")
    np.random.seed(42)
    # Simulate 200 embeddings from 5 natural clusters
    embeddings = np.vstack([
        np.random.randn(40, 128) + center
        for center in [np.random.randn(128) * 3 for _ in range(5)]
    ])

    clusterer = ConceptClusterer(n_concepts=5, method="kmeans")
    clusterer.fit(embeddings)
    print(clusterer.summary())

    # Find optimal k
    best_k, scores = clusterer.find_optimal_k(embeddings, k_range=range(2, 10))
    print(f"Optimal k (silhouette): {best_k}")
    print("ConceptClusterer OK!")
