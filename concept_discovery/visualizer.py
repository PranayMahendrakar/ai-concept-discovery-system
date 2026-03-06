"""
Concept Visualizer
==================
Visualize discovered concepts using t-SNE / UMAP projections,
concept grids, and similarity matrices.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import Dict, List, Optional, Tuple
import os

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

PALETTE = plt.cm.get_cmap("tab20")


class ConceptVisualizer:
    """
    Visualization toolkit for self-supervised concept discovery.

    Features:
        - t-SNE / UMAP embedding scatter plots colored by concept
        - Concept sample grid (top-K images per cluster)
        - Cosine similarity heatmap between concept centroids
        - Training loss curve
        - Concept distribution bar chart
    """

    def __init__(self, output_dir: str = "outputs/plots", figsize: tuple = (12, 8), dpi: int = 150):
        self.output_dir = output_dir
        self.figsize = figsize
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Embedding Scatter Plot (t-SNE / UMAP / PCA)
    # ------------------------------------------------------------------
    def plot_embeddings(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        concept_names: Optional[List[str]] = None,
        method: str = "tsne",
        title: str = "Concept Embedding Space",
        save_path: Optional[str] = None,
    ) -> str:
        """
        2-D projection of high-dimensional embeddings colored by cluster.

        Args:
            embeddings:    (N, D) feature array
            labels:        (N,) cluster labels
            concept_names: Optional list of concept name strings
            method:        'tsne', 'umap', or 'pca'
            title:         Plot title
            save_path:     Output path (auto-generated if None)

        Returns:
            Path to saved figure
        """
        print(f"  Computing {method.upper()} projection...")
        if method == "tsne":
            proj = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            coords = proj.fit_transform(embeddings)
        elif method == "umap" and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
            coords = reducer.fit_transform(embeddings)
        else:
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        unique_labels = np.unique(labels)
        legend_elements = []

        for i, label in enumerate(unique_labels):
            mask = labels == label
            color = PALETTE(i / max(len(unique_labels), 1))
            name = concept_names[label] if concept_names and label < len(concept_names) else f"Concept {label}"
            ax.scatter(coords[mask, 0], coords[mask, 1], c=[color], s=20, alpha=0.7, label=name)
            legend_elements.append(Patch(facecolor=color, label=name))

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel(f"{method.upper()} Dim 1")
        ax.set_ylabel(f"{method.upper()} Dim 2")
        ax.legend(handles=legend_elements, loc="best", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        path = save_path or os.path.join(self.output_dir, f"embeddings_{method}.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # 2. Concept Sample Grid
    # ------------------------------------------------------------------
    def plot_concept_grid(
        self,
        cluster_samples: Dict,
        title: str = "Discovered Concepts",
        save_path: Optional[str] = None,
        image_size: int = 64,
    ) -> str:
        """
        Grid showing top-K sample images per concept cluster.

        Args:
            cluster_samples: Output from ConceptClusterer.get_cluster_samples()
            title:           Plot title
            save_path:       Output path
            image_size:      Image display size in pixels
        """
        concept_ids = sorted(cluster_samples.keys())
        n_concepts = len(concept_ids)
        if n_concepts == 0:
            raise ValueError("No clusters to visualize.")

        # Get max samples per cluster
        max_samples = max(len(v["images"]) for v in cluster_samples.values() if v["images"])
        if max_samples == 0:
            print("  No images provided in cluster_samples, generating placeholder grid.")
            max_samples = 5

        fig = plt.figure(figsize=(max_samples * 1.5, n_concepts * 1.5 + 1), dpi=self.dpi)
        gs = gridspec.GridSpec(n_concepts + 1, max_samples, figure=fig, hspace=0.4, wspace=0.1)

        # Title row
        title_ax = fig.add_subplot(gs[0, :])
        title_ax.text(0.5, 0.5, title, ha="center", va="center", fontsize=14, fontweight="bold")
        title_ax.axis("off")

        for row, cid in enumerate(concept_ids):
            info = cluster_samples[cid]
            name = info.get("name", f"Concept {cid}")
            images = info.get("images", [])

            for col in range(max_samples):
                ax = fig.add_subplot(gs[row + 1, col])
                if col < len(images):
                    img = images[col]
                    if hasattr(img, "numpy"):
                        img = img.permute(1, 2, 0).numpy()
                        img = np.clip(img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]), 0, 1)
                    ax.imshow(img)
                else:
                    ax.set_facecolor("#1a1a2e")
                if col == 0:
                    ax.set_ylabel(f"{name}\n({info['size']})", fontsize=7, rotation=0, ha="right", va="center")
                ax.set_xticks([])
                ax.set_yticks([])

        plt.suptitle("", y=1.01)
        path = save_path or os.path.join(self.output_dir, "concept_grid.png")
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # 3. Training Loss Curve
    # ------------------------------------------------------------------
    def plot_loss_curve(
        self,
        losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "SimCLR Training Loss",
        save_path: Optional[str] = None,
    ) -> str:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=self.dpi)
        epochs = range(1, len(losses) + 1)
        ax.plot(epochs, losses, "b-o", markersize=4, label="Train NT-Xent Loss", linewidth=2)
        if val_losses:
            ax.plot(epochs, val_losses, "r--s", markersize=4, label="Val Loss", linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("NT-Xent Loss")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, "loss_curve.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # 4. Concept Similarity Heatmap
    # ------------------------------------------------------------------
    def plot_concept_similarity(
        self,
        cluster_centers: np.ndarray,
        concept_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        n = len(cluster_centers)
        norms = np.linalg.norm(cluster_centers, axis=1, keepdims=True)
        centers_norm = cluster_centers / (norms + 1e-8)
        sim_matrix = centers_norm @ centers_norm.T

        fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)), dpi=self.dpi)
        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

        names = concept_names or [f"C{i}" for i in range(n)]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(names, fontsize=8)
        ax.set_title("Concept Similarity Matrix", fontsize=13, fontweight="bold")

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{sim_matrix[i,j]:.2f}", ha="center", va="center", fontsize=7, color="black")

        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, "concept_similarity.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path

    # ------------------------------------------------------------------
    # 5. Concept Distribution Bar Chart
    # ------------------------------------------------------------------
    def plot_concept_distribution(
        self,
        labels: np.ndarray,
        concept_names: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> str:
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        names = [concept_names[i] if concept_names and i < len(concept_names) else f"Concept {i}" for i in unique]
        colors = [PALETTE(i / max(len(unique), 1)) for i in range(len(unique))]

        fig, ax = plt.subplots(figsize=(max(8, len(unique) * 0.8), 5), dpi=self.dpi)
        bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=0.5)

        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(count), ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Discovered Concepts")
        ax.set_ylabel("Number of Images")
        ax.set_title("Concept Distribution", fontsize=13, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        path = save_path or os.path.join(self.output_dir, "concept_distribution.png")
        fig.savefig(path)
        plt.close(fig)
        print(f"  Saved: {path}")
        return path
