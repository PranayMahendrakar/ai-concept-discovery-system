"""
discover_concepts.py - Concept Discovery Pipeline
==================================================
After training the SimCLR encoder, this script:
1. Extracts embeddings from all images
2. Clusters them to discover visual concepts
3. Generates visualizations (t-SNE, concept grid, similarity matrix)
4. Exports concept report

Usage:
    python discover_concepts.py --checkpoint outputs/checkpoints/best_model.pt
    python discover_concepts.py --checkpoint outputs/checkpoints/best_model.pt --n_concepts 15
    python discover_concepts.py --demo  # Run full demo with synthetic data
"""

import argparse
import os
import json
import time
import numpy as np
import torch

from concept_discovery.dataset import get_dataset, get_dataloader, SyntheticShapeDataset
from concept_discovery.trainer import SimCLRTrainer
from concept_discovery.concept_clusterer import ConceptClusterer
from concept_discovery.visualizer import ConceptVisualizer
from concept_discovery.augmentations import WeakAugmentation


def parse_args():
    parser = argparse.ArgumentParser(description="Discover visual concepts from unlabeled images")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained SimCLR checkpoint")
    parser.add_argument("--dataset", type=str, default="synthetic",
                        choices=["synthetic", "folder", "stl10", "cifar10"])
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to image folder")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Number of samples (synthetic dataset)")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Input image size")
    parser.add_argument("--n_concepts", type=int, default=10,
                        help="Number of concepts to discover")
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        choices=["kmeans", "agglomerative", "dbscan"],
                        help="Clustering algorithm")
    parser.add_argument("--auto_k", action="store_true",
                        help="Automatically find optimal number of concepts")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for results")
    parser.add_argument("--viz_method", type=str, default="tsne",
                        choices=["tsne", "umap", "pca"],
                        help="Dimensionality reduction for visualization")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of sample images per concept")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick demo with synthetic data (no checkpoint needed)")
    return parser.parse_args()


def run_demo(args):
    """Quick end-to-end demo: train SimCLR on synthetic shapes + discover concepts."""
    print("  Running DEMO mode with synthetic shape dataset...")
    from concept_discovery.trainer import SimCLRTrainer

    # Train a tiny model for demo
    demo_ds = get_dataset("synthetic", n_samples=500, image_size=64, mode="train")
    demo_loader = get_dataloader(demo_ds, batch_size=16)

    trainer = SimCLRTrainer(
        backbone="resnet18", projection_dim=128, temperature=0.5,
        learning_rate=3e-4, num_epochs=10, warmup_epochs=2,
        image_size=64,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        log_dir=os.path.join(args.output_dir, "logs"),
        device=args.device,
    )
    trainer.train(demo_loader)
    return trainer


def main():
    args = parse_args()

    print("=" * 60)
    print("  AI Concept Discovery System - Concept Discovery")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # ===========================================================
    # 1. Load Trainer / Model
    # ===========================================================
    print("\n[1/5] Loading model...")

    if args.demo:
        trainer = run_demo(args)
    else:
        trainer = SimCLRTrainer(
            backbone="resnet18",
            projection_dim=128,
            checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
            log_dir=os.path.join(args.output_dir, "logs"),
            device=args.device,
            num_epochs=1,  # not training, just loading
        )
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            print("  WARNING: No checkpoint provided. Using random weights.")
            print("  For meaningful concepts, train first with: python train.py")

    model = trainer.get_model()
    model.eval()
    print(f"  Model loaded. Feature dim: {model.feature_dim}")

    # ===========================================================
    # 2. Extract Embeddings
    # ===========================================================
    print("\n[2/5] Extracting embeddings...")
    t0 = time.time()

    # Eval dataset (no augmentation - just resize + normalize)
    eval_ds = get_dataset(
        dataset_type=args.dataset,
        root=args.data_path,
        image_size=args.image_size,
        n_samples=args.n_samples,
        mode="eval",
    )
    eval_loader = get_dataloader(eval_ds, batch_size=64, shuffle=False)

    embeddings = trainer.extract_embeddings(eval_loader)
    embeddings_np = embeddings.numpy()
    print(f"  Extracted {embeddings_np.shape[0]} embeddings of dim {embeddings_np.shape[1]}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Save embeddings
    emb_path = os.path.join(args.output_dir, "embeddings.npy")
    np.save(emb_path, embeddings_np)
    print(f"  Saved: {emb_path}")

    # ===========================================================
    # 3. Cluster Embeddings -> Discover Concepts
    # ===========================================================
    print("\n[3/5] Discovering concepts via clustering...")

    clusterer = ConceptClusterer(
        n_concepts=args.n_concepts,
        method=args.cluster_method,
        normalize_embeddings=True,
    )

    if args.auto_k:
        print("  Searching for optimal number of concepts...")
        best_k, scores = clusterer.find_optimal_k(embeddings_np, k_range=range(3, min(20, len(embeddings_np)//10)))
        print(f"  Optimal k: {best_k}")
        clusterer.n_concepts = best_k

    clusterer.fit(embeddings_np)
    print(clusterer.summary())

    # ===========================================================
    # 4. Visualize Concepts
    # ===========================================================
    print("\n[4/5] Generating visualizations...")
    viz = ConceptVisualizer(output_dir=plot_dir)

    # t-SNE / UMAP / PCA scatter
    print("  Plotting embedding space...")
    emb_plot = viz.plot_embeddings(
        embeddings_np,
        clusterer.labels_,
        concept_names=clusterer.concept_names_,
        method=args.viz_method,
        title=f"Discovered Concepts ({args.cluster_method.upper()}, k={clusterer.metrics_['n_clusters']})",
    )

    # Concept distribution
    print("  Plotting concept distribution...")
    dist_plot = viz.plot_concept_distribution(clusterer.labels_, clusterer.concept_names_)

    # Similarity matrix (if cluster centers available)
    if clusterer.cluster_centers_ is not None:
        print("  Plotting concept similarity matrix...")
        sim_plot = viz.plot_concept_similarity(clusterer.cluster_centers_, clusterer.concept_names_)

    # Concept grid (if we have images from synthetic dataset)
    if hasattr(eval_ds, 'metadata'):
        print("  Generating concept sample grid...")
        cluster_samples = clusterer.get_cluster_samples(embeddings_np, top_k=args.top_k)
        grid_plot = viz.plot_concept_grid(cluster_samples, title="Discovered Concept Clusters")

    # ===========================================================
    # 5. Export Concept Report
    # ===========================================================
    print("\n[5/5] Exporting concept report...")

    unique_labels = np.unique(clusterer.labels_[clusterer.labels_ >= 0])
    concepts_report = {
        "model": {
            "backbone": "resnet18",
            "checkpoint": args.checkpoint,
            "feature_dim": int(model.feature_dim),
        },
        "dataset": {
            "type": args.dataset,
            "n_samples": int(embeddings_np.shape[0]),
        },
        "discovery": {
            "method": args.cluster_method,
            "n_concepts_requested": args.n_concepts,
            "n_concepts_found": int(clusterer.metrics_["n_clusters"]),
            "quality_metrics": clusterer.metrics_,
        },
        "concepts": {
            str(cid): {
                "name": clusterer.concept_names_[cid] if cid < len(clusterer.concept_names_) else f"concept_{cid}",
                "size": int(np.sum(clusterer.labels_ == cid)),
                "fraction": float(np.mean(clusterer.labels_ == cid)),
            }
            for cid in unique_labels
        },
        "concept_categories": {
            "shapes": ["circles", "squares", "triangles", "ellipses", "polygons", "stars"],
            "patterns": ["stripes", "dots", "textures", "gradients", "noise"],
            "objects": ["organic", "geometric", "complex", "simple", "structured"],
        },
        "plots": {
            "embedding_space": emb_plot,
            "concept_distribution": dist_plot,
        }
    }

    report_path = os.path.join(args.output_dir, "concept_report.json")
    with open(report_path, "w") as f:
        json.dump(concepts_report, f, indent=2)

    print()
    print("=" * 60)
    print("  Concept Discovery Complete!")
    print(f"  Concepts found:   {clusterer.metrics_['n_clusters']}")
    if "silhouette" in clusterer.metrics_:
        print(f"  Silhouette score: {clusterer.metrics_['silhouette']:.4f}")
    print(f"  Embedding plot:   {emb_plot}")
    print(f"  Report:           {report_path}")
    print()
    print("  Discovered Concept Summary:")
    for cid in unique_labels[:10]:
        name = clusterer.concept_names_[cid] if cid < len(clusterer.concept_names_) else f"concept_{cid}"
        count = np.sum(clusterer.labels_ == cid)
        pct = 100 * count / len(embeddings_np)
        print(f"    Concept {cid:2d} ({name}): {count:4d} images ({pct:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    main()
