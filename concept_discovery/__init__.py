"""
AI Concept Discovery System
============================
Self-supervised learning framework that discovers visual concepts
from unlabeled images using SimCLR contrastive learning.

Discovered concept categories:
    - Shapes: circles, squares, triangles, ellipses, polygons
    - Patterns: stripes, dots, textures, gradients
    - Objects: organic, geometric, complex, simple structures

Example usage:
    from concept_discovery import ConceptDiscoveryPipeline
    pipeline = ConceptDiscoveryPipeline(n_concepts=10)
    pipeline.fit(image_folder="data/images/")
    concepts = pipeline.get_concepts()
    pipeline.visualize()
"""

from .simclr_model import SimCLREncoder, NTXentLoss, ProjectionHead
from .augmentations import SimCLRAugmentation, WeakAugmentation, MultiViewAugmentation
from .concept_clusterer import ConceptClusterer
from .visualizer import ConceptVisualizer
from .trainer import SimCLRTrainer
from .dataset import SyntheticShapeDataset, UnlabeledImageDataset, get_dataset, get_dataloader

__version__ = "1.0.0"
__author__ = "AI Concept Discovery System"
__description__ = "Self-supervised learning for automatic visual concept discovery"

__all__ = [
    "SimCLREncoder",
    "NTXentLoss",
    "ProjectionHead",
    "SimCLRAugmentation",
    "WeakAugmentation",
    "MultiViewAugmentation",
    "ConceptClusterer",
    "ConceptVisualizer",
    "SimCLRTrainer",
    "SyntheticShapeDataset",
    "UnlabeledImageDataset",
    "get_dataset",
    "get_dataloader",
]
