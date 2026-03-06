"""
Dataset Module for Self-Supervised Concept Discovery
=====================================================
Supports unlabeled image folders, STL-10, CIFAR-10, and synthetic shape datasets.
"""

import os
import glob
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dsets
import torchvision.transforms as T
from typing import Optional, Callable, Tuple, List

from .augmentations import SimCLRAugmentation, WeakAugmentation


# ================================================================
# 1. Unlabeled Image Folder Dataset
# ================================================================
class UnlabeledImageDataset(Dataset):
    """
    Dataset for loading images from a folder WITHOUT labels.
    Returns two augmented views for SimCLR training.

    Folder structure:
        data/
            images/
                img001.jpg
                img002.png
                ...
    """
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    def __init__(
        self,
        root: str,
        image_size: int = 64,
        mode: str = "train",
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            root:       Root directory containing images
            image_size: Resize all images to this size
            mode:       'train' (returns 2 augmented views) or 'eval' (returns 1 weak view)
            transform:  Custom transform (overrides default)
        """
        self.root = root
        self.mode = mode
        self.image_paths = self._scan_images(root)

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {root}")

        print(f"  Found {len(self.image_paths)} images in {root}")

        if transform is not None:
            self.transform = transform
        elif mode == "train":
            self.transform = SimCLRAugmentation(image_size=image_size)
        else:
            self.transform = WeakAugmentation(image_size=image_size)

    def _scan_images(self, root: str) -> List[str]:
        paths = []
        for ext in self.EXTENSIONS:
            paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
            paths.extend(glob.glob(os.path.join(root, "**", f"*{ext.upper()}"), recursive=True))
        return sorted(set(paths))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        path = self.image_paths[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load {path}: {e}")
            img = Image.new("RGB", (64, 64), color=(128, 128, 128))

        if self.mode == "train":
            view1, view2 = self.transform(img)
            return view1, view2
        return self.transform(img), path


# ================================================================
# 2. Synthetic Shape Dataset (for testing/demo)
# ================================================================
class SyntheticShapeDataset(Dataset):
    """
    Synthetic dataset with procedurally generated shapes.
    Useful for testing concept discovery without real images.

    Generates: circles, squares, triangles, ellipses, pentagons
    with random: colors, sizes, positions, backgrounds
    """
    SHAPES = ["circle", "square", "triangle", "ellipse", "pentagon", "hexagon", "star"]
    PATTERNS = ["solid", "striped", "dotted", "gradient"]

    def __init__(
        self,
        n_samples: int = 1000,
        image_size: int = 64,
        mode: str = "train",
        n_concepts: int = 7,
        seed: int = 42,
    ):
        self.n_samples = n_samples
        self.image_size = image_size
        self.mode = mode
        self.seed = seed
        self.n_concepts = min(n_concepts, len(self.SHAPES))
        random.seed(seed)
        np.random.seed(seed)

        # Generate metadata for each sample
        self.metadata = [
            {
                "shape": self.SHAPES[i % self.n_concepts],
                "color": tuple(np.random.randint(30, 230, 3).tolist()),
                "bg_color": tuple(np.random.randint(200, 255, 3).tolist()),
                "size": np.random.randint(10, image_size // 2),
                "x": np.random.randint(5, image_size - 5),
                "y": np.random.randint(5, image_size - 5),
            }
            for i in range(n_samples)
        ]

        self.aug = SimCLRAugmentation(image_size=image_size)
        self.eval_transform = WeakAugmentation(image_size=image_size)

    def _draw_shape(self, meta: dict) -> Image.Image:
        """Draw a single shape based on metadata."""
        img = Image.new("RGB", (self.image_size, self.image_size), color=meta["bg_color"])
        draw = ImageDraw.Draw(img)
        cx, cy = meta["x"], meta["y"]
        r = meta["size"]
        color = meta["color"]

        shape = meta["shape"]
        if shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=color)
        elif shape == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline=color)
        elif shape == "ellipse":
            draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2], fill=color)
        elif shape == "triangle":
            pts = [(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)]
            draw.polygon(pts, fill=color)
        elif shape == "pentagon":
            import math
            pts = [(cx + r * math.cos(2 * math.pi * k / 5 - math.pi / 2),
                    cy + r * math.sin(2 * math.pi * k / 5 - math.pi / 2)) for k in range(5)]
            draw.polygon(pts, fill=color)
        elif shape == "hexagon":
            import math
            pts = [(cx + r * math.cos(2 * math.pi * k / 6),
                    cy + r * math.sin(2 * math.pi * k / 6)) for k in range(6)]
            draw.polygon(pts, fill=color)
        elif shape == "star":
            import math
            pts = []
            for k in range(10):
                rad = r if k % 2 == 0 else r // 2
                angle = math.pi * k / 5 - math.pi / 2
                pts.append((cx + rad * math.cos(angle), cy + rad * math.sin(angle)))
            draw.polygon(pts, fill=color)
        return img

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        meta = self.metadata[idx]
        img = self._draw_shape(meta)
        shape_label = self.SHAPES.index(meta["shape"])

        if self.mode == "train":
            view1, view2 = self.aug(img)
            return view1, view2
        return self.eval_transform(img), shape_label


# ================================================================
# 3. Dataset Factory
# ================================================================
def get_dataset(
    dataset_type: str = "synthetic",
    root: Optional[str] = None,
    image_size: int = 64,
    n_samples: int = 2000,
    mode: str = "train",
) -> Dataset:
    """
    Factory function to get a dataset.

    Args:
        dataset_type: 'synthetic', 'folder', 'stl10', 'cifar10'
        root:         Path for folder dataset or download location
        image_size:   Image resize target
        n_samples:    Number of samples for synthetic datasets
        mode:         'train' or 'eval'

    Returns:
        Dataset instance
    """
    if dataset_type == "synthetic":
        return SyntheticShapeDataset(n_samples=n_samples, image_size=image_size, mode=mode)

    elif dataset_type == "folder":
        assert root is not None, "root must be provided for 'folder' dataset type"
        return UnlabeledImageDataset(root=root, image_size=image_size, mode=mode)

    elif dataset_type == "stl10":
        assert root is not None, "root must be provided for stl10"
        aug = SimCLRAugmentation(image_size=image_size)
        class STL10Pair(Dataset):
            def __init__(self, split="unlabeled"):
                self.base = dsets.STL10(root=root, split=split, download=True)
                self.aug = aug
                self.eval_t = WeakAugmentation(image_size)
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                img, label = self.base[i]
                if mode == "train": return self.aug(img)
                return self.eval_t(img), label
        return STL10Pair()

    elif dataset_type == "cifar10":
        assert root is not None, "root must be provided for cifar10"
        aug = SimCLRAugmentation(image_size=image_size)
        class CIFAR10Pair(Dataset):
            def __init__(self):
                self.base = dsets.CIFAR10(root=root, train=(mode=="train"), download=True)
                self.aug = aug
                self.eval_t = WeakAugmentation(image_size)
            def __len__(self): return len(self.base)
            def __getitem__(self, i):
                img, label = self.base[i]
                if mode == "train": return self.aug(img)
                return self.eval_t(img), label
        return CIFAR10Pair()

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")


def get_dataloader(dataset: Dataset, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader from a dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )


if __name__ == "__main__":
    print("=== Dataset Test ===")
    ds = SyntheticShapeDataset(n_samples=100, image_size=64, mode="train")
    print(f"Synthetic dataset: {len(ds)} samples")
    v1, v2 = ds[0]
    print(f"View shapes: {v1.shape}, {v2.shape}")

    ds_eval = SyntheticShapeDataset(n_samples=50, image_size=64, mode="eval")
    img, label = ds_eval[0]
    print(f"Eval sample: shape={img.shape}, label={label} ({SyntheticShapeDataset.SHAPES[label]})")

    loader = get_dataloader(ds, batch_size=8)
    batch = next(iter(loader))
    print(f"Batch: {len(batch)} views, each {batch[0].shape}")
    print("Dataset OK!")
