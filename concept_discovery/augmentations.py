"""
Augmentation Pipelines for Self-Supervised Concept Discovery
=============================================================
Strong random augmentations create multiple views of the same image,
enabling the model to learn invariant representations.
"""

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import random


class SimCLRAugmentation:
    """
    SimCLR augmentation pipeline.
    Applies two independently randomized transformations to each image,
    creating a positive pair for contrastive learning.

    Key augmentations (from SimCLR paper):
        1. Random Crop + Resize
        2. Random Horizontal Flip
        3. Color Jitter (brightness, contrast, saturation, hue)
        4. Random Grayscale
        5. Gaussian Blur
        6. Normalization
    """
    def __init__(
        self,
        image_size: int = 64,
        s: float = 1.0,
        gaussian_blur: bool = True,
    ):
        color_jitter = T.ColorJitter(
            brightness=0.8 * s,
            contrast=0.8 * s,
            saturation=0.8 * s,
            hue=0.2 * s,
        )
        transforms_list = [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([color_jitter], p=0.8),
            T.RandomGrayscale(p=0.2),
        ]
        if gaussian_blur:
            kernel_size = max(3, int(0.1 * image_size) | 1)
            transforms_list.append(
                T.RandomApply([T.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5)
            )
        transforms_list += [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        self.transform = T.Compose(transforms_list)

    def __call__(self, img):
        """Generate two augmented views of the same image."""
        return self.transform(img), self.transform(img)


class WeakAugmentation:
    """
    Weak augmentation for evaluation / feature extraction.
    Only resize + normalize, no random transforms.
    """
    def __init__(self, image_size: int = 64):
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img):
        return self.transform(img)


class MultiViewAugmentation:
    """
    Multi-crop augmentation strategy (from SwAV / DINO).
    Generates N global views + M local views for multi-scale concept learning.
    """
    def __init__(
        self,
        image_size: int = 64,
        n_global_views: int = 2,
        n_local_views: int = 4,
        local_scale: tuple = (0.05, 0.4),
        global_scale: tuple = (0.4, 1.0),
    ):
        self.n_global = n_global_views
        self.n_local = n_local_views

        base_transforms = [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        self.global_transform = T.Compose(
            [T.RandomResizedCrop(image_size, scale=global_scale)] + base_transforms
        )
        self.local_transform = T.Compose(
            [T.RandomResizedCrop(image_size // 2, scale=local_scale)] + base_transforms
        )

    def __call__(self, img):
        """Returns list of multi-scale views."""
        views = []
        for _ in range(self.n_global):
            views.append(self.global_transform(img))
        for _ in range(self.n_local):
            views.append(self.local_transform(img))
        return views


def denormalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Reverse normalization for visualization."""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean


if __name__ == "__main__":
    import numpy as np
    print("=== Augmentation Pipeline Test ===")

    # Create a dummy RGB image
    dummy_img = Image.fromarray(np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8))

    # Test SimCLR augmentation
    aug = SimCLRAugmentation(image_size=64)
    view1, view2 = aug(dummy_img)
    print(f"SimCLR view1 shape: {view1.shape}")
    print(f"SimCLR view2 shape: {view2.shape}")
    print(f"Views are different: {not torch.equal(view1, view2)}")

    # Test multi-view
    multi_aug = MultiViewAugmentation(image_size=64, n_global_views=2, n_local_views=4)
    views = multi_aug(dummy_img)
    print(f"Multi-view: {len(views)} views, global={views[0].shape}, local={views[2].shape}")
    print("Augmentation Pipeline OK!")
