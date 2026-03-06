# 🧠 AI Concept Discovery System

> **Self-supervised learning that discovers visual concepts from unlabeled images — no labels required.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-red.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Self-Supervised](https://img.shields.io/badge/Learning-Self--Supervised-purple.svg)]()

---

## 🎯 What It Does

The AI Concept Discovery System **learns visual concepts from images without any labels** using contrastive self-supervised learning. Given a folder of images, it automatically discovers:

| Concept Category | Examples |
|-----------------|----------|
| 🔷 **Shapes** | Circles, squares, triangles, ellipses, polygons, stars |
| 🌀 **Patterns** | Stripes, dots, grids, textures, gradients |
| 🌿 **Objects** | Organic, geometric, complex, simple structures |

The model learns by asking: *"What makes two views of the same image similar, and different from other images?"*

---

## 🏗️ Architecture

```
Unlabeled Images
       │
       ▼
┌─────────────────────────────────────────┐
│         Data Augmentation Pipeline       │
│  Random Crop │ Color Jitter │ Blur │ Flip │
└─────────────────────────────────────────┘
       │ view_1          │ view_2
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  ResNet-18  │   │  ResNet-18  │  ← Shared Encoder
│  Backbone   │   │  Backbone   │
└─────────────┘   └─────────────┘
       │                 │
       ▼                 ▼
┌─────────────┐   ┌─────────────┐
│  Projection │   │  Projection │  ← MLP Head
│    Head     │   │    Head     │
└─────────────┘   └─────────────┘
       │                 │
       └────────┬────────┘
                ▼
        NT-Xent Loss
   (Maximize positive pairs,
    Minimize negative pairs)
                │
                ▼
       Learned Embeddings
                │
                ▼
┌───────────────────────────────┐
│      K-Means Clustering       │
│  Silhouette Score Evaluation  │
└───────────────────────────────┘
                │
                ▼
     Discovered Visual Concepts
   (Shapes, Patterns, Objects...)
```

---

## 📁 Project Structure

```
ai-concept-discovery-system/
│
├── concept_discovery/              # Core package
│   ├── __init__.py                 # Package exports
│   ├── simclr_model.py             # SimCLR encoder + NT-Xent loss
│   ├── augmentations.py            # Multi-view augmentation pipelines
│   ├── concept_clusterer.py        # K-Means / DBSCAN concept clustering
│   ├── visualizer.py               # t-SNE, concept grid, similarity plots
│   ├── trainer.py                  # Self-supervised training loop
│   └── dataset.py                  # Dataset loaders (synthetic, folder, STL-10)
│
├── train.py                        # Entry point: Train SimCLR encoder
├── discover_concepts.py            # Entry point: Discover & visualize concepts
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
git clone https://github.com/PranayMahendrakar/ai-concept-discovery-system.git
cd ai-concept-discovery-system
pip install -r requirements.txt
```

### 2. Quick Demo (No Data Needed)

Run a complete self-supervised concept discovery demo with synthetic shapes:

```bash
python discover_concepts.py --demo
```

This will:
- Generate synthetic shapes (circles, squares, triangles, stars...)
- Train SimCLR encoder for 10 epochs
- Discover visual concepts via clustering
- Generate t-SNE visualization and concept grid

### 3. Train on Your Own Images

```bash
# Step 1: Train the encoder
python train.py --dataset folder \
                --data_path ./data/images \
                --epochs 100 \
                --batch_size 64

# Step 2: Discover concepts
python discover_concepts.py \
    --checkpoint outputs/checkpoints/best_model.pt \
    --dataset folder \
    --data_path ./data/images \
    --n_concepts 15
```

---

## 🔬 How Self-Supervised Learning Works

The system uses **SimCLR** (Simple Contrastive Learning of Representations):

**Step 1 — Augmentation:** For each image, create two randomly augmented versions (different crops, colors, flips, blur).

**Step 2 — Encoding:** Both views pass through the same ResNet encoder and projection head.

**Step 3 — NT-Xent Loss:** The loss maximizes agreement between the same image's two views (positive pair) and minimizes it with all other images in the batch (negative pairs).

```python
# Conceptually:
Loss = -log(
    exp(sim(z_i, z_j) / tau) /
    sum(exp(sim(z_i, z_k) / tau) for all k != i)
)
```

**Step 4 — Clustering:** After training, extract embeddings for all images and cluster them with K-Means. Each cluster = one discovered concept.

---

## 📊 Concept Discovery Output

After running `discover_concepts.py`, you get:

### t-SNE Embedding Space
Each point is an image. Colors = discovered concept clusters. Nearby points = visually similar images.

### Concept Distribution
Bar chart showing how many images belong to each discovered concept.

### Concept Similarity Matrix
Cosine similarity between concept centroids in embedding space.

### Concept Report (JSON)
```json
{
  "discovery": {
    "method": "kmeans",
    "n_concepts_found": 10,
    "quality_metrics": {
      "silhouette": 0.42,
      "davies_bouldin": 1.83
    }
  },
  "concepts": {
    "0": {"name": "concept_00", "size": 142, "fraction": 0.142},
    "1": {"name": "concept_01", "size": 98,  "fraction": 0.098},
    ...
  },
  "concept_categories": {
    "shapes":   ["circles", "squares", "triangles", "ellipses", "polygons"],
    "patterns": ["stripes", "dots", "textures", "gradients"],
    "objects":  ["organic", "geometric", "complex", "simple"]
  }
}
```

---

## ⚙️ Configuration

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backbone` | `resnet18` | Encoder: `resnet18` or `resnet50` |
| `--projection_dim` | `128` | Projection head output dim |
| `--temperature` | `0.5` | NT-Xent loss temperature |
| `--epochs` | `50` | Training epochs |
| `--batch_size` | `32` | Batch size |
| `--lr` | `3e-4` | Learning rate |
| `--warmup` | `5` | LR warmup epochs |

### Concept Discovery Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_concepts` | `10` | Number of concepts to discover |
| `--cluster_method` | `kmeans` | `kmeans`, `agglomerative`, `dbscan` |
| `--auto_k` | `False` | Auto-find optimal number of concepts |
| `--viz_method` | `tsne` | `tsne`, `umap`, or `pca` |

---

## 🧩 Supported Datasets

| Dataset | Usage |
|---------|-------|
| **Synthetic Shapes** | Built-in, no download needed. Great for testing |
| **Custom Folder** | Any folder of JPG/PNG images (unlabeled) |
| **STL-10** | Auto-downloaded, 96x96 unlabeled images |
| **CIFAR-10** | Auto-downloaded, used as unlabeled images |

---

## 🧪 Module Reference

### SimCLREncoder
```python
from concept_discovery import SimCLREncoder, NTXentLoss

model = SimCLREncoder(backbone="resnet18", projection_dim=128)
h, z = model(images)  # h=features, z=projected embeddings
```

### ConceptClusterer
```python
from concept_discovery import ConceptClusterer

clusterer = ConceptClusterer(n_concepts=10, method="kmeans")
clusterer.fit(embeddings)  # embeddings: numpy (N, D)
print(clusterer.summary())
```

### SimCLRTrainer
```python
from concept_discovery import SimCLRTrainer, get_dataloader, get_dataset

dataset = get_dataset("synthetic", n_samples=2000, image_size=64, mode="train")
loader = get_dataloader(dataset, batch_size=32)

trainer = SimCLRTrainer(num_epochs=100)
history = trainer.train(loader)
embeddings = trainer.extract_embeddings(eval_loader)
```

---

## 📈 Key Concepts

**Self-Supervised Learning:** The model learns representations without any human-annotated labels. Supervision comes from the structure of the data itself.

**Contrastive Learning:** The model is trained to bring together representations of different views of the same image (positive pairs) while pushing apart representations of different images (negative pairs).

**Concept Discovery:** After learning rich representations, clustering reveals naturally occurring visual concepts in the data — shapes, patterns, and objects emerge automatically.

**t-SNE Visualization:** After training, the high-dimensional embedding space is projected to 2D using t-SNE, showing how concepts form clusters.

---

## 📚 References

- **SimCLR:** Chen et al., "A Simple Framework for Contrastive Learning of Visual Representations", ICML 2020
- **NT-Xent Loss:** Normalized Temperature-scaled Cross Entropy Loss
- **t-SNE:** van der Maaten & Hinton, "Visualizing Data using t-SNE", JMLR 2008
- **BYOL:** Grill et al., "Bootstrap Your Own Latent", NeurIPS 2020
- **SwAV:** Caron et al., "Unsupervised Learning of Visual Features by Contrasting Cluster Assignments", NeurIPS 2020

---

## 🤝 Author

**Pranay M Mahendrakar** | [SONYTECH](https://sonytech.in/pranay/)

> *"The model sees without being told what to look for — and still learns to see."*

---

*Built as part of the AI Systems Series — #15 AI Concept Discovery System*
