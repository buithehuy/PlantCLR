
# PlantCLR: Leveraging Self-Supervised Contrastive Learning for Generalizable Plant Disease Detection

[![Paper: IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access-blue.svg)](YOUR_PAPER_LINK_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the **PlantCLR** framework. This repository provides a scalable, self-supervised pipeline for agricultural AI, focusing on feature extraction from unlabelled data to improve diagnostic accuracy in low-resource settings.

---

## 🔬 Research Overview & Methodology

The **PlantCLR** framework addresses two critical bottlenecks in plant pathology: the high cost of expert data annotation and the poor generalization of models from lab-controlled environments to noisy, real-world fields.

### 🛠 The Two-Stage Learning Pipeline
Our methodology decouples representation learning from classification to ensure the model captures intrinsic biological features rather than background noise.

1. **Self-Supervised Pretraining (SimCLR-style):**
   - **Backbone:** ConvNeXt-Tiny encoder $f(\cdot)$.
   - **Strategy:** Uses a stochastic augmentation pipeline to generate two correlated views of each plant leaf.
   - **Objective:** Minimize the **NT-Xent (Normalized Temperature-scaled Cross Entropy)** loss to maximize agreement between augmented versions of the same image in latent space.

2. **Supervised Fine-Tuning:**
   - The projection head is removed, and a linear classifier is attached to the pretrained backbone.
   - This allows the model to achieve high accuracy even with significantly reduced labeled training data.

<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/CLR_Dia.png" alt="SimCLR CNN Classifier Diagram" width="700"/>
  <br>
  <em>Figure 1: The PlantCLR Architectural Workflow - Transitioning from Contrastive Pretraining to Disease Diagnosis.</em>
</p>

### 📐 Mathematical Objective
To learn generalizable features, we optimize the **NT-Xent** loss function:

$$
\ell_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}
$$

*Where $\text{sim}(u, v)$ denotes cosine similarity and $\tau$ represents the temperature parameter.*

---
## 📊 Experimental Results & Interpretability

The **PlantCLR** framework was rigorously evaluated on the PlantVillage and Cassava Leaf datasets. Our results demonstrate that self-supervised pretraining significantly enhances the model's ability to cluster diseased states and localize pathogenic lesions.

### 📈 Performance Metrics
Our hybrid ConvNeXt-based approach achieves state-of-the-art results in classification accuracy and feature separation.

| Metric | Achievement | Visualization |
| :--- | :--- | :--- |
| **Accuracy** | **99.X%** | [Training Curves](#-accuracy--loss-curve) |
| **Separability** | **High Cluster Density** | [t-SNE Embeddings](#-t-sne-visualization) |
| **Robustness** | **Optimal ROC-AUC** | [ROC Curves](#-output-visualizations) |

### 🔍 Interpretability & Visual Analysis
A critical requirement for agricultural deployment is ensuring the model identifies actual disease markers rather than background artifacts.

#### ✅ Confusion Matrix
The confusion matrix confirms minimal inter-class leakage, even among visually similar symptoms (e.g., different types of leaf spots).
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/PL_CS.png" alt="Confusion Matrix" width="600"/>
</p>

#### 🌀 Feature Space: t-SNE Visualization
Using t-SNE (t-Distributed Stochastic Neighbor Embedding), we visualize the high-dimensional latent space. Notice how the self-supervised pretraining forces distinct diseases into well-defined clusters before a single label is even seen.
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/PL_tSNE.png" alt="t-SNE Plot" width="600"/>
</p>

#### 📈 Learning Dynamics: Accuracy & Loss
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/loss_accuracy_curve.png" alt="Training and Validation Curve" width="600"/>
</p>

#### 🔦 Explainable AI: Grad-CAM
We utilize **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps. This proves the model's "attention" is correctly localized on the necrotic tissue and leaf lesions, providing a biological basis for its decisions.
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/blob/main/Plots/gcPlantVillage%20(1).png" alt="Grad-CAM Attention Map" width="600"/>
</p>

---
## ⚙️ Implementation & Usage

This repository is designed with modularity in mind, allowing researchers to either use the pretrained weights or retrain the backbone on custom agricultural datasets.

### 1. Environment Setup
To ensure reproducibility, we recommend using a virtual environment (Python 3.8+).

```bash
git clone [https://github.com/ItsCodeBakery/PlantPathology.git](https://github.com/ItsCodeBakery/PlantPathology.git)
cd PlantPathology
pip install -r requirements.txt
---
### 2. Execution Pipeline
Phase I: Self-Supervised Pretraining
