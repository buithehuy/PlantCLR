
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
