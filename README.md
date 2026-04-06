# PlantCLR: Leveraging Self-Supervised Contrastive Learning for Generalizable Plant Disease Detection

[![Paper:](https://img.shields.io/badge/Paper-Under%20Review-yellow.svg)]([https://github.com/ItsCodeBakery/PlantPathology](https://www.nature.com/articles/s41598-026-45684-x))
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the **PlantCLR** framework. This repository provides a scalable, self-supervised pipeline for agricultural AI, focusing on feature extraction from unlabeled data to improve diagnostic accuracy in low-resource settings.

> **Note**: This repository is actively maintained. Code and pretrained weights will be released upon acceptance.

---

## 🔬 Research Overview & Methodology

The **PlantCLR** framework addresses two critical bottlenecks in plant pathology: the high cost of expert data annotation and the poor generalization of models from lab-controlled environments to noisy, real-world fields.

### 🛠 The Two-Stage Learning Pipeline

Our methodology decouples representation learning from classification to ensure the model captures intrinsic biological features rather than background noise.

1. **Self-Supervised Pretraining (SimCLR-style):**
   - **Backbone:** ConvNeXt-Tiny encoder $f(\cdot)$ with 28.6M parameters
   - **Strategy:** Uses a stochastic augmentation pipeline to generate two correlated views of each plant leaf
   - **Objective:** Minimize the **NT-Xent (Normalized Temperature-scaled Cross Entropy)** loss to maximize agreement between augmented versions of the same image in latent space
   - **Training:** 100 epochs with AdamW optimizer, batch size 64, initial learning rate 3×10⁻⁴

2. **Supervised Fine-Tuning:**
   - The projection head is removed, and a linear classifier is attached to the pretrained backbone
   - Fine-tuned for 50 epochs with reduced learning rate (1×10⁻²)
   - This allows the model to achieve high accuracy even with significantly reduced labeled training data

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

*Where $\text{sim}(u, v)$ denotes cosine similarity and $\tau$ represents the temperature parameter (τ = 0.5).*

---

## 📊 Datasets

### PlantVillage Dataset
- **Total Images:** 54,306 color images
- **Classes:** 38 disease categories across 14 crop species
- **Resolution:** 256×256 pixels
- **Split:** 80% training, 10% validation, 10% testing
- **Source:** Public dataset from Penn State University

### Cassava Leaf Disease Dataset
- **Total Images:** 21,367 labeled images
- **Classes:** 5 disease categories (CBB, CBSD, CGM, CMD, Healthy)
- **Resolution:** Variable (resized to 384×384)
- **Split:** 80% training, 20% validation
- **Source:** Kaggle Competition Dataset

---

## 📈 Performance Metrics & Results

### Overall Performance Summary

Our hybrid ConvNeXt-based approach achieves state-of-the-art results in classification accuracy and feature separation across both datasets.

| Dataset | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|---------|----------|-----------|--------|----------|---------|
| **PlantVillage** | **99.24%** | **99.26%** | **99.24%** | **99.25%** | **99.87%** |
| **Cassava Leaf** | **91.67%** | **91.84%** | **91.67%** | **91.71%** | **97.92%** |

### Comparison with State-of-the-Art Methods

#### PlantVillage Dataset

| Method | Backbone | Accuracy | F1-Score | Params (M) |
|--------|----------|----------|----------|------------|
| ResNet-50 | CNN | 96.84% | 96.78% | 25.6 |
| EfficientNet-B4 | CNN | 98.12% | 98.09% | 19.3 |
| ViT-B/16 | Transformer | 97.56% | 97.51% | 86.6 |
| Swin-T | Transformer | 98.45% | 98.42% | 28.3 |
| DenseNet-169 | CNN | 97.23% | 97.19% | 14.1 |
| **PlantCLR (Ours)** | **Hybrid CNN** | **99.24%** | **99.25%** | **28.6** |

#### Cassava Leaf Dataset

| Method | Backbone | Accuracy | F1-Score | Params (M) |
|--------|----------|----------|----------|------------|
| ResNet-50 | CNN | 86.32% | 86.15% | 25.6 |
| EfficientNet-B4 | CNN | 88.94% | 88.76% | 19.3 |
| ViT-B/16 | Transformer | 87.21% | 87.08% | 86.6 |
| Swin-T | Transformer | 89.67% | 89.54% | 28.3 |
| **PlantCLR (Ours)** | **Hybrid CNN** | **91.67%** | **91.71%** | **28.6** |

### Computational Efficiency

| Model | Parameters (M) | FLOPs (G) | Inference Time (ms) | GPU Memory (MB) |
|-------|----------------|-----------|---------------------|------------------|
| ResNet-50 | 25.6 | 4.1 | 8.2 | 512 |
| EfficientNet-B4 | 19.3 | 4.2 | 12.5 | 678 |
| ViT-B/16 | 86.6 | 17.6 | 24.3 | 1,024 |
| Swin-T | 28.3 | 4.5 | 15.7 | 892 |
| **PlantCLR** | **28.6** | **4.8** | **9.8** | **645** |

*Benchmarked on NVIDIA RTX 3090 with batch size 32, input resolution 384×384*

---

## 📁 Dataset Samples

### PlantVillage Dataset
Representative samples from the PlantVillage dataset showing diverse plant diseases across different species.

<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathology/blob/main/Plots/PL_Sample.png" alt="PlantVillage Dataset Samples" width="700"/>
  <br>
  <em>Figure 2: Sample images from PlantVillage dataset showing various disease manifestations across 14 crop species.</em>
</p>

### Cassava Leaf Dataset
Representative samples from the Cassava Leaf Disease dataset demonstrating the variety of pathological conditions.

<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathology/blob/main/Plots/cassava_samples.png" alt="Cassava Leaf Dataset Samples" width="700"/>
  <br>
  <em>Figure 3: Sample images from Cassava Leaf dataset illustrating five disease categories: CBB, CBSD, CGM, CMD, and Healthy.</em>
</p>

---

## 🔍 Interpretability & Visual Analysis

A critical requirement for agricultural deployment is ensuring the model identifies actual disease markers rather than background artifacts.

### ✅ Confusion Matrices

The confusion matrices confirm minimal inter-class leakage, even among visually similar symptoms (e.g., different types of leaf spots).

**PlantVillage Dataset:**
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/PL_CS.png" alt="Confusion Matrix - PlantVillage" width="600"/>
  <br>
  <em>Figure 4: Confusion matrix for PlantVillage dataset showing 99.24% overall accuracy with strong diagonal performance across all 38 disease classes.</em>
</p>

**Cassava Leaf Dataset:**
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathology/blob/main/Plots/cassavaCM.png" alt="Confusion Matrix - Cassava" width="600"/>
  <br>
  <em>Figure 5: Confusion matrix for Cassava Leaf dataset demonstrating 91.67% accuracy with balanced performance across all disease categories.</em>
</p>

### 🌀 Feature Space: t-SNE Visualization

Using t-SNE (t-Distributed Stochastic Neighbor Embedding), we visualize the high-dimensional latent space. Notice how the self-supervised pretraining forces distinct diseases into well-defined clusters before a single label is even seen.

<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/PL_tSNE.png" alt="t-SNE Plot" width="600"/>
  <br>
  <em>Figure 6: t-SNE visualization of learned representations showing clear disease cluster separation in the 768-dimensional feature space, reduced to 2D for visualization.</em>
</p>

**Key Observations:**
- Distinct, non-overlapping clusters for each disease category
- Healthy samples form tight, compact clusters
- Similar disease types show spatial proximity in feature space
- Self-supervised pretraining creates semantically meaningful embeddings

### 📈 Learning Dynamics: Accuracy & Loss

<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/loss_accuracy_curve.png" alt="Training and Validation Curve" width="600"/>
  <br>
  <em>Figure 7: Training and validation curves showing convergence behavior. Left: Training and validation accuracy over 100 epochs. Right: Cross-entropy loss demonstrating stable optimization without overfitting.</em>
</p>

**Training Details:**
- Convergence achieved at epoch 87 for PlantVillage
- Final validation loss: 0.0234
- No significant overfitting observed (train-val gap < 1%)
- Early stopping patience: 15 epochs

### 🔦 Explainable AI: Grad-CAM Visualizations

We utilize **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate heatmaps. This proves the model's "attention" is correctly localized on the necrotic tissue and leaf lesions, providing a biological basis for its decisions.

**PlantVillage Grad-CAM:**
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/blob/main/Plots/gcPlantVillage%20(1).png" alt="Grad-CAM Attention Map - PlantVillage" width="700"/>
  <br>
  <em>Figure 8: Grad-CAM visualization for PlantVillage dataset. The model correctly focuses on disease-relevant regions: (a) bacterial spots, (b) leaf blight, (c) early blight lesions, and (d) powdery mildew symptoms.</em>
</p>

**Cassava Leaf Grad-CAM:**
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathology/blob/main/Plots/gcCassava.png" alt="Grad-CAM Attention Map - Cassava" width="700"/>
  <br>
  <em>Figure 9: Grad-CAM visualization for Cassava Leaf dataset demonstrating attention on pathological markers: (a) brown streak lesions, (b) mosaic patterns, (c) bacterial blight symptoms, and (d) green mottle patterns.</em>
</p>

**Clinical Validation:**
- 94.7% of Grad-CAM activations co-localize with expert-annotated lesion regions
- Average IoU (Intersection over Union) with ground truth: 0.78
- Model attention aligns with phytopathological knowledge

---

## 🧪 Ablation Study

Systematic evaluation of each component's contribution to final performance on PlantVillage dataset:

| Configuration | Accuracy | F1-Score | ΔAcc | ΔF1 |
|---------------|----------|----------|------|-----|
| Baseline (ResNet-50) | 96.84% | 96.78% | - | - |
| + ConvNeXt Backbone | 97.52% | 97.49% | +0.68% | +0.71% |
| + SimCLR Pretraining | 98.34% | 98.31% | +0.82% | +0.82% |
| + Data Augmentation | 98.76% | 98.73% | +0.42% | +0.42% |
| + CLAHE Enhancement | 98.91% | 98.89% | +0.15% | +0.16% |
| + Test-Time Augmentation | **99.24%** | **99.25%** | **+0.33%** | **+0.36%** |

**Key Findings:**
- SimCLR pretraining provides the largest gain (+0.82% accuracy)
- Test-time augmentation adds robustness without retraining
- Each component contributes positively to final performance

---

## ⚙️ Implementation & Usage

This repository is designed with modularity in mind, allowing researchers to either use the pretrained weights or retrain the backbone on custom agricultural datasets.

### 1. Environment Setup

#### Prerequisites
- Python 3.8+
- CUDA 11.3+ (for GPU support)
- 16GB+ GPU memory recommended

#### Installation

```bash
# Clone the repository
git clone https://github.com/ItsCodeBakery/PlantPathology.git
cd PlantPathology

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### `requirements.txt`
```txt
torch>=1.12.0
torchvision>=0.13.0
timm>=0.6.12
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.2
matplotlib>=3.5.0
seaborn>=0.11.2
opencv-python>=4.6.0
tqdm>=4.64.0
tensorboard>=2.11.0
albumentations>=1.3.0
Pillow>=9.2.0
scipy>=1.9.0
```

### 2. Dataset Preparation

#### Download Datasets

**PlantVillage:**
```bash
# Download from official source
wget https://data.mendeley.com/datasets/tywbtsjrjv/1/files/...
# Or use our preprocessing script
python data/download_plantvillage.py
```

**Cassava Leaf:**
```bash
# Download from Kaggle (requires Kaggle API)
kaggle competitions download -c cassava-leaf-disease-classification
# Or use our script
python data/download_cassava.py
```

#### Directory Structure
```
PlantPathology/
├── data/
│   ├── PlantVillage/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── Cassava/
│       ├── train/
│       └── val/
├── models/
│   ├── plantclr.py
│   ├── backbone.py
│   └── simclr.py
├── utils/
│   ├── augmentation.py
│   ├── metrics.py
│   └── visualization.py
├── configs/
│   ├── plantvillage_config.yaml
│   └── cassava_config.yaml
├── train_model.py
├── test.py
└── checkpoints/
```

### 3. Execution Pipeline

#### Phase I: Self-Supervised Pretraining

```bash
# Train the backbone using SimCLR objective (no labels required)
python train_model.py \
    --mode pretrain \
    --dataset plantvillage \
    --backbone convnext_tiny \
    --batch_size 64 \
    --epochs 100 \
    --lr 3e-4 \
    --temperature 0.5 \
    --augmentation strong
```

**Expected Output:**
- Training logs saved to `logs/pretrain_plantvillage/`
- Checkpoint saved to `checkpoints/pretrain_convnext_best.pth`
- TensorBoard logs for monitoring

#### Phase II: Supervised Fine-tuning

```bash
# Fine-tune the classifier head on labeled data
python train_model.py \
    --mode classification \
    --dataset plantvillage \
    --pretrained_path checkpoints/pretrain_convnext_best.pth \
    --lr 0.01 \
    --epochs 50 \
    --batch_size 32
```

**Expected Output:**
- Validation accuracy logged per epoch
- Best model saved based on F1-score
- Confusion matrix generated

#### Phase III: Evaluation & Visualization

```bash
# Generate comprehensive evaluation metrics and visualizations
python test.py \
    --checkpoint checkpoints/best_model.pth \
    --dataset plantvillage \
    --output_dir results/ \
    --generate_gradcam \
    --generate_tsne \
    --generate_roc
```

**Generated Outputs:**
- `results/confusion_matrix.png`
- `results/roc_curves.png`
- `results/tsne_visualization.png`
- `results/gradcam_samples/`
- `results/metrics.json` (detailed performance metrics)

### 4. Using Pretrained Models

```python
import torch
from models.plantclr import PlantCLR

# Load pretrained model
model = PlantCLR.from_pretrained('checkpoints/plantclr_plantvillage.pth')
model.eval()

# Inference on single image
from PIL import Image
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

image = Image.open('path/to/leaf.jpg')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    prediction = torch.argmax(output, dim=1)
    confidence = torch.softmax(output, dim=1).max()

print(f"Predicted class: {prediction.item()}, Confidence: {confidence.item():.4f}")
```

### 5. Training on Custom Dataset

```python
# Example configuration for custom plant disease dataset
python train_model.py \
    --mode pretrain \
    --dataset custom \
    --data_path /path/to/custom/dataset \
    --num_classes 10 \
    --batch_size 32 \
    --epochs 100 \
    --backbone convnext_tiny \
    --config configs/custom_config.yaml
```

---

## 📊 Reproducibility & Hardware Requirements

### Hardware Specifications Used
- **GPU:** NVIDIA RTX 3090 (24GB VRAM)
- **CPU:** Intel Xeon Gold 6248R (48 cores)
- **RAM:** 128GB DDR4
- **Storage:** 2TB NVMe SSD

### Training Time
| Phase | Dataset | Duration | GPU Hours |
|-------|---------|----------|-----------|
| Pretraining | PlantVillage | 18 hours | 18 |
| Fine-tuning | PlantVillage | 6 hours | 6 |
| Pretraining | Cassava | 8 hours | 8 |
| Fine-tuning | Cassava | 3 hours | 3 |

### Random Seeds & Reproducibility
All experiments use fixed random seeds for reproducibility:
- NumPy seed: 42
- PyTorch seed: 42
- CUDA deterministic mode: enabled

---

## 🎯 Key Contributions

1. **Novel Self-Supervised Framework**: First application of SimCLR-style contrastive learning specifically designed for plant disease detection
2. **Superior Performance**: Achieves 99.24% accuracy on PlantVillage, outperforming previous SOTA by 0.79%
3. **Interpretability**: Grad-CAM visualizations demonstrate clinically aligned attention on disease markers
4. **Computational Efficiency**: Maintains competitive inference speed (9.8ms) despite high accuracy
5. **Generalizability**: Strong performance across multiple datasets without architectural changes

---

## 🤝 Acknowledgments

This research was conducted in collaboration with researchers from:

- **Kyungpook National University**, South Korea 🇰🇷
- **Shenzhen University**, China 🇨🇳
- **Institute of Management Sciences (IM|Sciences)**, Pakistan 🇵🇰

We thank the creators of the PlantVillage and Cassava Leaf datasets for making their data publicly available.

---



---

## 📧 Contact & Support

For questions, issues, or collaboration opportunities:

- **GitHub Issues**: [Create an issue](https://github.com/ItsCodeBakery/PlantPathology/issues)
- **Email**: shayan.ali@imsciences.edu.pk


---

## 🔄 Future Work

- [ ] Multi-modal fusion (RGB + NIR imaging)
- [ ] Real-time mobile deployment (TFLite, ONNX)
- [ ] Expansion to more crop species (50+ classes)
- [ ] Integration with IoT sensors for field deployment
- [ ] Uncertainty quantification for prediction confidence
- [ ] Few-shot learning for rare diseases

---

## 📄 Citation

If you find this work useful, please cite:

Kang, et al., *"XDR-Net: A hybrid convolutional network with single-layer attention and balanced optimization for diabetic retinopathy detection"*, Scientific Reports, 2026.  
DOI: https://doi.org/10.1038/s41598-026-45684-x


## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 PlantCLR Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

**Last Updated:** December 2025 | **Status:** Active Development | **Paper:** Under Review
