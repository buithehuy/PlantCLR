
# PlantCLR: Leveraging Self-Supervised Contrastive Learning for Generalizable Plant Disease Detection

[![Paper: IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access-blue.svg)](YOUR_PAPER_LINK_HERE)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-ee4c2c.svg)](https://pytorch.org/)

Official implementation of the **PlantCLR** framework. This repository provides a scalable, self-supervised pipeline for agricultural AI, focusing on feature extraction from unlabelled data to improve diagnostic accuracy in low-resource settings.

---

##  Project Structure

```
Code/
├── SimCLR_CNNClassifier.py         # Combined CNN model with SimCLR projection + classification head
├── data_loader.py                  # Data loading and augmentation (ImageFolder-based)
├── finetune_classifier.py          # Optional separate CNN model for classification only
├── generate_tsne.py                # t-SNE visualization of learned embeddings
├── gradcam.py                      # Grad-CAM visualization for model interpretability
├── plot_loss_accuracy_curves.py    # Training/validation loss and accuracy plots
├── plot_multiclass_roc.py          # Multi-class ROC and AUC curve plotting
├── test.py                         # Model evaluation and metric collection
├── train_model.py                  # Supervised training loop for classification
├── utils.py                        # Seed setting, accuracy calculation, and helpers
```

---

##  Model Overview

`SimCLR_CNNClassifier.py` defines a unified architecture:
- `mode='pretrain'`: Enables **SimCLR** contrastive learning using a projection head.
- `mode='classification'`: Switches to a **fully supervised classification head**.

Use `.set_mode('pretrain')` and `.set_mode('classification')` as needed.

<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/CLR_Dia.png" alt="SimCLR CNN Classifier Diagram" width="700"/>
</p>


---

##  Workflow

###  1. Setup

```bash
pip install -r requirements.txt
```

###  2. Data Format

Organize your dataset like this:
```
/data/
  ├── train/
  │     ├── class1/
  │     ├── class2/
  └── test/
        ├── class1/
        ├── class2/
```

###  3. Training


# Supervised fine-tuning
python train_model.py


###  4. Evaluation

python test.py

This will:
- Print classification report
- Save confusion matrix
- Save ROC-AUC plots
- Save t-SNE plot

### 📊 Output Visualizations

#### ✅ Confusion Matrix
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/PL_CS.png" alt="Confusion Matrix" width="600"/>
</p>

#### 🌀 t-SNE Visualization
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/PL_tSNE.png" alt="t-SNE Plot" width="600"/>
</p>

#### 📈 Accuracy & Loss Curve
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/raw/main/Plots/loss_accuracy_curve.png" alt="Training and Validation Curve" width="600"/>
</p>

#### 🔍 Grad-CAM Visualization
<p align="center">
  <img src="https://github.com/ItsCodeBakery/PlantPathalogy/blob/main/Plots/gcPlantVillage%20(1).png" alt="Grad-CAM Attention Map" width="600"/>
</p>


---

##  Visualization Tools

| Module | Description |
|--------|-------------|
| `plot_loss_accuracy_curves.py` | Dual-axis plot of loss and accuracy |
| `plot_multiclass_roc.py`      | ROC-AUC curve for each class and macro average |
| `generate_tsne.py`            | t-SNE of final embeddings |
| `gradcam.py`                  | Grad-CAM attention visualization on test images |
| `utils.py`                    | Utility functions (seed, accuracy, etc.) |

---

##  Key Features

-  Self-supervised learning via **SimCLR projection head**
-  Custom CNN architecture
-  Classification with full visualization suite
-  Modular design for ease of extension
-  Plots are saved in `plots/` (ensure the directory exists)
-  Feel free to contact me if you face any problem in running the code. 

---


