import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import confusion_matrix
import os

def plot_confusion_matrix(y_true, y_pred, class_names, output_path):
    """
    Save confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

def plot_tsne(features, labels, class_names=None, output_path='tsne.png'):
    """
    Reduce feature vectors locally and plot t-SNE latent space.
    """
    tsne = TSNE(n_components=2, random_state=42)
    reduced = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap='tab20', alpha=0.7)
    if class_names:
        # Create a legend
        handles, _ = scatter.legend_elements()
        plt.legend(handles, class_names, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.title('t-SNE Visualization of Latent Space')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
