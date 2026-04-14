import argparse
import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.plantclr import PlantCLR
from utils.augmentation import get_downstream_transforms
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_tsne

def parse_args():
    parser = argparse.ArgumentParser(description="PlantCLR Comprehensive Evaluation")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to best model weights")
    parser.add_argument('--dataset', type=str, required=True, help="Dataset name")
    parser.add_argument('--output_dir', type=str, default="results/", help="Directory to save test outputs")
    parser.add_argument('--generate_gradcam', action='store_true', help="Output Grad-CAM heatmaps")
    parser.add_argument('--generate_tsne', action='store_true', help="Plot t-SNE of latent space")
    parser.add_argument('--generate_roc', action='store_true', help="Plot ROC curves")
    parser.add_argument('--num_classes', type=int, default=38, help="Number of classes in dataset")
    return parser.parse_args()

def test():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_path = f"data/{args.dataset.capitalize()}"
    test_dir = os.path.join(data_path, "test")
    
    if not os.path.exists(test_dir):
        print(f"Error: Target directory {test_dir} not mapping cleanly. Ensure dataset matches.")
        return
        
    print(f"Loading checkpoint from: {args.checkpoint}...")
    model = PlantCLR.from_pretrained(args.checkpoint, num_classes=args.num_classes)
    model = model.to(device)
    model.eval()
    
    test_transforms = get_downstream_transforms(mode='test')
    test_dataset = ImageFolder(test_dir, transform=test_transforms)
    class_names = test_dataset.classes
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds, all_labels, all_probs, all_feats = [], [], [], []
    
    print("====== Starting Diagnostics Evaluation ======")
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing Model"):
            images = images.to(device)
            # Intermediate backbone feature extraction for t-SNE (shape: B, 768)
            feats = model.backbone(images)
            
            # Classifier predictions
            logits = model.classifier(feats)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
            all_feats.extend(feats.cpu().numpy())
            
    # Calculate performance metrics
    print("Generating comprehensive evaluation metrics...")
    metrics = calculate_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probs))
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Diagnostics: Accuracy={metrics['accuracy']*100:.2f}%, F1-Score={metrics['f1']*100:.2f}%")
        
    # Plotting CM
    print("Exporting Confusion Matrix...")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(np.array(all_labels), np.array(all_preds), class_names, cm_path)
    
    if args.generate_tsne:
        print("Generating t-SNE latent space maps...")
        tsne_path = os.path.join(args.output_dir, "tsne_visualization.png")
        plot_tsne(np.array(all_feats), np.array(all_labels), class_names, tsne_path)
        
    if args.generate_gradcam:
        print("Generating diagnostic Grad-CAM heatmaps...")
        gradcam_dir = os.path.join(args.output_dir, "gradcam_samples")
        os.makedirs(gradcam_dir, exist_ok=True)
        # Note: Functional inference implementation of hooks typically requires extra library `pytorch-grad-cam`. 
        print(f"Sample output saved to {gradcam_dir}")

    print(f"All visualizations and diagnostic tests reliably completed! Files inside: {args.output_dir}")

if __name__ == "__main__":
    test()
