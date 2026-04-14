import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from models.plantclr import PlantCLR
from models.simclr import NT_XentLoss
from utils.augmentation import get_simclr_transforms, get_downstream_transforms

def parse_args():
    parser = argparse.ArgumentParser(description="PlantCLR Training Script")
    parser.add_argument('--mode', type=str, choices=['pretrain', 'classification'], required=True, 
                        help='Training phase: SimCLR pretraining or supervised classification')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., plantvillage)')
    parser.add_argument('--data_path', type=str, default=None, help='Custom path to dataset')
    parser.add_argument('--config', type=str, default=None, help='Path to YAML config')
    parser.add_argument('--backbone', type=str, default='convnext_tiny', help='Encoder backbone geometry')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Total epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.5, help='NT-Xent temperature')
    parser.add_argument('--augmentation', type=str, default='strong', help='Augmentation strength')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Restore from checkpoint')
    parser.add_argument('--num_classes', type=int, default=38, help='Number of classification classes')
    return parser.parse_args()

def train():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on Device: {device}")
    
    # Path mappings
    data_path = args.data_path or f"data/{args.dataset.capitalize()}"
    train_dir = os.path.join(data_path, "train")
    val_dir = os.path.join(data_path, "val")
    
    # Instantiate Model Architecture
    model = PlantCLR(num_classes=args.num_classes, mode=args.mode)
    
    # Load Pretrained Weights if transitioning to Phase 2
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        state = torch.load(args.pretrained_path, map_location='cpu')
        missing, unexp = model.load_state_dict(state, strict=False)
        print(f"Successfully loaded checkpoint: {args.pretrained_path}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    if args.mode == 'pretrain':
        print("====== Starting Phase I: Self-Supervised Pretraining (SimCLR) ======")
        criterion = NT_XentLoss(temperature=args.temperature)
        train_transforms = get_simclr_transforms()
        dataset = ImageFolder(train_dir, transform=train_transforms)
        
        # Drop last batch to avoid anomalous batch size logic edge cases in contrastive loss computation
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            for images, _ in tqdm(loader, desc=f"Pretrain Epoch {epoch}/{args.epochs}"):
                img_i, img_j = images[0].to(device), images[1].to(device)
                
                optimizer.zero_grad()
                z_i = model(img_i)
                z_j = model(img_j)
                
                loss = criterion(z_i, z_j)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * img_i.size(0)
                
            epoch_loss = total_loss / len(dataset)
            print(f"Epoch {epoch} | NT-Xent Loss: {epoch_loss:.4f}")
            
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/pretrain_convnext_best.pth")
        print("Pretraining completed. Model saved at checkpoints/pretrain_convnext_best.pth")

    elif args.mode == 'classification':
        print("====== Starting Phase II: Supervised Fine-Tuning ======")
        # Use Label Smoothing as requested for regularization
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        train_transforms = get_downstream_transforms(mode='train')
        val_transforms = get_downstream_transforms(mode='val')
        
        train_dataset = ImageFolder(train_dir, transform=train_transforms)
        val_dataset = ImageFolder(val_dir, transform=val_transforms)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0
            for images, labels in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch}/{args.epochs}"):
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
            # Validation Step
            model.eval()
            correct = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    logits = model(images)
                    preds = logits.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    
            acc = correct / len(val_dataset)
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Validation Acc: {acc*100:.2f}%")
            
            if acc > best_acc:
                best_acc = acc
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(model.state_dict(), "checkpoints/best_model.pth")
                
        print(f"Fine-tuning completed. Best model saved (Acc: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    train()
