import torch
import torch.nn as nn
from .backbone import ConvNeXtTiny
from .simclr import SimCLRProjectionHead

class PlantCLR(nn.Module):
    """
    Main architecture of PlantCLR framing the ConvNeXt-Tiny backbone.
    Supports dual modes for SimCLR pretraining and down-stream classification.
    """
    def __init__(self, num_classes=38, pretrained_backbone=False, mode='classification', projection_dim=128):
        super().__init__()
        self.mode = mode
        
        # Encoder Backbone
        self.backbone = ConvNeXtTiny(pretrained=pretrained_backbone)
        feature_dim = self.backbone.feature_dim  # Default 768 for convnext_tiny
        
        # Pretraining-specific projection head (Removed during fine-tuning)
        self.projection_head = SimCLRProjectionHead(input_dim=feature_dim, output_dim=projection_dim)
        
        # Supervised classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        """
        Forward pass based on current mode.
        """
        features = self.backbone(x)
        
        if self.mode == 'pretrain':
            # SimCLR Contrastive Representation Mapping
            return self.projection_head(features)
        elif self.mode == 'classification':
            # Linear + MLP Classification
            return self.classifier(features)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def set_mode(self, mode):
        """
        Set PlantCLR mode to 'pretrain' or 'classification'.
        """
        assert mode in ['pretrain', 'classification']
        self.mode = mode

    @classmethod
    def from_pretrained(cls, path, num_classes=38):
        """
        Load a pretrained PlantCLR classification model from state dict.
        """
        model = cls(num_classes=num_classes, mode='classification')
        state_dict = torch.load(path, map_location='cpu')
        
        # Support loading just the backbone from a pretraining checkpoint if necessary.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded PlantCLR model from {path}.")
        if missing:
            print(f"Missing keys: {missing}")
        return model
