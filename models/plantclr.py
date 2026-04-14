import torch
import torch.nn as nn
from .backbone import ConvNeXtTiny
from .simclr import SimCLRProjectionHead

class PlantCLR(nn.Module):
    def __init__(self, num_classes=38, pretrained_backbone=False, mode='classification', projection_dim=128):
        super().__init__()
        self.mode = mode
        
        # Encoder Backbone (Feature Extractor)
        self.backbone = ConvNeXtTiny(pretrained=pretrained_backbone)
        feature_dim = self.backbone.feature_dim
        
        # Pretraining "Resize Feature" block (Global Average Pooling)
        self.resize_feature = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Pretraining projection head (MLP)
        self.projection_head = SimCLRProjectionHead(input_dim=feature_dim, output_dim=projection_dim)
        
        # Diagram: "CNN -> Feature Maps" block for classification
        self.cnn_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # Diagram: "FCN" Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim // 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x) # (B, 768, H/32, W/32)
        
        if self.mode == 'pretrain':
            # "Resize Feature" -> "MLP"
            resized = self.resize_feature(features)
            return self.projection_head(resized)
            
        elif self.mode == 'classification':
            # "CNN" -> Feature Maps -> "FCN"
            cnn_feats = self.cnn_block(features)
            return self.classifier(cnn_feats)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def freeze_backbone(self):
        """Freeze fe: Lock ConvNeXt-Tiny weights during fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze fe"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def set_mode(self, mode):
        assert mode in ['pretrain', 'classification']
        self.mode = mode

    @classmethod
    def from_pretrained(cls, path, num_classes=38):
        model = cls(num_classes=num_classes, mode='classification')
        state_dict = torch.load(path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded PlantCLR model from {path}.")
        return model
