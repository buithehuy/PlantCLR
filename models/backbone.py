import timm
import torch.nn as nn

class ConvNeXtTiny(nn.Module):
    def __init__(self, pretrained=False):
        """
        ConvNeXt-Tiny backbone using timm.
        Args:
            pretrained (bool): Whether to load timm's pretrained weights.
        """
        super().__init__()
        # Load ConvNeXt-Tiny from timm without the classification head, pooling globally.
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=0, global_pool='avg')
        self.feature_dim = self.backbone.num_features  # Should be 768 for convnext_tiny

    def forward(self, x):
        """
        Args:
            x (Tensor): Input images of shape (B, C, H, W)
        Returns:
            Tensor: Global average pooled features of shape (B, 768)
        """
        return self.backbone(x)
