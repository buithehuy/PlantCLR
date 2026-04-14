import timm
import torch.nn as nn

class ConvNeXtTiny(nn.Module):
    def __init__(self, pretrained=False):
        """
        ConvNeXt-Tiny backbone using timm.
        """
        super().__init__()
        # Remove classifier head AND global_pool to output spatial feature maps instead of 1D vector
        self.backbone = timm.create_model('convnext_tiny', pretrained=pretrained, num_classes=0, global_pool='')
        self.feature_dim = self.backbone.num_features  # 768

    def forward(self, x):
        """
        Returns:
            Tensor: Spatial features of shape (B, 768, H/32, W/32)
        """
        return self.backbone(x)
