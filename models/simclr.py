import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLRProjectionHead(nn.Module):
    """
    2-Layer MLP Projection Head for SimCLR Pretraining.
    """
    def __init__(self, input_dim=768, hidden_dim=512, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

class NT_XentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Used for SimCLR contrastive learning.
    """
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        # Reduction is sum, we'll divide by 2*N dynamically in forward
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def forward(self, z_i, z_j):
        """
        Args:
            z_i (Tensor): Embeddings from first augmented view (B, output_dim)
            z_j (Tensor): Embeddings from second augmented view (B, output_dim)
        """
        batch_size = z_i.size(0)
        
        # Concatenate embeddings: shape becomes (2*B, D)
        z = torch.cat([z_i, z_j], dim=0)
        
        # Cosine similarity matrix scaled by temperature
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        
        # Mask out self-similarities on the diagonal with a large negative value
        sim_matrix.fill_diagonal_(-1e9)
        
        # Automatically infer positive pairs: 
        # For sample i in [0..B-1], its augmented pair is at i + B.
        # For sample i in [B..2B-1], its augmented pair is at i - B.
        labels = torch.cat([torch.arange(batch_size) + batch_size, torch.arange(batch_size)], dim=0).to(z_i.device)
        
        # Calculate CrossEntropy loss where the target is the correct positive pair index.
        loss = self.criterion(sim_matrix, labels) / (2 * batch_size)
        return loss
