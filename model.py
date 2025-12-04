import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================
# SIAMESE NETWORK
# ======================
class Siamese1DNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared CNN để extract features
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv1d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Layer 2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            # Layer 3
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            
            # Fully connected
            nn.Flatten(),
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),  # Embedding dimension = 128
        )

    def forward_once(self, x):
        """Extract embedding từ một ảnh"""
        return self.conv(x)

    def forward(self, x1, x2):
        """Tính khoảng cách giữa 2 ảnh"""
        e1 = self.forward_once(x1)
        e2 = self.forward_once(x2)
        return F.pairwise_distance(e1, e2)

# ======================
# CONTRASTIVE LOSS
# ======================
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin
    
    def forward(self, distances, labels):
        loss_same = (1 - labels) * torch.pow(distances, 2)
        loss_diff = labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        return torch.mean(loss_same + loss_diff)