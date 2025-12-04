import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ======================
# SIAMESE MOBILENET
# ======================
class SiameseMobileNet(nn.Module):
    def __init__(self, embedding_dim=128, pretrained=True):
        super().__init__()
        
        # Load pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Lấy feature extractor (bỏ classifier)
        self.backbone = mobilenet.features
        
        # Embedding layers
        self.embedding = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1280, 512),  # MobileNetV2 output = 1280 channels
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
    
    def forward_once(self, x):
        """Extract embedding từ một ảnh"""
        x = self.backbone(x)
        x = self.embedding(x)
        return x
    
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
        """
        distances: khoảng cách giữa các cặp
        labels: 0 nếu cùng class, 1 nếu khác class
        """
        loss_same = (1 - labels) * torch.pow(distances, 2)
        loss_diff = labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        
        return torch.mean(loss_same + loss_diff)