import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.randn(num_classes, embedding_size))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        # Normalize embeddings and weights
        embeddings = F.normalize(embeddings)
        weights = F.normalize(self.weight)

        # Cos(theta)
        cosine = F.linear(embeddings, weights)  # [B, C]
        cosine = cosine.clamp(-1.0, 1.0)

        # sin(theta) = sqrt(1 - cos^2)
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encode labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Apply margin only to target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale
        logits *= self.s

        return logits
