import torch

def gram_matrix(feat: torch.Tensor):
    B, C, H, W = feat.shape
    F = feat.view(B, C, H*W)
    G = F @ F.transpose(1, 2)

    return G