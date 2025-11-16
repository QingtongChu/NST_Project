import torch
import torch.nn.functional as F
from nst.features import gram_matrix

def content_loss(gen_feats: dict, content_feats: dict, layer: str="conv4_2"):
    F_l = gen_feats[layer]
    P_l = content_feats[layer]

    return 0.5 * F.mse_loss(F_l, P_l, reduction="sum")

def style_loss(gen_feats: dict, style_feats: dict, style_layers: list[str], layer_weights: dict):
    if layer_weights is None:
        layer_weights = {l: 1.0 for l in style_layers}

    total = 0.0
    
    for layer in style_layers:
        F_l = gen_feats[layer]
        S_l = style_feats[layer]
        _, C, H, W = F_l.shape
        N_l = C
        M_l = H * W


        G_l = gram_matrix(F_l)
        A_l = gram_matrix(S_l)

        E_l = ((G_l - A_l)**2).sum() / (4 * N_l**2 * M_l**2)

        w_l = layer_weights.get(layer, 1.0)
        total = total + w_l * E_l
    return total