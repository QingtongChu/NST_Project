import os
import sys

# Add project root (NST_Project) to sys.path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from nst.models import VGG19FeatureExtractor
from nst.image_io import load_image, preprocess
from nst.features import gram_matrix
from nst.losses import content_loss, style_loss
from pathlib import Path

def main():
    device = "cuda" if torch.cuda.is_available else "cpu"
    print(f"device being used: {device}")

    c_path = Path("assets/content").glob("*.*")
    s_path = Path("assets/style").glob("*.*")

    c_img = load_image(next(c_path), max_size=512)
    s_img = load_image(next(s_path), max_size=512)
    try:
        c = preprocess(c_img, device=device)
        s = preprocess(s_img, device=device)
    except StopIteration:
        raise SystemExit("Input at least one image into assets/content and assets/style")

    vgg = VGG19FeatureExtractor().to(device).eval()

    with torch.no_grad():
        c_feats = vgg(c)
        s_feats = vgg(s)

    for k, v in c_feats.items():
        print(f"[content] {k}: {tuple(v.shape)}")
    for k, v in s_feats.items():
        G_s = gram_matrix(v)
        print(f"[style] {k}: {tuple(v.shape)} -> gram_matrix {G_s.shape}")
    
    # fake feature maps
    gen_feats = {
        "conv1_1": torch.rand(1, 64, 128, 128),
        "conv2_1": torch.rand(1, 128, 64, 64),
        "conv3_1": torch.rand(1, 256, 32, 32),
        "conv4_1": torch.rand(1, 512, 16, 16),
        "conv4_2": torch.rand(1, 512, 16, 16),   # for content loss
        "conv5_1": torch.rand(1, 512, 8, 8),
    }

    content_feats = {
        k: torch.rand_like(v) for k, v in gen_feats.items()
    }

    style_feats = {
        k: torch.rand_like(v) for k, v in gen_feats.items()
    }

    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    # test content loss
    Lc = content_loss(gen_feats, content_feats, "conv4_2")
    print("Content loss:", Lc.item())

    # test style loss
    Ls = style_loss(gen_feats, style_feats, style_layers, None)
    print("Style loss:", Ls.item())

    # test gram matrix
    test_feat = torch.rand(1, 64, 32, 32)
    G = gram_matrix(test_feat)
    print("Gram matrix size:", G.shape)

    print("NST test completed successfully.")

if __name__ == "__main__":
    main()















