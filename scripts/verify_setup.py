import torch
from nst.models import VGGFeatureExtractor
from nst.image_io import load_image, preprocess
from nst.features import gram_matrix
from pathlib import Path

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] using device: {device}")

    # Put any small test images in assets/content & assets/style
    content_path = Path("assets/content").glob("*.*")
    style_path   = Path("assets/style").glob("*.*")
    try:
        c_img = load_image(next(content_path), max_size=256)
        s_img = load_image(next(style_path),   max_size=256)
    except StopIteration:
        raise SystemExit("Put at least one content & one style image in assets/")

    c = preprocess(c_img, device=device)
    s = preprocess(s_img, device=device)

    vgg = VGGFeatureExtractor().to(device).eval()
    with torch.no_grad():
        c_feats = vgg(c)
        s_feats = vgg(s)

    # Print feature shapes and a Gram matrix shape to confirm
    for k, v in c_feats.items():
        print(f"[content] {k}: {tuple(v.shape)}")
    for k, v in s_feats.items():
        G = gram_matrix(v)
        print(f"[style] {k}: feat {tuple(v.shape)} -> gram {tuple(G.shape)}")

if __name__ == "__main__":
    main()