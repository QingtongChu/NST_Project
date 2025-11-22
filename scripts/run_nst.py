import os
import sys
from pathlib import Path

# Make sure project root is in path (same trick as verify_setup.py)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import torch
from nst.style_transfer import style_transfer

def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--content", type=str, required=True)
    parser.add_argument("--style",   type=str, required=True)
    parser.add_argument("--out",     type=str, default="outputs/stylized.png")
    parser.add_argument("--size",    type=int, default=512)
    parser.add_argument("--alpha",   type=float, default=1.0,  help="content weight")
    parser.add_argument("--beta",    type=float, default=1e5,  help="style weight")
    parser.add_argument("--tv",      type=float, default=1e-6, help="tv weight")
    parser.add_argument("--steps",   type=int, default=1000)
    parser.add_argument("--lr",      type=float, default=0.03)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    style_transfer(
        content_path=args.content,
        style_path=args.style,
        out_path=args.out,
        image_size=args.size,
        content_weight=args.alpha,
        style_weight=args.beta,
        tv_weight=args.tv,
        num_steps=args.steps,
        lr=args.lr,
        device=device,
    )

if __name__ == "__main__":
    main()
