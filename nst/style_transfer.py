
import torch
from pathlib import Path
from nst.image_io import preprocess, load_image, save_image
from nst.models import VGG19FeatureExtractor
from nst.features import gram_matrix
from nst.losses import content_loss, style_loss, variational_loss

def style_transfer(content_path: str, 
                   style_path: str,
                   out_path: str = "output/out.png",
                   image_size: int = 512,
                   content_weight: float = 1.0,
                   style_weight: float = 1e5,
                   tv_weight: float = 1e-6,
                   num_steps: int = 1000,
                   lr: float = 0.03,
                   device: str = None,
                   ):
    if device == None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    #1. load image
    c_img = load_image(content_path, max_size=image_size)
    s_img = load_image(style_path, max_size=image_size)
    
    #2. preprocess
    c = preprocess(c_img, device=device)
    s = preprocess(s_img, device=device)

    #3. extract content and style features
    model = VGG19FeatureExtractor().to(device).eval()
    with torch.no_grad():
        c_feats = model(c)
        s_feats = model(s)

    #4. define style layers and layer weights
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    layer_weights = {
    "conv1_1": 0.2,
    "conv2_1": 0.2,
    "conv3_1": 0.2,
    "conv4_1": 0.2,
    "conv5_1": 0.2,
    }

    #5. initialize generated image as the same as the content image
    x = c.clone().requires_grad_(True)

    #6. define optimizer (adam: params, lr)
    optimizer = torch.optim.Adam([x], lr=lr)

    #7. optimization loops:
    """
    a. reset the gradients
    b. get the genereated image features
    c. get the 3 losses
    d. total losses
    e. backprop
    f. update gen_image
    g. print 3 lossses and total loss every 50 steps
    """
    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        gen_feats = model(x)

        Lc = content_loss(gen_feats, c_feats, layer="conv4_2")
        Ls = style_loss(gen_feats, s_feats, style_layers=style_layers, layer_weights=layer_weights)
        Lv = variational_loss(x, tv_weight=tv_weight)

        loss = content_weight * Lc + style_weight * Ls + Lv
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == 1:
            print(f"step: {step}  "
                  f"content loss = {Lc.item() * content_weight:.2f}  "
                  f"style loss = {Ls.item() * style_weight:.2f}  "
                  f"variational loss = {Lv.item():.2f}  "
                  f"total loss = {loss.item():.2f}")
    
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    save_image(x.detach(), out_path)
    print(f"Saved final stylized image to {out_path}")