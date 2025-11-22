import torch
from PIL import Image
from torchvision import transforms

# imagenet mean and std for normalization
VGG_MEAN = [0.485, 0.456, 0.406]
VGG_STD = [0.229, 0.224, 0.225]

def load_image(path, max_size=512):
    img = Image.open(path).convert("RGB")
    
    # donwsampling for maintaining size
    if max(img.size) > max_size:
        scale = max_size / max(img.size)
        new_size = [int(img.width * scale), int(img.height * scale)]
        img = img.resize(new_size, Image.LANCZOS)

    return img

# make sure img is tensor with the right dimension
def preprocess(img: Image.Image, device="cpu"):
    # convert to tensor
    tfm = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD)
    ])

    # add batch dimension [1, 3, H, W]
    x = tfm(img).unsqueeze(0).to(device)   # add one dimension on index 0
    return x    

# convert back to viewable image
def deprocess(x: torch.Tensor):
    # match the mean and std with dim of img
    mean = torch.tensor(VGG_MEAN).view(1, 3, 1, 1).to(x.device)
    std = torch.tensor(VGG_STD).view(1, 3, 1, 1).to(x.device)

    # denorm
    y = x * std + mean
    y = y.clamp(0, 1)
    y = y.squeeze(0).detach().cpu()

    return transforms.ToPILImage()(y)

def save_image(x: torch.Tensor, out_path: str):
    deprocess(x).save(out_path)


