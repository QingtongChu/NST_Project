import torch
import torch.nn as nn
from torchvision import models

# define layers needed from VGG_19
VGG_LAYERS = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,    # used for content layer in the original paper
    "conv5_1": 28,
}

class VGG19FeatureExtractor(nn.Module):
    def __init__(self, layer_map=VGG_LAYERS):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # turn off ReLU inplce to prevent changing the input of conv layers
        for i,m in enumerate(vgg):
            if isinstance(m, nn.ReLU):
                vgg[i] = nn.ReLU(inplace=False)
            if isinstance(m, nn.MaxPool2d):
                vgg[i] = nn.AvgPool2d(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding)
        # turn on the evaluation mode
        self.features = vgg.eval()

        # freeze the model from being trained
        for p in self.features.parameters():
            p.requires_grad = False
        
        self.layer_map = layer_map
    
    # cpu or gpu
    @torch.no_grad()
    def device(self):
        return next(self.features.parameters()).device
    
    def forward(self, x: torch.Tensor):
        """
        x: [B, 3, H, W] in nomrlized VGG space. Returns dict(name: activation)
        """
        feats = {}
        t = x
        
        for i, layer in enumerate(self.features):
            t = layer(t)
            for name, idx in self.layer_map.items():
                if i == idx:
                    feats[name] = t
        return feats


    

        

        
        




    