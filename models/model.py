
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_activation(name: str):
    name = (name or "relu").lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu" or name == "swish":
        return nn.SiLU(inplace=True)
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01, inplace=True)
    return nn.ReLU(inplace=True)

class TinyMNISTNet(nn.Module):
    """
    Minimal CNN for MNIST with switches:
      - use_bn: BatchNorm after convs (except final 1x1)
      - dropout_p: Dropout prob after each activation (0 disables)
      - activation: 'relu' | 'gelu' | 'silu' | 'leakyrelu'
      - last_channels: width of the last conv block (kept small for <20k params)
    Architecture:
      1->8->12 -> MaxPool -> 12->16->16 -> MaxPool -> 16->24->last -> 1x1->10 -> GAP
    """
    def __init__(self, use_bn=True, dropout_p=0.05, activation="relu", last_channels=32):
        super().__init__()
        Act = lambda: get_activation(activation)
        self.use_bn = use_bn
        self.dropout_p = dropout_p

        def BN(c): return nn.BatchNorm2d(c) if use_bn else nn.Identity()
        def DO(): return nn.Dropout(dropout_p) if (dropout_p and dropout_p>0) else nn.Identity()

        self.c1 = nn.Conv2d(1, 8, 3, padding=1, bias=not use_bn);   self.b1, self.a1, self.d1 = BN(8), Act(), DO()
        self.c2 = nn.Conv2d(8,12, 3, padding=1, bias=not use_bn);   self.b2, self.a2, self.d2 = BN(12), Act(), DO()
        self.c3 = nn.Conv2d(12,16,3, padding=1, bias=not use_bn);   self.b3, self.a3, self.d3 = BN(16), Act(), DO()
        self.c4 = nn.Conv2d(16,16,3, padding=1, bias=not use_bn);   self.b4, self.a4, self.d4 = BN(16), Act(), DO()
        self.c5 = nn.Conv2d(16,24,3, padding=1, bias=not use_bn);   self.b5, self.a5, self.d5 = BN(24), Act(), DO()
        self.c6 = nn.Conv2d(24,last_channels,3,padding=1,bias=not use_bn); self.b6, self.a6, self.d6 = BN(last_channels), Act(), DO()

        self.pool = nn.MaxPool2d(2,2)
        self.clf  = nn.Conv2d(last_channels, 10, 1, bias=True)

    def forward(self, x):
        x = self.d1(self.a1(self.b1(self.c1(x))))
        x = self.d2(self.a2(self.b2(self.c2(x))))
        x = self.pool(x)
        x = self.d3(self.a3(self.b3(self.c3(x))))
        x = self.d4(self.a4(self.b4(self.c4(x))))
        x = self.pool(x)
        x = self.d5(self.a5(self.b5(self.c5(x))))
        x = self.d6(self.a6(self.b6(self.c6(x))))
        x = self.clf(x)                 # [B,10,7,7]
        x = F.adaptive_avg_pool2d(x, 1) # GAP
        x = x.view(x.size(0), -1)       # [B,10]
        return x

def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
