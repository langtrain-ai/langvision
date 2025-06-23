import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r=4, alpha=1.0, dropout=0.0):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(in_features, r) * 0.01)
            self.lora_B = nn.Parameter(torch.randn(r, out_features) * 0.01)
        else:
            self.lora_A = None
            self.lora_B = None
        self.scale = alpha / r if r > 0 else 1.0

    def forward(self, x):
        if self.r > 0:
            return self.dropout(x) @ self.lora_A @ self.lora_B * self.scale
        else:
            return 0.0 