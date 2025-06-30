import torch
import torch.nn as nn
from plimai.models.lora import LoRALinear

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, lora_config=None):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.lora_q = LoRALinear(self.head_dim, self.head_dim, **lora_config) if lora_config else None
        self.lora_v = LoRALinear(self.head_dim, self.head_dim, **lora_config) if lora_config else None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.lora_q:
            q = q + self.lora_q(q)
        if self.lora_v:
            v = v + self.lora_v(v)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, lora_config=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, lora_config)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, depth, dim, num_heads, mlp_ratio=4.0, lora_config=None):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(dim, num_heads, mlp_ratio, lora_config)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x 