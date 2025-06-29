import torch
import torch.nn as nn
from .lora import LoRALinear
from ..components.patch_embedding import PatchEmbedding
from ..components.attention import TransformerEncoder
from ..components.mlp import MLPHead

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, lora_config=None):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=0.1)
        self.encoder = TransformerEncoder(depth, embed_dim, num_heads, mlp_ratio, lora_config)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = MLPHead(embed_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.encoder(x)
        x = self.norm(x)
        return self.head(x[:, 0]) 