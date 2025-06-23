import torch
from plimai.models.lora import LoRALinear

def test_lora_linear():
    layer = LoRALinear(8, 16, r=2, alpha=1.0)
    x = torch.randn(4, 8)
    out = layer(x)
    assert out.shape == (4, 16)
    assert sum(p.numel() for p in layer.parameters() if p.requires_grad) == 8*2 + 2*16 