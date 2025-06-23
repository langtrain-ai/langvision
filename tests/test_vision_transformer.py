import torch
from plimai.models.vision_transformer import VisionTransformer
from plimai.utils.config import default_config

def test_vit_output_shape():
    model = VisionTransformer(
        img_size=default_config['img_size'],
        patch_size=default_config['patch_size'],
        in_chans=default_config['in_chans'],
        num_classes=default_config['num_classes'],
        embed_dim=default_config['embed_dim'],
        depth=1,  # shallow for test
        num_heads=2,
        mlp_ratio=2.0,
        lora_config=default_config['lora'],
    )
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, default_config['num_classes']) 