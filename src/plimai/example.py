"""
Example module demonstrating basic functionality.
"""

import torch
from plimai.models.vision_transformer import VisionTransformer
from plimai.utils.config import default_config

def hello(name: str = "World") -> str:
    """
    A simple greeting function.
    
    Args:
        name (str): Name to greet. Defaults to "World".
        
    Returns:
        str: Greeting message
    """
    return f"Hello, {name}!"

def get_version() -> str:
    """
    Get the current version of the package.
    
    Returns:
        str: Current version number
    """
    from plimai import __version__
    return __version__

if __name__ == '__main__':
    # Dummy image batch: batch_size=2, channels=3, height=224, width=224
    x = torch.randn(2, 3, 224, 224)
    model = VisionTransformer(
        img_size=default_config['img_size'],
        patch_size=default_config['patch_size'],
        in_chans=default_config['in_chans'],
        num_classes=default_config['num_classes'],
        embed_dim=default_config['embed_dim'],
        depth=default_config['depth'],
        num_heads=default_config['num_heads'],
        mlp_ratio=default_config['mlp_ratio'],
        lora_config=default_config['lora'],
    )
    out = model(x)
    print('Output shape:', out.shape) 