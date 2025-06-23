# plimai: Vision LLMs with LoRA Fine-Tuning

A modular framework for building and fine-tuning Vision Large Language Models (LLMs) with efficient Low-Rank Adaptation (LoRA) support.

## Features
- Modular Vision Transformer backbone
- LoRA for efficient fine-tuning on limited compute
- Easily extensible for new components and tasks
- Example and tests included

## Installation
```
pip install torch torchvision
```

## Usage Example
```python
import torch
from plimai.models.vision_transformer import VisionTransformer
from plimai.utils.config import default_config

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
```

## Running Tests
```
pytest tests/
```

## Directory Structure
```
plimai/
  models/
    vision_transformer.py
    lora.py
  components/
    patch_embedding.py
    attention.py
    mlp.py
  utils/
    data.py
    config.py
  example.py
```

## License
MIT 