# PlimAI: Vision LLMs with Efficient LoRA Fine-Tuning

[![PyPI version](https://img.shields.io/pypi/v/plimai.svg)](https://pypi.org/project/plimai/)
[![Downloads](https://pepy.tech/badge/plimai)](https://pepy.tech/project/plimai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

---

**PlimAI** is a modular, research-friendly framework for building and fine-tuning Vision Large Language Models (LLMs) with efficient Low-Rank Adaptation (LoRA) support. It is designed for:
- Researchers exploring new vision transformer architectures or fine-tuning strategies
- Practitioners who want to adapt large vision models to custom datasets with limited compute
- Developers looking for a clean, extensible codebase for vision-language AI

PlimAI provides a plug-and-play interface for LoRA, making it easy to experiment with parameter-efficient fine-tuning. The codebase is modular, so you can swap out or extend components like patch embedding, attention, or MLP heads.

---

## 🏗️ Architecture Overview

PlimAI is built around a modular Vision Transformer (ViT) backbone, with LoRA adapters injected into attention and MLP layers for efficient fine-tuning. The main components are:

```mermaid
graph TD
    A[Input Image] --> B[Patch Embedding]
    B --> C[+CLS Token & Positional Encoding]
    C --> D[Transformer Encoder]
    D --> E[LayerNorm]
    E --> F[MLP Head]
    F --> G[Output (e.g., Class logits)]
    subgraph LoRA Adapters
        D
    end
```

### Main Modules
- **PatchEmbedding**: Splits the image into patches and projects them into embedding space.
- **TransformerEncoder**: Stack of transformer layers, each with multi-head self-attention and MLP blocks. LoRA adapters can be injected here.
- **LoRALinear**: Low-rank adapters for efficient fine-tuning, only a small number of parameters are updated.
- **MLPHead**: Final classification or regression head.
- **Config & Utils**: Easy configuration and preprocessing utilities.

---

## 📦 Installation

```bash
pip install plimai
```
Or, for the latest version from source:
```bash
git clone https://github.com/plim-ai/plim.git
cd plim
pip install .
```

---

## 🧑‍💻 Quick Start

```python
import torch
from plimai.models.vision_transformer import VisionTransformer
from plimai.utils.config import default_config

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
```

---

## 📚 Documentation
- [API Reference](https://github.com/plim-ai/plim/tree/main/docs)
- [Vision Transformer with LoRA: Paper](https://arxiv.org/abs/2106.09685)
- [LoRA for Vision Models: HuggingFace PEFT](https://github.com/huggingface/peft)

---

## 🧩 Module Breakdown

| Module                | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| `PatchEmbedding`      | Converts images to patch embeddings for transformer input                    |
| `TransformerEncoder`  | Stack of transformer layers with optional LoRA adapters                      |
| `LoRALinear`          | Low-rank adapters for parameter-efficient fine-tuning                        |
| `MLPHead`             | Output head for classification or regression                                 |
| `data.py`             | Preprocessing and augmentation utilities                                     |
| `config.py`           | Centralized configuration for model/training hyperparameters                 |

---

## 🧪 Running Tests

```bash
pytest tests/
```

---

## 🤝 Contributing
We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Star ⭐ the repo if you find it useful!

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgements
- [PyTorch](https://pytorch.org/)
- [HuggingFace](https://huggingface.co/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

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