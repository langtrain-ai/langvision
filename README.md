# plimai: Vision LLMs with Efficient LoRA Fine-Tuning

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="static/plimai-use-dark.png">
    <img src="static/plimai-white.png" alt="Plimai Logo" width="full"/>
  </picture>
</p>

<p align="center">
  <b>Plimai</b> — Modular Vision LLMs with Efficient LoRA Fine-Tuning
</p>

<p align="center">
  <a href="https://pypi.org/project/plimai/"><img src="https://img.shields.io/pypi/v/plimai.svg"></a>
  <a href="https://pepy.tech/project/plimai"><img src="https://pepy.tech/badge/plimai"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

---

## ✨ Features

- **Plug-and-play LoRA adapters** for parameter-efficient fine-tuning
- **Modular Vision Transformer (ViT) backbone**
- **Unified model zoo** for open-source visual models
- **Easy configuration** and extensible codebase
- **Ready for research and production**

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

#### Python API

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

#### CLI Fine-tuning

```bash
python src/plimai/finetune_vit_lora.py --dataset cifar10 --epochs 10 --batch_size 64
```

---

## 🏗️ Architecture Overview

plimai is built around a modular Vision Transformer (ViT) backbone, with LoRA adapters injected into attention and MLP layers for efficient fine-tuning. The main components are:

```mermaid
graph TD
    A([Input Image]) --> B([Patch Embedding])
    B --> C([+CLS Token & Positional Encoding])
    C --> D([Transformer Encoder])
    D --> E([LayerNorm])
    E --> F([MLP Head])
    F --> G([Output<br/>(Class logits)])

    %% LoRA Adapters as a subgraph inside Transformer Encoder
    subgraph LoRA_Adapters["LoRA Adapters (in Attention & MLP)"]
        LA1[ ]
    end
    LA1 -.-> D

    style LoRA_Adapters fill:#f9f,stroke:#333,stroke-width:2px
    style D fill:#e0f7fa,stroke:#00796b,stroke-width:2px
    style F fill:#fffde7,stroke:#fbc02d,stroke-width:2px
    style G fill:#e8f5e9,stroke:#388e3c,stroke-width:2px
```

### Main Modules
- **PatchEmbedding**: Splits the image into patches and projects them into embedding space.
- **TransformerEncoder**: Stack of transformer layers, each with multi-head self-attention and MLP blocks. LoRA adapters can be injected here.
- **LoRALinear**: Low-rank adapters for efficient fine-tuning, only a small number of parameters are updated.
- **MLPHead**: Final classification or regression head.
- **Config & Utils**: Easy configuration and preprocessing utilities.

---

## 📚 Documentation
- [API Reference](docs/index.md)
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

## 🤝 Community & Contributing

- Open issues for bugs or feature requests
- Submit pull requests for improvements
- Join the discussion on [GitHub Discussions](https://github.com/plim-ai/plim/discussions)
- See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🌟 Acknowledgements
- [PyTorch](https://pytorch.org/)
- [HuggingFace](https://huggingface.co/)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)

---

## 📖 FAQ

**Q: I get a CUDA out of memory error!**  
A: Try reducing the batch size or use a smaller model configuration.

**Q: How do I add my own dataset?**  
A: See the `data.py` module and pass your dataset path to the CLI.

---

## 📂 Directory Structure

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
  ...
```

---

## 📑 Citation

If you use plimai in your research, please cite:

```bibtex
@software{plimai,
  author = {Pritesh Raj},
  title = {plimai: Vision LLMs with Efficient LoRA Fine-Tuning},
  url = {https://github.com/plim-ai/plim},
  year = {2024},
}
``` 