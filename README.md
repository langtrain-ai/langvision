<div align="center">

<img src="https://raw.githubusercontent.com/langtrain-ai/langvision/main/static/langvision-black.png" alt="Langvision" width="400" />

<h3>Fine-tune Vision LLMs with ease</h3>

<p>
  <strong>Train LLaVA, Qwen-VL, and other vision models in minutes.</strong><br>
  The simplest way to create custom multimodal AI.
</p>

<p>
  <a href="https://www.producthunt.com/products/langtrain-2" target="_blank"><img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=1049974&theme=light" alt="Product Hunt" width="200" /></a>
</p>

<p>
  <a href="https://pypi.org/project/langvision/"><img src="https://img.shields.io/pypi/v/langvision.svg?style=for-the-badge&logo=pypi&logoColor=white" alt="PyPI" /></a>
  <a href="https://pepy.tech/project/langvision"><img src="https://img.shields.io/pepy/dt/langvision?style=for-the-badge&logo=python&logoColor=white&label=downloads" alt="Downloads" /></a>
  <a href="https://github.com/langtrain-ai/langvision/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue?style=for-the-badge" alt="License" /></a>
</p>

<p>
  <a href="#quick-start">Quick Start</a> •
  <a href="#features">Features</a> •
  <a href="#supported-models">Models</a> •
  <a href="https://langtrain.xyz/docs">Docs</a>
</p>

</div>

---

## ⚡ Quick Start

### 1-Click Install (Recommended)
The fastest way to get started. Installs Langvision in an isolated environment.

```bash
curl -fsSL https://raw.githubusercontent.com/langtrain-ai/langvision/main/scripts/install.sh | bash
```

### Or using pip
```bash
pip install langvision
```

Fine-tune a vision model in **3 lines**:

```python
from langvision import LoRATrainer

trainer = LoRATrainer(model_name="llava-hf/llava-1.5-7b-hf")
trainer.train_from_file("image_data.jsonl")
```

Your custom vision model is ready.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🖼️ **Multimodal Training**
Train on images + text together. Perfect for VQA, image captioning, and visual reasoning.

### 🎯 **Smart Defaults**
Optimized configurations for each model architecture. Just point and train.

### 💾 **Efficient Memory**
LoRA + 4-bit quantization = Train 13B vision models on a single 24GB GPU.

</td>
<td width="50%">

### 🔧 **Battle-Tested**
Production-ready code used by teams building real-world vision applications.

### 🌐 **All Major Models**
LLaVA, Qwen-VL, CogVLM, InternVL, and more. Full compatibility.

### ☁️ **Deploy Anywhere**
Export to GGUF, ONNX, or deploy directly to Langtrain Cloud.

</td>
</tr>
</table>

---

## 🤖 Supported Models

| Model | Parameters | Memory Required |
|-------|-----------|-----------------|
| LLaVA 1.5 | 7B, 13B | 8GB, 16GB |
| Qwen-VL | 7B | 8GB |
| CogVLM | 17B | 24GB |
| InternVL | 6B, 26B | 8GB, 32GB |
| Phi-3 Vision | 4.2B | 6GB |

---

## 📖 Full Example

```python
from langvision import LoRATrainer
from langvision.config import TrainingConfig, LoRAConfig

# Configure training
config = TrainingConfig(
    num_epochs=3,
    batch_size=2,
    learning_rate=2e-4,
    lora=LoRAConfig(rank=16, alpha=32)
)

# Initialize trainer
trainer = LoRATrainer(
    model_name="llava-hf/llava-1.5-7b-hf",
    output_dir="./my-vision-model",
    config=config
)

# Train on image-text data
trainer.train_from_file("training_data.jsonl")
```

---

## 📝 Data Format

```jsonl
{"image": "path/to/image1.jpg", "conversations": [{"from": "human", "value": "What's in this image?"}, {"from": "assistant", "value": "A cat sitting on a couch."}]}
```

---

## 🤝 Community

<p align="center">
  <a href="https://discord.gg/langtrain">Discord</a> •
  <a href="https://twitter.com/langtrainai">Twitter</a> •
  <a href="https://langtrain.xyz">Website</a>
</p>

---

<div align="center">

**Built with ❤️ by [Langtrain AI](https://langtrain.xyz)**

*Making vision AI accessible to everyone.*

</div>
