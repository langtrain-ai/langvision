# Langvision CLI Guide

This guide provides comprehensive documentation for the Langvision command-line interface.

## Overview

The Langvision CLI provides a unified interface for all model operations including training, fine-tuning, evaluation, export, and configuration management.

## Installation

```bash
pip install langvision
```

## Basic Usage

```bash
langvision --help
```

## Commands

### 1. Training (`langvision train`)

Train a Vision Transformer model from scratch.

```bash
langvision train --dataset cifar10 --epochs 10 --batch_size 64
```

**Key Options:**
- `--dataset`: Dataset to use (cifar10, cifar100)
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--learning_rate`: Learning rate
- `--lora_rank`: LoRA rank for efficient training
- `--output_dir`: Directory to save checkpoints

### 2. Fine-tuning (`langvision finetune`)

Fine-tune a model with LoRA and advanced LLM concepts.

```bash
langvision finetune --dataset cifar100 --epochs 20 --lora_r 16 --rlhf
```

**Key Options:**
- All training options plus:
- `--rlhf`: Enable Reinforcement Learning from Human Feedback
- `--ppo`: Enable Proximal Policy Optimization
- `--dpo`: Enable Direct Preference Optimization
- `--lime`: Enable LIME explainability
- `--shap`: Enable SHAP explainability
- `--cot`: Enable Chain-of-Thought
- `--ccot`: Enable Contrastive Chain-of-Thought

### 3. Evaluation (`langvision evaluate`)

Evaluate a trained model on a dataset.

```bash
langvision evaluate --checkpoint model.pth --dataset cifar10
```

**Key Options:**
- `--checkpoint`: Path to model checkpoint
- `--dataset`: Dataset to evaluate on
- `--batch_size`: Batch size for evaluation
- `--save_predictions`: Save model predictions
- `--save_confusion_matrix`: Save confusion matrix plot

### 4. Export (`langvision export`)

Export a trained model to various formats.

```bash
langvision export --checkpoint model.pth --format onnx --output model.onnx
```

**Key Options:**
- `--checkpoint`: Path to model checkpoint
- `--format`: Export format (onnx, torchscript, state_dict)
- `--output`: Output file path
- `--batch_size`: Batch size for export
- `--opset_version`: ONNX opset version

### 5. Model Zoo (`langvision model-zoo`)

Browse and download pre-trained models.

```bash
# List available models
langvision model-zoo list

# Get model information
langvision model-zoo info vit_base_patch16_224

# Download a model
langvision model-zoo download vit_base_patch16_224

# Search for models
langvision model-zoo search "vit"
```

**Subcommands:**
- `list`: List all available models
- `info <model_name>`: Get detailed model information
- `download <model_name>`: Download a pre-trained model
- `search <query>`: Search for models

### 6. Configuration (`langvision config`)

Manage configuration files.

```bash
# Create a new configuration
langvision config create --template advanced --output config.yaml

# Validate a configuration
langvision config validate config.yaml

# Show default configuration
langvision config show --format yaml

# Convert between formats
langvision config convert config.yaml --format json

# Compare configurations
langvision config diff config1.yaml config2.yaml
```

**Subcommands:**
- `create`: Create a new configuration file
- `validate`: Validate a configuration file
- `show`: Show default configuration
- `convert`: Convert between YAML/JSON formats
- `diff`: Compare two configuration files

## Configuration Files

Langvision supports YAML and JSON configuration files. Example configurations are available in `examples/configs/`:

- `basic_config.yaml`: Simple configuration for basic training
- `advanced_config.yaml`: Advanced configuration with optimizations
- `custom_config.yaml`: Custom configuration for specific use cases

### Configuration Structure

```yaml
model:
  name: "vit_base"
  img_size: 224
  patch_size: 16
  num_classes: 10
  embed_dim: 768
  depth: 12
  num_heads: 12
  mlp_ratio: 4.0

data:
  dataset: "cifar10"
  data_dir: "./data"
  batch_size: 64
  num_workers: 2

training:
  epochs: 10
  learning_rate: 0.001
  optimizer: "adam"
  weight_decay: 0.01
  scheduler: "cosine"

lora:
  rank: 4
  alpha: 1.0
  dropout: 0.1
```

## Examples

### Complete Training Pipeline

```bash
# 1. Create configuration
langvision config create --template advanced --output my_config.yaml

# 2. Train model
langvision finetune --config my_config.yaml

# 3. Evaluate model
langvision evaluate --checkpoint outputs/vit_lora_best.pth --dataset cifar10

# 4. Export model
langvision export --checkpoint outputs/vit_lora_best.pth --format onnx --output model.onnx
```

### Using Pre-trained Models

```bash
# 1. Browse available models
langvision model-zoo list

# 2. Download a model
langvision model-zoo download vit_base_patch16_224

# 3. Fine-tune the downloaded model
langvision finetune --checkpoint models/vit_base_patch16_224.pth --dataset cifar100
```

## Tips and Best Practices

1. **Start with Basic Configuration**: Use `langvision config create --template basic` for simple tasks
2. **Use Advanced Features**: Enable LoRA, early stopping, and advanced LLM concepts for better results
3. **Monitor Training**: Use the built-in progress bars and logging to monitor training progress
4. **Validate Configurations**: Always validate your configuration files before training
5. **Export for Production**: Export trained models to ONNX or TorchScript for deployment

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Slow Training**: Enable CUDA benchmark mode and use multiple workers
3. **Configuration Errors**: Use `langvision config validate` to check your configuration
4. **Missing Dependencies**: Install optional dependencies with `pip install langvision[dev,docs,examples,gpu]`

### Getting Help

- Use `--help` with any command for detailed options
- Check the logs for detailed error messages
- Validate configurations before running
- Ensure all dependencies are installed

## Advanced Usage

### Custom Datasets

```bash
# Use custom dataset directory
langvision train --dataset custom --data_dir /path/to/custom/dataset
```

### Distributed Training

```bash
# Use multiple GPUs (if available)
langvision finetune --device cuda --batch_size 128
```

### Hyperparameter Tuning

```bash
# Create multiple configurations and compare
langvision config create --template advanced --output config1.yaml
langvision config create --template advanced --output config2.yaml
# Modify config2.yaml with different hyperparameters
langvision config diff config1.yaml config2.yaml
```

This CLI provides a complete solution for vision model training and deployment with Langvision!
