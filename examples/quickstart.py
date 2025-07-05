#!/usr/bin/env python3
"""
Quickstart example for langvision.

This script demonstrates basic usage of the langvision library for
vision transformer training with LoRA fine-tuning.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from langvision import VisionTransformer, Trainer, default_config


def create_dummy_dataset(num_samples=1000, num_classes=10):
    """Create a dummy dataset for demonstration."""
    # Generate random images
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Split into train/val
    train_size = int(0.8 * num_samples)
    train_images = images[:train_size]
    train_labels = labels[:train_size]
    val_images = images[train_size:]
    val_labels = labels[train_size:]
    
    return (train_images, train_labels), (val_images, val_labels)


def main():
    """Main function demonstrating langvision usage."""
    print("🚀 Langvision Quickstart Example")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    print("\n📦 Creating Vision Transformer model...")
    model = VisionTransformer(
        img_size=default_config['img_size'],
        patch_size=default_config['patch_size'],
        in_chans=default_config['in_chans'],
        num_classes=10,  # 10 classes for our dummy dataset
        embed_dim=default_config['embed_dim'],
        depth=6,  # Shallow for quick demo
        num_heads=default_config['num_heads'],
        mlp_ratio=default_config['mlp_ratio'],
        lora_config=default_config['lora'],
    ).to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dummy dataset
    print("\n📊 Creating dummy dataset...")
    (train_images, train_labels), (val_images, val_labels) = create_dummy_dataset()
    
    train_dataset = TensorDataset(train_images, train_labels)
    val_dataset = TensorDataset(val_images, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Setup training
    print("\n⚙️ Setting up training...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=5,  # Quick demo
        save_dir="./checkpoints",
        log_interval=10,
    )
    
    # Train the model
    print("\n🎯 Starting training...")
    try:
        trainer.train()
        print("✅ Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    # Test inference
    print("\n🔍 Testing inference...")
    model.eval()
    with torch.no_grad():
        test_input = torch.randn(1, 3, 224, 224).to(device)
        output = model(test_input)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {probabilities[0, predicted_class]:.3f}")
    
    print("\n🎉 Quickstart example completed!")
    print("\nNext steps:")
    print("1. Try with your own dataset")
    print("2. Experiment with different LoRA configurations")
    print("3. Use pre-trained models from the model zoo")
    print("4. Check the documentation for advanced features")


if __name__ == "__main__":
    main() 