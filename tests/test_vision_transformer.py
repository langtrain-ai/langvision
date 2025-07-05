"""
Tests for VisionTransformer model.
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict, Any

from langvision.models.vision_transformer import VisionTransformer


class TestVisionTransformer:
    """Test cases for VisionTransformer model."""

    def test_model_creation(self, sample_config: Dict[str, Any]):
        """Test that VisionTransformer can be created with valid config."""
        model = VisionTransformer(
            img_size=sample_config["img_size"],
            patch_size=sample_config["patch_size"],
            in_chans=sample_config["in_chans"],
            num_classes=sample_config["num_classes"],
            embed_dim=sample_config["embed_dim"],
            depth=sample_config["depth"],
            num_heads=sample_config["num_heads"],
            mlp_ratio=sample_config["mlp_ratio"],
            lora_config=sample_config["lora"],
        )
        
        assert isinstance(model, VisionTransformer)
        assert model.num_classes == sample_config["num_classes"]
        assert model.embed_dim == sample_config["embed_dim"]

    def test_forward_pass(self, sample_model: VisionTransformer, sample_batch: torch.Tensor):
        """Test forward pass through the model."""
        model = sample_model
        batch_size = sample_batch.shape[0]
        
        with torch.no_grad():
            output = model(sample_batch)
        
        expected_shape = (batch_size, model.num_classes)
        assert output.shape == expected_shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_parameters(self, sample_model: VisionTransformer):
        """Test that model has trainable parameters."""
        total_params = sum(p.numel() for p in sample_model.parameters())
        trainable_params = sum(p.numel() for p in sample_model.parameters() if p.requires_grad)
        
        assert total_params > 0
        assert trainable_params > 0
        assert trainable_params <= total_params

    def test_lora_integration(self, sample_config: Dict[str, Any]):
        """Test that LoRA adapters are properly integrated."""
        model = VisionTransformer(
            img_size=sample_config["img_size"],
            patch_size=sample_config["patch_size"],
            in_chans=sample_config["in_chans"],
            num_classes=sample_config["num_classes"],
            embed_dim=sample_config["embed_dim"],
            depth=sample_config["depth"],
            num_heads=sample_config["num_heads"],
            mlp_ratio=sample_config["mlp_ratio"],
            lora_config=sample_config["lora"],
        )
        
        # Check that LoRA layers exist
        lora_layers = []
        for module in model.modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                lora_layers.append(module)
        
        assert len(lora_layers) > 0, "No LoRA layers found in model"

    @pytest.mark.parametrize("img_size", [224, 384, 512])
    def test_different_image_sizes(self, img_size: int, sample_config: Dict[str, Any]):
        """Test model with different input image sizes."""
        config = sample_config.copy()
        config["img_size"] = img_size
        
        model = VisionTransformer(
            img_size=config["img_size"],
            patch_size=config["patch_size"],
            in_chans=config["in_chans"],
            num_classes=config["num_classes"],
            embed_dim=config["embed_dim"],
            depth=config["depth"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            lora_config=config["lora"],
        )
        
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, img_size, img_size)
        
        with torch.no_grad():
            output = model(input_tensor)
        
        assert output.shape == (batch_size, config["num_classes"])

    def test_gradient_flow(self, sample_model: VisionTransformer, sample_batch: torch.Tensor, sample_labels: torch.Tensor):
        """Test that gradients flow through the model during training."""
        model = sample_model
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Forward pass
        output = model(sample_batch)
        loss = criterion(output, sample_labels)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "No gradients found after backward pass"

    @pytest.mark.slow
    def test_model_memory_usage(self, sample_model: VisionTransformer, sample_batch: torch.Tensor):
        """Test memory usage during forward pass."""
        model = sample_model
        
        if torch.cuda.is_available():
            model = model.cuda()
            sample_batch = sample_batch.cuda()
            
            # Get initial memory usage
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # Forward pass
            with torch.no_grad():
                _ = model(sample_batch)
            
            # Check memory usage
            final_memory = torch.cuda.memory_allocated()
            memory_used = final_memory - initial_memory
            
            assert memory_used > 0, "Model should use some GPU memory"
        else:
            # CPU test - just check that it runs without error
            with torch.no_grad():
                _ = model(sample_batch)

    def test_model_save_load(self, sample_model: VisionTransformer, tmp_path):
        """Test that model can be saved and loaded."""
        model = sample_model
        save_path = tmp_path / "model.pth"
        
        # Save model
        torch.save(model.state_dict(), save_path)
        assert save_path.exists()
        
        # Load model
        new_model = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=10,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            lora_config={"rank": 16, "alpha": 32, "dropout": 0.1},
        )
        new_model.load_state_dict(torch.load(save_path))
        
        # Test that loaded model produces same output
        test_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            original_output = model(test_input)
            loaded_output = new_model(test_input)
        
        torch.testing.assert_close(original_output, loaded_output)

    def test_invalid_config(self):
        """Test that invalid configurations raise appropriate errors."""
        with pytest.raises(ValueError):
            VisionTransformer(
                img_size=224,
                patch_size=32,  # Should be divisible by img_size
                in_chans=3,
                num_classes=10,
                embed_dim=768,
                depth=12,
                num_heads=13,  # Should be divisible by embed_dim
                mlp_ratio=4.0,
                lora_config={"rank": 16, "alpha": 32, "dropout": 0.1},
            ) 