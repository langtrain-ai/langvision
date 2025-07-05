"""
Pytest configuration and fixtures for langvision tests.
"""

import pytest
import torch
import numpy as np
from typing import Generator, Dict, Any

from langvision.models.vision_transformer import VisionTransformer
from langvision.utils.config import default_config


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Get the device to run tests on."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def sample_config() -> Dict[str, Any]:
    """Get a sample configuration for testing."""
    return {
        "img_size": 224,
        "patch_size": 16,
        "in_chans": 3,
        "num_classes": 10,
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "lora": {
            "rank": 16,
            "alpha": 32,
            "dropout": 0.1,
        }
    }


@pytest.fixture(scope="session")
def sample_model(sample_config: Dict[str, Any]) -> VisionTransformer:
    """Create a sample VisionTransformer model for testing."""
    return VisionTransformer(
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


@pytest.fixture
def sample_batch(device: torch.device) -> torch.Tensor:
    """Create a sample batch of images for testing."""
    batch_size = 4
    channels = 3
    height = 224
    width = 224
    return torch.randn(batch_size, channels, height, width, device=device)


@pytest.fixture
def sample_labels(device: torch.device) -> torch.Tensor:
    """Create sample labels for testing."""
    batch_size = 4
    num_classes = 10
    return torch.randint(0, num_classes, (batch_size,), device=device)


@pytest.fixture(autouse=True)
def set_random_seed() -> Generator[None, None, None]:
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    yield


@pytest.fixture(scope="session")
def test_data_dir(tmp_path_factory) -> str:
    """Create a temporary directory for test data."""
    return str(tmp_path_factory.mktemp("test_data"))


# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Skip GPU tests if CUDA is not available."""
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu) 