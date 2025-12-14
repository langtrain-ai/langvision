"""
Langvision Dataset Classes

Provides dataset classes for Vision LLM fine-tuning as documented in README:
- CIFAR10Dataset, CIFAR100Dataset
- ImageFolderDataset for custom datasets
- VQADataset for Visual QA tasks
- CaptioningDataset for image captioning
"""

import os
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List, Tuple, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


@dataclass
class DatasetConfig:
    """Configuration for datasets."""
    img_size: int = 224
    normalize: bool = True
    augment: bool = False
    cache_images: bool = False


class BaseDataset(Dataset):
    """Base class for all Langvision datasets."""
    
    def __init__(
        self,
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        self.transform = transform
        self.config = config or DatasetConfig()
    
    @property
    def default_transform(self) -> Callable:
        """Get default transform for the dataset."""
        transform_list = [
            transforms.Resize((self.config.img_size, self.config.img_size)),
            transforms.ToTensor(),
        ]
        
        if self.config.normalize:
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        return transforms.Compose(transform_list)


class CIFAR10Dataset(BaseDataset):
    """
    CIFAR-10 dataset wrapper for Langvision.
    
    Usage:
        dataset = CIFAR10Dataset(train=True)
        image, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__(transform, config)
        
        self._transform = transform or self.default_transform
        
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=download,
            transform=self._transform,
        )
        
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class CIFAR100Dataset(BaseDataset):
    """
    CIFAR-100 dataset wrapper for Langvision.
    
    Usage:
        dataset = CIFAR100Dataset(train=True)
        image, label = dataset[0]
    """
    
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__(transform, config)
        
        self._transform = transform or self.default_transform
        
        self.dataset = datasets.CIFAR100(
            root=root,
            train=train,
            download=download,
            transform=self._transform,
        )
        
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class ImageFolderDataset(BaseDataset):
    """
    Custom image folder dataset.
    
    Expects folder structure:
        root/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
    
    Usage:
        dataset = ImageFolderDataset(
            root="/path/to/dataset",
            split="train",
            transform=my_transform
        )
    """
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__(transform, config)
        
        # Handle split subdirectory
        root_path = Path(root)
        if split and (root_path / split).exists():
            root_path = root_path / split
        
        self._transform = transform or self.default_transform
        
        self.dataset = datasets.ImageFolder(
            root=str(root_path),
            transform=self._transform,
        )
        
        self.classes = self.dataset.classes
        self.num_classes = len(self.classes)
        self.class_to_idx = self.dataset.class_to_idx
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]


class VQADataset(BaseDataset):
    """
    Visual Question Answering dataset for Vision LLM fine-tuning.
    
    Expected data format (JSON):
    [
        {
            "image": "path/to/image.jpg",
            "question": "What is in the image?",
            "answer": "A cat sitting on a sofa"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer=None,
        max_length: int = 512,
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__(transform, config)
        
        import json
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._transform = transform or self.default_transform
        
        # Load data
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load image
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        
        if self._transform:
            image = self._transform(image)
        
        result = {
            "image": image,
            "question": item["question"],
            "answer": item.get("answer", ""),
        }
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            prompt = f"Question: {item['question']}\nAnswer:"
            target = item.get("answer", "")
            
            result["input_ids"] = self.tokenizer(
                prompt, 
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze()
            
            if target:
                result["labels"] = self.tokenizer(
                    target,
                    max_length=self.max_length,
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"].squeeze()
        
        return result


class CaptioningDataset(BaseDataset):
    """
    Image captioning dataset for Vision LLM fine-tuning.
    
    Expected data format (JSON):
    [
        {
            "image": "path/to/image.jpg",
            "caption": "A beautiful sunset over the ocean"
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer=None,
        max_length: int = 256,
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__(transform, config)
        
        import json
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._transform = transform or self.default_transform
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load image
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        
        if self._transform:
            image = self._transform(image)
        
        result = {
            "image": image,
            "caption": item["caption"],
        }
        
        # Tokenize if tokenizer provided
        if self.tokenizer is not None:
            result["labels"] = self.tokenizer(
                item["caption"],
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt"
            )["input_ids"].squeeze()
        
        return result


class PreferenceDataset(BaseDataset):
    """
    Preference dataset for DPO/RLHF training.
    
    Expected data format (JSON):
    [
        {
            "image": "path/to/image.jpg",
            "prompt": "Describe this image",
            "chosen": "A detailed accurate description...",
            "rejected": "A vague incorrect description..."
        },
        ...
    ]
    """
    
    def __init__(
        self,
        data_path: str,
        image_dir: str,
        tokenizer=None,
        max_length: int = 512,
        transform: Optional[Callable] = None,
        config: Optional[DatasetConfig] = None,
    ):
        super().__init__(transform, config)
        
        import json
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._transform = transform or self.default_transform
        
        with open(data_path, 'r') as f:
            self.data = json.load(f)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Load image
        image_path = self.image_dir / item["image"]
        image = Image.open(image_path).convert("RGB")
        
        if self._transform:
            image = self._transform(image)
        
        result = {
            "image": image,
            "prompt": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
        
        return result


# Legacy function for backward compatibility
def get_dataset(name: str, data_dir: str, train: bool, img_size: int = 224):
    """Get a dataset by name (legacy function)."""
    config = DatasetConfig(img_size=img_size)
    
    if name.lower() == 'cifar10':
        return CIFAR10Dataset(root=data_dir, train=train, config=config)
    elif name.lower() == 'cifar100':
        return CIFAR100Dataset(root=data_dir, train=train, config=config)
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    "DatasetConfig",
    "BaseDataset",
    "CIFAR10Dataset",
    "CIFAR100Dataset",
    "ImageFolderDataset",
    "VQADataset",
    "CaptioningDataset",
    "PreferenceDataset",
    "get_dataset",
]