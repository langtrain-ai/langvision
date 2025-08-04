"""
Enhanced dataset classes with comprehensive data validation, augmentation, and multimodal support.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100
from PIL import Image
import pandas as pd
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Callable, Any
import numpy as np
import logging
from dataclasses import dataclass
import warnings
from collections import Counter
import cv2


@dataclass
class DatasetConfig:
    """Configuration for dataset parameters."""
    # Basic settings
    root_dir: str
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Augmentation settings
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Multimodal settings
    text_max_length: int = 77
    text_tokenizer: Optional[str] = None
    
    # Validation settings
    validate_images: bool = True
    min_image_size: Tuple[int, int] = (32, 32)
    max_image_size: Tuple[int, int] = (4096, 4096)
    allowed_formats: List[str] = None
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Validate splits sum to 1.0
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")


class ImageValidator:
    """Comprehensive image validation utilities."""
    
    def __init__(self, 
                 min_size: Tuple[int, int] = (32, 32),
                 max_size: Tuple[int, int] = (4096, 4096),
                 allowed_formats: List[str] = None):
        self.min_size = min_size
        self.max_size = max_size
        self.allowed_formats = allowed_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.logger = logging.getLogger(__name__)
    
    def validate_image_file(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate a single image file."""
        image_path = Path(image_path)
        result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'metadata': {}
        }
        
        # Check if file exists
        if not image_path.exists():
            result['valid'] = False
            result['errors'].append(f"File does not exist: {image_path}")
            return result
        
        # Check file extension
        if image_path.suffix.lower() not in self.allowed_formats:
            result['valid'] = False
            result['errors'].append(f"Unsupported format: {image_path.suffix}")
            return result
        
        try:
            # Try to open and validate image
            with Image.open(image_path) as img:
                width, height = img.size
                result['metadata'].update({
                    'width': width,
                    'height': height,
                    'mode': img.mode,
                    'format': img.format,
                    'file_size': image_path.stat().st_size
                })
                
                # Check image dimensions
                if width < self.min_size[0] or height < self.min_size[1]:
                    result['warnings'].append(f"Image too small: {width}x{height} < {self.min_size}")
                
                if width > self.max_size[0] or height > self.max_size[1]:
                    result['warnings'].append(f"Image very large: {width}x{height} > {self.max_size}")
                
                # Check if image is corrupted by trying to load it
                img.verify()
                
        except Exception as e:
            result['valid'] = False
            result['errors'].append(f"Failed to open image: {str(e)}")
        
        return result
    
    def validate_dataset(self, image_paths: List[Union[str, Path]]) -> Dict[str, Any]:
        """Validate an entire dataset."""
        results = {
            'total_images': len(image_paths),
            'valid_images': 0,
            'invalid_images': 0,
            'warnings_count': 0,
            'errors': [],
            'warnings': [],
            'metadata_stats': {}
        }
        
        valid_images = []
        widths, heights, file_sizes = [], [], []
        
        for image_path in image_paths:
            validation_result = self.validate_image_file(image_path)
            
            if validation_result['valid']:
                results['valid_images'] += 1
                valid_images.append(image_path)
                
                # Collect metadata for statistics
                metadata = validation_result['metadata']
                widths.append(metadata['width'])
                heights.append(metadata['height'])
                file_sizes.append(metadata['file_size'])
            else:
                results['invalid_images'] += 1
                results['errors'].extend(validation_result['errors'])
            
            results['warnings'].extend(validation_result['warnings'])
            results['warnings_count'] += len(validation_result['warnings'])
        
        # Compute metadata statistics
        if valid_images:
            results['metadata_stats'] = {
                'width_stats': {
                    'mean': np.mean(widths),
                    'std': np.std(widths),
                    'min': np.min(widths),
                    'max': np.max(widths)
                },
                'height_stats': {
                    'mean': np.mean(heights),
                    'std': np.std(heights),
                    'min': np.min(heights),
                    'max': np.max(heights)
                },
                'file_size_stats': {
                    'mean_mb': np.mean(file_sizes) / (1024**2),
                    'std_mb': np.std(file_sizes) / (1024**2),
                    'min_mb': np.min(file_sizes) / (1024**2),
                    'max_mb': np.max(file_sizes) / (1024**2)
                }
            }
        
        return results


class SmartAugmentation:
    """Intelligent data augmentation with adaptive strategies."""
    
    def __init__(self, 
                 image_size: Tuple[int, int] = (224, 224),
                 strength: float = 0.5,
                 preserve_aspect_ratio: bool = True):
        self.image_size = image_size
        self.strength = strength
        self.preserve_aspect_ratio = preserve_aspect_ratio
        
        # Base transforms
        self.base_transforms = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Augmentation transforms based on strength
        self.augmentation_transforms = self._create_augmentation_transforms()
    
    def _create_augmentation_transforms(self) -> transforms.Compose:
        """Create augmentation transforms based on strength parameter."""
        aug_list = []
        
        if self.preserve_aspect_ratio:
            aug_list.append(transforms.Resize(min(self.image_size)))
            aug_list.append(transforms.CenterCrop(self.image_size))
        else:
            aug_list.append(transforms.Resize(self.image_size))
        
        # Add augmentations based on strength
        if self.strength > 0.1:
            aug_list.extend([
                transforms.RandomHorizontalFlip(p=0.5 * self.strength),
                transforms.RandomRotation(degrees=10 * self.strength),
            ])
        
        if self.strength > 0.3:
            aug_list.extend([
                transforms.ColorJitter(
                    brightness=0.2 * self.strength,
                    contrast=0.2 * self.strength,
                    saturation=0.2 * self.strength,
                    hue=0.1 * self.strength
                ),
                transforms.RandomAffine(
                    degrees=5 * self.strength,
                    translate=(0.1 * self.strength, 0.1 * self.strength),
                    scale=(1 - 0.1 * self.strength, 1 + 0.1 * self.strength)
                )
            ])
        
        if self.strength > 0.5:
            aug_list.extend([
                transforms.RandomPerspective(distortion_scale=0.2 * self.strength, p=0.3),
                transforms.RandomErasing(p=0.2 * self.strength, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        
        # Always add tensor conversion and normalization
        aug_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transforms.Compose(aug_list)
    
    def get_train_transforms(self) -> transforms.Compose:
        """Get training transforms with augmentation."""
        return self.augmentation_transforms
    
    def get_val_transforms(self) -> transforms.Compose:
        """Get validation transforms without augmentation."""
        return self.base_transforms


class EnhancedImageDataset(Dataset):
    """Enhanced image dataset with validation, smart augmentation, and error handling."""
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 config: Optional[DatasetConfig] = None,
                 transform: Optional[Callable] = None,
                 validate_on_init: bool = True,
                 cache_images: bool = False):
        
        self.root_dir = Path(root_dir)
        self.config = config or DatasetConfig(root_dir=str(root_dir))
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Find all image files
        self.image_paths = self._find_image_files()
        self.logger.info(f"Found {len(self.image_paths)} image files")
        
        # Setup transforms
        if transform is None:
            augmentation = SmartAugmentation(
                image_size=self.config.image_size,
                strength=self.config.augmentation_strength
            )
            self.transform = augmentation.get_train_transforms() if self.config.use_augmentation else augmentation.get_val_transforms()
        else:
            self.transform = transform
        
        # Validate dataset if requested
        if validate_on_init and self.config.validate_images:
            self._validate_dataset()
        
        # Create class mapping if this is a classification dataset
        self.classes, self.class_to_idx = self._create_class_mapping()
        
        # Create labels
        self.labels = self._create_labels()
    
    def _find_image_files(self) -> List[Path]:
        """Find all valid image files in the directory."""
        image_paths = []
        
        for ext in self.config.allowed_formats:
            pattern = f"**/*{ext}"
            image_paths.extend(self.root_dir.glob(pattern))
            # Also check uppercase extensions
            pattern = f"**/*{ext.upper()}"
            image_paths.extend(self.root_dir.glob(pattern))
        
        return sorted(list(set(image_paths)))  # Remove duplicates and sort
    
    def _validate_dataset(self):
        """Validate the entire dataset."""
        validator = ImageValidator(
            min_size=self.config.min_image_size,
            max_size=self.config.max_image_size,
            allowed_formats=self.config.allowed_formats
        )
        
        validation_results = validator.validate_dataset(self.image_paths)
        
        # Log validation results
        self.logger.info(f"Dataset validation completed:")
        self.logger.info(f"  Total images: {validation_results['total_images']}")
        self.logger.info(f"  Valid images: {validation_results['valid_images']}")
        self.logger.info(f"  Invalid images: {validation_results['invalid_images']}")
        self.logger.info(f"  Warnings: {validation_results['warnings_count']}")
        
        # Remove invalid images
        if validation_results['invalid_images'] > 0:
            self.logger.warning(f"Removing {validation_results['invalid_images']} invalid images")
            valid_paths = []
            for image_path in self.image_paths:
                result = validator.validate_image_file(image_path)
                if result['valid']:
                    valid_paths.append(image_path)
            self.image_paths = valid_paths
        
        # Log metadata statistics
        if 'metadata_stats' in validation_results:
            stats = validation_results['metadata_stats']
            self.logger.info(f"Image statistics:")
            self.logger.info(f"  Width: {stats['width_stats']['mean']:.1f}±{stats['width_stats']['std']:.1f}")
            self.logger.info(f"  Height: {stats['height_stats']['mean']:.1f}±{stats['height_stats']['std']:.1f}")
            self.logger.info(f"  File size: {stats['file_size_stats']['mean_mb']:.2f}±{stats['file_size_stats']['std_mb']:.2f} MB")
    
    def _create_class_mapping(self) -> Tuple[List[str], Dict[str, int]]:
        """Create class mapping from directory structure."""
        # Assume directory structure: root_dir/class_name/image_files
        classes = set()
        
        for image_path in self.image_paths:
            # Get parent directory name as class
            class_name = image_path.parent.name
            if class_name != self.root_dir.name:  # Skip if image is directly in root
                classes.add(class_name)
        
        classes = sorted(list(classes))
        class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
        
        return classes, class_to_idx
    
    def _create_labels(self) -> List[int]:
        """Create labels for each image based on directory structure."""
        labels = []
        
        for image_path in self.image_paths:
            class_name = image_path.parent.name
            if class_name in self.class_to_idx:
                labels.append(self.class_to_idx[class_name])
            else:
                labels.append(-1)  # Unknown class
        
        return labels
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single item from the dataset."""
        image_path = self.image_paths[idx]
        
        try:
            # Load image from cache or disk
            if self.cache_images and str(image_path) in self.image_cache:
                image = self.image_cache[str(image_path)]
            else:
                image = Image.open(image_path).convert('RGB')
                if self.cache_images:
                    self.image_cache[str(image_path)] = image
            
            # Apply transforms
            if self.transform:
                image = self.transform(image)
            
            # Create sample dictionary
            sample = {
                'images': image,
                'labels': self.labels[idx] if idx < len(self.labels) else -1,
                'image_paths': str(image_path),
                'class_names': self.classes[self.labels[idx]] if self.labels[idx] >= 0 and self.labels[idx] < len(self.classes) else 'unknown'
            }
            
            return sample
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a dummy sample to avoid breaking the dataloader
            dummy_image = torch.zeros(3, *self.config.image_size)
            return {
                'images': dummy_image,
                'labels': -1,
                'image_paths': str(image_path),
                'class_names': 'error'
            }
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get the distribution of classes in the dataset."""
        if not self.classes:
            return {}
        
        label_counts = Counter(self.labels)
        return {self.classes[label]: count for label, count in label_counts.items() if label >= 0}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get comprehensive dataset information."""
        return {
            'num_samples': len(self),
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': self.get_class_distribution(),
            'image_size': self.config.image_size,
            'root_directory': str(self.root_dir),
            'cache_enabled': self.cache_images
        }


class MultimodalDataset(EnhancedImageDataset):
    """Multimodal dataset supporting both images and text."""
    
    def __init__(self,
                 root_dir: Union[str, Path],
                 annotations_file: Optional[Union[str, Path]] = None,
                 config: Optional[DatasetConfig] = None,
                 **kwargs):
        
        super().__init__(root_dir, config, **kwargs)
        
        self.annotations_file = Path(annotations_file) if annotations_file else None
        self.annotations = self._load_annotations()
        
        # Setup text processing
        if self.config.text_tokenizer:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_tokenizer)
        else:
            self.tokenizer = None
    
    def _load_annotations(self) -> Dict[str, Any]:
        """Load text annotations from file."""
        if not self.annotations_file or not self.annotations_file.exists():
            self.logger.warning("No annotations file found, using image filenames as text")
            return {}
        
        try:
            if self.annotations_file.suffix == '.json':
                with open(self.annotations_file, 'r') as f:
                    return json.load(f)
            elif self.annotations_file.suffix == '.csv':
                df = pd.read_csv(self.annotations_file)
                return df.to_dict('records')
            else:
                self.logger.error(f"Unsupported annotations format: {self.annotations_file.suffix}")
                return {}
        except Exception as e:
            self.logger.error(f"Failed to load annotations: {str(e)}")
            return {}
    
    def _get_text_for_image(self, image_path: Path) -> str:
        """Get text description for an image."""
        image_name = image_path.name
        
        # Try to find annotation by filename
        if isinstance(self.annotations, dict):
            return self.annotations.get(image_name, image_path.stem)
        elif isinstance(self.annotations, list):
            for annotation in self.annotations:
                if annotation.get('filename') == image_name or annotation.get('image_path') == str(image_path):
                    return annotation.get('caption', annotation.get('text', image_path.stem))
        
        # Fallback to filename
        return image_path.stem.replace('_', ' ').replace('-', ' ')
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a multimodal sample."""
        sample = super().__getitem__(idx)
        
        # Add text information
        image_path = Path(sample['image_paths'])
        text = self._get_text_for_image(image_path)
        
        sample['texts'] = text
        
        # Tokenize text if tokenizer is available
        if self.tokenizer:
            tokens = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=self.config.text_max_length,
                return_tensors='pt'
            )
            sample['text_tokens'] = {k: v.squeeze(0) for k, v in tokens.items()}
        
        return sample


def create_enhanced_dataloaders(config: DatasetConfig,
                              dataset_type: str = "image",
                              annotations_file: Optional[str] = None) -> Dict[str, DataLoader]:
    """Factory function to create enhanced dataloaders."""
    
    # Create dataset
    if dataset_type == "multimodal":
        dataset_class = MultimodalDataset
        dataset_kwargs = {"annotations_file": annotations_file}
    else:
        dataset_class = EnhancedImageDataset
        dataset_kwargs = {}
    
    # Create full dataset
    full_dataset = dataset_class(
        root_dir=config.root_dir,
        config=config,
        **dataset_kwargs
    )
    
    # Split dataset
    total_size = len(full_dataset)
    train_size = int(config.train_split * total_size)
    val_size = int(config.val_split * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size]
    )
    
    # Create dataloaders
    dataloaders = {}
    
    if train_size > 0:
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True
        )
    
    if val_size > 0:
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
    
    if test_size > 0:
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory
        )
    
    return dataloaders
