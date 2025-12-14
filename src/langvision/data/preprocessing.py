"""
Data Preprocessing Utilities for Vision LLMs

Preprocessing pipelines for different tasks:
- VQA dataset preparation
- Image captioning formatting
- Multi-modal conversation formatting
- OCR data preparation
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import re
from PIL import Image
import logging

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing."""
    # Image
    image_size: Tuple[int, int] = (384, 384)
    normalize: bool = True
    
    # Text
    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"
    
    # Prompt format
    prompt_template: str = "default"  # default, llava, qwen, chatml
    
    # Labels
    ignore_index: int = -100


class ConversationFormatter:
    """
    Format multi-turn conversations for different Vision LLMs.
    
    Supports multiple prompt formats:
    - default: Simple "User: ... Assistant: ..." format
    - llava: LLaVA conversation format
    - qwen: Qwen-VL format
    - chatml: ChatML format
    """
    
    TEMPLATES = {
        "default": {
            "system": "System: {content}\n",
            "user": "User: {content}\n",
            "assistant": "Assistant: {content}\n",
            "image_token": "<image>",
        },
        "llava": {
            "system": "{content}\n",
            "user": "USER: {content}\n",
            "assistant": "ASSISTANT: {content}\n",
            "image_token": "<image>",
        },
        "qwen": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
            "image_token": "<img></img>",
        },
        "chatml": {
            "system": "<|im_start|>system\n{content}<|im_end|>\n",
            "user": "<|im_start|>user\n{content}<|im_end|>\n",
            "assistant": "<|im_start|>assistant\n{content}<|im_end|>\n",
            "image_token": "<image>",
        },
    }
    
    def __init__(self, template: str = "default"):
        if template not in self.TEMPLATES:
            raise ValueError(f"Unknown template: {template}. Choose from {list(self.TEMPLATES.keys())}")
        
        self.template = self.TEMPLATES[template]
    
    def format_conversation(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a conversation into a prompt string.
        
        Args:
            messages: List of {"role": "user/assistant/system", "content": "..."}
            add_generation_prompt: Add prompt for model to generate
        
        Returns:
            Formatted conversation string
        """
        formatted = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role in self.template:
                formatted += self.template[role].format(content=content)
        
        if add_generation_prompt:
            formatted += self.template["assistant"].format(content="").rstrip()
        
        return formatted
    
    def format_single_turn(
        self,
        question: str,
        answer: Optional[str] = None,
        system_prompt: Optional[str] = None,
        has_image: bool = True,
    ) -> str:
        """Format a single-turn QA as conversation."""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add image token if needed
        if has_image:
            question = f"{self.template['image_token']}\n{question}"
        
        messages.append({"role": "user", "content": question})
        
        if answer:
            messages.append({"role": "assistant", "content": answer})
            return self.format_conversation(messages, add_generation_prompt=False)
        
        return self.format_conversation(messages, add_generation_prompt=True)


class VQAPreprocessor:
    """
    Preprocessor for Visual Question Answering datasets.
    
    Converts VQA data to model-ready format with:
    - Image preprocessing
    - Prompt formatting
    - Tokenization
    - Label masking
    """
    
    def __init__(
        self,
        tokenizer,
        processor=None,
        config: Optional[PreprocessingConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or PreprocessingConfig()
        self.formatter = ConversationFormatter(self.config.prompt_template)
    
    def preprocess(
        self,
        image: Union[str, Image.Image],
        question: str,
        answer: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a single VQA example.
        
        Args:
            image: Image path or PIL Image
            question: Question text
            answer: Optional answer (for training)
        
        Returns:
            Dictionary with input_ids, attention_mask, pixel_values, labels
        """
        # Load image
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Format prompt
        prompt = self.formatter.format_single_turn(question, answer, has_image=True)
        
        # Process with processor or manually
        if self.processor is not None:
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=self.config.truncation,
                padding=self.config.padding,
            )
        else:
            # Manual processing
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=self.config.truncation,
                padding=self.config.padding,
            )
            
            # Process image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ) if self.config.normalize else transforms.Lambda(lambda x: x),
            ])
            inputs["pixel_values"] = transform(image).unsqueeze(0)
        
        # Create labels (mask prompt tokens)
        if answer is not None:
            labels = inputs["input_ids"].clone()
            
            # Find where the answer starts
            prompt_only = self.formatter.format_single_turn(question, None, has_image=True)
            prompt_ids = self.tokenizer(prompt_only, return_tensors="pt")["input_ids"]
            prompt_len = prompt_ids.shape[1]
            
            # Mask prompt tokens
            labels[:, :prompt_len] = self.config.ignore_index
            inputs["labels"] = labels
        
        # Squeeze batch dimension
        return {k: v.squeeze(0) for k, v in inputs.items()}
    
    def preprocess_batch(
        self,
        images: List[Union[str, Image.Image]],
        questions: List[str],
        answers: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a batch of VQA examples."""
        batch = [
            self.preprocess(img, q, a)
            for img, q, a in zip(
                images,
                questions,
                answers or [None] * len(images),
            )
        ]
        
        # Stack into batch
        return {
            k: torch.stack([item[k] for item in batch])
            for k in batch[0].keys()
        }


class CaptioningPreprocessor:
    """Preprocessor for image captioning datasets."""
    
    def __init__(
        self,
        tokenizer,
        processor=None,
        config: Optional[PreprocessingConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or PreprocessingConfig()
        self.formatter = ConversationFormatter(self.config.prompt_template)
    
    def preprocess(
        self,
        image: Union[str, Image.Image],
        caption: Optional[str] = None,
        prompt: str = "Describe this image in detail:",
    ) -> Dict[str, torch.Tensor]:
        """Preprocess a captioning example."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Format as QA
        full_prompt = self.formatter.format_single_turn(prompt, caption, has_image=True)
        
        if self.processor is not None:
            inputs = self.processor(
                text=full_prompt,
                images=image,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
            )
        else:
            inputs = self.tokenizer(
                full_prompt,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
            )
            
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
            ])
            inputs["pixel_values"] = transform(image).unsqueeze(0)
        
        # Create labels
        if caption is not None:
            labels = inputs["input_ids"].clone()
            prompt_only = self.formatter.format_single_turn(prompt, None, has_image=True)
            prompt_ids = self.tokenizer(prompt_only, return_tensors="pt")["input_ids"]
            labels[:, :prompt_ids.shape[1]] = self.config.ignore_index
            inputs["labels"] = labels
        
        return {k: v.squeeze(0) for k, v in inputs.items()}


class PreferencePreprocessor:
    """
    Preprocessor for preference/DPO datasets.
    
    Formats data for Direct Preference Optimization training.
    """
    
    def __init__(
        self,
        tokenizer,
        processor=None,
        config: Optional[PreprocessingConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or PreprocessingConfig()
        self.formatter = ConversationFormatter(self.config.prompt_template)
    
    def preprocess(
        self,
        image: Union[str, Image.Image],
        prompt: str,
        chosen: str,
        rejected: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess a preference example.
        
        Returns dict with chosen_*, rejected_*, and pixel_values.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Format chosen and rejected
        chosen_prompt = self.formatter.format_single_turn(prompt, chosen, has_image=True)
        rejected_prompt = self.formatter.format_single_turn(prompt, rejected, has_image=True)
        
        # Tokenize
        chosen_inputs = self.tokenizer(
            chosen_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=self.config.padding,
        )
        
        rejected_inputs = self.tokenizer(
            rejected_prompt,
            return_tensors="pt",
            max_length=self.config.max_length,
            truncation=True,
            padding=self.config.padding,
        )
        
        # Process image
        if self.processor is not None:
            img_inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = img_inputs["pixel_values"]
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize(self.config.image_size),
                transforms.ToTensor(),
            ])
            pixel_values = transform(image).unsqueeze(0)
        
        return {
            "chosen_input_ids": chosen_inputs["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_inputs["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_inputs["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values.squeeze(0),
        }


def load_and_preprocess_dataset(
    dataset_path: str,
    task: str = "vqa",
    tokenizer=None,
    processor=None,
    **kwargs,
) -> Dataset:
    """
    Load and preprocess a dataset for training.
    
    Args:
        dataset_path: Path to dataset (JSON/JSONL file or directory)
        task: Task type (vqa, captioning, preference)
        tokenizer: Tokenizer for text
        processor: Optional multi-modal processor
    
    Returns:
        Preprocessed dataset
    """
    # Load raw data
    path = Path(dataset_path)
    
    if path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
    elif path.suffix == ".jsonl":
        data = []
        with open(path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Get preprocessor
    config = PreprocessingConfig(**kwargs)
    
    if task == "vqa":
        preprocessor = VQAPreprocessor(tokenizer, processor, config)
    elif task == "captioning":
        preprocessor = CaptioningPreprocessor(tokenizer, processor, config)
    elif task == "preference":
        preprocessor = PreferencePreprocessor(tokenizer, processor, config)
    else:
        raise ValueError(f"Unknown task: {task}")
    
    # Create dataset class
    class PreprocessedDataset(Dataset):
        def __init__(self, data, preprocessor, task):
            self.data = data
            self.preprocessor = preprocessor
            self.task = task
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            item = self.data[idx]
            
            if self.task == "vqa":
                return self.preprocessor.preprocess(
                    item["image"],
                    item["question"],
                    item.get("answer"),
                )
            elif self.task == "captioning":
                return self.preprocessor.preprocess(
                    item["image"],
                    item.get("caption"),
                )
            elif self.task == "preference":
                return self.preprocessor.preprocess(
                    item["image"],
                    item["prompt"],
                    item["chosen"],
                    item["rejected"],
                )
    
    return PreprocessedDataset(data, preprocessor, task)
