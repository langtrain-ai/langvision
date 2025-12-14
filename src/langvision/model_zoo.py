"""
Model Zoo for Langvision - Pre-trained Vision LLM models for fine-tuning.

Supports popular Vision LLMs including LLaVA, BLIP-2, Qwen-VL, InternVL, and more.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# Vision LLM Model Categories
MODEL_CATEGORIES = {
    "vision_language": "Vision-Language Models (VLMs)",
    "vision_transformer": "Vision Transformers (ViT)",
    "multimodal": "Multimodal Foundation Models",
}

# Default model configurations - Vision LLMs for fine-tuning
DEFAULT_MODELS = {
    # ========================
    # Vision-Language Models (VLMs)
    # ========================
    "llava-v1.6-7b": {
        "name": "llava-v1.6-7b",
        "type": "vision_language",
        "family": "LLaVA",
        "size": "7B",
        "description": "LLaVA 1.6 - Large Language and Vision Assistant for visual instruction tuning",
        "use_cases": ["Visual QA", "Image Captioning", "Visual Reasoning", "OCR"],
        "base_llm": "Vicuna-7B",
        "vision_encoder": "CLIP ViT-L/14",
        "config": {
            "vision_tower": "openai/clip-vit-large-patch14-336",
            "mm_projector_type": "mlp2x_gelu",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "max_position_embeddings": 4096,
        },
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended_lora_r": 64,
    },
    "llava-v1.6-13b": {
        "name": "llava-v1.6-13b",
        "type": "vision_language",
        "family": "LLaVA",
        "size": "13B",
        "description": "LLaVA 1.6 13B - Enhanced visual understanding with larger language model",
        "use_cases": ["Visual QA", "Image Captioning", "Visual Reasoning", "Document Understanding"],
        "base_llm": "Vicuna-13B",
        "vision_encoder": "CLIP ViT-L/14",
        "config": {
            "vision_tower": "openai/clip-vit-large-patch14-336",
            "mm_projector_type": "mlp2x_gelu",
            "hidden_size": 5120,
            "num_attention_heads": 40,
            "max_position_embeddings": 4096,
        },
        "lora_targets": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended_lora_r": 64,
    },
    "qwen-vl-chat": {
        "name": "qwen-vl-chat",
        "type": "vision_language",
        "family": "Qwen-VL",
        "size": "9.6B",
        "description": "Qwen-VL Chat - Alibaba's vision-language model with strong multilingual support",
        "use_cases": ["Visual QA", "Image Captioning", "Chinese VQA", "Grounding"],
        "base_llm": "Qwen-7B",
        "vision_encoder": "ViT-bigG",
        "config": {
            "visual_encoder": "openclip_g",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "image_size": 448,
        },
        "lora_targets": ["c_attn", "c_proj", "w1", "w2"],
        "recommended_lora_r": 64,
    },
    "qwen2-vl-7b": {
        "name": "qwen2-vl-7b",
        "type": "vision_language",
        "family": "Qwen2-VL",
        "size": "7B",
        "description": "Qwen2-VL 7B - Next-gen vision-language model with dynamic resolution",
        "use_cases": ["Visual QA", "Video Understanding", "Multi-image", "Document OCR"],
        "base_llm": "Qwen2-7B",
        "vision_encoder": "ViT-600M",
        "config": {
            "hidden_size": 3584,
            "num_attention_heads": 28,
            "dynamic_resolution": True,
            "max_pixels": 1280 * 28 * 28,
        },
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended_lora_r": 64,
    },
    "blip2-flan-t5-xl": {
        "name": "blip2-flan-t5-xl",
        "type": "vision_language",
        "family": "BLIP-2",
        "size": "3.7B",
        "description": "BLIP-2 with Flan-T5-XL - Efficient vision-language pre-training with Q-Former",
        "use_cases": ["Image Captioning", "Visual QA", "Image-Text Retrieval"],
        "base_llm": "Flan-T5-XL",
        "vision_encoder": "EVA-CLIP ViT-G",
        "config": {
            "vision_model": "eva_clip_g",
            "text_model": "google/flan-t5-xl",
            "qformer_layers": 12,
            "num_query_tokens": 32,
        },
        "lora_targets": ["q", "v", "wi_0", "wi_1", "wo"],
        "recommended_lora_r": 32,
    },
    "blip2-opt-6.7b": {
        "name": "blip2-opt-6.7b",
        "type": "vision_language",
        "family": "BLIP-2",
        "size": "7.8B",
        "description": "BLIP-2 with OPT-6.7B - Large-scale vision-language model",
        "use_cases": ["Image Captioning", "Visual QA", "Zero-shot Classification"],
        "base_llm": "OPT-6.7B",
        "vision_encoder": "EVA-CLIP ViT-G",
        "config": {
            "vision_model": "eva_clip_g",
            "text_model": "facebook/opt-6.7b",
            "qformer_layers": 12,
            "num_query_tokens": 32,
        },
        "lora_targets": ["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
        "recommended_lora_r": 32,
    },
    "internvl2-8b": {
        "name": "internvl2-8b",
        "type": "vision_language",
        "family": "InternVL",
        "size": "8B",
        "description": "InternVL2 8B - Scaling up vision-language models with InternLM2",
        "use_cases": ["Visual QA", "OCR", "Chart/Document Understanding", "Multi-image"],
        "base_llm": "InternLM2-Chat-7B",
        "vision_encoder": "InternViT-300M",
        "config": {
            "vision_hidden_size": 1024,
            "llm_hidden_size": 4096,
            "dynamic_image_size": True,
            "max_num_tiles": 12,
        },
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended_lora_r": 128,
    },
    "cogvlm2-llama3-chat": {
        "name": "cogvlm2-llama3-chat",
        "type": "vision_language",
        "family": "CogVLM",
        "size": "8B",
        "description": "CogVLM2 with Llama-3 - High-resolution visual understanding",
        "use_cases": ["Visual QA", "GUI Agent", "Fine-grained Recognition", "Multi-turn Dialog"],
        "base_llm": "Llama-3-8B",
        "vision_encoder": "EVA2-CLIP-E",
        "config": {
            "image_size": 1344,
            "hidden_size": 4096,
            "num_attention_heads": 32,
        },
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended_lora_r": 64,
    },
    "phi3-vision-128k": {
        "name": "phi3-vision-128k",
        "type": "vision_language",
        "family": "Phi-3",
        "size": "4.2B",
        "description": "Phi-3 Vision - Microsoft's efficient vision-language model with 128K context",
        "use_cases": ["Visual QA", "Document Understanding", "Long-context Vision"],
        "base_llm": "Phi-3-mini",
        "vision_encoder": "CLIP ViT-L/14",
        "config": {
            "hidden_size": 3072,
            "num_attention_heads": 32,
            "max_position_embeddings": 131072,
        },
        "lora_targets": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
        "recommended_lora_r": 32,
    },
    "idefics2-8b": {
        "name": "idefics2-8b",
        "type": "vision_language",
        "family": "IDEFICS",
        "size": "8B",
        "description": "IDEFICS2 - HuggingFace's open vision-language model",
        "use_cases": ["Visual QA", "Image Captioning", "Document Understanding"],
        "base_llm": "Mistral-7B",
        "vision_encoder": "SigLIP-400M",
        "config": {
            "vision_hidden_size": 1152,
            "text_hidden_size": 4096,
            "perceiver_resampler": True,
        },
        "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "recommended_lora_r": 64,
    },

    # ========================
    # Vision Transformers (for vision encoders)
    # ========================
    "vit_base_patch16_224": {
        "name": "vit_base_patch16_224",
        "type": "vision_transformer",
        "family": "ViT",
        "size": "86M",
        "description": "Vision Transformer Base - Standard ViT for image classification",
        "use_cases": ["Image Classification", "Feature Extraction", "Vision Encoder"],
        "config": {
            "img_size": 224,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "num_heads": 12,
            "mlp_ratio": 4.0,
            "num_classes": 1000
        },
        "lora_targets": ["qkv", "proj", "fc1", "fc2"],
        "recommended_lora_r": 16,
    },
    "vit_large_patch14_336": {
        "name": "vit_large_patch14_336",
        "type": "vision_transformer",
        "family": "ViT",
        "size": "304M",
        "description": "Vision Transformer Large - High-resolution vision encoder",
        "use_cases": ["Image Classification", "Feature Extraction", "Vision Encoder"],
        "config": {
            "img_size": 336,
            "patch_size": 14,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
            "mlp_ratio": 4.0,
            "num_classes": 1000
        },
        "lora_targets": ["qkv", "proj", "fc1", "fc2"],
        "recommended_lora_r": 32,
    },
    "clip_vit_large_patch14": {
        "name": "clip_vit_large_patch14",
        "type": "vision_transformer",
        "family": "CLIP",
        "size": "428M",
        "description": "CLIP ViT-L/14 - OpenAI's contrastive vision encoder",
        "use_cases": ["Image-Text Matching", "Zero-shot Classification", "Vision Encoder for VLMs"],
        "config": {
            "img_size": 224,
            "patch_size": 14,
            "embed_dim": 1024,
            "depth": 24,
            "num_heads": 16,
        },
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "mlp.c_fc", "mlp.c_proj"],
        "recommended_lora_r": 32,
    },
    "siglip_so400m_patch14_384": {
        "name": "siglip_so400m_patch14_384",
        "type": "vision_transformer",
        "family": "SigLIP",
        "size": "400M",
        "description": "SigLIP SO400M - Google's improved CLIP variant with sigmoid loss",
        "use_cases": ["Image-Text Matching", "Vision Encoder", "Zero-shot Classification"],
        "config": {
            "img_size": 384,
            "patch_size": 14,
            "embed_dim": 1152,
            "depth": 27,
            "num_heads": 16,
        },
        "lora_targets": ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        "recommended_lora_r": 32,
    },
}


def get_available_models() -> List[Dict[str, Any]]:
    """Get list of all available models."""
    return list(DEFAULT_MODELS.values())


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(DEFAULT_MODELS.keys())}")
    
    return DEFAULT_MODELS[model_name]


def download_model(model_name: str, output_dir: str = "./models", force: bool = False) -> str:
    """Download a pre-trained model (placeholder implementation)."""
    if model_name not in DEFAULT_MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available models: {list(DEFAULT_MODELS.keys())}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # For now, just save the model configuration
    # In a real implementation, this would download actual model weights from HuggingFace
    model_info = DEFAULT_MODELS[model_name]
    output_path = os.path.join(output_dir, f"{model_name}.json")
    
    if os.path.exists(output_path) and not force:
        raise FileExistsError(f"Model already exists at {output_path}. Use --force to overwrite.")
    
    with open(output_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return output_path


def list_models_by_type(model_type: str = None) -> List[Dict[str, Any]]:
    """List models filtered by type."""
    models = get_available_models()
    if model_type:
        models = [m for m in models if m.get('type') == model_type]
    return models


def list_models_by_family(family: str) -> List[Dict[str, Any]]:
    """List models filtered by family (e.g., LLaVA, BLIP-2, Qwen-VL)."""
    models = get_available_models()
    return [m for m in models if m.get('family', '').lower() == family.lower()]


def search_models(query: str) -> List[Dict[str, Any]]:
    """Search models by name, description, or use case."""
    models = get_available_models()
    query_lower = query.lower()
    
    results = []
    for model in models:
        if (query_lower in model.get('name', '').lower() or 
            query_lower in model.get('description', '').lower() or
            query_lower in model.get('family', '').lower() or
            any(query_lower in uc.lower() for uc in model.get('use_cases', []))):
            results.append(model)
    
    return results


def get_lora_config(model_name: str) -> Dict[str, Any]:
    """Get recommended LoRA configuration for a model."""
    model = get_model_info(model_name)
    return {
        "target_modules": model.get("lora_targets", []),
        "r": model.get("recommended_lora_r", 16),
        "lora_alpha": model.get("recommended_lora_r", 16) * 2,
        "lora_dropout": 0.05,
    }


class ModelZoo:
    """Model Zoo manager for Langvision Vision LLMs."""
    
    def __init__(self, cache_dir: str = "./models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
    
    def list_models(self, model_type: str = None) -> List[Dict[str, Any]]:
        """List available models, optionally filtered by type."""
        return list_models_by_type(model_type)
    
    def list_vlm_models(self) -> List[Dict[str, Any]]:
        """List Vision-Language Models only."""
        return list_models_by_type("vision_language")
    
    def list_by_family(self, family: str) -> List[Dict[str, Any]]:
        """List models by family (LLaVA, BLIP-2, etc.)."""
        return list_models_by_family(family)
    
    def get_model(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        return get_model_info(model_name)
    
    def get_lora_config(self, model_name: str) -> Dict[str, Any]:
        """Get recommended LoRA configuration."""
        return get_lora_config(model_name)
    
    def download(self, model_name: str, force: bool = False) -> str:
        """Download a model."""
        return download_model(model_name, str(self.cache_dir), force)
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """Search for models."""
        return search_models(query)
    
    def is_downloaded(self, model_name: str) -> bool:
        """Check if model is downloaded."""
        model_path = self.cache_dir / f"{model_name}.json"
        return model_path.exists()
    
    def get_downloaded_models(self) -> List[str]:
        """Get list of downloaded models."""
        downloaded = []
        for model_file in self.cache_dir.glob("*.json"):
            model_name = model_file.stem
            if model_name in DEFAULT_MODELS:
                downloaded.append(model_name)
        return downloaded
    
    def print_model_card(self, model_name: str) -> None:
        """Print a formatted model card."""
        model = self.get_model(model_name)
        
        print(f"\n{'='*60}")
        print(f"  {model['name']}")
        print(f"{'='*60}")
        print(f"  Family:      {model.get('family', 'N/A')}")
        print(f"  Size:        {model.get('size', 'N/A')}")
        print(f"  Type:        {model.get('type', 'N/A')}")
        print(f"  Description: {model.get('description', 'N/A')}")
        
        if 'base_llm' in model:
            print(f"  Base LLM:    {model['base_llm']}")
        if 'vision_encoder' in model:
            print(f"  Vision:      {model['vision_encoder']}")
        
        if 'use_cases' in model:
            print(f"  Use Cases:   {', '.join(model['use_cases'])}")
        
        if 'lora_targets' in model:
            print(f"\n  Recommended LoRA Config:")
            print(f"    - Rank (r):        {model.get('recommended_lora_r', 16)}")
            print(f"    - Target modules:  {', '.join(model['lora_targets'][:4])}...")
        
        print(f"{'='*60}\n")
