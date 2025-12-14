"""
Model Loading Utilities for Vision LLMs

Unified interface for loading Vision LLMs from various sources:
- HuggingFace Hub
- Local checkpoints
- Custom model configurations
- Automatic LoRA integration
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelLoadConfig:
    """Configuration for model loading."""
    # Model source
    model_name_or_path: str = "llava-v1.6-7b"
    revision: str = "main"
    trust_remote_code: bool = True
    
    # Quantization
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # Precision
    torch_dtype: str = "auto"  # auto, float16, bfloat16, float32
    
    # Device
    device_map: str = "auto"
    max_memory: Optional[Dict[int, str]] = None
    
    # LoRA
    lora_path: Optional[str] = None
    merge_lora: bool = False
    
    # Flash Attention
    use_flash_attention: bool = True


# Model registry with loading configurations
MODEL_REGISTRY = {
    "llava-v1.6-7b": {
        "hf_id": "llava-hf/llava-1.5-7b-hf",
        "model_type": "llava",
        "processor": "llava-hf/llava-1.5-7b-hf",
    },
    "llava-v1.6-13b": {
        "hf_id": "llava-hf/llava-1.5-13b-hf",
        "model_type": "llava",
        "processor": "llava-hf/llava-1.5-13b-hf",
    },
    "qwen-vl-chat": {
        "hf_id": "Qwen/Qwen-VL-Chat",
        "model_type": "qwen_vl",
        "processor": "Qwen/Qwen-VL-Chat",
    },
    "qwen2-vl-7b": {
        "hf_id": "Qwen/Qwen2-VL-7B-Instruct",
        "model_type": "qwen2_vl",
        "processor": "Qwen/Qwen2-VL-7B-Instruct",
    },
    "blip2-flan-t5-xl": {
        "hf_id": "Salesforce/blip2-flan-t5-xl",
        "model_type": "blip2",
        "processor": "Salesforce/blip2-flan-t5-xl",
    },
    "internvl2-8b": {
        "hf_id": "OpenGVLab/InternVL2-8B",
        "model_type": "internvl",
        "processor": "OpenGVLab/InternVL2-8B",
    },
    "phi-3-vision": {
        "hf_id": "microsoft/Phi-3-vision-128k-instruct",
        "model_type": "phi3_vision",
        "processor": "microsoft/Phi-3-vision-128k-instruct",
    },
    "idefics2-8b": {
        "hf_id": "HuggingFaceM4/idefics2-8b",
        "model_type": "idefics2",
        "processor": "HuggingFaceM4/idefics2-8b",
    },
}


class ModelLoader:
    """
    Unified model loader for Vision LLMs.
    
    Features:
    - Automatic model detection
    - Quantization support (4-bit, 8-bit)
    - LoRA adapter loading
    - Flash Attention 2 support
    - Multi-GPU distribution
    
    Usage:
        loader = ModelLoader("llava-v1.6-7b", load_in_4bit=True)
        model, tokenizer, processor = loader.load()
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        config: Optional[ModelLoadConfig] = None,
        **kwargs,
    ):
        if config is None:
            config = ModelLoadConfig(model_name_or_path=model_name_or_path)
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        
        # Resolve model info from registry
        if model_name_or_path in MODEL_REGISTRY:
            self.model_info = MODEL_REGISTRY[model_name_or_path]
            self.hf_id = self.model_info["hf_id"]
        else:
            # Assume it's a HuggingFace ID or local path
            self.model_info = None
            self.hf_id = model_name_or_path
    
    def load(self) -> Tuple[nn.Module, Any, Any]:
        """
        Load model, tokenizer, and processor.
        
        Returns:
            Tuple of (model, tokenizer, processor)
        """
        print(f"\n{'='*60}")
        print(f"  🔄 Loading Model")
        print(f"{'='*60}")
        print(f"  Model: {self.hf_id}")
        print(f"  4-bit: {'✓' if self.config.load_in_4bit else '✗'}")
        print(f"  8-bit: {'✓' if self.config.load_in_8bit else '✗'}")
        print(f"{'='*60}\n")
        
        # Get dtype
        dtype = self._get_dtype()
        
        # Build quantization config
        quantization_config = self._get_quantization_config()
        
        # Load model
        model = self._load_model(dtype, quantization_config)
        
        # Load tokenizer and processor
        tokenizer = self._load_tokenizer()
        processor = self._load_processor()
        
        # Load LoRA if specified
        if self.config.lora_path:
            model = self._load_lora(model)
        
        print(f"  ✅ Model loaded successfully!")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model, tokenizer, processor
    
    def _get_dtype(self) -> torch.dtype:
        """Get torch dtype from config."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        if self.config.torch_dtype == "auto":
            # Use bfloat16 if available, else float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        
        return dtype_map.get(self.config.torch_dtype, torch.float32)
    
    def _get_quantization_config(self) -> Optional[Any]:
        """Get BitsAndBytes quantization config."""
        if not (self.config.load_in_4bit or self.config.load_in_8bit):
            return None
        
        try:
            from transformers import BitsAndBytesConfig
            
            compute_dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            
            return BitsAndBytesConfig(
                load_in_4bit=self.config.load_in_4bit,
                load_in_8bit=self.config.load_in_8bit,
                bnb_4bit_compute_dtype=compute_dtype_map.get(
                    self.config.bnb_4bit_compute_dtype, torch.bfloat16
                ),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
        except ImportError:
            logger.warning("BitsAndBytes not installed. Skipping quantization.")
            return None
    
    def _load_model(
        self,
        dtype: torch.dtype,
        quantization_config: Optional[Any],
    ) -> nn.Module:
        """Load the model."""
        try:
            from transformers import AutoModelForVision2Seq, AutoModel
            
            # Determine model class based on type
            model_type = self.model_info["model_type"] if self.model_info else None
            
            # Build kwargs
            kwargs = {
                "pretrained_model_name_or_path": self.hf_id,
                "revision": self.config.revision,
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": self.config.device_map,
            }
            
            if quantization_config:
                kwargs["quantization_config"] = quantization_config
            else:
                kwargs["torch_dtype"] = dtype
            
            if self.config.max_memory:
                kwargs["max_memory"] = self.config.max_memory
            
            # Try Flash Attention 2
            if self.config.use_flash_attention:
                try:
                    kwargs["attn_implementation"] = "flash_attention_2"
                except:
                    pass
            
            # Load model
            try:
                model = AutoModelForVision2Seq.from_pretrained(**kwargs)
            except:
                model = AutoModel.from_pretrained(**kwargs)
            
            return model
            
        except ImportError:
            raise ImportError(
                "transformers not installed. Install with: pip install transformers"
            )
    
    def _load_tokenizer(self) -> Any:
        """Load tokenizer."""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                self.hf_id,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Ensure pad token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
            
        except ImportError:
            logger.warning("Could not load tokenizer")
            return None
    
    def _load_processor(self) -> Any:
        """Load multi-modal processor."""
        try:
            from transformers import AutoProcessor
            
            processor_id = self.model_info.get("processor", self.hf_id) if self.model_info else self.hf_id
            
            processor = AutoProcessor.from_pretrained(
                processor_id,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            return processor
            
        except Exception as e:
            logger.warning(f"Could not load processor: {e}")
            return None
    
    def _load_lora(self, model: nn.Module) -> nn.Module:
        """Load and apply LoRA adapter."""
        lora_path = Path(self.config.lora_path)
        
        if not lora_path.exists():
            raise FileNotFoundError(f"LoRA path not found: {lora_path}")
        
        print(f"  Loading LoRA from: {lora_path}")
        
        # Load LoRA weights
        if (lora_path / "adapter_model.pt").exists():
            lora_weights = torch.load(lora_path / "adapter_model.pt")
        elif (lora_path / "lora_weights.pt").exists():
            lora_weights = torch.load(lora_path / "lora_weights.pt")
        else:
            # Try PEFT format
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, str(lora_path))
                
                if self.config.merge_lora:
                    model = model.merge_and_unload()
                
                return model
            except:
                raise FileNotFoundError(f"No LoRA weights found in {lora_path}")
        
        # Load using our custom format
        from ..training.fast_lora import load_lora_state_dict
        load_lora_state_dict(model, lora_weights)
        
        print(f"  ✅ LoRA loaded successfully!")
        
        return model


def load_model(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    lora_path: Optional[str] = None,
    **kwargs,
) -> Tuple[nn.Module, Any, Any]:
    """
    Convenience function to load a Vision LLM.
    
    Args:
        model_name: Model name (from registry) or HuggingFace ID
        load_in_4bit: Use 4-bit quantization
        load_in_8bit: Use 8-bit quantization
        lora_path: Path to LoRA adapter
        **kwargs: Additional loading options
    
    Returns:
        Tuple of (model, tokenizer, processor)
    """
    loader = ModelLoader(
        model_name,
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        lora_path=lora_path,
        **kwargs,
    )
    return loader.load()


def list_available_models() -> List[str]:
    """List all models in the registry."""
    return list(MODEL_REGISTRY.keys())


def get_model_info(model_name: str) -> Optional[Dict[str, Any]]:
    """Get information about a registered model."""
    return MODEL_REGISTRY.get(model_name)
