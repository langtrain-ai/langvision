"""
Vision LLM Inference Module

High-performance inference utilities for Vision LLMs:
- Batched inference with automatic memory management
- Streaming generation
- KV cache optimization
- Multi-GPU inference
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Union, Iterator, Tuple
from dataclasses import dataclass
from pathlib import Path
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference."""
    # Generation
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.0
    
    # Batching
    batch_size: int = 1
    
    # Optimization
    use_cache: bool = True
    use_flash_attention: bool = True
    
    # Precision
    dtype: str = "bfloat16"  # bfloat16, float16, float32
    
    # Streaming
    stream: bool = False


class VisionLLMInference:
    """
    High-performance inference engine for Vision LLMs.
    
    Features:
    - Automatic batching
    - KV cache management
    - Streaming generation
    - Memory optimization
    
    Usage:
        inference = VisionLLMInference(model, tokenizer)
        
        # Single image
        response = inference.generate(image, "What is in this image?")
        
        # Batch processing
        responses = inference.batch_generate(images, questions)
        
        # Streaming
        for token in inference.stream_generate(image, "Describe this image"):
            print(token, end="", flush=True)
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        processor=None,
        config: Optional[InferenceConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config or InferenceConfig()
        
        self.device = self._get_device()
        self.dtype = self._get_dtype()
        
        # Move model to device and set dtype
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()
        
        # KV cache
        self.kv_cache = None
        
        logger.info(f"Inference engine initialized on {self.device} with {self.dtype}")
    
    def _get_device(self) -> torch.device:
        """Get best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _get_dtype(self) -> torch.dtype:
        """Get dtype from config."""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.config.dtype, torch.bfloat16)
    
    @torch.inference_mode()
    def generate(
        self,
        image: Union[torch.Tensor, "PIL.Image.Image", str],
        prompt: str,
        **kwargs,
    ) -> str:
        """
        Generate response for a single image and prompt.
        
        Args:
            image: Input image (tensor, PIL Image, or path)
            prompt: Text prompt
            **kwargs: Override generation parameters
        
        Returns:
            Generated text response
        """
        # Prepare inputs
        inputs = self._prepare_inputs([image], [prompt])
        
        # Merge config with kwargs
        gen_config = self._get_generation_config(**kwargs)
        
        # Generate
        start_time = time.time()
        
        outputs = self.model.generate(
            **inputs,
            **gen_config,
        )
        
        # Decode
        response = self._decode_outputs(outputs)[0]
        
        elapsed = time.time() - start_time
        tokens = outputs.shape[-1]
        logger.debug(f"Generated {tokens} tokens in {elapsed:.2f}s ({tokens/elapsed:.1f} tok/s)")
        
        return response
    
    @torch.inference_mode()
    def batch_generate(
        self,
        images: List[Union[torch.Tensor, "PIL.Image.Image", str]],
        prompts: List[str],
        **kwargs,
    ) -> List[str]:
        """
        Generate responses for a batch of images and prompts.
        
        Args:
            images: List of input images
            prompts: List of text prompts
            **kwargs: Override generation parameters
        
        Returns:
            List of generated responses
        """
        if len(images) != len(prompts):
            raise ValueError("Number of images must match number of prompts")
        
        all_responses = []
        
        # Process in batches
        for i in range(0, len(images), self.config.batch_size):
            batch_images = images[i:i + self.config.batch_size]
            batch_prompts = prompts[i:i + self.config.batch_size]
            
            inputs = self._prepare_inputs(batch_images, batch_prompts)
            gen_config = self._get_generation_config(**kwargs)
            
            outputs = self.model.generate(**inputs, **gen_config)
            responses = self._decode_outputs(outputs)
            
            all_responses.extend(responses)
        
        return all_responses
    
    @torch.inference_mode()
    def stream_generate(
        self,
        image: Union[torch.Tensor, "PIL.Image.Image", str],
        prompt: str,
        **kwargs,
    ) -> Iterator[str]:
        """
        Stream generation token by token.
        
        Args:
            image: Input image
            prompt: Text prompt
            **kwargs: Override generation parameters
        
        Yields:
            Generated tokens one at a time
        """
        inputs = self._prepare_inputs([image], [prompt])
        gen_config = self._get_generation_config(**kwargs)
        
        # Initialize KV cache
        past_key_values = None
        generated_tokens = []
        
        # Get initial logits
        outputs = self.model(**inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        
        # Sample first token
        next_token = self._sample_token(outputs.logits[:, -1, :], gen_config)
        generated_tokens.append(next_token)
        
        # Decode and yield
        token_str = self.tokenizer.decode([next_token.item()])
        yield token_str
        
        # Continue generation
        for _ in range(gen_config.get("max_new_tokens", 512) - 1):
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Forward with KV cache
            outputs = self.model(
                input_ids=next_token.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            
            # Sample next token
            next_token = self._sample_token(outputs.logits[:, -1, :], gen_config)
            generated_tokens.append(next_token)
            
            # Decode and yield
            token_str = self.tokenizer.decode([next_token.item()])
            yield token_str
    
    def _prepare_inputs(
        self,
        images: List[Any],
        prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for the model."""
        # Load images if paths
        processed_images = []
        for img in images:
            if isinstance(img, str):
                from PIL import Image
                img = Image.open(img).convert("RGB")
            processed_images.append(img)
        
        # Use processor if available
        if self.processor is not None:
            inputs = self.processor(
                text=prompts,
                images=processed_images,
                return_tensors="pt",
                padding=True,
            )
        else:
            # Basic tokenization
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
            )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        return inputs
    
    def _get_generation_config(self, **kwargs) -> Dict[str, Any]:
        """Get generation configuration."""
        config = {
            "max_new_tokens": self.config.max_new_tokens,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
            "repetition_penalty": self.config.repetition_penalty,
            "use_cache": self.config.use_cache,
            "do_sample": self.config.temperature > 0,
        }
        config.update(kwargs)
        return config
    
    def _sample_token(
        self,
        logits: torch.Tensor,
        config: Dict[str, Any],
    ) -> torch.Tensor:
        """Sample next token from logits."""
        temperature = config.get("temperature", 1.0)
        top_p = config.get("top_p", 1.0)
        top_k = config.get("top_k", 0)
        
        if temperature == 0:
            # Greedy
            return logits.argmax(dim=-1)
        
        # Apply temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    def _decode_outputs(self, outputs: torch.Tensor) -> List[str]:
        """Decode model outputs to text."""
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
    
    def clear_cache(self):
        """Clear KV cache and GPU memory."""
        self.kv_cache = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ChatInterface:
    """
    Chat interface for Vision LLM conversations.
    
    Maintains conversation history and supports multi-turn dialogue.
    
    Usage:
        chat = ChatInterface(model, tokenizer)
        
        # Start conversation
        response = chat.chat(image, "What is this?")
        
        # Continue conversation
        response = chat.chat(None, "Tell me more about it")
        
        # Reset
        chat.reset()
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer=None,
        processor=None,
        system_prompt: Optional[str] = None,
    ):
        self.inference = VisionLLMInference(model, tokenizer, processor)
        self.tokenizer = tokenizer
        
        self.system_prompt = system_prompt or (
            "You are a helpful assistant that can understand and describe images."
        )
        
        self.history: List[Dict[str, str]] = []
        self.current_image = None
    
    def chat(
        self,
        image: Optional[Any] = None,
        message: str = "",
        **kwargs,
    ) -> str:
        """
        Send a message in the conversation.
        
        Args:
            image: Optional new image to discuss
            message: User message
            **kwargs: Generation parameters
        
        Returns:
            Assistant response
        """
        # Update image if provided
        if image is not None:
            self.current_image = image
        
        # Add user message to history
        self.history.append({"role": "user", "content": message})
        
        # Build prompt with history
        prompt = self._build_prompt()
        
        # Generate response
        if self.current_image is not None:
            response = self.inference.generate(self.current_image, prompt, **kwargs)
        else:
            # No image - text only
            response = self.inference.generate(
                torch.zeros(1, 3, 224, 224),  # Dummy image
                prompt,
                **kwargs,
            )
        
        # Extract assistant response
        response = self._extract_response(response)
        
        # Add to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def _build_prompt(self) -> str:
        """Build prompt from conversation history."""
        parts = [f"System: {self.system_prompt}\n"]
        
        for turn in self.history:
            role = turn["role"].capitalize()
            content = turn["content"]
            parts.append(f"{role}: {content}")
        
        parts.append("Assistant:")
        
        return "\n".join(parts)
    
    def _extract_response(self, full_response: str) -> str:
        """Extract assistant response from full generation."""
        # Try to find the last "Assistant:" and get text after
        if "Assistant:" in full_response:
            parts = full_response.split("Assistant:")
            return parts[-1].strip()
        return full_response.strip()
    
    def reset(self):
        """Reset conversation history."""
        self.history = []
        self.current_image = None
        self.inference.clear_cache()
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history.copy()
