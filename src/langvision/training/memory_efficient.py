"""
Memory Efficient Training Utilities for Vision LLM Fine-Tuning

Implements techniques for reducing memory usage during training:
- Gradient checkpointing with selective recomputation
- CPU offloading for optimizer states
- Memory-efficient attention patterns
- Dynamic batch size adjustment
- Activation memory optimization
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from typing import Optional, Callable, List, Dict, Any, Tuple
from dataclasses import dataclass
from contextlib import contextmanager
import gc


@dataclass
class MemoryConfig:
    """Configuration for memory optimizations."""
    
    # Gradient checkpointing
    use_gradient_checkpointing: bool = True
    checkpoint_ratio: float = 0.5  # Fraction of layers to checkpoint
    
    # CPU offloading
    offload_optimizer: bool = False
    offload_gradients: bool = False
    
    # Memory management
    empty_cache_frequency: int = 10  # Empty CUDA cache every N steps
    pin_memory: bool = True
    
    # Activation optimization
    use_activation_checkpointing: bool = True
    recompute_attention: bool = True  # Recompute attention instead of storing
    
    # Mixed precision
    use_bf16: bool = True
    use_fp16: bool = False
    
    # Dynamic batching
    auto_batch_size: bool = False
    max_memory_fraction: float = 0.9


class MemoryTracker:
    """Track and report GPU memory usage."""
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history: List[Dict[str, float]] = []
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get current memory statistics."""
        if not torch.cuda.is_available():
            return {"allocated": 0, "reserved": 0, "max_allocated": 0}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated(self.device) / 1e9,
            "reserved_gb": torch.cuda.memory_reserved(self.device) / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated(self.device) / 1e9,
            "free_gb": (torch.cuda.get_device_properties(self.device).total_memory - 
                       torch.cuda.memory_reserved(self.device)) / 1e9,
        }
    
    def log(self, step: int, phase: str = ""):
        """Log memory usage at a step."""
        stats = self.get_memory_stats()
        stats["step"] = step
        stats["phase"] = phase
        self.history.append(stats)
    
    def clear_cache(self):
        """Clear CUDA cache and run garbage collection."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    
    def print_stats(self):
        """Print current memory statistics."""
        stats = self.get_memory_stats()
        print(f"\n{'='*50}")
        print(f"  GPU Memory Usage")
        print(f"{'='*50}")
        print(f"  Allocated:     {stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:      {stats['reserved_gb']:.2f} GB")
        print(f"  Max Allocated: {stats['max_allocated_gb']:.2f} GB")
        print(f"  Free:          {stats['free_gb']:.2f} GB")
        print(f"{'='*50}\n")


class GradientCheckpointer:
    """
    Smart gradient checkpointing for Vision LLMs.
    
    Selectively applies checkpointing to balance memory savings vs compute overhead.
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.checkpointed_layers: List[str] = []
    
    def apply_to_model(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to a model."""
        if not self.config.use_gradient_checkpointing:
            return model
        
        # Find transformer blocks or similar repeating structures
        blocks = self._find_checkpointable_blocks(model)
        
        # Checkpoint every nth block based on ratio
        n_to_checkpoint = int(len(blocks) * self.config.checkpoint_ratio)
        step = max(1, len(blocks) // n_to_checkpoint) if n_to_checkpoint > 0 else len(blocks) + 1
        
        for i, (name, block) in enumerate(blocks):
            if i % step == 0:
                self._wrap_with_checkpoint(model, name, block)
                self.checkpointed_layers.append(name)
        
        return model
    
    def _find_checkpointable_blocks(self, model: nn.Module) -> List[Tuple[str, nn.Module]]:
        """Find blocks that can be checkpointed (transformer layers, etc.)."""
        blocks = []
        
        for name, module in model.named_modules():
            # Common patterns for transformer blocks
            if any(pattern in name.lower() for pattern in 
                   ['layer.', 'block.', 'encoder.', 'decoder.', 'transformer.']):
                if hasattr(module, 'forward') and len(list(module.children())) > 0:
                    blocks.append((name, module))
        
        return blocks
    
    def _wrap_with_checkpoint(self, model: nn.Module, name: str, block: nn.Module):
        """Wrap a block with gradient checkpointing."""
        original_forward = block.forward
        
        def checkpointed_forward(*args, **kwargs):
            # Use checkpoint for backward pass
            def custom_forward(*inputs):
                return original_forward(*inputs, **kwargs)
            
            return checkpoint(custom_forward, *args, use_reentrant=False)
        
        block.forward = checkpointed_forward


class CPUOffloader:
    """
    Offload tensors to CPU to save GPU memory.
    
    Supports:
    - Optimizer state offloading
    - Gradient offloading
    - Model weight offloading (for large models)
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.offloaded_tensors: Dict[str, torch.Tensor] = {}
    
    def offload_optimizer_state(self, optimizer: torch.optim.Optimizer):
        """Move optimizer state to CPU."""
        if not self.config.offload_optimizer:
            return
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key, val in state.items():
                        if isinstance(val, torch.Tensor):
                            state[key] = val.cpu()
    
    def restore_optimizer_state(self, optimizer: torch.optim.Optimizer, device: torch.device):
        """Move optimizer state back to GPU."""
        if not self.config.offload_optimizer:
            return
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in optimizer.state:
                    state = optimizer.state[p]
                    for key, val in state.items():
                        if isinstance(val, torch.Tensor):
                            state[key] = val.to(device)
    
    def offload_gradients(self, model: nn.Module):
        """Offload gradients to CPU."""
        if not self.config.offload_gradients:
            return
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.offloaded_tensors[name] = param.grad.cpu()
                param.grad = None
    
    def restore_gradients(self, model: nn.Module, device: torch.device):
        """Restore gradients from CPU."""
        if not self.config.offload_gradients:
            return
        
        for name, param in model.named_parameters():
            if name in self.offloaded_tensors:
                param.grad = self.offloaded_tensors[name].to(device)
        
        self.offloaded_tensors.clear()


@contextmanager
def memory_efficient_mode(config: Optional[MemoryConfig] = None):
    """
    Context manager for memory-efficient training.
    
    Usage:
        with memory_efficient_mode():
            loss = model(inputs)
            loss.backward()
    """
    config = config or MemoryConfig()
    
    # Enable memory-efficient settings
    prev_grad_enabled = torch.is_grad_enabled()
    
    try:
        # Set memory-efficient defaults
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        yield
        
    finally:
        # Restore previous state
        if config.empty_cache_frequency > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()


class ActivationOptimizer:
    """
    Optimize activation memory during training.
    
    Implements:
    - In-place operations where safe
    - Activation recomputation
    - Memory-efficient attention computation
    """
    
    @staticmethod
    def optimize_for_training(model: nn.Module) -> nn.Module:
        """Apply activation optimizations to model."""
        
        for module in model.modules():
            # Enable in-place ReLU operations
            if isinstance(module, nn.ReLU):
                module.inplace = True
            
            # Enable in-place GELU if using custom implementation
            if hasattr(module, 'inplace'):
                module.inplace = True
        
        return model
    
    @staticmethod
    def compute_memory_efficient_attention(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Memory-efficient attention computation.
        
        Uses chunked computation to reduce peak memory.
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        scale = scale or (head_dim ** -0.5)
        
        # For short sequences, use standard attention
        if seq_len <= 2048:
            attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            if dropout_p > 0:
                attn_weights = torch.dropout(attn_weights, p=dropout_p, train=True)
            
            return torch.matmul(attn_weights, value)
        
        # For long sequences, compute in chunks
        chunk_size = 1024
        output = torch.zeros_like(query)
        
        for i in range(0, seq_len, chunk_size):
            end_i = min(i + chunk_size, seq_len)
            query_chunk = query[:, :, i:end_i, :]
            
            # Compute attention for this chunk against all keys
            attn_weights = torch.matmul(query_chunk, key.transpose(-2, -1)) * scale
            
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, i:end_i, :]
                attn_weights = attn_weights + mask_chunk
            
            attn_weights = torch.softmax(attn_weights, dim=-1)
            
            if dropout_p > 0:
                attn_weights = torch.dropout(attn_weights, p=dropout_p, train=True)
            
            output[:, :, i:end_i, :] = torch.matmul(attn_weights, value)
        
        return output


def estimate_memory_usage(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    dtype: torch.dtype = torch.float16,
) -> Dict[str, float]:
    """
    Estimate memory usage for a model during training.
    
    Returns memory estimates in GB for:
    - Model parameters
    - Gradients
    - Optimizer states (AdamW)
    - Activations (approximate)
    """
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Bytes per element
    bytes_per_elem = 2 if dtype in [torch.float16, torch.bfloat16] else 4
    
    # Model size
    model_size = total_params * bytes_per_elem / 1e9
    
    # Gradients (only for trainable params)
    gradient_size = trainable_params * bytes_per_elem / 1e9
    
    # Optimizer states (AdamW: 2 states per param - momentum and variance)
    # Stored in FP32 for stability
    optimizer_size = trainable_params * 4 * 2 / 1e9
    
    # Activations (rough estimate: proportional to batch size and sequence length)
    # This is a rough estimate; actual usage depends on model architecture
    activation_multiplier = 2.0  # Conservative estimate
    activation_size = (batch_size * seq_length * total_params * bytes_per_elem * 
                      activation_multiplier / total_params) / 1e9
    
    total = model_size + gradient_size + optimizer_size + activation_size
    
    return {
        "model_gb": model_size,
        "gradients_gb": gradient_size,
        "optimizer_gb": optimizer_size,
        "activations_gb": activation_size,
        "total_gb": total,
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def find_optimal_batch_size(
    model: nn.Module,
    sample_input: torch.Tensor,
    max_batch_size: int = 64,
    target_memory_fraction: float = 0.85,
) -> int:
    """
    Find the optimal batch size that fits in GPU memory.
    
    Uses binary search to find the largest batch size that doesn't OOM.
    """
    if not torch.cuda.is_available():
        return max_batch_size
    
    device = next(model.parameters()).device
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * target_memory_fraction
    
    # Binary search for optimal batch size
    low, high = 1, max_batch_size
    optimal = 1
    
    while low <= high:
        mid = (low + high) // 2
        
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create batch of this size
            batch = sample_input.repeat(mid, *([1] * (sample_input.dim() - 1)))[:mid]
            batch = batch.to(device)
            
            # Forward pass
            with torch.no_grad():
                _ = model(batch)
            
            # Check memory usage
            peak_memory = torch.cuda.max_memory_allocated()
            
            if peak_memory < target_memory:
                optimal = mid
                low = mid + 1
            else:
                high = mid - 1
            
            del batch
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise
    
    return optimal
