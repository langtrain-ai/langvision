"""
Fast Attention Implementations for Vision LLM Fine-Tuning

Provides optimized attention implementations:
- Flash Attention-like memory-efficient attention
- Chunked attention for long sequences
- Sliding window attention
- Sparse attention patterns for efficiency
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FastAttention(nn.Module):
    """
    Memory-efficient attention implementation inspired by Flash Attention.
    
    Key optimizations:
    1. Tiling/chunking to reduce memory footprint
    2. Fused softmax and dropout
    3. Online softmax computation (no full attention matrix storage)
    4. Recomputation during backward pass
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        chunk_size: int = 1024,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.chunk_size = chunk_size
        self.scale = self.head_dim ** -0.5
        
        # Validate dimensions
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute memory-efficient attention.
        
        Args:
            query: (batch, seq_len, hidden_size) or (batch, heads, seq_len, head_dim)
            key: Same shape as query
            value: Same shape as query
            attention_mask: Optional mask tensor
        
        Returns:
            Attention output of same shape as input
        """
        # Handle different input shapes
        if query.dim() == 3:
            batch_size, seq_len, _ = query.shape
            
            # Reshape to (batch, heads, seq_len, head_dim)
            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            reshape_output = True
        else:
            batch_size, _, seq_len, _ = query.shape
            reshape_output = False
        
        # Reshape back if needed
        reshape_output = reshape_output
        """
        We inject the accelerator check here.
        """
        from langvision.training.acceleration import Accelerator
        accelerator = Accelerator()
        
        # Try fused kernel first
        fused_output = None
        if accelerator.is_available() and query.is_cuda:
            fused_output = accelerator.fused_attention(
                query, key, value, 
                is_causal=self.is_causal, 
                scale=self.scale
            )
            
        if fused_output is not None:
            output = fused_output
        elif seq_len <= 2048:
            output = self._standard_attention(query, key, value, attention_mask)
        else:
            output = self._chunked_attention(query, key, value, attention_mask)
        
        # Reshape back if needed
        if reshape_output:
            output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return output
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard attention for shorter sequences."""
        # Compute attention scores
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        # Apply causal mask if needed
        if self.is_causal:
            seq_len = query.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device, dtype=torch.bool),
                diagonal=1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        if self.dropout > 0 and self.training:
            attn_weights = F.dropout(attn_weights, p=self.dropout)
        
        # Compute output
        return torch.matmul(attn_weights, value)
    
    def _chunked_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Chunked attention for long sequences.
        
        Computes attention in chunks to reduce peak memory usage.
        Uses online softmax normalization for correctness.
        """
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Initialize output and normalization factors
        output = torch.zeros_like(query)
        row_max = torch.full((batch_size, num_heads, seq_len, 1), 
                            float('-inf'), device=query.device, dtype=query.dtype)
        row_sum = torch.zeros((batch_size, num_heads, seq_len, 1), 
                             device=query.device, dtype=query.dtype)
        
        # Process key-value pairs in chunks
        for j in range(0, seq_len, self.chunk_size):
            j_end = min(j + self.chunk_size, seq_len)
            
            key_chunk = key[:, :, j:j_end, :]
            value_chunk = value[:, :, j:j_end, :]
            
            # Compute attention scores for this chunk
            attn_chunk = torch.matmul(query, key_chunk.transpose(-2, -1)) * self.scale
            
            # Apply causal mask if needed
            if self.is_causal:
                q_idx = torch.arange(seq_len, device=query.device).unsqueeze(1)
                k_idx = torch.arange(j, j_end, device=query.device).unsqueeze(0)
                causal_mask = k_idx > q_idx
                attn_chunk = attn_chunk.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply attention mask
            if attention_mask is not None:
                mask_chunk = attention_mask[:, :, :, j:j_end]
                attn_chunk = attn_chunk + mask_chunk
            
            # Online softmax: update max and compute adjusted scores
            chunk_max = attn_chunk.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, chunk_max)
            
            # Compute exponentials with numerical stability
            exp_old = torch.exp(row_max - new_max)
            exp_chunk = torch.exp(attn_chunk - new_max)
            
            # Update running sum
            new_sum = row_sum * exp_old + exp_chunk.sum(dim=-1, keepdim=True)
            
            # Update output
            output = output * (row_sum / new_sum) * exp_old
            output = output + torch.matmul(exp_chunk / new_sum, value_chunk)
            
            # Update tracking variables
            row_max = new_max
            row_sum = new_sum
        
        return output


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention for efficient processing of long sequences.
    
    Each position only attends to a local window of tokens,
    reducing complexity from O(n²) to O(n * window_size).
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        window_size: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        self.dropout = dropout
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sliding window attention."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        
        # Pad sequences to handle window edges
        padding = self.window_size // 2
        key_padded = F.pad(key, (0, 0, padding, padding), value=0)
        value_padded = F.pad(value, (0, 0, padding, padding), value=0)
        
        # Compute attention in sliding windows
        output = torch.zeros_like(query)
        
        for i in range(seq_len):
            # Extract window
            start = i
            end = i + self.window_size
            
            key_window = key_padded[:, :, start:end, :]  # (batch, heads, window, dim)
            value_window = value_padded[:, :, start:end, :]
            
            # Compute attention for this position
            query_i = query[:, :, i:i+1, :]  # (batch, heads, 1, dim)
            
            attn_weights = torch.matmul(query_i, key_window.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(attn_weights, dim=-1)
            
            if self.dropout > 0 and self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            
            output[:, :, i:i+1, :] = torch.matmul(attn_weights, value_window)
        
        return output


class MultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention with various efficiency options.
    
    Supports:
    - Standard attention
    - Flash-like memory-efficient attention
    - Sliding window attention
    - Grouped query attention (GQA)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,  # For GQA
        dropout: float = 0.0,
        is_causal: bool = False,
        use_fast_attention: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads  # Default to MHA
        self.head_dim = hidden_size // num_heads
        self.kv_head_dim = hidden_size // self.num_kv_heads
        self.dropout = dropout
        self.is_causal = is_causal
        self.use_fast_attention = use_fast_attention
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Attention implementation
        if use_fast_attention:
            self.attention = FastAttention(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                is_causal=is_causal,
            )
        else:
            self.attention = None
        
        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional KV caching."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute Q, K, V projections
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=2)
            value = torch.cat([past_value, value], dim=2)
        
        # Expand KV heads for GQA
        if self.num_kv_heads < self.num_heads:
            key = self._repeat_kv(key, self.num_heads // self.num_kv_heads)
            value = self._repeat_kv(value, self.num_heads // self.num_kv_heads)
        
        # Compute attention
        if self.use_fast_attention and self.attention is not None:
            attn_output = self.attention(query, key, value, attention_mask)
        else:
            attn_output = self._standard_attention(query, key, value, attention_mask)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Prepare cache for next iteration
        new_cache = (key, value) if use_cache else None
        
        return attn_output, new_cache
    
    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """Repeat key/value heads for grouped query attention."""
        if n_rep == 1:
            return x
        batch, num_heads, seq_len, head_dim = x.shape
        x = x[:, :, None, :, :].expand(batch, num_heads, n_rep, seq_len, head_dim)
        return x.reshape(batch, num_heads * n_rep, seq_len, head_dim)
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard attention computation."""
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        
        if self.is_causal:
            seq_len = query.size(2)
            kv_len = key.size(2)
            causal_mask = torch.triu(
                torch.ones(seq_len, kv_len, device=query.device, dtype=torch.bool),
                diagonal=kv_len - seq_len + 1
            )
            attn_weights = attn_weights.masked_fill(causal_mask, float('-inf'))
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        return torch.matmul(attn_weights, value)


def create_attention_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    is_causal: bool = True,
) -> torch.Tensor:
    """Create attention mask for causal or non-causal attention."""
    if is_causal:
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=dtype) * float('-inf'),
            diagonal=1
        )
    else:
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
    
    return mask
