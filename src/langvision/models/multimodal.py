"""
Multimodal Vision-Language Models with LoRA fine-tuning support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
from transformers import AutoTokenizer, AutoModel
from .vision_transformer import VisionTransformer
from .lora import LoRALinear, LoRAConfig
import math


class CrossAttention(nn.Module):
    """Cross-attention mechanism for vision-language fusion."""
    
    def __init__(self, 
                 vision_dim: int, 
                 text_dim: int, 
                 hidden_dim: int = 512,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        if lora_config and lora_config.r > 0:
            self.q_proj = LoRALinear(vision_dim, hidden_dim, 
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.k_proj = LoRALinear(text_dim, hidden_dim,
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.v_proj = LoRALinear(text_dim, hidden_dim,
                                   r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.out_proj = LoRALinear(hidden_dim, vision_dim,
                                     r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        else:
            self.q_proj = nn.Linear(vision_dim, hidden_dim)
            self.k_proj = nn.Linear(text_dim, hidden_dim)
            self.v_proj = nn.Linear(text_dim, hidden_dim)
            self.out_proj = nn.Linear(hidden_dim, vision_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(vision_dim)
    
    def forward(self, 
                vision_features: torch.Tensor, 
                text_features: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            vision_features: (B, N_v, D_v) vision tokens
            text_features: (B, N_t, D_t) text tokens
            attention_mask: (B, N_t) mask for text tokens
        """
        B, N_v, D_v = vision_features.shape
        B, N_t, D_t = text_features.shape
        
        # Project to query, key, value
        Q = self.q_proj(vision_features)  # (B, N_v, hidden_dim)
        K = self.k_proj(text_features)    # (B, N_t, hidden_dim)
        V = self.v_proj(text_features)    # (B, N_t, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_v, head_dim)
        K = K.view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_t, head_dim)
        V = V.view(B, N_t, self.num_heads, self.head_dim).transpose(1, 2)  # (B, num_heads, N_t, head_dim)
        
        # Compute attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, num_heads, N_v, N_t)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N_t)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, N_v, head_dim)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N_v, self.hidden_dim)
        output = self.out_proj(attn_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(vision_features + output)
        
        return output


class VisionLanguageModel(nn.Module):
    """Vision-Language Model with cross-modal attention and LoRA fine-tuning."""
    
    def __init__(self,
                 vision_model: str = "vit_base",
                 text_model: str = "bert-base-uncased",
                 vision_dim: int = 768,
                 text_dim: int = 768,
                 hidden_dim: int = 512,
                 num_classes: int = 1000,
                 max_text_length: int = 77,
                 lora_config: Optional[LoRAConfig] = None):
        super().__init__()
        
        self.vision_dim = vision_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.max_text_length = max_text_length
        
        # Vision encoder
        if vision_model == "vit_base":
            self.vision_encoder = VisionTransformer(
                embed_dim=vision_dim,
                lora_config=lora_config
            )
        else:
            raise ValueError(f"Unsupported vision model: {vision_model}")
        
        # Text encoder
        self.tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_encoder = AutoModel.from_pretrained(text_model)
        
        # Freeze text encoder if using LoRA
        if lora_config and lora_config.r > 0:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Cross-modal fusion
        self.cross_attention = CrossAttention(
            vision_dim=vision_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            lora_config=lora_config
        )
        
        # Classification head
        if lora_config and lora_config.r > 0:
            self.classifier = LoRALinear(vision_dim, num_classes,
                                       r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        else:
            self.classifier = nn.Linear(vision_dim, num_classes)
        
        # Text projection for contrastive learning
        if lora_config and lora_config.r > 0:
            self.text_projection = LoRALinear(text_dim, hidden_dim,
                                            r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
            self.vision_projection = LoRALinear(vision_dim, hidden_dim,
                                              r=lora_config.r, alpha=lora_config.alpha, dropout=lora_config.dropout)
        else:
            self.text_projection = nn.Linear(text_dim, hidden_dim)
            self.vision_projection = nn.Linear(vision_dim, hidden_dim)
        
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to feature representations."""
        return self.vision_encoder(images)
    
    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode texts to feature representations."""
        # Tokenize texts
        tokens = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        # Get text features
        with torch.no_grad() if hasattr(self, 'text_encoder') else torch.enable_grad():
            text_outputs = self.text_encoder(**tokens)
            text_features = text_outputs.last_hidden_state  # (B, seq_len, text_dim)
        
        return text_features, tokens['attention_mask']
    
    def forward(self, 
                images: torch.Tensor, 
                texts: Optional[List[str]] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass for vision-language model.
        
        Args:
            images: (B, C, H, W) input images
            texts: List of text descriptions
            return_features: Whether to return intermediate features
        """
        # Encode images
        vision_features = self.encode_image(images)  # (B, N_patches, vision_dim)
        
        outputs = {"vision_features": vision_features}
        
        if texts is not None:
            # Encode texts
            text_features, attention_mask = self.encode_text(texts)  # (B, seq_len, text_dim)
            outputs["text_features"] = text_features
            
            # Cross-modal fusion
            fused_features = self.cross_attention(
                vision_features, text_features, attention_mask
            )  # (B, N_patches, vision_dim)
            outputs["fused_features"] = fused_features
            
            # Global pooling for classification
            pooled_vision = fused_features.mean(dim=1)  # (B, vision_dim)
            pooled_text = text_features.mean(dim=1)     # (B, text_dim)
            
            # Classification
            logits = self.classifier(pooled_vision)
            outputs["logits"] = logits
            
            # Contrastive learning projections
            vision_proj = F.normalize(self.vision_projection(pooled_vision), dim=-1)
            text_proj = F.normalize(self.text_projection(pooled_text), dim=-1)
            
            # Compute contrastive logits
            logit_scale = self.logit_scale.exp()
            contrastive_logits = logit_scale * vision_proj @ text_proj.T
            
            outputs.update({
                "vision_proj": vision_proj,
                "text_proj": text_proj,
                "contrastive_logits": contrastive_logits,
                "logit_scale": logit_scale
            })
        else:
            # Image-only classification
            pooled_vision = vision_features.mean(dim=1)
            logits = self.classifier(pooled_vision)
            outputs["logits"] = logits
        
        if not return_features:
            # Return only essential outputs for training
            essential_keys = ["logits", "contrastive_logits"] if texts else ["logits"]
            outputs = {k: v for k, v in outputs.items() if k in essential_keys}
        
        return outputs


class CLIPLoss(nn.Module):
    """CLIP-style contrastive loss for vision-language learning."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, vision_proj: torch.Tensor, text_proj: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between vision and text projections.
        
        Args:
            vision_proj: (B, D) normalized vision projections
            text_proj: (B, D) normalized text projections
        """
        batch_size = vision_proj.shape[0]
        
        # Compute similarity matrix
        logits = vision_proj @ text_proj.T / self.temperature
        
        # Labels are diagonal (each image matches its corresponding text)
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric loss (image-to-text and text-to-image)
        loss_i2t = self.cross_entropy(logits, labels)
        loss_t2i = self.cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2


def create_multimodal_model(model_config: Dict[str, Any]) -> VisionLanguageModel:
    """Factory function to create multimodal models with different configurations."""
    
    # Default configuration
    default_config = {
        "vision_model": "vit_base",
        "text_model": "bert-base-uncased",
        "vision_dim": 768,
        "text_dim": 768,
        "hidden_dim": 512,
        "num_classes": 1000,
        "max_text_length": 77,
        "lora_config": None
    }
    
    # Update with user config
    config = {**default_config, **model_config}
    
    return VisionLanguageModel(**config)
