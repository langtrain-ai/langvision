import torch
import torch.nn as nn

class NEFTuneEmbedding(nn.Module):
    """
    NEFTune: Noise Embedding Fine-Tuning
    
    Adds uniform noise to embeddings during training for better generalization.
    Paper: https://arxiv.org/abs/2310.05914
    """
    
    def __init__(self, embedding: nn.Module, noise_alpha: float = 5.0):
        super().__init__()
        self.embedding = embedding
        self.noise_alpha = noise_alpha
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(input_ids)
        
        if self.training:
            # Add noise scaled by 1/sqrt(seq_len * embed_dim)
            dims = torch.tensor(embeds.size(1) * embeds.size(2))
            magnitude = self.noise_alpha / torch.sqrt(dims)
            noise = torch.zeros_like(embeds).uniform_(-1, 1) * magnitude
            embeds = embeds + noise
        
        return embeds
