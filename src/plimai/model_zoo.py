import timm
import torch
from torchvision import models as tv_models
from transformers import AutoModel, AutoImageProcessor

class VisualModelLoader:
    def __init__(self, model_name, source='timm', pretrained=True, device='cpu'):
        self.model_name = model_name
        self.source = source
        self.pretrained = pretrained
        self.device = device
        self.model = self.load_model()

    def load_model(self):
        if self.source == 'timm':
            model = timm.create_model(self.model_name, pretrained=self.pretrained)
        elif self.source == 'torchvision':
            model_fn = getattr(tv_models, self.model_name)
            model = model_fn(pretrained=self.pretrained)
        elif self.source == 'hf':
            model = AutoModel.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unknown source: {self.source}")
        return model.to(self.device)

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)

# Example usage:
# loader = VisualModelLoader('vit_base_patch16_224', source='timm')
# loader = VisualModelLoader('resnet50', source='torchvision')
# loader = VisualModelLoader('google/vit-base-patch16-224', source='hf')
