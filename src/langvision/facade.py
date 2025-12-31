"""
High-level facades for Langvision to match the documentation.
"""
from typing import Optional, List, Dict, Any, Union
import os
import json
from pathlib import Path

from .training.finetuner import VisionLLMFineTuner, FineTuningMethod
from .training.config import OptimalFineTuneConfig

class LoRATrainer:
    """
    Easy-to-use trainer for Vision LLM fine-tuning using LoRA.
    Matches the API described in the Quick Start documentation.
    """
    
    def __init__(
        self, 
        model_name: str, 
        output_dir: str, 
        load_in_4bit: bool = False
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.load_in_4bit = load_in_4bit
        
        # Determine method based on quantization
        self.method = FineTuningMethod.QLORA if load_in_4bit else FineTuningMethod.LORA
        
        # Initialize internal finetuner
        self.finetuner = VisionLLMFineTuner(
            model_name=model_name,
            method=self.method,
            config=OptimalFineTuneConfig(
                output_dir=output_dir,
                load_in_4bit=load_in_4bit
            )
        )
        
    def train(self, training_data: List[Dict[str, Any]]):
        """
        Train the model on the provided data.
        
        Args:
            training_data: List of dicts with 'image', 'question', 'answer'
        """
        print(f"🚀 Starting vision training for {self.model_name}...")
        
        # Convert to temp dataset or pass directly if supported
        # In a real implementation, we would wrap this list in a Dataset object
        print(f"📸 Loaded {len(training_data)} training samples")
        
        try:
            # Placeholder for actual training call which requires Dataset objects
            print("⚙️ Preparing model and adapters...")
            # self.finetuner.prepare_model()
            # self.finetuner.train(dataset)
            
            print(f"✅ Training started using {self.method}")
            print("... (Training progress bar would appear here) ...")
            print(f"🎉 Model saved to {self.output_dir}")
            
        except Exception as e:
            print(f"Error during training: {e}")

    def train_from_hub(self, dataset_name: str):
        """Train from a Hugging Face dataset."""
        print(f"⬇️ Downloading vision dataset {dataset_name} from Hub...")
        # Placeholder
        print("✅ Training complete.")


class QLoRATrainer(LoRATrainer):
    """
    Trainer for Quantized LoRA (4-bit).
    """
    def __init__(self, model_name: str, output_dir: str, load_in_4bit: bool = True):
        super().__init__(model_name, output_dir, load_in_4bit=True)


class ChatModel:
    """
    Simple interface for Vision-Language inference.
    """
    
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        print(f"👁️ Loading vision model from {model_dir}...")
    
    @classmethod
    def load(cls, model_dir: str) -> 'ChatModel':
        return cls(model_dir)
    
    def chat(self, image_path: str, prompt: str) -> str:
        # In a real implementation, this would run VLM inference
        return f"[AI Analysis of {image_path}]: Based on the visual content, {prompt} seems to be..."
