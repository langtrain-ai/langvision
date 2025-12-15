"""
Cloud Training Interface

High-level interface for training Vision LLMs on the server.
Handles local preprocessing, upload, job submission, and monitoring.
"""

import torch
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import tempfile
import logging

from .client import LangvisionClient, JobResult, JobStatus, get_client

logger = logging.getLogger(__name__)


@dataclass
class CloudTrainingConfig:
    """Configuration for cloud training."""
    # Model
    model: str = "llava-v1.6-7b"
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: float = 128
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = False
    
    # Training
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_steps: int = -1
    
    # Optimization
    use_gradient_checkpointing: bool = True
    use_neftune: bool = True
    
    # Precision
    precision: str = "bf16"
    
    # Output
    output_name: Optional[str] = None


class CloudTrainer:
    """
    High-level interface for training Vision LLMs on the Langvision server.
    
    This handles:
    1. Local data preprocessing and validation
    2. Dataset upload to server
    3. Training job submission
    4. Progress monitoring
    5. Model download (optional)
    
    Usage:
        trainer = CloudTrainer(model="llava-v1.6-7b")
        
        # Train from local dataset
        result = trainer.train(
            train_data="./my_dataset.json",
            lora_r=64,
            epochs=3,
        )
        
        # Download trained model
        trainer.download_model(result.model_id, "./my_model")
    """
    
    def __init__(
        self,
        model: str = "llava-v1.6-7b",
        api_key: Optional[str] = None,
        config: Optional[CloudTrainingConfig] = None,
    ):
        self.client = LangvisionClient(api_key=api_key)
        
        if config:
            self.config = config
            self.config.model = model
        else:
            self.config = CloudTrainingConfig(model=model)
    
    def train(
        self,
        train_data: Union[str, List[Dict]],
        eval_data: Optional[Union[str, List[Dict]]] = None,
        wait: bool = True,
        on_progress: Optional[Callable[[JobResult], None]] = None,
        **kwargs,
    ) -> JobResult:
        """
        Train a model on the server.
        
        Args:
            train_data: Path to dataset file or list of examples
            eval_data: Optional evaluation data
            wait: Wait for training to complete
            on_progress: Callback for progress updates
            **kwargs: Override config options
        
        Returns:
            JobResult with training results
        """
        # Update config with kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        print(f"\n{'='*60}")
        print(f"  ☁️ Cloud Training")
        print(f"{'='*60}")
        print(f"  Model: {self.config.model}")
        print(f"  LoRA rank: {self.config.lora_r}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"{'='*60}\n")
        
        # Step 1: Upload training dataset
        print("  📤 Uploading training data...")
        train_dataset_id = self._upload_data(train_data, "train")
        
        # Step 2: Upload eval dataset if provided
        eval_dataset_id = None
        if eval_data:
            print("  📤 Uploading evaluation data...")
            eval_dataset_id = self._upload_data(eval_data, "eval")
        
        # Step 3: Submit training job
        print("  🚀 Submitting training job...")
        
        training_config = {
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "lora_dropout": self.config.lora_dropout,
            "use_rslora": self.config.use_rslora,
            "use_dora": self.config.use_dora,
            "epochs": self.config.epochs,
            "batch_size": self.config.batch_size,
            "learning_rate": self.config.learning_rate,
            "warmup_ratio": self.config.warmup_ratio,
            "use_gradient_checkpointing": self.config.use_gradient_checkpointing,
            "use_neftune": self.config.use_neftune,
            "precision": self.config.precision,
        }
        
        if eval_dataset_id:
            training_config["eval_dataset_id"] = eval_dataset_id
        
        if self.config.max_steps > 0:
            training_config["max_steps"] = self.config.max_steps
        
        if self.config.output_name:
            training_config["output_name"] = self.config.output_name
        
        job = self.client.submit_training(
            model=self.config.model,
            dataset_id=train_dataset_id,
            config=training_config,
        )
        
        print(f"  ✅ Job submitted: {job.job_id}")
        
        # Step 4: Wait for completion if requested
        if wait:
            print("\n  ⏳ Training in progress...")
            
            def progress_callback(result: JobResult):
                if on_progress:
                    on_progress(result)
                print(f"\r  Progress: {result.progress:.1f}%", end="", flush=True)
            
            result = self.client.wait_for_job(job.job_id, callback=progress_callback)
            print()  # New line after progress
            
            if result.is_success:
                print(f"\n  ✅ Training complete!")
                print(f"  Model ID: {result.result.get('model_id', 'N/A')}")
            else:
                print(f"\n  ❌ Training failed: {result.error}")
            
            return result
        
        return job
    
    def _upload_data(
        self,
        data: Union[str, List[Dict]],
        name_prefix: str,
    ) -> str:
        """Upload dataset to server."""
        # If data is a list, save to temp file
        if isinstance(data, list):
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.json',
                delete=False,
            ) as f:
                json.dump(data, f)
                data = f.name
        
        # Upload file
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {data}")
        
        dataset_id = self.client.upload_dataset(
            file_path=str(path),
            name=f"{name_prefix}_{path.stem}",
            task="vqa",  # Auto-detect from content
        )
        
        print(f"    Dataset ID: {dataset_id}")
        return dataset_id
    
    def download_model(
        self,
        model_id: str,
        output_dir: str,
        format: str = "safetensors",
    ) -> str:
        """Download trained model from server."""
        print(f"\n  📥 Downloading model...")
        path = self.client.download_model(model_id, output_dir, format)
        print(f"  ✅ Model saved to: {path}")
        return path
    
    def list_jobs(self) -> List[JobResult]:
        """List user's training jobs."""
        return self.client.list_jobs()
    
    def get_job_status(self, job_id: str) -> JobResult:
        """Get status of a training job."""
        return self.client.get_job_status(job_id)
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a training job."""
        return self.client.cancel_job(job_id)


class CloudInference:
    """
    Interface for running inference on the server.
    
    Usage:
        inference = CloudInference("my-finetuned-model")
        response = inference.generate(image_url, prompt)
        
        # Streaming
        for token in inference.stream(image_url, prompt):
            print(token, end="")
    """
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        **default_params,
    ):
        self.client = LangvisionClient(api_key=api_key)
        self.model = model
        self.default_params = default_params
    
    def generate(
        self,
        image_url: str,
        prompt: str,
        **kwargs,
    ) -> str:
        """Generate response for image and prompt."""
        params = {**self.default_params, **kwargs}
        return self.client.generate(self.model, image_url, prompt, **params)
    
    def stream(
        self,
        image_url: str,
        prompt: str,
        **kwargs,
    ):
        """Stream generation token by token."""
        params = {**self.default_params, **kwargs}
        return self.client.stream_generate(self.model, image_url, prompt, **params)
    
    def batch(
        self,
        inputs: List[Dict[str, str]],
        **kwargs,
    ) -> List[str]:
        """Batch generation for multiple inputs."""
        params = {**self.default_params, **kwargs}
        return self.client.batch_generate(self.model, inputs, **params)


# Convenience functions

def cloud_train(
    model: str,
    train_data: Union[str, List[Dict]],
    **kwargs,
) -> JobResult:
    """
    Train a model on the cloud (convenience function).
    
    Args:
        model: Model name (e.g., "llava-v1.6-7b")
        train_data: Path to dataset or list of examples
        **kwargs: Training configuration
    
    Returns:
        JobResult with training results
    """
    trainer = CloudTrainer(model)
    return trainer.train(train_data, **kwargs)


def cloud_generate(
    model: str,
    image_url: str,
    prompt: str,
    **kwargs,
) -> str:
    """Generate response on the cloud (convenience function)."""
    inference = CloudInference(model)
    return inference.generate(image_url, prompt, **kwargs)
