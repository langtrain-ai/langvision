"""
Basic training example for Langvision with LoRA fine-tuning.
This example demonstrates how to use the enhanced Langvision framework for image classification.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging

# Langvision imports
from langvision.models.vision_transformer import VisionTransformer
from langvision.models.resnet import resnet50
from langvision.models.lora import LoRAConfig
from langvision.training.advanced_trainer import AdvancedTrainer, TrainingConfig
from langvision.data.enhanced_datasets import create_enhanced_dataloaders, DatasetConfig
from langvision.callbacks.base import Callback
from langvision.utils.metrics import EvaluationSuite
from langvision.utils.device import get_device


class MetricsLoggingCallback(Callback):
    """Custom callback for logging training metrics."""
    
    def __init__(self):
        super().__init__("MetricsLogger")
        self.epoch_metrics = []
    
    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        """Log metrics at the end of each epoch."""
        self.logger.info(f"Epoch {epoch} completed:")
        for metric_name, metric_value in metrics.items():
            self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        self.epoch_metrics.append({
            'epoch': epoch,
            **metrics
        })
    
    def on_train_end(self, trainer):
        """Save metrics to file at the end of training."""
        import json
        metrics_file = Path(trainer.output_dir) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        self.logger.info(f"Training metrics saved to {metrics_file}")


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )


def create_model_with_lora(model_type: str = "vit", num_classes: int = 10):
    """Create a model with LoRA configuration."""
    
    # Configure LoRA parameters
    lora_config = LoRAConfig(
        r=16,                    # LoRA rank
        alpha=32,                # LoRA alpha
        dropout=0.1,             # LoRA dropout
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention modules
        bias="none",             # Don't adapt bias terms
        task_type="FEATURE_EXTRACTION"
    )
    
    if model_type == "vit":
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            num_classes=num_classes,
            embed_dim=768,
            depth=12,
            num_heads=12,
            lora_config=lora_config
        )
    elif model_type == "resnet":
        model = resnet50(
            num_classes=num_classes,
            lora_config=lora_config
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    return model, lora_config


def main():
    """Main training function."""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting Langvision training example")
    
    # Configuration
    dataset_config = DatasetConfig(
        root_dir="./data/cifar10",  # Replace with your dataset path
        image_size=(224, 224),
        batch_size=32,
        num_workers=4,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        use_augmentation=True,
        augmentation_strength=0.5,
        validate_images=True
    )
    
    training_config = TrainingConfig(
        epochs=50,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_epochs=5,
        optimizer="adamw",
        scheduler="cosine",
        use_amp=True,
        gradient_clip_norm=1.0,
        early_stopping_patience=10,
        early_stopping_metric="val_loss",
        output_dir="./outputs",
        experiment_name="langvision_basic_training",
        log_interval=10,
        eval_interval=1,
        save_interval=5
    )
    
    try:
        # Create datasets and dataloaders
        logger.info("Creating datasets...")
        dataloaders = create_enhanced_dataloaders(
            config=dataset_config,
            dataset_type="image"
        )
        
        logger.info(f"Dataset created with {len(dataloaders['train'].dataset)} training samples")
        if 'val' in dataloaders:
            logger.info(f"Validation set: {len(dataloaders['val'].dataset)} samples")
        
        # Create model with LoRA
        logger.info("Creating model with LoRA fine-tuning...")
        model, lora_config = create_model_with_lora(
            model_type="vit",  # or "resnet"
            num_classes=10     # CIFAR-10 has 10 classes
        )
        
        # Update training config with LoRA settings
        training_config.lora_config = lora_config
        training_config.freeze_backbone = True
        
        # Create callbacks
        callbacks = [
            MetricsLoggingCallback()
        ]
        
        # Create trainer
        logger.info("Initializing trainer...")
        trainer = AdvancedTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders.get('val'),
            config=training_config,
            callbacks=callbacks
        )
        
        # Start training
        logger.info("Starting training...")
        trainer.train()
        
        # Evaluate model
        if 'val' in dataloaders:
            logger.info("Evaluating model...")
            device = get_device()
            evaluation_suite = EvaluationSuite(
                model=trainer.model,
                device=device,
                class_names=[f"class_{i}" for i in range(10)]  # CIFAR-10 class names
            )
            
            eval_results = evaluation_suite.evaluate_classification(
                dataloader=dataloaders['val'],
                return_predictions=False
            )
            
            logger.info("Evaluation Results:")
            for metric_name, metric_value in eval_results.items():
                if isinstance(metric_value, (int, float)):
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
            
            # Benchmark inference speed
            logger.info("Benchmarking inference speed...")
            benchmark_results = evaluation_suite.benchmark_inference(
                dataloader=dataloaders['val'],
                num_warmup=10,
                num_benchmark=100
            )
            
            logger.info("Benchmark Results:")
            for metric_name, metric_value in benchmark_results.items():
                logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
