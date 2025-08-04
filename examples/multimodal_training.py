"""
Multimodal Vision-Language training example with CLIP-style contrastive learning.
This example demonstrates how to train a vision-language model using the enhanced Langvision framework.
"""

import torch
import torch.nn as nn
from pathlib import Path
import logging
import json

# Langvision imports
from langvision.models.multimodal import VisionLanguageModel, CLIPLoss, create_multimodal_model
from langvision.models.lora import LoRAConfig
from langvision.training.advanced_trainer import AdvancedTrainer, TrainingConfig
from langvision.data.enhanced_datasets import create_enhanced_dataloaders, DatasetConfig
from langvision.callbacks.base import Callback
from langvision.utils.metrics import EvaluationSuite, ContrastiveMetrics
from langvision.utils.device import get_device


class MultimodalLoss(nn.Module):
    """Combined loss for multimodal training."""
    
    def __init__(self, 
                 contrastive_weight: float = 1.0,
                 classification_weight: float = 0.5,
                 temperature: float = 0.07):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.classification_weight = classification_weight
        
        self.clip_loss = CLIPLoss(temperature=temperature)
        self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: dict, targets: torch.Tensor) -> dict:
        """Compute combined loss."""
        losses = {}
        total_loss = 0
        
        # Contrastive loss
        if 'vision_proj' in outputs and 'text_proj' in outputs:
            contrastive_loss = self.clip_loss(outputs['vision_proj'], outputs['text_proj'])
            losses['contrastive_loss'] = contrastive_loss
            total_loss += self.contrastive_weight * contrastive_loss
        
        # Classification loss
        if 'logits' in outputs:
            classification_loss = self.classification_loss(outputs['logits'], targets)
            losses['classification_loss'] = classification_loss
            total_loss += self.classification_weight * classification_loss
        
        losses['total_loss'] = total_loss
        return losses


class MultimodalMetricsCallback(Callback):
    """Callback for logging multimodal training metrics."""
    
    def __init__(self):
        super().__init__("MultimodalMetrics")
        self.epoch_metrics = []
        self.contrastive_metrics = ContrastiveMetrics()
    
    def on_epoch_end(self, trainer, epoch: int, metrics: dict):
        """Log comprehensive multimodal metrics."""
        self.logger.info(f"Epoch {epoch} Multimodal Metrics:")
        
        # Log all metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        # Store metrics
        self.epoch_metrics.append({
            'epoch': epoch,
            **metrics
        })
    
    def on_validation_end(self, trainer, metrics: dict):
        """Compute additional contrastive metrics during validation."""
        if hasattr(trainer, 'last_validation_outputs'):
            outputs = trainer.last_validation_outputs
            if 'vision_proj' in outputs and 'text_proj' in outputs:
                # Compute contrastive accuracy
                contrastive_acc = self.contrastive_metrics.contrastive_accuracy(
                    outputs['vision_proj'], outputs['text_proj']
                )
                
                # Compute retrieval metrics
                retrieval_metrics = self.contrastive_metrics.retrieval_metrics(
                    outputs['vision_proj'], outputs['text_proj']
                )
                
                # Log additional metrics
                self.logger.info("Additional Contrastive Metrics:")
                for metric_name, metric_value in {**contrastive_acc, **retrieval_metrics}.items():
                    self.logger.info(f"  {metric_name}: {metric_value:.4f}")


def create_sample_annotations():
    """Create sample annotations for demonstration."""
    annotations = {
        "sample1.jpg": "A red car driving on the highway",
        "sample2.jpg": "A cat sitting on a windowsill",
        "sample3.jpg": "Mountains covered with snow",
        "sample4.jpg": "A person walking in the park",
        "sample5.jpg": "A beautiful sunset over the ocean"
    }
    
    # Save to file
    annotations_file = Path("./data/annotations.json")
    annotations_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    return str(annotations_file)


def main():
    """Main multimodal training function."""
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('multimodal_training.log')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Langvision multimodal training example")
    
    # Create sample annotations
    annotations_file = create_sample_annotations()
    logger.info(f"Created sample annotations: {annotations_file}")
    
    # Configuration
    dataset_config = DatasetConfig(
        root_dir="./data/images",  # Replace with your dataset path
        image_size=(224, 224),
        batch_size=16,  # Smaller batch size for multimodal training
        num_workers=4,
        train_split=0.8,
        val_split=0.2,
        test_split=0.0,
        use_augmentation=True,
        augmentation_strength=0.3,  # Lighter augmentation for multimodal
        validate_images=True,
        text_max_length=77,
        text_tokenizer="bert-base-uncased"
    )
    
    # LoRA configuration for efficient fine-tuning
    lora_config = LoRAConfig(
        r=8,                     # Lower rank for multimodal
        alpha=16,
        dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    training_config = TrainingConfig(
        epochs=30,
        batch_size=16,
        learning_rate=5e-5,      # Lower learning rate for multimodal
        weight_decay=1e-4,
        warmup_epochs=3,
        optimizer="adamw",
        scheduler="cosine",
        use_amp=True,
        gradient_clip_norm=1.0,
        early_stopping_patience=8,
        early_stopping_metric="val_loss",
        output_dir="./outputs",
        experiment_name="langvision_multimodal_training",
        log_interval=5,
        eval_interval=1,
        save_interval=3,
        lora_config=lora_config,
        freeze_backbone=True
    )
    
    try:
        # Create multimodal datasets
        logger.info("Creating multimodal datasets...")
        dataloaders = create_enhanced_dataloaders(
            config=dataset_config,
            dataset_type="multimodal",
            annotations_file=annotations_file
        )
        
        logger.info(f"Dataset created with {len(dataloaders['train'].dataset)} training samples")
        if 'val' in dataloaders:
            logger.info(f"Validation set: {len(dataloaders['val'].dataset)} samples")
        
        # Create multimodal model
        logger.info("Creating vision-language model...")
        model_config = {
            "vision_model": "vit_base",
            "text_model": "bert-base-uncased",
            "vision_dim": 768,
            "text_dim": 768,
            "hidden_dim": 512,
            "num_classes": 10,
            "max_text_length": 77,
            "lora_config": lora_config
        }
        
        model = create_multimodal_model(model_config)
        
        # Create custom loss function
        criterion = MultimodalLoss(
            contrastive_weight=1.0,
            classification_weight=0.5,
            temperature=0.07
        )
        
        # Create callbacks
        callbacks = [
            MultimodalMetricsCallback()
        ]
        
        # Create custom trainer for multimodal training
        class MultimodalTrainer(AdvancedTrainer):
            """Custom trainer for multimodal models."""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.multimodal_criterion = criterion
                self.last_validation_outputs = {}
            
            def train_step(self, batch):
                """Custom training step for multimodal data."""
                images = batch['images']
                texts = batch.get('texts', None)
                labels = batch['labels']
                
                # Forward pass
                outputs = self.model(images, texts, return_features=True)
                
                # Compute losses
                loss_dict = self.multimodal_criterion(outputs, labels)
                
                return loss_dict['total_loss'], outputs, loss_dict
            
            def validation_step(self, batch):
                """Custom validation step for multimodal data."""
                images = batch['images']
                texts = batch.get('texts', None)
                labels = batch['labels']
                
                # Forward pass
                outputs = self.model(images, texts, return_features=True)
                
                # Store outputs for metrics computation
                if not hasattr(self, 'validation_outputs'):
                    self.validation_outputs = {'vision_proj': [], 'text_proj': [], 'logits': []}
                
                if 'vision_proj' in outputs:
                    self.validation_outputs['vision_proj'].append(outputs['vision_proj'].cpu())
                if 'text_proj' in outputs:
                    self.validation_outputs['text_proj'].append(outputs['text_proj'].cpu())
                if 'logits' in outputs:
                    self.validation_outputs['logits'].append(outputs['logits'].cpu())
                
                # Compute losses
                loss_dict = self.multimodal_criterion(outputs, labels)
                
                return loss_dict['total_loss'], outputs, loss_dict
            
            def on_validation_epoch_end(self):
                """Process validation outputs at the end of validation."""
                if hasattr(self, 'validation_outputs'):
                    # Concatenate all validation outputs
                    self.last_validation_outputs = {}
                    for key, values in self.validation_outputs.items():
                        if values:
                            self.last_validation_outputs[key] = torch.cat(values, dim=0)
                    
                    # Clear for next epoch
                    self.validation_outputs = {'vision_proj': [], 'text_proj': [], 'logits': []}
        
        # Initialize trainer
        logger.info("Initializing multimodal trainer...")
        trainer = MultimodalTrainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders.get('val'),
            config=training_config,
            callbacks=callbacks
        )
        
        # Start training
        logger.info("Starting multimodal training...")
        trainer.train()
        
        # Evaluate model
        if 'val' in dataloaders:
            logger.info("Evaluating multimodal model...")
            device = get_device()
            evaluation_suite = EvaluationSuite(
                model=trainer.model,
                device=device
            )
            
            # Evaluate contrastive performance
            eval_results = evaluation_suite.evaluate_contrastive(
                dataloader=dataloaders['val']
            )
            
            logger.info("Multimodal Evaluation Results:")
            for metric_name, metric_value in eval_results.items():
                if isinstance(metric_value, (int, float)):
                    logger.info(f"  {metric_name}: {metric_value:.4f}")
        
        logger.info("Multimodal training completed successfully!")
        
    except Exception as e:
        logger.error(f"Multimodal training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
