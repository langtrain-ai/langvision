"""
Advanced training utilities with LoRA fine-tuning, mixed precision, and comprehensive logging.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, Any, Optional, List, Callable, Union
import logging
import time
import os
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb
from dataclasses import dataclass, asdict

from ..models.lora import LoRAConfig
from ..utils.metrics import MetricsTracker
from ..callbacks.base import Callback
from ..utils.device import get_device, to_device


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training settings
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    
    # LoRA settings
    lora_config: Optional[LoRAConfig] = None
    freeze_backbone: bool = True
    
    # Optimization settings
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    gradient_clip_norm: float = 1.0
    
    # Mixed precision and performance
    use_amp: bool = True
    compile_model: bool = False
    
    # Regularization
    dropout: float = 0.1
    label_smoothing: float = 0.0
    
    # Logging and checkpointing
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 5
    save_top_k: int = 3
    
    # Paths
    output_dir: str = "./outputs"
    experiment_name: str = "langvision_experiment"
    
    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # "min" or "max"


class AdvancedTrainer:
    """Advanced trainer with comprehensive features for vision model fine-tuning."""
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 config: Optional[TrainingConfig] = None,
                 callbacks: Optional[List[Callback]] = None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.callbacks = callbacks or []
        
        # Setup device and distributed training
        self.device = get_device()
        self.model = to_device(self.model, self.device)
        
        # Setup LoRA if configured
        if self.config.lora_config:
            self._setup_lora()
        
        # Setup optimization
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Setup loss function
        self.criterion = self._create_criterion()
        
        # Setup logging
        self.logger = self._setup_logging()
        self.metrics_tracker = MetricsTracker()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if self.config.early_stopping_mode == 'min' else float('-inf')
        self.patience_counter = 0
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir) / self.config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", "w") as f:
            json.dump(asdict(self.config), f, indent=2)
        
        # Model compilation for PyTorch 2.0+
        if self.config.compile_model and hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
    
    def _setup_lora(self):
        """Setup LoRA fine-tuning by freezing backbone parameters."""
        if self.config.freeze_backbone:
            # Freeze all parameters first
            for param in self.model.parameters():
                param.requires_grad = False
            
            # Unfreeze LoRA parameters
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') or hasattr(module, 'lora_B'):
                    for param in module.parameters():
                        if 'lora' in param.name if hasattr(param, 'name') else True:
                            param.requires_grad = True
        
        # Log trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"Total parameters: {total_params:,}")
        self.logger.info(f"Trainable parameters: {trainable_params:,}")
        self.logger.info(f"Trainable ratio: {trainable_params/total_params:.2%}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        if self.config.optimizer.lower() == "adam":
            return optim.Adam(trainable_params, 
                            lr=self.config.learning_rate,
                            weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "adamw":
            return optim.AdamW(trainable_params,
                             lr=self.config.learning_rate,
                             weight_decay=self.config.weight_decay)
        elif self.config.optimizer.lower() == "sgd":
            return optim.SGD(trainable_params,
                           lr=self.config.learning_rate,
                           momentum=0.9,
                           weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.config.epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.config.early_stopping_mode,
                patience=self.config.early_stopping_patience // 2,
                factor=0.5
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create loss function with label smoothing support."""
        if self.config.label_smoothing > 0:
            return nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        else:
            return nn.CrossEntropyLoss()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(f"langvision.{self.config.experiment_name}")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {}
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = to_device(batch, self.device)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(batch['images'])
                loss = self.criterion(outputs, batch['labels'])
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.gradient_clip_norm
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                # Gradient clipping
                if self.config.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_norm
                    )
                
                self.optimizer.step()
            
            # Update metrics
            batch_size = batch['images'].size(0)
            self.metrics_tracker.update('train_loss', loss.item(), batch_size)
            
            # Calculate accuracy
            with torch.no_grad():
                pred = outputs.argmax(dim=1)
                acc = (pred == batch['labels']).float().mean()
                self.metrics_tracker.update('train_acc', acc.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{acc.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}, Batch {batch_idx}: "
                    f"Loss={loss.item():.4f}, Acc={acc.item():.4f}"
                )
            
            self.global_step += 1
        
        # Get epoch metrics
        epoch_metrics = self.metrics_tracker.get_averages(['train_loss', 'train_acc'])
        self.metrics_tracker.reset()
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            
            for batch in pbar:
                batch = to_device(batch, self.device)
                
                with autocast(enabled=self.config.use_amp):
                    outputs = self.model(batch['images'])
                    loss = self.criterion(outputs, batch['labels'])
                
                # Update metrics
                batch_size = batch['images'].size(0)
                self.metrics_tracker.update('val_loss', loss.item(), batch_size)
                
                # Calculate accuracy
                pred = outputs.argmax(dim=1)
                acc = (pred == batch['labels']).float().mean()
                self.metrics_tracker.update('val_acc', acc.item(), batch_size)
                
                pbar.set_postfix({
                    'val_loss': f"{loss.item():.4f}",
                    'val_acc': f"{acc.item():.4f}"
                })
        
        # Get validation metrics
        val_metrics = self.metrics_tracker.get_averages(['val_loss', 'val_acc'])
        self.metrics_tracker.reset()
        
        return val_metrics
    
    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'metrics': metrics,
            'config': asdict(self.config)
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with {self.config.early_stopping_metric}: {metrics.get(self.config.early_stopping_metric, 'N/A')}")
        
        # Clean up old checkpoints (keep only top-k)
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent ones."""
        checkpoint_files = list(self.output_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoint_files) > self.config.save_top_k:
            # Sort by epoch number
            checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
            # Remove oldest checkpoints
            for old_checkpoint in checkpoint_files[:-self.config.save_top_k]:
                old_checkpoint.unlink()
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.logger.info(f"Training for {self.config.epochs} epochs")
        
        # Call training start callbacks
        for callback in self.callbacks:
            callback.on_train_start(self)
        
        try:
            for epoch in range(self.config.epochs):
                self.current_epoch = epoch
                
                # Call epoch start callbacks
                for callback in self.callbacks:
                    callback.on_epoch_start(self, epoch)
                
                # Training
                train_metrics = self.train_epoch()
                
                # Validation
                val_metrics = {}
                if epoch % self.config.eval_interval == 0:
                    val_metrics = self.validate()
                
                # Combine metrics
                all_metrics = {**train_metrics, **val_metrics}
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                        metric_value = all_metrics.get(self.config.early_stopping_metric)
                        if metric_value is not None:
                            self.scheduler.step(metric_value)
                    else:
                        self.scheduler.step()
                
                # Logging
                self.logger.info(f"Epoch {epoch} completed:")
                for metric_name, metric_value in all_metrics.items():
                    self.logger.info(f"  {metric_name}: {metric_value:.4f}")
                
                # Check for best model
                current_metric = all_metrics.get(self.config.early_stopping_metric)
                is_best = False
                
                if current_metric is not None:
                    if self.config.early_stopping_mode == 'min':
                        is_best = current_metric < self.best_metric
                    else:
                        is_best = current_metric > self.best_metric
                    
                    if is_best:
                        self.best_metric = current_metric
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                
                # Save checkpoint
                if epoch % self.config.save_interval == 0 or is_best:
                    self.save_checkpoint(all_metrics, is_best)
                
                # Call epoch end callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(self, epoch, all_metrics)
                
                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {self.patience_counter} epochs without improvement")
                    break
        
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            raise
        
        finally:
            # Call training end callbacks
            for callback in self.callbacks:
                callback.on_train_end(self)
            
            self.logger.info("Training completed!")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
