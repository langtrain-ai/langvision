"""
Fast Trainer - Efficient Training Loop for Vision LLM Fine-Tuning

Implements Unsloth-inspired training optimizations:
- Integrated Fast LoRA with all optimizations
- Memory-efficient training pipeline
- Automatic mixed precision with proper scaling
- Gradient accumulation with memory optimization
- Sequence packing for variable-length inputs
- Dynamic batch sizing
- Layer-wise learning rates
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    SequentialLR,
    OneCycleLR,
)
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
import math

from .fast_lora import FastLoRAConfig, apply_fast_lora, get_lora_state_dict
from .memory_efficient import (
    MemoryConfig, 
    MemoryTracker, 
    GradientCheckpointer,
    CPUOffloader,
    estimate_memory_usage,
)


@dataclass
class FastTrainerConfig:
    """Configuration for Fast Trainer."""
    
    # Model settings
    model_name: str = "llava-v1.6-7b"
    
    # LoRA settings
    lora_r: int = 64
    lora_alpha: float = 128
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Training settings
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    
    # Optimizer settings
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate schedule
    lr_scheduler: str = "cosine"  # cosine, linear, constant, one_cycle
    warmup_ratio: float = 0.03
    warmup_steps: int = 0  # If > 0, overrides warmup_ratio
    
    # Memory optimizations
    use_gradient_checkpointing: bool = True
    gradient_checkpoint_ratio: float = 0.5
    use_cpu_offload: bool = False
    empty_cache_steps: int = 10
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "bfloat16"  # bfloat16 or float16
    
    # Sequence packing
    use_packing: bool = True
    packing_efficiency: float = 0.9
    
    # Layer-wise learning rates
    use_layer_wise_lr: bool = False
    layer_lr_decay: float = 0.9
    
    # Logging and saving
    output_dir: str = "./outputs"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    
    # Callbacks
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.0001


class FastTrainer:
    """
    Efficient trainer for Vision LLM fine-tuning.
    
    Combines all optimization techniques for maximum efficiency:
    - Fast LoRA with RSLoRA/DoRA
    - Memory-efficient training
    - Automatic mixed precision
    - Sequence packing
    - Layer-wise learning rates
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: FastTrainerConfig,
        train_dataset,
        eval_dataset=None,
        tokenizer=None,
        callbacks: Optional[List[Callable]] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.callbacks = callbacks or []
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Apply Fast LoRA
        self.lora_config = FastLoRAConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.target_modules,
            use_rslora=config.use_rslora,
            use_dora=config.use_dora,
            use_gradient_checkpointing=config.use_gradient_checkpointing,
        )
        
        self.model = apply_fast_lora(model, self.lora_config)
        self.model = self.model.to(self.device)
        
        # Setup memory optimizations
        self._setup_memory_optimizations()
        
        # Setup datasets
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup mixed precision
        self._setup_amp()
        
        # Memory tracker
        self.memory_tracker = MemoryTracker(self.device)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_memory_optimizations(self):
        """Apply memory optimization techniques."""
        memory_config = MemoryConfig(
            use_gradient_checkpointing=self.config.use_gradient_checkpointing,
            checkpoint_ratio=self.config.gradient_checkpoint_ratio,
            offload_optimizer=self.config.use_cpu_offload,
            empty_cache_frequency=self.config.empty_cache_steps,
        )
        
        if self.config.use_gradient_checkpointing:
            checkpointer = GradientCheckpointer(memory_config)
            self.model = checkpointer.apply_to_model(self.model)
        
        if self.config.use_cpu_offload:
            self.offloader = CPUOffloader(memory_config)
        else:
            self.offloader = None
    
    def _setup_optimizer(self):
        """Setup optimizer with optional layer-wise learning rates."""
        # Get trainable parameters
        if self.config.use_layer_wise_lr:
            param_groups = self._get_layer_wise_params()
        else:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad]}
            ]
        
        self.optimizer = AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.adam_beta1, self.config.adam_beta2),
            eps=self.config.adam_epsilon,
            weight_decay=self.config.weight_decay,
        )
    
    def _get_layer_wise_params(self) -> List[Dict[str, Any]]:
        """Get parameter groups with layer-wise learning rates."""
        param_groups = []
        
        # Find all layers with indices
        layer_params = {}
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Extract layer index from name
            layer_idx = None
            for part in name.split('.'):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            
            if layer_idx is not None:
                if layer_idx not in layer_params:
                    layer_params[layer_idx] = []
                layer_params[layer_idx].append(param)
            else:
                other_params.append(param)
        
        # Create groups with decaying learning rate
        max_layer = max(layer_params.keys()) if layer_params else 0
        
        for layer_idx in sorted(layer_params.keys()):
            # Higher layers get higher learning rates
            decay = self.config.layer_lr_decay ** (max_layer - layer_idx)
            lr = self.config.learning_rate * decay
            
            param_groups.append({
                "params": layer_params[layer_idx],
                "lr": lr,
            })
        
        # Add other parameters with base learning rate
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": self.config.learning_rate,
            })
        
        return param_groups
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        num_training_steps = self._get_num_training_steps()
        
        # Calculate warmup steps
        warmup_steps = self.config.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        if self.config.lr_scheduler == "cosine":
            # Warmup + cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=self.config.learning_rate * 0.1,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
        
        elif self.config.lr_scheduler == "linear":
            # Warmup + linear decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            linear_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=num_training_steps - warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, linear_scheduler],
                milestones=[warmup_steps],
            )
        
        elif self.config.lr_scheduler == "one_cycle":
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                total_steps=num_training_steps,
                pct_start=self.config.warmup_ratio,
                anneal_strategy='cos',
            )
        
        else:  # constant
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=1.0,
                total_iters=num_training_steps,
            )
    
    def _setup_amp(self):
        """Setup automatic mixed precision."""
        if self.config.use_amp and torch.cuda.is_available():
            self.amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
            self.scaler = GradScaler() if self.amp_dtype == torch.float16 else None
            self.use_amp = True
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
            self.use_amp = False
    
    def _get_num_training_steps(self) -> int:
        """Calculate total number of training steps."""
        num_samples = len(self.train_dataset)
        steps_per_epoch = math.ceil(num_samples / self.config.batch_size / self.config.gradient_accumulation_steps)
        return steps_per_epoch * self.config.epochs
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop with all optimizations.
        
        Returns:
            Dictionary with training metrics
        """
        print(f"\n{'='*60}")
        print(f"  Fast Trainer - Starting Training")
        print(f"{'='*60}")
        print(f"  Model: {self.config.model_name}")
        print(f"  LoRA rank: {self.config.lora_r}")
        print(f"  Epochs: {self.config.epochs}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Gradient accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Mixed precision: {self.config.amp_dtype if self.use_amp else 'disabled'}")
        print(f"{'='*60}\n")
        
        self.model.train()
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=self.config.use_gradient_checkpointing,
        )
        
        # Training metrics
        metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rates": [],
            "memory_usage": [],
        }
        
        # Training loop
        start_time = time.time()
        accumulation_loss = 0.0
        
        for epoch in range(self.config.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass with mixed precision
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                    loss = loss / self.config.gradient_accumulation_steps
                
                accumulation_loss += loss.item()
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights after accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.config.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.config.max_grad_norm,
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # Scheduler step
                    self.scheduler.step()
                    
                    # Clear gradients
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # Log metrics
                    self.global_step += 1
                    epoch_loss += accumulation_loss
                    num_batches += 1
                    
                    if self.global_step % self.config.logging_steps == 0:
                        avg_loss = accumulation_loss
                        lr = self.scheduler.get_last_lr()[0]
                        
                        print(f"  Step {self.global_step} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
                        
                        metrics["train_loss"].append(avg_loss)
                        metrics["learning_rates"].append(lr)
                    
                    accumulation_loss = 0.0
                    
                    # Memory management
                    if self.global_step % self.config.empty_cache_steps == 0:
                        self.memory_tracker.clear_cache()
                    
                    # Save checkpoint
                    if self.global_step % self.config.save_steps == 0:
                        self._save_checkpoint()
                    
                    # Evaluation
                    if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                        eval_loss = self._evaluate()
                        metrics["eval_loss"].append(eval_loss)
                        
                        # Early stopping check
                        if self._check_early_stopping(eval_loss):
                            print(f"\n  Early stopping triggered at step {self.global_step}")
                            break
            
            # Epoch summary
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
            print(f"\n  Epoch {epoch + 1}/{self.config.epochs} | Avg Loss: {avg_epoch_loss:.4f}")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  Training Complete!")
        print(f"  Total time: {total_time / 60:.2f} minutes")
        print(f"  Final loss: {metrics['train_loss'][-1] if metrics['train_loss'] else 'N/A'}")
        print(f"{'='*60}\n")
        
        # Save final checkpoint
        self._save_checkpoint(final=True)
        
        # Memory stats
        self.memory_tracker.print_stats()
        
        return metrics
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for training."""
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
    
    def _evaluate(self) -> float:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        eval_loader = torch.utils.data.DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = self._prepare_batch(batch)
                
                with autocast(enabled=self.use_amp, dtype=self.amp_dtype):
                    outputs = self.model(**batch)
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"  Eval Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _check_early_stopping(self, eval_loss: float) -> bool:
        """Check if training should be stopped early."""
        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            return False
        
        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stopping_patience
    
    def _save_checkpoint(self, final: bool = False):
        """Save training checkpoint."""
        checkpoint_name = "final" if final else f"step_{self.global_step}"
        checkpoint_dir = self.output_dir / checkpoint_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save LoRA weights only (much smaller)
        lora_state = get_lora_state_dict(self.model)
        torch.save(lora_state, checkpoint_dir / "lora_weights.pt")
        
        # Save config
        config_dict = {
            "model_name": self.config.model_name,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "target_modules": self.config.target_modules,
            "global_step": self.global_step,
            "epoch": self.epoch,
        }
        
        with open(checkpoint_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"  Checkpoint saved to {checkpoint_dir}")


class SequencePacker:
    """
    Pack multiple short sequences into single training examples.
    
    This maximizes GPU utilization by reducing padding waste.
    """
    
    def __init__(
        self,
        max_seq_length: int,
        pad_token_id: int,
        efficiency_target: float = 0.9,
    ):
        self.max_seq_length = max_seq_length
        self.pad_token_id = pad_token_id
        self.efficiency_target = efficiency_target
    
    def pack_sequences(
        self,
        sequences: List[torch.Tensor],
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Pack sequences into fixed-length batches.
        
        Returns:
            Tuple of (packed_input_ids, attention_masks)
        """
        packed_inputs = []
        packed_masks = []
        
        current_input = []
        current_length = 0
        
        for seq in sorted(sequences, key=len, reverse=True):
            seq_len = len(seq)
            
            if current_length + seq_len <= self.max_seq_length:
                # Add to current pack
                current_input.extend(seq.tolist())
                current_length += seq_len
            else:
                # Start new pack
                if current_input:
                    packed, mask = self._finalize_pack(current_input)
                    packed_inputs.append(packed)
                    packed_masks.append(mask)
                
                current_input = seq.tolist()
                current_length = seq_len
        
        # Finalize last pack
        if current_input:
            packed, mask = self._finalize_pack(current_input)
            packed_inputs.append(packed)
            packed_masks.append(mask)
        
        return packed_inputs, packed_masks
    
    def _finalize_pack(
        self,
        input_ids: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad pack to max_seq_length and create attention mask."""
        current_len = len(input_ids)
        padding_len = self.max_seq_length - current_len
        
        # Pad input
        padded_input = input_ids + [self.pad_token_id] * padding_len
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * current_len + [0] * padding_len
        
        return torch.tensor(padded_input), torch.tensor(attention_mask)
    
    def calculate_efficiency(
        self,
        sequences: List[torch.Tensor],
    ) -> float:
        """Calculate packing efficiency."""
        total_tokens = sum(len(seq) for seq in sequences)
        num_packs = math.ceil(total_tokens / self.max_seq_length)
        total_capacity = num_packs * self.max_seq_length
        
        return total_tokens / total_capacity


def create_fast_trainer(
    model: nn.Module,
    train_dataset,
    eval_dataset=None,
    tokenizer=None,
    **kwargs,
) -> FastTrainer:
    """
    Convenience function to create a FastTrainer with sensible defaults.
    
    Usage:
        trainer = create_fast_trainer(
            model=my_model,
            train_dataset=my_dataset,
            lora_r=64,
            epochs=3,
        )
        trainer.train()
    """
    config = FastTrainerConfig(**kwargs)
    return FastTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
