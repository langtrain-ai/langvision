import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import GradScaler, autocast
from typing import Optional, Dict, Any, List, Union, Callable, Tuple
from pathlib import Path
import math
import time
import logging

from .config import OptimalFineTuneConfig, FineTuningMethod
from .modules import NEFTuneEmbedding
from .optimizers import PagedAdamW

logger = logging.getLogger(__name__)

class VisionLLMFineTuner:
    """
    Optimal Vision LLM Fine-Tuner.
    
    Combines the best techniques for efficient and effective fine-tuning:
    - QLoRA for memory efficiency
    - RSLoRA/DoRA for better training
    - Flash Attention for speed
    - NEFTune for generalization
    - Layer-wise LR for VLMs
    
    Usage:
        finetuner = VisionLLMFineTuner("llava-v1.6-7b", method="qlora")
        finetuner.prepare_model()
        finetuner.train(train_dataset, eval_dataset)
        finetuner.save("./my_model")
    """
    
    def __init__(
        self,
        model_name: str = "llava-v1.6-7b",
        method: Union[str, FineTuningMethod] = "qlora",
        config: Optional[OptimalFineTuneConfig] = None,
        **kwargs,
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: Name of the Vision LLM to fine-tune
            method: Fine-tuning method (lora, qlora, rslora, dora)
            config: Full configuration object
            **kwargs: Override config parameters
        """
        if config is None:
            config = OptimalFineTuneConfig(model_name=model_name)
        
        # Apply method
        if isinstance(method, str):
            method = FineTuningMethod(method.lower())
        config.method = method
        
        # Apply kwargs overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        self.config = config
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Device setup
        self.device = self._get_device()
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._print_config()
    
    def _get_device(self) -> torch.device:
        """Get the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    
    def _print_config(self):
        """Print configuration summary."""
        c = self.config
        print(f"\n{'='*60}")
        print(f"  🎯 Vision LLM Fine-Tuner Configuration")
        print(f"{'='*60}")
        print(f"  Model:           {c.model_name}")
        print(f"  Method:          {c.method.value.upper()}")
        print(f"  Objective:       {c.objective.value.upper()}")
        print(f"  LoRA Rank:       {c.lora_r}")
        print(f"  LoRA Alpha:      {c.lora_alpha}")
        print(f"  RSLoRA:          {'✓' if c.use_rslora else '✗'}")
        print(f"  DoRA:            {'✓' if c.use_dora else '✗'}")
        print(f"  NEFTune:         {'✓' if c.use_neftune else '✗'}")
        print(f"  Flash Attention: {'✓' if c.use_flash_attention else '✗'}")
        print(f"  Batch Size:      {c.effective_batch_size} (effective)")
        print(f"  Learning Rate:   {c.learning_rate}")
        print(f"  Precision:       {'BF16' if c.use_bf16 else 'FP16' if c.use_fp16 else 'FP32'}")
        print(f"  Device:          {self.device}")
        print(f"{'='*60}\n")
    
    def prepare_model(self, model: Optional[nn.Module] = None):
        """
        Prepare model for fine-tuning.
        
        Applies:
        1. Quantization (for QLoRA)
        2. LoRA adapters
        3. Gradient checkpointing
        4. NEFTune noise
        5. Flash attention (if available)
        """
        c = self.config
        
        if model is not None:
            self.model = model
        else:
            # Load model (placeholder - in reality would load from HF)
            logger.info(f"Loading model: {c.model_name}")
            # self.model = AutoModelForVision2Seq.from_pretrained(...)
            raise NotImplementedError(
                "Automatic model loading requires HuggingFace transformers. "
                "Please pass a pre-loaded model to prepare_model()."
            )
        
        # Apply LoRA
        self._apply_lora()
        
        # Apply gradient checkpointing
        if c.use_gradient_checkpointing:
            self._enable_gradient_checkpointing()
        
        # Apply NEFTune
        if c.use_neftune:
            self._apply_neftune()
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup mixed precision
        if c.use_bf16 or c.use_fp16:
            self.amp_dtype = torch.bfloat16 if c.use_bf16 else torch.float16
            if c.use_fp16:
                self.scaler = GradScaler()
        else:
            self.amp_dtype = torch.float32
            self.scaler = None
        
        # Count parameters
        total, trainable = self._count_parameters()
        print(f"  📊 Parameters: {trainable:,} trainable / {total:,} total "
              f"({100*trainable/total:.2f}%)")
        
        return self
    
    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        c = self.config
        
        # Import Fast LoRA
        from .fast_lora import FastLoRAConfig, apply_fast_lora
        
        lora_config = FastLoRAConfig(
            r=c.lora_r,
            lora_alpha=c.lora_alpha,
            lora_dropout=c.lora_dropout,
            target_modules=c.target_modules,
            use_rslora=c.use_rslora,
            use_dora=c.use_dora,
        )
        
        self.model = apply_fast_lora(self.model, lora_config, verbose=True)
    
    def _enable_gradient_checkpointing(self):
        """Enable gradient checkpointing."""
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
        else:
            from .memory_efficient import GradientCheckpointer, MemoryConfig
            
            mem_config = MemoryConfig(
                use_gradient_checkpointing=True,
                checkpoint_ratio=self.config.gradient_checkpointing_ratio,
            )
            checkpointer = GradientCheckpointer(mem_config)
            self.model = checkpointer.apply_to_model(self.model)
    
    def _apply_neftune(self):
        """Apply NEFTune noise to embeddings."""
        # Find embedding layer
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Embedding) and 'embed' in name.lower():
                # Wrap with NEFTune
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                if parent_name:
                    parent = self.model.get_submodule(parent_name)
                else:
                    parent = self.model
                
                neftune_embed = NEFTuneEmbedding(
                    module, 
                    noise_alpha=self.config.neftune_noise_alpha
                )
                setattr(parent, attr_name, neftune_embed)
                logger.info(f"Applied NEFTune to {name}")
                break
    
    def _setup_optimizer(self):
        """Setup optimizer with layer-wise learning rates."""
        c = self.config
        
        if c.use_layer_wise_lr:
            param_groups = self._get_layer_wise_params()
        else:
            param_groups = [
                {"params": [p for p in self.model.parameters() if p.requires_grad]}
            ]
        
        # Choose optimizer
        if c.use_paged_adamw:
            self.optimizer = PagedAdamW(
                param_groups,
                lr=c.learning_rate,
                betas=(c.adam_beta1, c.adam_beta2),
                eps=c.adam_epsilon,
                weight_decay=c.weight_decay,
                offload_to_cpu=True,
            )
        else:
            self.optimizer = AdamW(
                param_groups,
                lr=c.learning_rate,
                betas=(c.adam_beta1, c.adam_beta2),
                eps=c.adam_epsilon,
                weight_decay=c.weight_decay,
            )
    
    def _get_layer_wise_params(self) -> List[Dict[str, Any]]:
        """Get parameter groups with layer-wise learning rates."""
        c = self.config
        param_groups = []
        
        # Categorize parameters
        vision_params = []
        llm_layer_params = {}
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Vision encoder gets lower LR
            if 'vision' in name.lower() or 'visual' in name.lower():
                vision_params.append(param)
            # LLM layers
            elif any(f'layer.{i}.' in name or f'layers.{i}.' in name 
                    for i in range(100)):
                # Extract layer index
                for i in range(100):
                    if f'layer.{i}.' in name or f'layers.{i}.' in name:
                        if i not in llm_layer_params:
                            llm_layer_params[i] = []
                        llm_layer_params[i].append(param)
                        break
            else:
                other_params.append(param)
        
        # Vision encoder group (lower LR)
        if vision_params:
            param_groups.append({
                "params": vision_params,
                "lr": c.learning_rate * c.vision_lr_multiplier,
                "name": "vision_encoder",
            })
        
        # LLM layers with decaying LR
        if llm_layer_params:
            max_layer = max(llm_layer_params.keys())
            for layer_idx in sorted(llm_layer_params.keys()):
                decay = c.layer_lr_decay ** (max_layer - layer_idx)
                lr = c.learning_rate * decay
                
                param_groups.append({
                    "params": llm_layer_params[layer_idx],
                    "lr": lr,
                    "name": f"layer_{layer_idx}",
                })
        
        # Other params with base LR
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": c.learning_rate,
                "name": "other",
            })
        
        return param_groups
    
    def _count_parameters(self) -> Tuple[int, int]:
        """Count total and trainable parameters."""
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return total, trainable
    
    def train(
        self,
        train_dataset,
        eval_dataset=None,
        resume_from_checkpoint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            resume_from_checkpoint: Path to checkpoint to resume from
        
        Returns:
            Training metrics
        """
        c = self.config
        
        if self.model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")
        
        # Setup scheduler
        num_training_steps = self._get_num_training_steps(train_dataset)
        self._setup_scheduler(num_training_steps)
        
        # Create data loader
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=c.per_device_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        
        print(f"\n{'='*60}")
        print(f"  🚀 Starting Training")
        print(f"{'='*60}")
        print(f"  Training samples: {len(train_dataset)}")
        print(f"  Steps per epoch:  {len(train_loader) // c.gradient_accumulation_steps}")
        print(f"  Total steps:      {num_training_steps}")
        print(f"{'='*60}\n")
        
        # Training loop
        self.model.train()
        metrics = {"train_loss": [], "eval_loss": [], "learning_rate": []}
        
        accumulation_loss = 0.0
        early_stop = False
        start_time = time.time()
        
        for epoch in range(c.epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_loader):
                # Move to device
                batch = self._prepare_batch(batch)
                
                # Forward pass with AMP
                with autocast(enabled=self.amp_dtype != torch.float32, dtype=self.amp_dtype):
                    outputs = self._forward_step(batch)
                    loss = outputs["loss"] / c.gradient_accumulation_steps
                
                accumulation_loss += loss.item()
                
                # Backward pass
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # Update weights
                if (batch_idx + 1) % c.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if c.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), c.max_grad_norm
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    self.global_step += 1
                    epoch_loss += accumulation_loss
                    num_batches += 1
                    
                    # Logging
                    if self.global_step % c.logging_steps == 0:
                        lr = self.scheduler.get_last_lr()[0]
                        print(f"  Step {self.global_step} | "
                              f"Loss: {accumulation_loss:.4f} | "
                              f"LR: {lr:.2e}")
                        
                        metrics["train_loss"].append(accumulation_loss)
                        metrics["learning_rate"].append(lr)
                    
                    accumulation_loss = 0.0
                    
                    # Save checkpoint
                    if self.global_step % c.save_steps == 0:
                        self._save_checkpoint()
                    
                    # Evaluation
                    if eval_dataset and self.global_step % c.eval_steps == 0:
                        eval_loss = self._evaluate(eval_dataset)
                        metrics["eval_loss"].append(eval_loss)
                        
                        if c.early_stopping and self._check_early_stop(eval_loss):
                            print(f"  ⚠️ Early stopping at step {self.global_step}")
                            early_stop = True
                            break
                
                if early_stop or (c.max_steps > 0 and self.global_step >= c.max_steps):
                    break
            
            if early_stop or (c.max_steps > 0 and self.global_step >= c.max_steps):
                break
            
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            print(f"\n  Epoch {epoch + 1}/{c.epochs} | Avg Loss: {avg_epoch_loss:.4f}\n")
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"  ✅ Training Complete!")
        print(f"  Time: {total_time / 60:.2f} minutes")
        print(f"  Steps: {self.global_step}")
        print(f"{'='*60}\n")
        
        # Save final model
        self._save_checkpoint(final=True)
        
        return metrics
    
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare batch for training."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
    
    def _forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a forward step."""
        outputs = self.model(**batch)
        
        if hasattr(outputs, 'loss'):
            return {"loss": outputs.loss, "outputs": outputs}
        elif isinstance(outputs, dict) and 'loss' in outputs:
            return outputs
        elif isinstance(outputs, tuple):
            return {"loss": outputs[0], "outputs": outputs}
        else:
            raise ValueError("Model must return loss")
    
    def _get_num_training_steps(self, dataset) -> int:
        """Calculate total training steps."""
        c = self.config
        num_samples = len(dataset)
        steps_per_epoch = math.ceil(
            num_samples / c.per_device_batch_size / c.gradient_accumulation_steps
        )
        
        if c.max_steps > 0:
            return c.max_steps
        return steps_per_epoch * c.epochs
    
    def _setup_scheduler(self, num_training_steps: int):
        """Setup learning rate scheduler."""
        c = self.config
        
        warmup_steps = c.warmup_steps
        if warmup_steps == 0:
            warmup_steps = int(num_training_steps * c.warmup_ratio)
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        min_lr = c.learning_rate * c.min_lr_ratio
        
        if c.lr_scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps - warmup_steps,
                eta_min=min_lr,
            )
        else:
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=c.min_lr_ratio,
                total_iters=num_training_steps - warmup_steps,
            )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def _save_checkpoint(self, final: bool = False):
        """
        Save model checkpoint.
        
        Args:
            final: Whether this is the final model
        """
        if final:
            save_path = self.output_dir / "final_model"
        else:
            save_path = self.output_dir / f"checkpoint-{self.global_step}"
        
        save_path.mkdir(exist_ok=True)
        
        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(save_path)
        else:
            torch.save(self.model.state_dict(), save_path / "model.pt")
            
        # Save tokenizer
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_path)
            
        # Save config
        if self.config:
            with open(save_path / "config.json", "w") as f:
                # Basic dump of config
                import json
                # Handle non-serializable objects manually if needed
                config_dict = {
                    k: v.value if isinstance(v, Enum) else v 
                    for k, v in self.config.__dict__.items() 
                    if not k.startswith('_')
                }
                json.dump(config_dict, f, indent=2, default=str)
                
        print(f"  💾 Saved {'final model' if final else 'checkpoint'} to {save_path}")

    def _evaluate(self, eval_dataset) -> float:
        """
        Evaluate the model.
        
        Args:
            eval_dataset: Dataset to evaluate on
            
        Returns:
            Average loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        c = self.config
        
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=c.per_device_batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        with torch.no_grad():
            for batch in eval_loader:
                batch = self._prepare_batch(batch)
                
                with autocast(enabled=self.amp_dtype != torch.float32, dtype=self.amp_dtype):
                    outputs = self._forward_step(batch)
                    loss = outputs["loss"]
                
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def _check_early_stop(self, current_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            current_loss: Current evaluation loss
            
        Returns:
            True if training should stop
        """
        c = self.config
        
        if current_loss < self.best_eval_loss - c.early_stopping_threshold:
            self.best_eval_loss = current_loss
            self.patience_counter = 0
        else:
            if not hasattr(self, 'patience_counter'):
                self.patience_counter = 0
            self.patience_counter += 1
            
        return self.patience_counter >= c.early_stopping_patience
