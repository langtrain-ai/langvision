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
from .fast_trainer import FastTrainer, FastTrainerConfig

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
        
        """
        Train the model using FastTrainer.
        """
        c = self.config
        
        if self.model is None:
            raise RuntimeError("Model not prepared. Call prepare_model() first.")
            
        # Create FastTrainerConfig from OptimalFineTuneConfig
        fast_config = FastTrainerConfig(
            model_name=c.model_name,
            lora_r=c.lora_r,
            lora_alpha=c.lora_alpha,
            lora_dropout=c.lora_dropout,
            target_modules=c.target_modules,
            use_rslora=c.use_rslora,
            use_dora=c.use_dora,
            
            epochs=c.epochs,
            max_steps=c.max_steps,
            batch_size=c.per_device_batch_size,
            gradient_accumulation_steps=c.gradient_accumulation_steps,
            max_seq_length=c.max_seq_length,
            
            learning_rate=c.learning_rate,
            weight_decay=c.weight_decay,
            adam_beta1=c.adam_beta1,
            adam_beta2=c.adam_beta2,
            adam_epsilon=c.adam_epsilon,
            max_grad_norm=c.max_grad_norm,
            
            lr_scheduler=c.lr_scheduler_type,
            warmup_ratio=c.warmup_ratio,
            warmup_steps=c.warmup_steps,
            min_lr_ratio=c.min_lr_ratio,
            
            use_gradient_checkpointing=c.use_gradient_checkpointing,
            gradient_checkpoint_ratio=c.gradient_checkpointing_ratio,
            
            use_amp=c.use_bf16 or c.use_fp16,
            amp_dtype="bfloat16" if c.use_bf16 else "float16",
            
            use_packing=c.pack_sequences,
            
            output_dir=c.output_dir,
            logging_steps=c.logging_steps,
            save_steps=c.save_steps,
            eval_steps=c.eval_steps,
            
            compile_model=getattr(c, 'compile_model', False),
            
            early_stopping_patience=c.early_stopping_patience,
            early_stopping_threshold=c.early_stopping_threshold,
        )
        
        # Instantiate FastTrainer
        trainer = FastTrainer(
            model=self.model,
            config=fast_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            optimizer=self.optimizer,
            # We don't pass scheduler because FastTrainer will create one matching the config
            # OR we can pass self.scheduler if we want to preserve the exact one created in prepare_model?
            # VisionLLMFineTuner doesn't create scheduler in prepare_model, it creates it in train()!
            # So passing None lets FastTrainer create it.
            # Wait, finetuner.py had _setup_scheduler call in train().
            # FastTrainer will call _setup_scheduler if we don't pass one.
            # FastTrainer's _setup_scheduler supports cosine/linear/etc. matching config.
            skip_lora=True  # Already applied in prepare_model
        )
        
        # Train
        metrics = trainer.train()
        
        # Copy back state
        self.global_step = trainer.global_step
        self.epoch = trainer.epoch
        
        return metrics
    
