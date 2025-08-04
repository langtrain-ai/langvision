"""
Complete CLI interface for Langvision framework.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from ..utils.setup import initialize_langvision, quick_setup
from ..models.lora import LoRAConfig
from ..training.advanced_trainer import AdvancedTrainer, TrainingConfig
from ..data.enhanced_datasets import create_enhanced_dataloaders, DatasetConfig
from ..models.vision_transformer import VisionTransformer
from ..models.resnet import resnet50
from ..models.multimodal import create_multimodal_model
from ..utils.device import get_device


def create_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser for Langvision CLI."""
    
    parser = argparse.ArgumentParser(
        prog="langvision",
        description="Langvision: Advanced Vision-Language Models with Efficient LoRA Fine-Tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic image classification training
  langvision train --model vit --dataset cifar10 --data-dir ./data --epochs 50
  
  # LoRA fine-tuning with custom parameters
  langvision train --model resnet50 --lora-r 16 --lora-alpha 32 --freeze-backbone
  
  # Multimodal training
  langvision train-multimodal --vision-model vit --text-model bert-base-uncased --data-dir ./data
  
  # Evaluation
  langvision evaluate --model-path ./checkpoints/best_model.pt --data-dir ./test_data
        """
    )
    
    parser.add_argument("--version", action="version", version="langvision 0.1.0")
    
    # Create subparsers
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a vision model")
    add_training_args(train_parser)
    
    # Multimodal training command
    multimodal_parser = subparsers.add_parser("train-multimodal", help="Train a multimodal vision-language model")
    add_multimodal_args(multimodal_parser)
    
    # Evaluation command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    add_evaluation_args(eval_parser)
    
    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Setup and validate Langvision environment")
    add_setup_args(setup_parser)
    
    return parser


def add_training_args(parser: argparse.ArgumentParser) -> None:
    """Add training-specific arguments."""
    
    # Model arguments
    model_group = parser.add_argument_group("Model Configuration")
    model_group.add_argument("--model", choices=["vit", "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
                           default="vit", help="Model architecture to use")
    model_group.add_argument("--num-classes", type=int, default=10, help="Number of output classes")
    model_group.add_argument("--img-size", type=int, default=224, help="Input image size")
    
    # LoRA arguments
    lora_group = parser.add_argument_group("LoRA Configuration")
    lora_group.add_argument("--use-lora", action="store_true", help="Enable LoRA fine-tuning")
    lora_group.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    lora_group.add_argument("--lora-alpha", type=float, default=32.0, help="LoRA alpha")
    lora_group.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    lora_group.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone parameters")
    
    # Data arguments
    data_group = parser.add_argument_group("Data Configuration")
    data_group.add_argument("--dataset", choices=["cifar10", "cifar100", "imagefolder"], 
                          default="cifar10", help="Dataset to use")
    data_group.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    data_group.add_argument("--batch-size", type=int, default=32, help="Batch size")
    data_group.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    data_group.add_argument("--augmentation", action="store_true", help="Enable data augmentation")
    data_group.add_argument("--augmentation-strength", type=float, default=0.5, help="Augmentation strength")
    
    # Training arguments
    train_group = parser.add_argument_group("Training Configuration")
    train_group.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_group.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_group.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    train_group.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw", help="Optimizer")
    train_group.add_argument("--scheduler", choices=["cosine", "step", "plateau"], default="cosine", help="LR scheduler")
    train_group.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    train_group.add_argument("--use-amp", action="store_true", help="Use mixed precision training")
    train_group.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping norm")
    
    # Output arguments
    output_group = parser.add_argument_group("Output Configuration")
    output_group.add_argument("--output-dir", type=str, default="./outputs", help="Output directory")
    output_group.add_argument("--experiment-name", type=str, default="langvision_experiment", help="Experiment name")
    output_group.add_argument("--save-interval", type=int, default=5, help="Save checkpoint every N epochs")
    output_group.add_argument("--log-interval", type=int, default=10, help="Log every N batches")
    
    # System arguments
    system_group = parser.add_argument_group("System Configuration")
    system_group.add_argument("--seed", type=int, default=42, help="Random seed")
    system_group.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda, mps)")
    system_group.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                            default="INFO", help="Logging level")


def add_multimodal_args(parser: argparse.ArgumentParser) -> None:
    """Add multimodal training-specific arguments."""
    
    # Inherit basic training args
    add_training_args(parser)
    
    # Multimodal-specific arguments
    multimodal_group = parser.add_argument_group("Multimodal Configuration")
    multimodal_group.add_argument("--vision-model", choices=["vit_base"], default="vit_base", 
                                 help="Vision model architecture")
    multimodal_group.add_argument("--text-model", type=str, default="bert-base-uncased", 
                                 help="Text model from HuggingFace")
    multimodal_group.add_argument("--vision-dim", type=int, default=768, help="Vision feature dimension")
    multimodal_group.add_argument("--text-dim", type=int, default=768, help="Text feature dimension")
    multimodal_group.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension for fusion")
    multimodal_group.add_argument("--max-text-length", type=int, default=77, help="Maximum text sequence length")
    multimodal_group.add_argument("--annotations-file", type=str, help="Path to text annotations file")
    multimodal_group.add_argument("--contrastive-weight", type=float, default=1.0, 
                                 help="Weight for contrastive loss")
    multimodal_group.add_argument("--classification-weight", type=float, default=0.5, 
                                 help="Weight for classification loss")


def add_evaluation_args(parser: argparse.ArgumentParser) -> None:
    """Add evaluation-specific arguments."""
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagefolder"], 
                       default="cifar10", help="Dataset type")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--output-dir", type=str, default="./eval_results", help="Output directory for results")
    parser.add_argument("--save-predictions", action="store_true", help="Save model predictions")
    parser.add_argument("--benchmark", action="store_true", help="Run inference speed benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")


def add_setup_args(parser: argparse.ArgumentParser) -> None:
    """Add setup-specific arguments."""
    
    parser.add_argument("--check-deps", action="store_true", help="Check all dependencies")
    parser.add_argument("--validate-env", action="store_true", help="Validate environment")
    parser.add_argument("--setup-cuda", action="store_true", help="Setup CUDA environment")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for testing")


def handle_train_command(args: argparse.Namespace) -> None:
    """Handle training command."""
    
    logger = logging.getLogger("langvision.cli")
    logger.info("Starting training...")
    
    # Initialize framework
    quick_setup(seed=args.seed, log_level=args.log_level)
    
    # Create LoRA config if requested
    lora_config = None
    if args.use_lora:
        lora_config = LoRAConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout
        )
        logger.info(f"Using LoRA with r={args.lora_r}, alpha={args.lora_alpha}")
    
    # Create model
    if args.model == "vit":
        model = VisionTransformer(
            img_size=args.img_size,
            num_classes=args.num_classes,
            lora_config=lora_config
        )
    elif args.model.startswith("resnet"):
        model_fn = {
            "resnet18": "resnet18", "resnet34": "resnet34", "resnet50": "resnet50",
            "resnet101": "resnet101", "resnet152": "resnet152"
        }[args.model]
        from ..models import resnet
        model = getattr(resnet, model_fn)(num_classes=args.num_classes, lora_config=lora_config)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    logger.info(f"Created {args.model} model with {args.num_classes} classes")
    
    # Create dataset config
    dataset_config = DatasetConfig(
        root_dir=args.data_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=args.augmentation,
        augmentation_strength=args.augmentation_strength
    )
    
    # Create dataloaders
    dataloaders = create_enhanced_dataloaders(dataset_config)
    logger.info(f"Created dataloaders for {args.dataset}")
    
    # Create training config
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        warmup_epochs=args.warmup_epochs,
        use_amp=args.use_amp,
        gradient_clip_norm=args.gradient_clip,
        output_dir=args.output_dir,
        experiment_name=args.experiment_name,
        save_interval=args.save_interval,
        log_interval=args.log_interval,
        lora_config=lora_config,
        freeze_backbone=args.freeze_backbone
    )
    
    # Create trainer
    trainer = AdvancedTrainer(
        model=model,
        train_loader=dataloaders['train'],
        val_loader=dataloaders.get('val'),
        config=training_config
    )
    
    # Start training
    trainer.train()
    logger.info("Training completed!")


def handle_setup_command(args: argparse.Namespace) -> None:
    """Handle setup command."""
    
    config = {
        "log_level": "INFO",
        "seed": args.seed
    }
    
    if args.setup_cuda:
        config["setup_cuda"] = True
    
    results = initialize_langvision(config)
    
    print("\n=== Langvision Environment Setup ===")
    print(f"Python: {results['validation_results']['python_version']}")
    print(f"PyTorch: {results['validation_results']['pytorch_version']}")
    
    if results['validation_results']['cuda_available']:
        print(f"CUDA: {results['validation_results']['cuda_version']}")
        print(f"GPUs: {results['validation_results']['gpu_count']}")
    else:
        print("CUDA: Not available")
    
    if args.check_deps:
        print("\n=== Dependencies ===")
        for dep, available in results['dependencies'].items():
            status = "✓" if available else "✗"
            print(f"{status} {dep}")
    
    print("\nSetup completed successfully!")


def main() -> None:
    """Main CLI entry point."""
    
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.command == "train":
            handle_train_command(args)
        elif args.command == "train-multimodal":
            # TODO: Implement multimodal training handler
            print("Multimodal training not yet implemented in CLI")
            sys.exit(1)
        elif args.command == "evaluate":
            # TODO: Implement evaluation handler
            print("Evaluation not yet implemented in CLI")
            sys.exit(1)
        elif args.command == "setup":
            handle_setup_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    
    except Exception as e:
        logger = logging.getLogger("langvision.cli")
        logger.error(f"Command failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
