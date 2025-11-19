#!/usr/bin/env python3
"""
Evaluation CLI for Langvision models.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Import from langvision modules
try:
    from langvision.models.vision_transformer import VisionTransformer
    from langvision.utils.data import get_dataset
    from langvision.utils.device import setup_cuda, set_seed
    from langvision.training.trainer import Trainer
except ImportError as e:
    print(f"❌ Error importing langvision modules: {e}")
    print("Please ensure langvision is properly installed:")
    print("  pip install langvision")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained VisionTransformer model',
        epilog='''\nExamples:\n  langvision evaluate --checkpoint model.pth --dataset cifar10\n  langvision evaluate --checkpoint model.pth --dataset cifar100 --batch_size 128\n''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    model_group.add_argument('--img_size', type=int, default=224, help='Input image size')
    model_group.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT')
    model_group.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    model_group.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    model_group.add_argument('--depth', type=int, default=12, help='Number of transformer layers')
    model_group.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    model_group.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP hidden dim ratio')
    
    # Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to evaluate on')
    data_group.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    data_group.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    data_group.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output_dir', type=str, default='./evaluation_results', help='Directory to save evaluation results')
    output_group.add_argument('--save_predictions', action='store_true', help='Save model predictions to file')
    output_group.add_argument('--save_confusion_matrix', action='store_true', help='Save confusion matrix plot')
    
    # Device
    device_group = parser.add_argument_group('Device')
    device_group.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    
    # Misc
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    misc_group.add_argument('--log_level', type=str, default='info', help='Logging level')
    
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(
        level=numeric_level, 
        format='[%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )


def main():
    """Main function for model evaluation."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting model evaluation...")
    
    # Setup
    setup_cuda(seed=args.seed)
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    logger.info(f"📊 Loading {args.dataset} dataset...")
    try:
        test_dataset = get_dataset(args.dataset, args.data_dir, train=False, img_size=args.img_size)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=args.num_workers
        )
        logger.info(f"✅ Dataset loaded: {len(test_dataset)} test samples")
    except Exception as e:
        logger.error(f"❌ Failed to load dataset: {e}")
        return 1
    
    # Load model
    logger.info("🤖 Loading model...")
    try:
        model = VisionTransformer(
            img_size=args.img_size,
            patch_size=args.patch_size,
            in_chans=3,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
        ).to(args.device)
        
        # Load checkpoint
        if not os.path.isfile(args.checkpoint):
            logger.error(f"❌ Checkpoint file not found: {args.checkpoint}")
            return 1
            
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        logger.info("✅ Model loaded successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return 1
    
    # Evaluate
    logger.info("📈 Running evaluation...")
    try:
        trainer = Trainer(model, device=args.device)
        test_loss, test_acc = trainer.evaluate(test_loader)
        
        logger.info(f"🎯 Evaluation Results:")
        logger.info(f"   Test Loss: {test_loss:.4f}")
        logger.info(f"   Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'dataset': args.dataset,
            'checkpoint': args.checkpoint,
            'model_config': {
                'img_size': args.img_size,
                'patch_size': args.patch_size,
                'num_classes': args.num_classes,
                'embed_dim': args.embed_dim,
                'depth': args.depth,
                'num_heads': args.num_heads,
                'mlp_ratio': args.mlp_ratio,
            }
        }
        
        import json
        results_file = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"💾 Results saved to {results_file}")
        
        if args.save_predictions:
            logger.info("💾 Saving predictions...")
            # TODO: Implement prediction saving
            pass
            
        if args.save_confusion_matrix:
            logger.info("📊 Saving confusion matrix...")
            # TODO: Implement confusion matrix
            pass
            
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return 1
    
    logger.info("✅ Evaluation completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
