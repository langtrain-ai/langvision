#!/usr/bin/env python3
"""
Export CLI for Langvision models.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.onnx

# Import from langvision modules
try:
    from langvision.models.vision_transformer import VisionTransformer
except ImportError as e:
    print(f"❌ Error importing langvision modules: {e}")
    print("Please ensure langvision is properly installed:")
    print("  pip install langvision")
    sys.exit(1)


def parse_args():
    """Parse command-line arguments for model export."""
    parser = argparse.ArgumentParser(
        description='Export a trained VisionTransformer model to various formats',
        epilog='''\nExamples:\n  langvision export --checkpoint model.pth --format onnx --output model.onnx\n  langvision export --checkpoint model.pth --format torchscript --output model.pt\n''',
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
    
    # Export
    export_group = parser.add_argument_group('Export')
    export_group.add_argument('--format', type=str, required=True, choices=['onnx', 'torchscript', 'state_dict'], help='Export format')
    export_group.add_argument('--output', type=str, required=True, help='Output file path')
    export_group.add_argument('--batch_size', type=int, default=1, help='Batch size for export (ONNX/TorchScript)')
    export_group.add_argument('--opset_version', type=int, default=11, help='ONNX opset version')
    
    # Device
    device_group = parser.add_argument_group('Device')
    device_group.add_argument('--device', type=str, default='cpu', help='Device to use for export (use CPU for ONNX)')
    
    # Misc
    misc_group = parser.add_argument_group('Misc')
    misc_group.add_argument('--log_level', type=str, default='info', help='Logging level')
    
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')


def main():
    """Main function for model export."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting model export...")
    
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
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Export model
    logger.info(f"📦 Exporting model to {args.format.upper()} format...")
    try:
        if args.format == 'onnx':
            # Create dummy input
            dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(args.device)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                args.output,
                export_params=True,
                opset_version=args.opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            logger.info(f"✅ ONNX model exported to {args.output}")
            
        elif args.format == 'torchscript':
            # Create dummy input
            dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(args.device)
            
            # Export to TorchScript
            traced_model = torch.jit.trace(model, dummy_input)
            traced_model.save(args.output)
            logger.info(f"✅ TorchScript model exported to {args.output}")
            
        elif args.format == 'state_dict':
            # Export state dict
            torch.save(model.state_dict(), args.output)
            logger.info(f"✅ State dict exported to {args.output}")
            
    except Exception as e:
        logger.error(f"❌ Export failed: {e}")
        return 1
    
    # Verify export
    if args.format in ['onnx', 'torchscript']:
        logger.info("🔍 Verifying exported model...")
        try:
            if args.format == 'onnx':
                import onnx
                onnx_model = onnx.load(args.output)
                onnx.checker.check_model(onnx_model)
                logger.info("✅ ONNX model verification passed")
                
            elif args.format == 'torchscript':
                loaded_model = torch.jit.load(args.output)
                logger.info("✅ TorchScript model verification passed")
                
        except Exception as e:
            logger.warning(f"⚠️ Model verification failed: {e}")
    
    logger.info("✅ Model export completed successfully!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
