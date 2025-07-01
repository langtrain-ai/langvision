import argparse
import torch
import logging
import os
import random
import numpy as np
from plimai.models.vision_transformer import VisionTransformer
from plimai.data.datasets import get_dataset
from plimai.training.trainer import Trainer
from plimai.callbacks.early_stopping import EarlyStopping
from plimai.utils.device import get_device
from plimai.utils.cuda import setup_cuda


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for fine-tuning."""
    parser = argparse.ArgumentParser(description='Fine-tune VisionTransformer with LoRA')
    # Data
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224, help='Input image size')
    parser.add_argument('--patch_size', type=int, default=16, help='Patch size for ViT')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loader workers')
    # Model
    parser.add_argument('--embed_dim', type=int, default=768, help='Embedding dimension')
    parser.add_argument('--depth', type=int, default=12, help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=12, help='Number of attention heads')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='MLP ratio')
    parser.add_argument('--lora_r', type=int, default=4, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    # Training
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'], help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay (L2 regularization)')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'], help='LR scheduler')
    parser.add_argument('--step_size', type=int, default=5, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation')
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save outputs and checkpoints')
    parser.add_argument('--save_name', type=str, default='vit_lora_best.pth', help='Checkpoint file name')
    # Callbacks
    parser.add_argument('--early_stopping', action='store_true', help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    # CUDA
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--cuda_deterministic', action='store_true', help='Enable deterministic CUDA (reproducible, slower)')
    parser.add_argument('--cuda_benchmark', action='store_true', default=True, help='Enable cudnn.benchmark for fast training (default: True)')
    parser.add_argument('--cuda_max_split_size_mb', type=int, default=None, help='Set CUDA max split size in MB (for large models, PyTorch >=1.10)')
    # Misc
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--log_level', type=str, default='info', help='Logging level (debug, info, warning, error)')
    return parser.parse_args()


def setup_logging(log_level: str) -> None:
    """Set up logging with the specified log level."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        numeric_level = logging.INFO
    logging.basicConfig(level=numeric_level, format='[%(levelname)s] %(message)s')


def main() -> None:
    """Main function for fine-tuning VisionTransformer with LoRA."""
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {args.device}")

    setup_cuda(seed=args.seed, deterministic=args.cuda_deterministic, benchmark=args.cuda_benchmark, max_split_size_mb=args.cuda_max_split_size_mb)
    set_seed(args.seed)

    # Data
    try:
        train_dataset = get_dataset(args.dataset, args.data_dir, train=True, img_size=args.img_size)
        val_dataset = get_dataset(args.dataset, args.data_dir, train=False, img_size=args.img_size)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
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
            lora_config={
                'r': args.lora_r,
                'alpha': args.lora_alpha,
                'dropout': args.lora_dropout,
            },
        ).to(args.device)
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return

    # Optimizer
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(lora_params, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(lora_params, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if args.device == 'cuda' else None

    # Callbacks
    callbacks = []
    if args.early_stopping:
        callbacks.append(EarlyStopping(patience=args.patience))

    # Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        scaler=scaler,
        callbacks=callbacks,
        device=args.device,
    )

    # Optionally resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        logger.info(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        val_loss, val_acc = trainer.evaluate(val_loader)
        logger.info(f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_acc:.4f}")
        return

    # Training
    best_acc = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        start_epoch=start_epoch,
        best_acc=best_acc,
        checkpoint_path=os.path.join(args.output_dir, args.save_name),
    )
    logger.info(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main() 