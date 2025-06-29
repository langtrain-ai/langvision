import argparse
import torch
from plimai.models.vision_transformer import VisionTransformer
from plimai.data.datasets import get_dataset
from plimai.training.trainer import Trainer
from plimai.callbacks.early_stopping import EarlyStopping
import os
from plimai.utils.device import get_device
from plimai.utils.cuda import setup_cuda


def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune VisionTransformer with LoRA')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step'])
    parser.add_argument('--step_size', type=int, default=5, help='StepLR step size')
    parser.add_argument('--gamma', type=float, default=0.5, help='StepLR gamma')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_name', type=str, default='vit_lora_best.pth')
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--cuda_deterministic', action='store_true', help='Enable deterministic CUDA (reproducible, slower)')
    parser.add_argument('--cuda_benchmark', action='store_true', default=True, help='Enable cudnn.benchmark for fast training (default: True)')
    parser.add_argument('--cuda_max_split_size_mb', type=int, default=None, help='Set CUDA max split size in MB (for large models, PyTorch >=1.10)')
    return parser.parse_args()

def set_seed(seed):
    import random, numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()
    device = get_device()
    setup_cuda(seed=args.seed if hasattr(args, 'seed') else 42,
               deterministic=args.cuda_deterministic,
               benchmark=args.cuda_benchmark,
               max_split_size_mb=args.cuda_max_split_size_mb)
    set_seed(args.seed)
    print(f"Using device: {device}")

    # Data
    train_dataset = get_dataset(args.dataset, args.data_dir, train=True, img_size=args.img_size)
    val_dataset = get_dataset(args.dataset, args.data_dir, train=False, img_size=args.img_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Model
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
    ).to(device)

    # Optimizer
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(lora_params, lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(lora_params, lr=args.lr)
    else:
        optimizer = torch.optim.SGD(lora_params, lr=args.lr, momentum=0.9)

    # Scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    criterion = torch.nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

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
        device=device,
    )

    # Optionally resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if args.eval_only:
        val_loss, val_acc = trainer.evaluate(val_loader)
        print(f"Eval Loss: {val_loss:.4f}, Eval Acc: {val_acc:.4f}")
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
    print(f"Best validation accuracy: {best_acc:.4f}")

if __name__ == '__main__':
    main() 