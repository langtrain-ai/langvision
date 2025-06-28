import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from plimai.models.vision_transformer import VisionTransformer
from plimai.utils.config import default_config
from plimai.utils.data import get_preprocessing
import os
import argparse
import random
import numpy as np
from plimai.banner import print_banner
from plimai import __version__

# ----------------------
# Argument Parsing
# ----------------------
def get_args():
    parser = argparse.ArgumentParser(description='Fine-tune VisionTransformer with LoRA')
    # Dataset/model
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'], help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Dataset directory')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=768)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    # LoRA
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_alpha', type=float, default=1.0)
    parser.add_argument('--lora_dropout', type=float, default=0.1)
    # Training
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
    # Logging/output
    parser.add_argument('--log_file', type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--save_name', type=str, default='vit_lora_best.pth')
    parser.add_argument('--verbose', action='store_true')
    # Early stopping
    parser.add_argument('--early_stopping', action='store_true')
    parser.add_argument('--patience', type=int, default=5)
    # Eval only
    parser.add_argument('--eval_only', action='store_true')
    return parser.parse_args()

# ----------------------
# Seed Setting
# ----------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------
# Training and Validation
# ----------------------
def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += imgs.size(0)
    return total_loss / total, correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += imgs.size(0)
    return total_loss / total, correct / total

# ----------------------
# Main
# ----------------------
def main():
    print_banner(__version__)
    args = get_args()
    set_seed(args.seed)
    print(f"Using device: {args.device}")

    # Data
    transform = get_preprocessing(args.img_size)
    train_dataset = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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
    ).to(args.device)

    # Only LoRA parameters are trainable
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n and p.requires_grad]
    optimizer = optim.Adam(lora_params, lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler() if args.device.startswith('cuda') else None

    # Optionally resume
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, args.device, scaler)
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'best_acc': best_acc,
            }, args.save_name)
            print(f"Saved new best model to {args.save_name}")

    print(f"Best validation accuracy: {best_acc:.4f}")

    # Test set evaluation
    test_dataset = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    # Load best model
    if os.path.isfile(args.save_name):
        checkpoint = torch.load(args.save_name, map_location=args.device)
        model.load_state_dict(checkpoint['model'])
        print(f"Loaded best model from {args.save_name} for test evaluation.")
    test_loss, test_acc = validate(model, test_loader, criterion, args.device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main() 