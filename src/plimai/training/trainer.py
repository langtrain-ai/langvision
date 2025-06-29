import torch
import os
import logging
from plimai.callbacks.base import Callback

logger = logging.getLogger("plimai.trainer")

class Trainer:
    """
    Modular Trainer for fine-tuning with support for:
    - Mixed precision (AMP)
    - Checkpointing
    - Callbacks
    - Distributed/multi-GPU (use torch.nn.parallel.DistributedDataParallel)
    - TPU (use torch_xla)
    """
    def __init__(self, model, optimizer, criterion, scheduler=None, scaler=None, callbacks=None, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.callbacks = callbacks or []
        self.device = device
        # GPU optimization
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
        # Ensure all callbacks inherit from Callback
        self.callbacks = [cb if isinstance(cb, Callback) else Callback() for cb in self.callbacks]

    def fit(self, train_loader, val_loader=None, epochs=10, start_epoch=0, best_acc=0.0, checkpoint_path=None):
        for cb in self.callbacks:
            cb.on_train_begin(self)
        for epoch in range(start_epoch, epochs):
            for cb in self.callbacks:
                cb.on_epoch_begin(self, epoch)
            self.model.train()
            total_loss, correct, total = 0, 0, 0
            for batch, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                self.optimizer.zero_grad()
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(imgs)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(imgs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                total_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += imgs.size(0)
                for cb in self.callbacks:
                    cb.on_batch_end(self, batch, logs={"loss": loss.item()})
            train_acc = correct / total
            train_loss = total_loss / total
            val_loss, val_acc = None, None
            if val_loader:
                val_loss, val_acc = self.evaluate(val_loader)
            if self.scheduler:
                self.scheduler.step()
            if checkpoint_path and (val_acc is not None and val_acc > best_acc):
                best_acc = val_acc
                self.save_checkpoint(epoch, best_acc, checkpoint_path)
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss}, Acc: {val_acc}")
            for cb in self.callbacks:
                cb.on_epoch_end(self, epoch, logs={"train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc})
        for cb in self.callbacks:
            cb.on_train_end(self)
        return best_acc

    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * imgs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += imgs.size(0)
        return total_loss / total, correct / total

    def save_checkpoint(self, epoch, best_acc, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
        }, path) 