import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import time

from ..callbacks.base import Callback
from ..utils.device import get_device, to_device

logger = logging.getLogger("langvision.trainer")

class Trainer:
    """
    Modular Trainer for fine-tuning with support for:
    - Mixed precision (AMP) 
    - Checkpointing
    - Callbacks
    - Distributed/multi-GPU (use torch.nn.parallel.DistributedDataParallel)
    - TPU (use torch_xla)

    Integration points for advanced LLM concepts:
    - RLHF: Use RLHF-based feedback in the training loop
    - PPO/DPO/GRPO/RLVR: Use RL-based optimization for policy/model updates
    - LIME/SHAP: Use for model interpretability during/after training
    """
    def __init__(self, model, optimizer, criterion, scheduler=None, scaler=None, callbacks=None, device='cpu', rlhf=None, ppo=None, dpo=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.callbacks = callbacks or []
        self.device = device
        self.rlhf = rlhf  # RLHF integration (optional)
        self.ppo = ppo    # PPO integration (optional)
        self.dpo = dpo    # DPO integration (optional)
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
                # RLHF integration example (stub)
                if self.rlhf is not None:
                    # TODO: Use self.rlhf.train(data, feedback) for RLHF-based updates
                    pass
                # PPO integration example (stub)
                if self.ppo is not None:
                    # TODO: Use self.ppo.step(state, action, reward) for PPO-based updates
                    pass
                # DPO integration example (stub)
                if self.dpo is not None:
                    # TODO: Use self.dpo.optimize_with_preferences(preferences) for DPO-based updates
                    pass
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