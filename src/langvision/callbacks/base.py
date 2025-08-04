"""Enhanced callback system for training and evaluation hooks."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import traceback


class Callback(ABC):
    """Base class for all callbacks with comprehensive training hooks."""
    
    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"langvision.callbacks.{self.name}")
    
    def on_train_start(self, trainer) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, trainer) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, trainer, epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, trainer, batch_idx: int, batch: Dict[str, Any]) -> None:
        """Called at the beginning of each batch."""
        pass
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """Called at the end of each batch."""
        pass
    
    def on_validation_start(self, trainer) -> None:
        """Called at the beginning of validation."""
        pass
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """Called at the end of validation."""
        pass
    
    def on_test_start(self, trainer) -> None:
        """Called at the beginning of testing."""
        pass
    
    def on_test_end(self, trainer, metrics: Dict[str, Any]) -> None:
        """Called at the end of testing."""
        pass
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str, metrics: Dict[str, Any]) -> None:
        """Called when a checkpoint is saved."""
        pass
    
    def on_checkpoint_load(self, trainer, checkpoint_path: str) -> None:
        """Called when a checkpoint is loaded."""
        pass
    
    def on_lr_schedule(self, trainer, old_lr: float, new_lr: float) -> None:
        """Called when learning rate is scheduled."""
        pass
    
    def on_exception(self, trainer, exception: Exception) -> bool:
        """Called when an exception occurs during training.
        
        Returns:
            bool: True if the exception was handled and training should continue,
                  False if training should stop.
        """
        return False


class CallbackManager:
    """Manager for handling multiple callbacks with error handling."""
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
        self.logger = logging.getLogger("langvision.callbacks.manager")
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the manager."""
        if not isinstance(callback, Callback):
            raise TypeError(f"Expected Callback instance, got {type(callback)}")
        self.callbacks.append(callback)
        self.logger.info(f"Added callback: {callback.name}")
    
    def remove_callback(self, callback_name: str) -> bool:
        """Remove a callback by name."""
        for i, callback in enumerate(self.callbacks):
            if callback.name == callback_name:
                del self.callbacks[i]
                self.logger.info(f"Removed callback: {callback_name}")
                return True
        return False
    
    def _call_callbacks(self, method_name: str, *args, **kwargs) -> None:
        """Safely call a method on all callbacks."""
        for callback in self.callbacks:
            try:
                method = getattr(callback, method_name, None)
                if method and callable(method):
                    method(*args, **kwargs)
            except Exception as e:
                self.logger.error(
                    f"Error in callback {callback.name}.{method_name}: {str(e)}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                # Continue with other callbacks even if one fails
    
    def on_train_start(self, trainer) -> None:
        self._call_callbacks('on_train_start', trainer)
    
    def on_train_end(self, trainer) -> None:
        self._call_callbacks('on_train_end', trainer)
    
    def on_epoch_start(self, trainer, epoch: int) -> None:
        self._call_callbacks('on_epoch_start', trainer, epoch)
    
    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        self._call_callbacks('on_epoch_end', trainer, epoch, metrics)
    
    def on_batch_start(self, trainer, batch_idx: int, batch: Dict[str, Any]) -> None:
        self._call_callbacks('on_batch_start', trainer, batch_idx, batch)
    
    def on_batch_end(self, trainer, batch_idx: int, batch: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self._call_callbacks('on_batch_end', trainer, batch_idx, batch, outputs)
    
    def on_validation_start(self, trainer) -> None:
        self._call_callbacks('on_validation_start', trainer)
    
    def on_validation_end(self, trainer, metrics: Dict[str, Any]) -> None:
        self._call_callbacks('on_validation_end', trainer, metrics)
    
    def on_test_start(self, trainer) -> None:
        self._call_callbacks('on_test_start', trainer)
    
    def on_test_end(self, trainer, metrics: Dict[str, Any]) -> None:
        self._call_callbacks('on_test_end', trainer, metrics)
    
    def on_checkpoint_save(self, trainer, checkpoint_path: str, metrics: Dict[str, Any]) -> None:
        self._call_callbacks('on_checkpoint_save', trainer, checkpoint_path, metrics)
    
    def on_checkpoint_load(self, trainer, checkpoint_path: str) -> None:
        self._call_callbacks('on_checkpoint_load', trainer, checkpoint_path)
    
    def on_lr_schedule(self, trainer, old_lr: float, new_lr: float) -> None:
        self._call_callbacks('on_lr_schedule', trainer, old_lr, new_lr)
    
    def on_exception(self, trainer, exception: Exception) -> bool:
        """Handle exceptions through callbacks.
        
        Returns:
            bool: True if any callback handled the exception and training should continue.
        """
        handled = False
        for callback in self.callbacks:
            try:
                if callback.on_exception(trainer, exception):
                    handled = True
                    self.logger.info(f"Exception handled by callback: {callback.name}")
            except Exception as callback_error:
                self.logger.error(
                    f"Error in callback {callback.name}.on_exception: {str(callback_error)}"
                )
        return handled 