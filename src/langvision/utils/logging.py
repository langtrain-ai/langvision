"""
Training Logger and Monitoring

Comprehensive logging utilities for training Vision LLMs:
- TensorBoard integration
- Weights & Biases support
- Console progress bars
- Metric tracking and visualization
"""

import torch
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import sys
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class LoggerConfig:
    """Configuration for training logger."""
    log_dir: str = "./logs"
    project_name: str = "langvision"
    run_name: Optional[str] = None
    
    # Backends
    use_tensorboard: bool = True
    use_wandb: bool = False
    use_console: bool = True
    
    # Logging frequency
    log_steps: int = 10
    
    # What to log
    log_gradients: bool = False
    log_parameters: bool = False
    log_images: bool = True


class MetricTracker:
    """
    Track and aggregate metrics during training.
    
    Features:
    - Running averages
    - Min/max tracking
    - Metric history
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.running_sums: Dict[str, float] = {}
        self.running_counts: Dict[str, int] = {}
    
    def update(self, name: str, value: float, n: int = 1):
        """Update a metric with new value."""
        if name not in self.metrics:
            self.metrics[name] = []
            self.running_sums[name] = 0.0
            self.running_counts[name] = 0
        
        self.metrics[name].append(value)
        self.running_sums[name] += value * n
        self.running_counts[name] += n
    
    def get_average(self, name: str) -> float:
        """Get running average of a metric."""
        if name not in self.running_counts or self.running_counts[name] == 0:
            return 0.0
        return self.running_sums[name] / self.running_counts[name]
    
    def get_last(self, name: str) -> float:
        """Get last value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return self.metrics[name][-1]
    
    def get_min(self, name: str) -> float:
        """Get minimum value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return float('inf')
        return min(self.metrics[name])
    
    def get_max(self, name: str) -> float:
        """Get maximum value of a metric."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return float('-inf')
        return max(self.metrics[name])
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.running_sums = {}
        self.running_counts = {}
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics."""
        summary = {}
        for name in self.metrics:
            summary[name] = {
                "last": self.get_last(name),
                "avg": self.get_average(name),
                "min": self.get_min(name),
                "max": self.get_max(name),
            }
        return summary


class ProgressBar:
    """
    Console progress bar for training.
    """
    
    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 40,
        show_eta: bool = True,
    ):
        self.total = total
        self.desc = desc
        self.width = width
        self.show_eta = show_eta
        
        self.current = 0
        self.start_time = time.time()
        self.metrics: Dict[str, float] = {}
    
    def update(self, n: int = 1, **metrics):
        """Update progress bar."""
        self.current += n
        self.metrics.update(metrics)
        self._render()
    
    def _render(self):
        """Render progress bar to console."""
        percent = self.current / self.total
        filled = int(self.width * percent)
        bar = "█" * filled + "░" * (self.width - filled)
        
        # Calculate ETA
        elapsed = time.time() - self.start_time
        if self.current > 0 and self.show_eta:
            eta = elapsed * (self.total - self.current) / self.current
            eta_str = self._format_time(eta)
        else:
            eta_str = "--:--"
        
        # Build metric string
        metric_str = " | ".join(f"{k}: {v:.4f}" for k, v in self.metrics.items())
        
        # Print
        line = f"\r{self.desc} |{bar}| {self.current}/{self.total} [{eta_str}]"
        if metric_str:
            line += f" | {metric_str}"
        
        sys.stdout.write(line)
        sys.stdout.flush()
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS."""
        if seconds < 3600:
            return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def close(self):
        """Close progress bar."""
        self._render()
        print()


class TrainingLogger:
    """
    Unified training logger with multiple backends.
    
    Supports:
    - TensorBoard
    - Weights & Biases
    - Console output
    - JSON logs
    """
    
    def __init__(self, config: Optional[LoggerConfig] = None):
        self.config = config or LoggerConfig()
        
        # Generate run name if not provided
        if self.config.run_name is None:
            self.config.run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup log directory
        self.log_dir = Path(self.config.log_dir) / self.config.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        self.tensorboard_writer = None
        self.wandb_run = None
        
        if self.config.use_tensorboard:
            self._setup_tensorboard()
        
        if self.config.use_wandb:
            self._setup_wandb()
        
        # Metric tracker
        self.metrics = MetricTracker()
        
        # Step counter
        self.global_step = 0
        
        # JSON log file
        self.json_log_path = self.log_dir / "metrics.jsonl"
    
    def _setup_tensorboard(self):
        """Setup TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(str(self.log_dir / "tensorboard"))
            logger.info(f"TensorBoard logging to {self.log_dir / 'tensorboard'}")
        except ImportError:
            logger.warning("TensorBoard not installed. Install with: pip install tensorboard")
    
    def _setup_wandb(self):
        """Setup Weights & Biases."""
        try:
            import wandb
            self.wandb_run = wandb.init(
                project=self.config.project_name,
                name=self.config.run_name,
                dir=str(self.log_dir),
            )
            logger.info(f"W&B run: {wandb.run.url}")
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
    
    def log(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = "",
    ):
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric names and values
            step: Global step (uses internal counter if not provided)
            prefix: Prefix for metric names
        """
        step = step if step is not None else self.global_step
        
        # Add prefix
        if prefix:
            metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        # Update tracker
        for name, value in metrics.items():
            self.metrics.update(name, value)
        
        # Log to TensorBoard
        if self.tensorboard_writer is not None:
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(name, value, step)
        
        # Log to W&B
        if self.wandb_run is not None:
            import wandb
            wandb.log(metrics, step=step)
        
        # Log to JSON
        self._log_json(metrics, step)
        
        # Console output
        if self.config.use_console:
            self._log_console(metrics, step)
        
        self.global_step = step
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: Optional[int] = None,
    ):
        """Log an image."""
        step = step if step is not None else self.global_step
        
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_image(tag, image, step)
        
        if self.wandb_run is not None:
            import wandb
            wandb.log({tag: wandb.Image(image)}, step=step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: Optional[int] = None,
    ):
        """Log a histogram."""
        step = step if step is not None else self.global_step
        
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_histogram(tag, values, step)
    
    def log_model_gradients(self, model: torch.nn.Module, step: Optional[int] = None):
        """Log model gradient statistics."""
        if not self.config.log_gradients:
            return
        
        step = step if step is not None else self.global_step
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                self.log({
                    f"gradients/{name}/mean": grad.mean().item(),
                    f"gradients/{name}/std": grad.std().item(),
                    f"gradients/{name}/norm": grad.norm().item(),
                }, step=step)
    
    def _log_json(self, metrics: Dict[str, float], step: int):
        """Log metrics to JSON file."""
        entry = {"step": step, "timestamp": time.time(), **metrics}
        
        with open(self.json_log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def _log_console(self, metrics: Dict[str, float], step: int):
        """Log metrics to console."""
        parts = [f"Step {step}"]
        for name, value in metrics.items():
            # Shorten name for console
            short_name = name.split("/")[-1]
            parts.append(f"{short_name}: {value:.4f}")
        
        print(" | ".join(parts))
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get metric summary."""
        return self.metrics.get_summary()
    
    def close(self):
        """Close all logging backends."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
        
        # Save final summary
        summary = self.get_summary()
        with open(self.log_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)


def create_logger(
    log_dir: str = "./logs",
    use_tensorboard: bool = True,
    use_wandb: bool = False,
    **kwargs,
) -> TrainingLogger:
    """Create a training logger with specified configuration."""
    config = LoggerConfig(
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        **kwargs,
    )
    return TrainingLogger(config)
