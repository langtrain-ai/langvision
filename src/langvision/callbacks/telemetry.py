import requests
import logging
from typing import Dict, Any, Optional
from .base import Callback

logger = logging.getLogger(__name__)

class ServerTelemetryCallback(Callback):
    """
    Streams vision training metrics directly to langtrain-server via HTTP POST.
    """
    def __init__(self, api_url: str, job_id: str, log_every: int = 10, name: Optional[str] = None):
        super().__init__(name=name)
        self.api_url = api_url.rstrip('/')
        self.job_id = job_id
        self.log_every = log_every
        self.step = 0
        self.api_token = None # Optional: Set from env if needed for auth
        
    def _send_telemetry(self, payload: Dict[str, Any]):
        try:
            headers = {"Content-Type": "application/json"}
            if self.api_token:
                headers["Authorization"] = f"Bearer {self.api_token}"
                
            requests.post(
                f"{self.api_url}/v1/training/jobs/{self.job_id}/telemetry",
                json=payload,
                headers=headers,
                timeout=2.0
            )
        except Exception as e:
            self.logger.debug(f"Telemetry failed to send: {e}")

    def on_batch_end(self, trainer, batch_idx: int, batch: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        self.step += 1
        loss = outputs.get("loss", 0.0)
        
        # In pytorch-lightning or similar trainers, learning rate is often accessed via optimizer
        lr = None
        if hasattr(trainer, "optimizers"):
            opts = trainer.optimizers()
            if isinstance(opts, list) and len(opts) > 0:
                lr = opts[0].param_groups[0].get('lr')
        elif hasattr(trainer, "optimizer"):
            lr = trainer.optimizer.param_groups[0].get('lr')

        if self.step % self.log_every == 0:
            self._send_telemetry({
                "step": self.step,
                "loss": float(loss) if hasattr(loss, "item") else loss,
                "learning_rate": lr
            })

    def on_epoch_end(self, trainer, epoch: int, metrics: Dict[str, Any]) -> None:
        self._send_telemetry({
            "step": self.step,
            "loss": metrics.get("train_loss", metrics.get("loss", 0.0)),
            "epoch": epoch + 1,
            "log_message": f"Epoch {epoch + 1} completed"
        })

    def on_train_end(self, trainer) -> None:
        self._send_telemetry({
            "step": self.step,
            "loss": 0.0,
            "progress": 100,
            "log_message": "Training completed"
        })
