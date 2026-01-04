import torch
import os

def is_tpu_available():
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False

def get_device(prefer_tpu=True):
    if prefer_tpu and is_tpu_available():
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def to_device(data, device: torch.device):
    """
    Recursively move tensors to the specified device.
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, (list, tuple)):
        return type(data)([to_device(item, device) for item in data]) # type: ignore
    elif isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    return data 