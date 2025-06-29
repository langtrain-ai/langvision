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