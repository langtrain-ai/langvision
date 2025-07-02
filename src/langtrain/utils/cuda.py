import torch
import os

def setup_cuda(seed=None, deterministic=False, benchmark=True, max_split_size_mb=None):
    """
    Set up CUDA for best performance and reproducibility.
    Args:
        seed (int): Random seed for reproducibility.
        deterministic (bool): If True, sets deterministic mode (slower, but reproducible).
        benchmark (bool): If True, enables cudnn.benchmark for faster training.
        max_split_size_mb (int): If set, limits max split size for CUDA memory allocator (PyTorch >=1.10).
    """
    if torch.cuda.is_available():
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = benchmark
        if max_split_size_mb is not None and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(max_split_size_mb / 1024)
        # Optionally, set max split size for allocator (PyTorch >=1.10)
        if max_split_size_mb is not None and hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(max_split_size_mb / 1024)
        # Optionally, set environment variable for reproducibility
        if deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 