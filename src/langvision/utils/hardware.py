"""
Hardware Detection and Optimization for Vision LLM Training

Provides comprehensive detection and optimization for:
- NVIDIA GPUs (CUDA) with compute capability detection
- Google TPUs (via torch_xla)
- Apple Silicon (MPS)
- AMD GPUs (ROCm)
- CPU fallback with optimizations

Automatically configures optimal settings based on detected hardware.
"""

import os
import sys
import platform
import subprocess
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


class AcceleratorType(Enum):
    """Types of supported accelerators."""
    NVIDIA_CUDA = "nvidia_cuda"
    GOOGLE_TPU = "google_tpu"
    APPLE_MPS = "apple_mps"
    AMD_ROCM = "amd_rocm"
    CPU = "cpu"
    UNKNOWN = "unknown"


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    index: int
    name: str
    memory_total_gb: float
    memory_free_gb: float
    compute_capability: Optional[Tuple[int, int]] = None
    driver_version: Optional[str] = None
    cuda_version: Optional[str] = None
    
    def __str__(self) -> str:
        return f"GPU {self.index}: {self.name} ({self.memory_total_gb:.1f}GB)"


@dataclass
class TPUInfo:
    """Information about detected TPU."""
    tpu_type: str  # v2, v3, v4, v5
    num_cores: int
    topology: str  # e.g., "2x2" for v3-8
    zone: Optional[str] = None
    
    def __str__(self) -> str:
        return f"TPU {self.tpu_type} ({self.num_cores} cores, {self.topology})"


@dataclass
class HardwareConfig:
    """Complete hardware configuration."""
    accelerator_type: AcceleratorType
    device_count: int
    total_memory_gb: float
    
    # Device-specific info
    gpus: List[GPUInfo] = field(default_factory=list)
    tpu: Optional[TPUInfo] = None
    
    # System info
    cpu_cores: int = 0
    system_memory_gb: float = 0
    platform: str = ""
    
    # Recommended settings
    recommended_dtype: str = "float32"
    recommended_batch_size: int = 1
    supports_flash_attention: bool = False
    supports_bf16: bool = False
    supports_fp16: bool = False
    
    def __str__(self) -> str:
        return (
            f"Hardware: {self.accelerator_type.value}\n"
            f"Devices: {self.device_count}\n"
            f"Memory: {self.total_memory_gb:.1f}GB\n"
            f"Recommended dtype: {self.recommended_dtype}"
        )


class HardwareDetector:
    """
    Detect available hardware accelerators and configure optimal settings.
    
    Usage:
        detector = HardwareDetector()
        config = detector.detect()
        print(config)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._torch_available = False
        self._cuda_available = False
        self._tpu_available = False
        self._mps_available = False
        
        self._check_torch()
    
    def _check_torch(self):
        """Check PyTorch availability and capabilities."""
        try:
            import torch
            self._torch_available = True
            
            # Check CUDA
            self._cuda_available = torch.cuda.is_available()
            
            # Check MPS (Apple Silicon)
            self._mps_available = (
                hasattr(torch.backends, 'mps') and 
                torch.backends.mps.is_available()
            )
            
        except ImportError:
            self._torch_available = False
    
    def _check_tpu(self) -> bool:
        """Check if running on Google TPU."""
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
            
            # Try to get TPU device
            device = xm.xla_device()
            self._tpu_available = True
            return True
            
        except (ImportError, RuntimeError):
            self._tpu_available = False
            return False
    
    def detect(self) -> HardwareConfig:
        """
        Detect all available hardware and return configuration.
        
        Priority order:
        1. Google TPU
        2. NVIDIA CUDA
        3. Apple MPS
        4. AMD ROCm
        5. CPU
        """
        # Get system info
        cpu_cores = os.cpu_count() or 1
        system_memory = self._get_system_memory()
        plat = f"{platform.system()} {platform.machine()}"
        
        # Check TPU first (highest priority for cloud training)
        if self._check_tpu():
            return self._configure_tpu(cpu_cores, system_memory, plat)
        
        # Check NVIDIA CUDA
        if self._cuda_available:
            return self._configure_nvidia(cpu_cores, system_memory, plat)
        
        # Check Apple MPS
        if self._mps_available:
            return self._configure_mps(cpu_cores, system_memory, plat)
        
        # Check AMD ROCm
        if self._check_rocm():
            return self._configure_rocm(cpu_cores, system_memory, plat)
        
        # Fallback to CPU
        return self._configure_cpu(cpu_cores, system_memory, plat)
    
    def _get_system_memory(self) -> float:
        """Get total system memory in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            # Fallback for systems without psutil
            if platform.system() == "Darwin":
                try:
                    output = subprocess.check_output(["sysctl", "-n", "hw.memsize"])
                    return int(output.strip()) / (1024 ** 3)
                except:
                    pass
            elif platform.system() == "Linux":
                try:
                    with open("/proc/meminfo") as f:
                        for line in f:
                            if "MemTotal" in line:
                                return int(line.split()[1]) / (1024 ** 2)
                except:
                    pass
            return 8.0  # Default assumption
    
    def _check_rocm(self) -> bool:
        """Check for AMD ROCm support."""
        try:
            import torch
            return torch.version.hip is not None
        except:
            return False
    
    def _configure_nvidia(
        self,
        cpu_cores: int,
        system_memory: float,
        plat: str,
    ) -> HardwareConfig:
        """Configure for NVIDIA CUDA GPUs."""
        import torch
        
        gpus = []
        total_memory = 0.0
        supports_bf16 = False
        supports_flash = False
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            
            # Get memory info
            total_mem = props.total_memory / (1024 ** 3)
            free_mem = (props.total_memory - torch.cuda.memory_allocated(i)) / (1024 ** 3)
            
            # Get compute capability
            cc = (props.major, props.minor)
            
            gpu = GPUInfo(
                index=i,
                name=props.name,
                memory_total_gb=total_mem,
                memory_free_gb=free_mem,
                compute_capability=cc,
                driver_version=None,
                cuda_version=torch.version.cuda,
            )
            gpus.append(gpu)
            total_memory += total_mem
            
            # Check capabilities
            if cc[0] >= 8:  # Ampere and newer
                supports_bf16 = True
                supports_flash = True
            elif cc[0] >= 7:  # Volta/Turing
                supports_flash = True
        
        # Determine recommended settings
        if supports_bf16:
            dtype = "bfloat16"
        elif gpus:
            dtype = "float16"
        else:
            dtype = "float32"
        
        # Recommend batch size based on available memory
        if total_memory >= 80:
            batch_size = 16
        elif total_memory >= 40:
            batch_size = 8
        elif total_memory >= 24:
            batch_size = 4
        elif total_memory >= 16:
            batch_size = 2
        else:
            batch_size = 1
        
        config = HardwareConfig(
            accelerator_type=AcceleratorType.NVIDIA_CUDA,
            device_count=len(gpus),
            total_memory_gb=total_memory,
            gpus=gpus,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory,
            platform=plat,
            recommended_dtype=dtype,
            recommended_batch_size=batch_size,
            supports_flash_attention=supports_flash,
            supports_bf16=supports_bf16,
            supports_fp16=True,
        )
        
        if self.verbose:
            self._print_nvidia_info(config)
        
        return config
    
    def _configure_tpu(
        self,
        cpu_cores: int,
        system_memory: float,
        plat: str,
    ) -> HardwareConfig:
        """Configure for Google TPU."""
        import torch_xla.core.xla_model as xm
        
        # Detect TPU type
        tpu_type = os.environ.get("TPU_NAME", "unknown")
        
        # Try to detect TPU configuration
        try:
            num_devices = xm.xrt_world_size()
        except:
            num_devices = 8  # Default for TPU v3-8
        
        # Determine TPU version from environment
        accelerator_type = os.environ.get("ACCELERATOR_TYPE", "v3-8")
        
        if "v5" in accelerator_type:
            tpu_version = "v5"
            memory_per_core = 16  # GB
        elif "v4" in accelerator_type:
            tpu_version = "v4"
            memory_per_core = 32
        elif "v3" in accelerator_type:
            tpu_version = "v3"
            memory_per_core = 16
        else:
            tpu_version = "v2"
            memory_per_core = 8
        
        total_memory = num_devices * memory_per_core
        
        tpu_info = TPUInfo(
            tpu_type=tpu_version,
            num_cores=num_devices,
            topology=accelerator_type,
            zone=os.environ.get("TPU_ZONE"),
        )
        
        # TPU always uses bfloat16
        config = HardwareConfig(
            accelerator_type=AcceleratorType.GOOGLE_TPU,
            device_count=num_devices,
            total_memory_gb=total_memory,
            tpu=tpu_info,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory,
            platform=plat,
            recommended_dtype="bfloat16",
            recommended_batch_size=8 * num_devices,  # TPU loves large batches
            supports_flash_attention=False,  # Different attention on TPU
            supports_bf16=True,
            supports_fp16=False,  # TPU prefers bf16
        )
        
        if self.verbose:
            self._print_tpu_info(config)
        
        return config
    
    def _configure_mps(
        self,
        cpu_cores: int,
        system_memory: float,
        plat: str,
    ) -> HardwareConfig:
        """Configure for Apple Silicon MPS."""
        import torch
        
        # Apple Silicon shares memory with system
        # Estimate available GPU memory as fraction of system memory
        gpu_memory = system_memory * 0.7  # Rough estimate
        
        config = HardwareConfig(
            accelerator_type=AcceleratorType.APPLE_MPS,
            device_count=1,
            total_memory_gb=gpu_memory,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory,
            platform=plat,
            recommended_dtype="float16",
            recommended_batch_size=2 if gpu_memory >= 16 else 1,
            supports_flash_attention=False,
            supports_bf16=False,  # MPS doesn't fully support bf16 yet
            supports_fp16=True,
        )
        
        if self.verbose:
            self._print_mps_info(config)
        
        return config
    
    def _configure_rocm(
        self,
        cpu_cores: int,
        system_memory: float,
        plat: str,
    ) -> HardwareConfig:
        """Configure for AMD ROCm."""
        import torch
        
        gpus = []
        total_memory = 0.0
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / (1024 ** 3)
            
            gpu = GPUInfo(
                index=i,
                name=props.name,
                memory_total_gb=total_mem,
                memory_free_gb=total_mem,  # Approximate
            )
            gpus.append(gpu)
            total_memory += total_mem
        
        config = HardwareConfig(
            accelerator_type=AcceleratorType.AMD_ROCM,
            device_count=len(gpus),
            total_memory_gb=total_memory,
            gpus=gpus,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory,
            platform=plat,
            recommended_dtype="float16",
            recommended_batch_size=4 if total_memory >= 16 else 2,
            supports_flash_attention=False,
            supports_bf16=True,
            supports_fp16=True,
        )
        
        if self.verbose:
            self._print_rocm_info(config)
        
        return config
    
    def _configure_cpu(
        self,
        cpu_cores: int,
        system_memory: float,
        plat: str,
    ) -> HardwareConfig:
        """Configure for CPU-only training."""
        config = HardwareConfig(
            accelerator_type=AcceleratorType.CPU,
            device_count=1,
            total_memory_gb=system_memory,
            cpu_cores=cpu_cores,
            system_memory_gb=system_memory,
            platform=plat,
            recommended_dtype="float32",
            recommended_batch_size=1,
            supports_flash_attention=False,
            supports_bf16=False,
            supports_fp16=False,
        )
        
        if self.verbose:
            self._print_cpu_info(config)
        
        return config
    
    def _print_nvidia_info(self, config: HardwareConfig):
        """Print NVIDIA GPU information."""
        print(f"\n{'='*60}")
        print(f"  🎮 NVIDIA CUDA GPUs Detected")
        print(f"{'='*60}")
        
        for gpu in config.gpus:
            cc = f"SM {gpu.compute_capability[0]}.{gpu.compute_capability[1]}" if gpu.compute_capability else "Unknown"
            print(f"  GPU {gpu.index}: {gpu.name}")
            print(f"    Memory: {gpu.memory_total_gb:.1f} GB")
            print(f"    Compute Capability: {cc}")
            print(f"    CUDA Version: {gpu.cuda_version}")
        
        print(f"\n  📊 Recommended Settings:")
        print(f"    Precision: {config.recommended_dtype}")
        print(f"    Batch Size: {config.recommended_batch_size}")
        print(f"    Flash Attention: {'✓' if config.supports_flash_attention else '✗'}")
        print(f"    BF16 Support: {'✓' if config.supports_bf16 else '✗'}")
        print(f"{'='*60}\n")
    
    def _print_tpu_info(self, config: HardwareConfig):
        """Print Google TPU information."""
        print(f"\n{'='*60}")
        print(f"  ☁️ Google TPU Detected")
        print(f"{'='*60}")
        
        if config.tpu:
            print(f"  TPU Type: {config.tpu.tpu_type}")
            print(f"  Cores: {config.tpu.num_cores}")
            print(f"  Topology: {config.tpu.topology}")
            if config.tpu.zone:
                print(f"  Zone: {config.tpu.zone}")
        
        print(f"\n  📊 Recommended Settings:")
        print(f"    Precision: {config.recommended_dtype} (TPU-optimized)")
        print(f"    Batch Size: {config.recommended_batch_size}")
        print(f"    Note: TPUs perform best with large batch sizes")
        print(f"{'='*60}\n")
    
    def _print_mps_info(self, config: HardwareConfig):
        """Print Apple Silicon MPS information."""
        print(f"\n{'='*60}")
        print(f"  🍎 Apple Silicon (MPS) Detected")
        print(f"{'='*60}")
        print(f"  Platform: {config.platform}")
        print(f"  Estimated GPU Memory: {config.total_memory_gb:.1f} GB")
        print(f"  CPU Cores: {config.cpu_cores}")
        
        print(f"\n  📊 Recommended Settings:")
        print(f"    Precision: {config.recommended_dtype}")
        print(f"    Batch Size: {config.recommended_batch_size}")
        print(f"{'='*60}\n")
    
    def _print_rocm_info(self, config: HardwareConfig):
        """Print AMD ROCm information."""
        print(f"\n{'='*60}")
        print(f"  🔴 AMD ROCm GPUs Detected")
        print(f"{'='*60}")
        
        for gpu in config.gpus:
            print(f"  GPU {gpu.index}: {gpu.name}")
            print(f"    Memory: {gpu.memory_total_gb:.1f} GB")
        
        print(f"\n  📊 Recommended Settings:")
        print(f"    Precision: {config.recommended_dtype}")
        print(f"    Batch Size: {config.recommended_batch_size}")
        print(f"{'='*60}\n")
    
    def _print_cpu_info(self, config: HardwareConfig):
        """Print CPU-only information."""
        print(f"\n{'='*60}")
        print(f"  💻 CPU-Only Mode")
        print(f"{'='*60}")
        print(f"  ⚠️  No GPU/TPU detected - training will be slow")
        print(f"  Platform: {config.platform}")
        print(f"  CPU Cores: {config.cpu_cores}")
        print(f"  System Memory: {config.system_memory_gb:.1f} GB")
        
        print(f"\n  📊 Recommended Settings:")
        print(f"    Precision: {config.recommended_dtype}")
        print(f"    Batch Size: {config.recommended_batch_size}")
        print(f"\n  💡 Tips:")
        print(f"    - Consider using cloud GPU/TPU for faster training")
        print(f"    - Use smaller models (Phi-3 Vision, Qwen2-VL-2B)")
        print(f"{'='*60}\n")


def get_device() -> str:
    """
    Get the best available device string.
    
    Returns:
        Device string: 'cuda', 'xla', 'mps', or 'cpu'
    """
    detector = HardwareDetector(verbose=False)
    config = detector.detect()
    
    if config.accelerator_type == AcceleratorType.NVIDIA_CUDA:
        return "cuda"
    elif config.accelerator_type == AcceleratorType.GOOGLE_TPU:
        return "xla"
    elif config.accelerator_type == AcceleratorType.APPLE_MPS:
        return "mps"
    elif config.accelerator_type == AcceleratorType.AMD_ROCM:
        return "cuda"  # ROCm uses CUDA interface
    else:
        return "cpu"


def get_optimal_dtype():
    """Get the optimal dtype for the detected hardware."""
    detector = HardwareDetector(verbose=False)
    config = detector.detect()
    
    import torch
    
    if config.recommended_dtype == "bfloat16":
        return torch.bfloat16
    elif config.recommended_dtype == "float16":
        return torch.float16
    else:
        return torch.float32


def auto_configure_training(
    model_size_b: float = 7,
) -> Dict[str, Any]:
    """
    Automatically configure training settings based on hardware.
    
    Args:
        model_size_b: Model size in billions of parameters
    
    Returns:
        Dictionary of recommended training settings
    """
    detector = HardwareDetector(verbose=True)
    config = detector.detect()
    
    # Estimate memory requirements
    # Rule of thumb: ~2 bytes/param for model + ~8 bytes/param for optimizer (AdamW) + activations
    model_memory_gb = model_size_b * 2  # fp16
    optimizer_memory_gb = model_size_b * 8  # AdamW states in fp32
    
    # With LoRA, we only train ~1-3% of parameters
    lora_training_memory = model_memory_gb + optimizer_memory_gb * 0.03
    
    # Calculate batch size
    available_memory = config.total_memory_gb * 0.9  # 90% of available
    activation_memory_per_sample = 0.5  # Rough estimate per sample
    
    max_batch_size = max(1, int(
        (available_memory - lora_training_memory) / activation_memory_per_sample
    ))
    
    # Apply hardware-specific adjustments
    if config.accelerator_type == AcceleratorType.GOOGLE_TPU:
        # TPU prefers larger batches, rounded to power of 2
        batch_size = min(max_batch_size, 128)
        batch_size = 2 ** int(batch_size).bit_length() // 2 or 1
        gradient_accumulation = 1  # TPU handles large batches well
    else:
        batch_size = min(max_batch_size, config.recommended_batch_size * 2)
        gradient_accumulation = max(1, 32 // batch_size)  # Target effective batch of 32
    
    return {
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "effective_batch_size": batch_size * gradient_accumulation,
        "dtype": config.recommended_dtype,
        "use_gradient_checkpointing": config.total_memory_gb < 40,
        "use_flash_attention": config.supports_flash_attention,
        "device": get_device(),
        "device_count": config.device_count,
        "hardware_type": config.accelerator_type.value,
    }


# Convenience function for CLI
def print_hardware_info():
    """Print detected hardware information."""
    detector = HardwareDetector(verbose=True)
    detector.detect()
