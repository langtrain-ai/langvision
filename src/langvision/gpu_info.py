"""
gpu_info.py — Langvision GPU Detection & Auto-Configuration
============================================================

Detects available hardware and automatically selects the optimal
kernel stack, attention implementation, and data type for vision LLMs.

Vision-specific considerations vs langtune:
  - Vision encoders (ViT/CLIP) are larger than text-only models
  - Image tokens consume additional VRAM (~256–1024 tokens per image)
  - Sequence length recommendations are more conservative
  - Flash Attention 2 is applied only to the language decoder (not ViT)
  - MPS support: limited — vision encoders often fail on MPS fp16

Works seamlessly in:
  - Jupyter notebooks (rich HTML or plain-text output)
  - Python scripts (terminal ANSI color or plain)
  - Colab / Kaggle / SageMaker / Modal

Usage:
    from langvision.gpu_info import get_gpu_info, print_gpu_info, auto_config

    info = get_gpu_info()
    cfg  = auto_config()          # prints banner + returns config dict
    cfg  = auto_config(silent=True)  # no print
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GPUInfo:
    has_cuda: bool = False
    has_mps: bool = False
    has_tpu: bool = False
    is_cpu_only: bool = False

    gpu_name: str = ""
    gpu_count: int = 0
    gpu_memory_gb: float = 0.0
    compute_capability: Tuple[int, int] = (0, 0)
    cuda_version: str = ""
    multi_gpu: bool = False

    mps_version: str = ""
    apple_chip: str = ""

    tpu_version: str = ""
    tpu_cores: int = 0

    # Capabilities
    supports_flash_attention_2: bool = False
    supports_bfloat16: bool = False
    supports_4bit_quant: bool = False
    supports_8bit_quant: bool = False

    # Vision-specific: enough VRAM for vision encoder?
    supports_vision_encoder_lora: bool = False  # needs ~4 GB extra above base model

    # Packages
    flash_attn_available: bool = False
    bitsandbytes_available: bool = False
    triton_available: bool = False
    peft_available: bool = False
    trl_available: bool = False
    timm_available: bool = False       # vision model library
    transformers_vision: bool = False  # transformers >= 4.37 for VLMs

    # Recommended
    recommended_dtype: str = "float32"
    recommended_attn: str = "eager"
    recommended_load_in_4bit: bool = False
    recommended_max_seq_length: int = 1024   # conservative for vision (image tokens)
    recommended_image_size: int = 336


@lru_cache(maxsize=1)
def get_gpu_info() -> GPUInfo:
    """Detect hardware and capabilities. Cached after first call."""
    info = GPUInfo()

    try:
        import torch
    except ImportError:
        info.is_cpu_only = True
        return info

    # ── NVIDIA CUDA ──────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        info.has_cuda = True
        info.gpu_count = torch.cuda.device_count()
        info.multi_gpu = info.gpu_count > 1

        try:
            props = torch.cuda.get_device_properties(0)
            info.gpu_name = props.name
            info.gpu_memory_gb = props.total_memory / (1024 ** 3)
            info.compute_capability = (props.major, props.minor)
        except Exception:
            info.gpu_name = torch.cuda.get_device_name(0)

        info.cuda_version = torch.version.cuda or ""
        cc = info.compute_capability[0]

        info.supports_flash_attention_2 = cc >= 8
        info.supports_bfloat16 = cc >= 8
        info.supports_4bit_quant = True
        info.supports_8bit_quant = True

        # Vision models need extra VRAM — be conservative with 4-bit threshold
        # 7B VLM: ~14 GB fp16, ~7 GB 4-bit + image tokens
        info.recommended_load_in_4bit = info.gpu_memory_gb < 40
        info.recommended_dtype = "bfloat16" if info.supports_bfloat16 else "float16"

        # Sequence lengths for vision: images take 256–1024 tokens
        if info.gpu_memory_gb >= 80:
            info.recommended_max_seq_length = 4096
            info.recommended_image_size = 448
        elif info.gpu_memory_gb >= 40:
            info.recommended_max_seq_length = 2048
            info.recommended_image_size = 336
        elif info.gpu_memory_gb >= 24:
            info.recommended_max_seq_length = 2048
            info.recommended_image_size = 336
        elif info.gpu_memory_gb >= 16:
            info.recommended_max_seq_length = 1024
            info.recommended_image_size = 224
        else:
            info.recommended_max_seq_length = 512
            info.recommended_image_size = 224

        # Vision encoder LoRA needs ~4 GB headroom
        info.supports_vision_encoder_lora = info.gpu_memory_gb >= 20

    # ── Apple MPS ────────────────────────────────────────────────────────────
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        info.has_mps = True
        info.supports_4bit_quant = False
        info.supports_8bit_quant = False

        try:
            import subprocess
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, timeout=3,
            ).decode().strip()
            for chip in ("M4", "M3", "M2", "M1"):
                if chip in brand:
                    info.apple_chip = chip
                    break
            info.apple_chip = info.apple_chip or "Apple Silicon"
        except Exception:
            info.apple_chip = "Apple Silicon"

        # M2+ supports bfloat16; M1 is float16 only
        info.supports_bfloat16 = info.apple_chip not in ("M1",)
        info.recommended_dtype = "bfloat16" if info.supports_bfloat16 else "float16"
        info.recommended_attn = "sdpa"
        info.recommended_max_seq_length = 512
        info.recommended_image_size = 224
        # Vision models on MPS can be unstable — warn user
        info.supports_vision_encoder_lora = False

    # ── TPU ──────────────────────────────────────────────────────────────────
    elif _check_tpu():
        info.has_tpu = True
        try:
            import torch_xla.core.xla_model as xm
            info.tpu_cores = xm.xrt_world_size()
            tpu_name = os.environ.get("TPU_NAME", "")
            for v in ("v5", "v4", "v3", "v2"):
                if v in tpu_name.lower():
                    info.tpu_version = v
                    break
            info.tpu_version = info.tpu_version or "v4"
        except Exception:
            info.tpu_cores = 1
            info.tpu_version = "?"
        info.supports_bfloat16 = True
        info.recommended_dtype = "bfloat16"
        info.recommended_max_seq_length = 1024

    else:
        info.is_cpu_only = True
        info.recommended_dtype = "float32"
        info.recommended_max_seq_length = 256

    # ── Flash Attention availability ──────────────────────────────────────────
    if info.has_cuda and info.supports_flash_attention_2:
        if _pkg("flash_attn"):
            info.flash_attn_available = True
            info.recommended_attn = "flash_attention_2"
        else:
            info.recommended_attn = "sdpa"
    elif info.has_cuda:
        info.recommended_attn = "sdpa"

    # ── Package checks ────────────────────────────────────────────────────────
    info.bitsandbytes_available = _pkg("bitsandbytes") and info.has_cuda
    info.triton_available = _pkg("triton") and info.has_cuda
    info.peft_available = _pkg("peft")
    info.trl_available = _pkg("trl")
    info.timm_available = _pkg("timm")

    # Check transformers version for VLM support (needs >= 4.37)
    try:
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
        info.transformers_vision = ver >= (4, 37)
    except Exception:
        info.transformers_vision = False

    return info


def _check_tpu() -> bool:
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        return "xla" in str(dev).lower() or "tpu" in str(dev).lower()
    except Exception:
        return False


def _pkg(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Jupyter detection
# ─────────────────────────────────────────────────────────────────────────────

def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and ("ZMQ" in type(shell).__name__ or "Colab" in str(type(shell)))
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

_BANNER_PRINTED = False


def print_gpu_info(info: Optional[GPUInfo] = None, force: bool = False) -> None:
    global _BANNER_PRINTED
    if _BANNER_PRINTED and not force:
        return
    _BANNER_PRINTED = True

    if info is None:
        info = get_gpu_info()

    if _in_jupyter():
        _print_jupyter(info)
    else:
        _print_terminal(info)


def _accel_label(info: GPUInfo) -> str:
    if info.has_cuda:
        mem = f"{info.gpu_memory_gb:.0f}GB" if info.gpu_memory_gb else ""
        cc = f"sm_{info.compute_capability[0]}{info.compute_capability[1]}"
        count = f" ×{info.gpu_count}" if info.multi_gpu else ""
        return f"NVIDIA {info.gpu_name}{count} ({mem}, CUDA {info.cuda_version}, {cc})"
    elif info.has_mps:
        return f"Apple {info.apple_chip} (MPS)"
    elif info.has_tpu:
        return f"Google TPU {info.tpu_version} ({info.tpu_cores} cores)"
    return "CPU only (no GPU detected)"


def _print_terminal(info: GPUInfo) -> None:
    RESET = "\033[0m"
    BOLD  = "\033[1m"
    DIM   = "\033[2m"
    GREEN = "\033[92m"
    YELLOW= "\033[93m"
    BLUE  = "\033[94m"
    CYAN  = "\033[96m"

    def tick(ok: bool) -> str:
        return f"{GREEN}✓{RESET}" if ok else f"{YELLOW}✗{RESET}"

    mps_warning = ""
    if info.has_mps:
        mps_warning = (
            f"\n  {YELLOW}⚠ MPS: Some vision models may require CPU fallback for fp16 ops.{RESET}"
            f"\n  {DIM}  Set PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 if you hit OOM.{RESET}"
        )

    fa2_hint = ""
    if not info.flash_attn_available and info.has_cuda and info.supports_flash_attention_2:
        fa2_hint = (
            f"\n  {YELLOW}💡 Install flash-attn for 2–4× faster attention:{RESET}"
            f"\n     {DIM}pip install flash-attn --no-build-isolation{RESET}"
        )

    lines = [
        f"\n{BLUE}{BOLD}  langvision{RESET}{DIM} — Vision LLMs · Unsloth-compatible API{RESET}",
        "",
        f"  {BOLD}Hardware{RESET}",
        f"    {tick(not info.is_cpu_only)} {_accel_label(info)}",
        mps_warning,
        "",
        f"  {BOLD}Kernel Stack{RESET}",
        f"    {tick(info.flash_attn_available)}  FlashAttention 2  (language decoder, 2–4×)",
        f"    {tick(info.triton_available)}  Triton JIT kernels (RMSNorm, RoPE, FusedCE)",
        f"    {tick(info.bitsandbytes_available)}  BitsAndBytes 4-bit",
        f"    {tick(info.peft_available)}  PEFT",
        f"    {tick(info.trl_available)}  TRL",
        f"    {tick(info.timm_available)}  timm (vision model zoo)",
        f"    {tick(info.transformers_vision)}  transformers VLM support",
        "",
        f"  {BOLD}Recommended config{RESET}",
        f"    dtype           : {CYAN}{info.recommended_dtype}{RESET}",
        f"    attention       : {CYAN}{info.recommended_attn}{RESET}",
        f"    load_in_4bit    : {CYAN}{info.recommended_load_in_4bit}{RESET}",
        f"    max_seq_length  : {CYAN}{info.recommended_max_seq_length}{RESET}",
        f"    image_size      : {CYAN}{info.recommended_image_size}{RESET}",
        f"    vision_enc_lora : {CYAN}{info.supports_vision_encoder_lora}{RESET}",
        fa2_hint,
        "",
        f"  {DIM}Set LANGVISION_NO_BANNER=1 to suppress this message{RESET}\n",
    ]

    print("\n".join(l for l in lines if l is not None))


def _print_jupyter(info: GPUInfo) -> None:
    try:
        from IPython.display import display, HTML
    except ImportError:
        _print_terminal(info)
        return

    def badge(ok: bool, label: str, detail: str = "") -> str:
        color = "#22c55e" if ok else "#6b7280"
        icon = "✓" if ok else "✗"
        d = f" <span style='color:#94a3b8;font-size:11px'>{detail}</span>" if detail else ""
        return (
            f"<span style='display:inline-flex;align-items:center;gap:6px;margin:2px 0'>"
            f"<span style='color:{color};font-weight:900'>{icon}</span>"
            f"<span style='color:#e2e8f0;font-size:13px'>{label}</span>{d}</span>"
        )

    mps_warn = ""
    if info.has_mps:
        mps_warn = (
            "<div style='margin-top:10px;padding:8px 12px;background:#1e293b;"
            "border-left:3px solid #fbbf24;border-radius:6px;font-size:12px;color:#fbbf24'>"
            "⚠ MPS: Some vision models may need CPU fallback for fp16 ops. "
            "Set <code style='background:#0f172a;padding:2px 6px;border-radius:4px'>"
            "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0</code> if you hit OOM."
            "</div>"
        )

    fa2_hint = ""
    if not info.flash_attn_available and info.has_cuda and info.supports_flash_attention_2:
        fa2_hint = (
            "<div style='margin-top:10px;padding:8px 12px;background:#1e293b;"
            "border-left:3px solid #fbbf24;border-radius:6px;font-size:12px;color:#fbbf24'>"
            "💡 Install flash-attn for 2–4× faster attention (language decoder): "
            "<code style='background:#0f172a;padding:2px 6px;border-radius:4px'>"
            "pip install flash-attn --no-build-isolation</code>"
            "</div>"
        )

    accel_color = "#60a5fa" if not info.is_cpu_only else "#fbbf24"

    html = f"""
<div style="font-family:'JetBrains Mono',monospace;background:#050505;
            border:1px solid rgba(96,165,250,0.2);border-radius:16px;
            padding:24px 28px;margin:12px 0;max-width:700px">
  <div style="color:#60a5fa;font-size:22px;font-weight:900;letter-spacing:-0.5px;margin-bottom:4px">
    langvision
  </div>
  <div style="color:#475569;font-size:12px;margin-bottom:20px">
    Vision LLMs · Unsloth-compatible API
  </div>

  <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
    <div>
      <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:0.1em;margin-bottom:10px">HARDWARE</div>
      <div style="color:{accel_color};font-weight:700;font-size:13px">{_accel_label(info)}</div>
      {mps_warn}
    </div>
    <div>
      <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:0.1em;margin-bottom:10px">RECOMMENDED</div>
      <div style="color:#94a3b8;font-size:12px;line-height:1.8">
        dtype: <span style="color:#60a5fa">{info.recommended_dtype}</span><br>
        attention: <span style="color:#60a5fa">{info.recommended_attn}</span><br>
        4-bit: <span style="color:#60a5fa">{info.recommended_load_in_4bit}</span><br>
        max_seq_len: <span style="color:#60a5fa">{info.recommended_max_seq_length}</span><br>
        image_size: <span style="color:#60a5fa">{info.recommended_image_size}</span>
      </div>
    </div>
  </div>

  <div style="border-top:1px solid rgba(255,255,255,0.06);margin:16px 0 14px"></div>
  <div style="color:#64748b;font-size:11px;font-weight:700;letter-spacing:0.1em;margin-bottom:10px">KERNEL STACK</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 24px;font-size:13px">
    {badge(info.flash_attn_available, "FlashAttention 2", "decoder only, 2–4×")}
    {badge(info.bitsandbytes_available, "BitsAndBytes 4-bit")}
    {badge(info.triton_available, "Triton JIT", "RMSNorm, RoPE, FusedCE")}
    {badge(info.peft_available, "PEFT")}
    {badge(info.trl_available, "TRL")}
    {badge(info.timm_available, "timm")}
  </div>
  {fa2_hint}
</div>
"""
    display(HTML(html))


# ─────────────────────────────────────────────────────────────────────────────
# Auto-config
# ─────────────────────────────────────────────────────────────────────────────

_AUTO_CONFIG_CALLED = False


def auto_config(
    *,
    silent: bool = False,
    load_in_4bit: Optional[bool] = None,
    use_flash_attention_2: Optional[bool] = None,
    dtype=None,
    max_seq_length: Optional[int] = None,
    image_size: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Detect GPU, print banner (first call only), return optimal config.

    Returns:
        {
            "load_in_4bit": bool,
            "use_flash_attention_2": bool,
            "attn_implementation": str,
            "dtype": torch.dtype,
            "max_seq_length": int,
            "image_size": int,
            "device_map": str,
            "info": GPUInfo,
        }
    """
    global _AUTO_CONFIG_CALLED

    info = get_gpu_info()

    if not silent and not _AUTO_CONFIG_CALLED:
        no_banner = os.environ.get("LANGVISION_NO_BANNER", "0") == "1"
        if not no_banner:
            print_gpu_info(info)
    _AUTO_CONFIG_CALLED = True

    try:
        import torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16":  torch.float16,
            "float32":  torch.float32,
        }
        auto_dtype = dtype_map.get(info.recommended_dtype, torch.float32)
    except ImportError:
        auto_dtype = None

    resolved_4bit = load_in_4bit if load_in_4bit is not None else (
        info.recommended_load_in_4bit and info.bitsandbytes_available
    )
    resolved_fa2 = use_flash_attention_2 if use_flash_attention_2 is not None else (
        info.flash_attn_available
    )
    resolved_dtype = dtype if dtype is not None else auto_dtype
    resolved_seqlen = max_seq_length if max_seq_length is not None else info.recommended_max_seq_length
    resolved_img = image_size if image_size is not None else info.recommended_image_size

    attn_impl = "flash_attention_2" if resolved_fa2 else info.recommended_attn

    return {
        "load_in_4bit":          resolved_4bit,
        "use_flash_attention_2": resolved_fa2,
        "attn_implementation":   attn_impl,
        "dtype":                 resolved_dtype,
        "max_seq_length":        resolved_seqlen,
        "image_size":            resolved_img,
        "device_map":            "auto" if (info.has_cuda or info.has_mps) else "cpu",
        "info":                  info,
    }
