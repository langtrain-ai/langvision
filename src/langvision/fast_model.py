"""
fast_model.py — FastVisionModel
=================================
Unsloth-style high-level API for Langvision (vision LLMs).

Two execution modes, same API:
  LOCAL  — trains on the current machine's GPU (uses Triton/CUDA kernels directly)
  REMOTE — dispatches to langtrain-server GPU infrastructure via REST API

Usage:
    from langvision import FastVisionModel

    # Local GPU (like Unsloth for vision)
    model, processor = FastVisionModel.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        load_in_4bit=True,
    )
    model = FastVisionModel.get_peft_model(model, r=16, method="qlora")
    FastVisionModel.train(model, processor, dataset, output_dir="./output")

    # Remote — dispatches to langtrain-server
    model, processor = FastVisionModel.from_pretrained(
        "llava-hf/llava-1.5-7b-hf",
        api_key="lt_...",
        load_in_4bit=True,
    )
    model = FastVisionModel.get_peft_model(model, r=16, method="qlora")
    job = FastVisionModel.train(model, processor, dataset)
    for step in job.stream():
        print(step)

Supported models:
  LLaVA, InstructBLIP, Qwen-VL, InternVL, PaliGemma, BLIP-2,
  Idefics, mPLUG-Owl, CogVLM, MiniGPT-4, and any HF vision-language model.

Training methods (vision-specific):
  'sft'        — Supervised fine-tuning on captioning / VQA
  'lora'       — LoRA on vision + language projections
  'qlora'      — 4-bit QLoRA (default, most memory efficient)
  'dora'       — DoRA (weight decomposition LoRA)
  'dpo'        — Direct Preference Optimization for vision preferences
  'orpo'       — Odds Ratio Preference Optimization
  'simpo'      — Simple Preference Optimization
  'kto'        — Kahneman-Tversky Optimization
  'rlhf'       — RLHF / PPO with a vision reward model
  'grpo'       — Group Relative Policy Optimization (vision RLVR)
  'prefix'     — Prefix Tuning (vision-language)
  'ia3'        — IA³ (Infused Adapter by Inhibiting and Amplifying Activations)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Remote job handle (identical to langtune's RemoteJob)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class VisionTrainingStep:
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    progress: int = 0
    status: str = "running"
    message: str = ""


class VisionRemoteJob:
    """
    Handle for a vision training job running on langtrain-server.
    Poll, stream, cancel, and download the trained model.
    """

    def __init__(self, job_id: str, client: "LangvisionServerClient"):
        self.job_id = job_id
        self._client = client

    def status(self) -> Dict[str, Any]:
        return self._client.get_job(self.job_id)

    def stream(self, interval_s: float = 5.0) -> Generator[VisionTrainingStep, None, None]:
        """Yield VisionTrainingStep updates until the job completes."""
        import time

        last_step = -1
        while True:
            info = self.status()
            current_status = info.get("status", "running")

            telemetry = self._client.get_telemetry(self.job_id, after_step=last_step)
            for point in telemetry:
                s = point.get("step", last_step + 1)
                if s > last_step:
                    last_step = s
                    yield VisionTrainingStep(
                        step=s,
                        loss=point.get("loss"),
                        learning_rate=point.get("learning_rate"),
                        epoch=point.get("epoch"),
                        progress=info.get("progress", 0),
                        status=current_status,
                        message=point.get("message", ""),
                    )

            if current_status in ("completed", "failed", "cancelled"):
                yield VisionTrainingStep(
                    step=last_step,
                    progress=100 if current_status == "completed" else 0,
                    status=current_status,
                    message=f"Job {current_status}",
                )
                break

            time.sleep(interval_s)

    def wait(self, poll_interval_s: float = 10.0) -> Dict[str, Any]:
        """Block until the job finishes. Returns final job info."""
        import time

        while True:
            info = self.status()
            if info.get("status") in ("completed", "failed", "cancelled"):
                return info
            time.sleep(poll_interval_s)

    def cancel(self) -> bool:
        return self._client.cancel_job(self.job_id)

    def download(self, output_dir: str = "./model") -> str:
        return self._client.download_model(self.job_id, output_dir)

    def __repr__(self) -> str:
        return f"VisionRemoteJob(id={self.job_id!r})"


# ─────────────────────────────────────────────────────────────────────────────
# langtrain-server REST client (vision flavour)
# ─────────────────────────────────────────────────────────────────────────────

class LangvisionServerClient:
    """
    Thin HTTP client for langtrain-server — vision endpoints.
    Mirrors LangtrainServerClient from langtune with vision-specific headers.
    """

    DEFAULT_BASE_URL = "https://api.langtrain.xyz"

    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.api_key = api_key
        self.base_url = (
            base_url
            or os.environ.get("LANGTRAIN_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self._session = None

    def _get_session(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-SDK": "langvision",
            })
            self._session = s
        return self._session

    def _post(self, path: str, payload: Dict) -> Dict:
        r = self._get_session().post(f"{self.base_url}/v1{path}", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: Dict = None) -> Any:
        r = self._get_session().get(f"{self.base_url}/v1{path}", params=params or {}, timeout=30)
        r.raise_for_status()
        return r.json()

    def upload_dataset(self, dataset_path: str, name: str = None) -> str:
        """Upload a local dataset file, return dataset_id."""
        with open(dataset_path, "rb") as f:
            r = self._get_session().post(
                f"{self.base_url}/v1/datasets/upload",
                files={"file": (os.path.basename(dataset_path), f)},
                data={"name": name or os.path.basename(dataset_path)},
                timeout=600,
            )
        r.raise_for_status()
        return r.json()["id"]

    def upload_images(self, image_paths: List[str]) -> List[str]:
        """Upload image files in batch, return list of image_ids."""
        ids = []
        for path in image_paths:
            with open(path, "rb") as f:
                r = self._get_session().post(
                    f"{self.base_url}/v1/datasets/images/upload",
                    files={"file": (os.path.basename(path), f)},
                    timeout=120,
                )
            r.raise_for_status()
            ids.append(r.json()["id"])
        return ids

    def create_job(self, payload: Dict) -> Dict:
        return self._post("/finetune/vision/jobs", payload)

    def get_job(self, job_id: str) -> Dict:
        return self._get(f"/finetune/vision/jobs/{job_id}")

    def get_telemetry(self, job_id: str, after_step: int = -1) -> List[Dict]:
        try:
            return self._get(
                f"/finetune/vision/jobs/{job_id}/telemetry",
                {"after_step": after_step}
            ) or []
        except Exception:
            return []

    def cancel_job(self, job_id: str) -> bool:
        try:
            self._post(f"/finetune/vision/jobs/{job_id}/cancel", {})
            return True
        except Exception:
            return False

    def download_model(self, job_id: str, output_dir: str) -> str:
        import zipfile
        os.makedirs(output_dir, exist_ok=True)
        r = self._get_session().get(
            f"{self.base_url}/v1/finetune/vision/jobs/{job_id}/download",
            stream=True,
            timeout=600,
        )
        r.raise_for_status()
        zip_path = os.path.join(output_dir, "adapter.zip")
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        with __import__("zipfile").ZipFile(zip_path, "r") as z:
            z.extractall(output_dir)
        os.remove(zip_path)
        logger.info(f"Vision model downloaded to {output_dir}")
        return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# Wrapped model handles
# ─────────────────────────────────────────────────────────────────────────────

class _WrappedVisionLocalModel:
    """
    Thin wrapper around a HuggingFace vision-language model.
    Returned by FastVisionModel.from_pretrained() in local mode.
    """

    def __init__(self, model, processor=None, lora_config=None, hyperparameters=None):
        self._model = model
        self._processor = processor
        self._lora_config = lora_config
        self._hyperparameters = hyperparameters or {}

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return self._model.parameters(*args, **kwargs)

    def named_parameters(self, *args, **kwargs):
        return self._model.named_parameters(*args, **kwargs)

    def train(self, mode=True):
        return self._model.train(mode)

    def eval(self):
        return self._model.eval()

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self


class _WrappedVisionRemoteModel:
    """
    Placeholder returned by FastVisionModel.from_pretrained() in remote mode.
    Holds model config for dispatching to langtrain-server.
    """

    def __init__(
        self,
        model_id: str,
        client: LangvisionServerClient,
        hyperparameters: Dict,
        lora_config: Dict = None,
    ):
        self.model_id = model_id
        self._client = client
        self._hyperparameters = hyperparameters
        self._lora_config = lora_config or {}
        self._method = "qlora"


# ─────────────────────────────────────────────────────────────────────────────
# Kernel integration helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_server_root() -> Optional[str]:
    """Locate langtrain-server directory for Triton kernel access."""
    env_path = os.environ.get("LANGTRAIN_SERVER_PATH")
    if env_path and os.path.isdir(os.path.join(env_path, "app", "training")):
        return env_path

    here = os.path.dirname(__file__)
    for rel in [
        "../../../langtrain-server",
        "../../../../langtrain-server",
        "~/langtrain-server",
        "/opt/langtrain-server",
    ]:
        candidate = os.path.abspath(os.path.expanduser(os.path.join(here, rel)))
        if os.path.isdir(os.path.join(candidate, "app", "training")):
            return candidate
    return None


_SERVER_ROOT = _find_server_root()


def _apply_vision_kernels(model) -> Any:
    """
    Apply Langtrain Triton/CUDA kernel stack to a vision-language model.

    Applies (when langtrain-server is available + CUDA present):
      1. FusedRMSNorm — patches all norm layers in the language decoder
      2. Triton RoPE  — patches rotary embeddings in the decoder
      3. FusedCE      — active via LangvisionSFTTrainer mixin

    Vision encoder (ViT / CLIP) layers are NOT patched — they use their own
    attention patterns and are generally kept in bf16/fp16 as-is.
    """
    if _SERVER_ROOT is None:
        return model

    try:
        import sys
        for p in [_SERVER_ROOT, os.path.join(_SERVER_ROOT, "app", "training", "triton_kernels")]:
            if p not in sys.path:
                sys.path.insert(0, p)
        from app.training.kernel_integration import apply_all_triton_kernels
        model = apply_all_triton_kernels(model, {})
        logger.info("[Langvision] Triton kernel stack applied to language decoder")
    except Exception as e:
        logger.debug(f"[Langvision] Kernel application skipped: {e}")

    return model


def _make_vision_sft_trainer_cls():
    """Build LangvisionSFTTrainer using TRL SFTTrainer when available."""
    try:
        from trl import SFTTrainer
        base = SFTTrainer
    except ImportError:
        from transformers import Trainer
        base = Trainer

    # Try to pull in fused CE mixin from langtune kernels
    try:
        import sys
        if _SERVER_ROOT:
            for p in [_SERVER_ROOT, os.path.join(_SERVER_ROOT, "app", "training", "triton_kernels")]:
                if p not in sys.path:
                    sys.path.insert(0, p)

        import torch

        class _FusedCEMixin:
            _fused_ce_fn = None
            _checked = False

            def _get_fused_ce(self):
                if not self.__class__._checked:
                    self.__class__._checked = True
                    if torch.cuda.is_available():
                        try:
                            from fused_cross_entropy import chunked_cross_entropy_loss
                            self.__class__._fused_ce_fn = chunked_cross_entropy_loss
                        except ImportError:
                            pass
                return self.__class__._fused_ce_fn

            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                fused_ce = self._get_fused_ce()
                if fused_ce is None:
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
                labels = inputs.pop("labels", None)
                outputs = model(**inputs)
                if labels is None:
                    return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
                shift_logits = outputs.logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = fused_ce(shift_logits, shift_labels, ignore_index=-100)
                return (loss, outputs) if return_outputs else loss

        class _LangvisionSFTTrainer(_FusedCEMixin, base):
            pass

        return _LangvisionSFTTrainer

    except Exception:
        # Fallback: plain SFTTrainer / Trainer without fused CE
        return base


# ─────────────────────────────────────────────────────────────────────────────
# FastVisionModel — the Unsloth-style entry point for vision LLMs
# ─────────────────────────────────────────────────────────────────────────────

class FastVisionModel:
    """
    Langvision high-level API — Unsloth-compatible interface for vision LLMs.

    Supports all vision-language models on HuggingFace Hub and langtrain-server's
    full training suite with Triton/CUDA kernel acceleration.

    Execution modes:
      LOCAL  — no api_key (or api_key=None). Trains on local GPU with Triton kernels.
      REMOTE — api_key="lt_...". Dispatches to langtrain-server GPU cloud.

    Supported training methods:
      'sft', 'lora', 'qlora', 'dora', 'dpo', 'orpo', 'simpo', 'kto',
      'rlhf', 'grpo', 'prefix', 'ia3'
    """

    # ── from_pretrained ──────────────────────────────────────────────────────

    @staticmethod
    def from_pretrained(
        model_name: str,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        dtype: Optional[Any] = None,
        use_flash_attention_2: bool = True,
        use_gradient_checkpointing: bool = True,
        max_seq_length: int = 2048,
        image_size: int = 336,
        device_map: str = "auto",
        trust_remote_code: bool = True,
        token: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Any, Any]:
        """
        Load a vision-language model + processor with full Langvision optimization stack.

        Args:
            model_name: HuggingFace model ID (e.g. "llava-hf/llava-1.5-7b-hf")
            api_key: langtrain-server API key. If provided, returns a remote handle.
            load_in_4bit: NF4 QLoRA quantization (local mode only)
            load_in_8bit: 8-bit LLM.int8() quantization (local mode only)
            use_flash_attention_2: Enable FlashAttention2 kernel on language decoder
            use_gradient_checkpointing: Gradient checkpointing (local mode)
            max_seq_length: Maximum sequence length for training
            image_size: Input image resolution (square). Overridden by model config if present.
            token: HuggingFace Hub token for gated models

        Returns:
            (model, processor) — processor is AutoProcessor or AutoImageProcessor.
            In remote mode, model is a _WrappedVisionRemoteModel.
        """
        _key = api_key or os.environ.get("LANGTRAIN_API_KEY")
        hyperparameters = {
            "max_seq_length": max_seq_length,
            "image_size": image_size,
            "use_flash_attention_2": use_flash_attention_2,
            "use_gradient_checkpointing": use_gradient_checkpointing,
        }

        if _key:
            # ── REMOTE MODE ──────────────────────────────────────────────
            logger.info(f"[Langvision] Remote mode — dispatching to langtrain-server")
            client = LangvisionServerClient(_key, base_url)
            model = _WrappedVisionRemoteModel(model_name, client, hyperparameters)
            processor = FastVisionModel._load_processor(model_name, token)
            return model, processor

        # ── LOCAL MODE ───────────────────────────────────────────────────
        logger.info(f"[Langvision] Local mode — loading {model_name}")
        import torch

        # Auto-detect GPU hardware and resolve optimal settings
        try:
            from langvision.gpu_info import auto_config
            cfg = auto_config(
                load_in_4bit=load_in_4bit,
                use_flash_attention_2=use_flash_attention_2,
                dtype=dtype,
                max_seq_length=max_seq_length,
                image_size=image_size,
            )
            info = cfg["info"]
            _load_in_4bit = load_in_4bit if load_in_4bit else cfg["load_in_4bit"]
            _use_fa2 = cfg["use_flash_attention_2"]
            _attn_impl = cfg["attn_implementation"]
            _dtype = dtype if dtype is not None else cfg["dtype"]
            _image_size = image_size if image_size != 336 else cfg.get("image_size", image_size)
            _bnb_available = info.bitsandbytes_available
        except Exception:
            # Fallback to manual detection
            _load_in_4bit = load_in_4bit
            _use_fa2 = use_flash_attention_2 and torch.cuda.is_available()
            _attn_impl = "flash_attention_2" if _use_fa2 else "eager"
            _dtype = dtype if dtype is not None else (
                torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float16
            )
            _image_size = image_size
            _bnb_available = True  # assume available; BnB import will fail naturally

        load_kwargs: Dict[str, Any] = {
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            "torch_dtype": _dtype,
        }
        if token:
            load_kwargs["token"] = token
        if _load_in_4bit and not load_in_8bit:
            if _bnb_available:
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=_dtype,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                logger.warning("[Langvision] bitsandbytes not available — skipping 4-bit quantization")
        elif load_in_8bit:
            load_kwargs["load_in_8bit"] = True

        if _attn_impl and _attn_impl != "eager":
            load_kwargs["attn_implementation"] = _attn_impl

        model = FastVisionModel._load_with_kernels(model_name, load_kwargs, hyperparameters)

        if use_gradient_checkpointing:
            try:
                model.gradient_checkpointing_enable()
            except Exception:
                pass

        processor = FastVisionModel._load_processor(model_name, token)
        return _WrappedVisionLocalModel(model, processor, hyperparameters=hyperparameters), processor

    @staticmethod
    def _load_with_kernels(model_name: str, load_kwargs: Dict, hyperparameters: Dict) -> Any:
        from transformers import AutoModelForVision2Seq, AutoModelForCausalLM

        # Try vision-language model loaders in priority order
        for loader_cls in [AutoModelForVision2Seq, AutoModelForCausalLM]:
            try:
                model = loader_cls.from_pretrained(model_name, **load_kwargs)
                model = _apply_vision_kernels(model)
                return model
            except (OSError, ValueError, KeyError):
                continue
            except Exception as e:
                # Flash attention may not be supported — retry without it
                if "flash_attention_2" in str(e).lower() or "attn_implementation" in str(e).lower():
                    load_kwargs_no_flash = {k: v for k, v in load_kwargs.items()
                                           if k != "attn_implementation"}
                    try:
                        model = loader_cls.from_pretrained(model_name, **load_kwargs_no_flash)
                        model = _apply_vision_kernels(model)
                        return model
                    except Exception:
                        continue
                raise

        raise ValueError(f"Could not load {model_name} with any vision model loader")

    @staticmethod
    def _load_processor(model_name: str, token: Optional[str] = None) -> Any:
        """Load AutoProcessor or AutoImageProcessor — whichever works."""
        from transformers import AutoProcessor
        try:
            kwargs = {"trust_remote_code": True}
            if token:
                kwargs["token"] = token
            return AutoProcessor.from_pretrained(model_name, **kwargs)
        except Exception:
            from transformers import AutoTokenizer
            kwargs = {"trust_remote_code": True}
            if token:
                kwargs["token"] = token
            return AutoTokenizer.from_pretrained(model_name, **kwargs)

    # ── get_peft_model ───────────────────────────────────────────────────────

    @staticmethod
    def get_peft_model(
        model: Any,
        r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[List[str]] = None,
        method: str = "qlora",
        bias: str = "none",
        use_dora: bool = False,
        use_rslora: bool = False,
        # Vision-specific
        vision_tower_modules: Optional[List[str]] = None,
        train_vision_encoder: bool = False,
        # IA³ / prefix
        num_prefix_tokens: int = 10,
        **kwargs,
    ) -> Any:
        """
        Wrap the model with a PEFT adapter for efficient fine-tuning.

        Args:
            model: Model returned by from_pretrained()
            r: LoRA rank
            lora_alpha: LoRA scaling factor
            lora_dropout: LoRA dropout probability
            target_modules: Modules to apply LoRA to. Defaults to language decoder
                            attention + MLP (vision encoder excluded unless
                            train_vision_encoder=True).
            method: 'lora', 'qlora', 'dora', 'ia3', 'prefix'
            train_vision_encoder: Also apply LoRA to ViT/CLIP vision encoder layers.
            vision_tower_modules: Specific module names in the vision encoder to target.

        Returns:
            model with PEFT adapter attached (in-place for local, config stored for remote)
        """
        if isinstance(model, _WrappedVisionRemoteModel):
            model._lora_config = {
                "r": r, "lora_alpha": lora_alpha, "lora_dropout": lora_dropout,
                "method": method, "use_dora": use_dora, "use_rslora": use_rslora,
                "train_vision_encoder": train_vision_encoder,
                "num_prefix_tokens": num_prefix_tokens,
            }
            model._method = method
            return model

        # Local mode — apply PEFT
        inner = model._model if isinstance(model, _WrappedVisionLocalModel) else model

        # Default target modules: language decoder attention + MLP
        if target_modules is None:
            target_modules = FastVisionModel._default_target_modules(inner, train_vision_encoder)

        if method in ("ia3",):
            from peft import IA3Config, get_peft_model as peft_get
            cfg = IA3Config(
                target_modules=target_modules,
                feedforward_modules=["mlp.fc2", "fc2", "down_proj", "o_proj"],
                bias=bias,
            )
        elif method == "prefix":
            from peft import PrefixTuningConfig, get_peft_model as peft_get
            cfg = PrefixTuningConfig(
                num_virtual_tokens=num_prefix_tokens,
                encoder_hidden_size=inner.config.hidden_size,
            )
        else:
            from peft import LoraConfig, get_peft_model as peft_get
            cfg = LoraConfig(
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias=bias,
                use_dora=use_dora if method in ("dora",) else False,
                use_rslora=use_rslora,
                task_type="CAUSAL_LM",
            )

        patched = peft_get(inner, cfg)

        if isinstance(model, _WrappedVisionLocalModel):
            model._model = patched
            model._lora_config = cfg
            return model
        return patched

    @staticmethod
    def _default_target_modules(model, train_vision_encoder: bool) -> List[str]:
        """Auto-detect LoRA target modules from model architecture."""
        # Standard language decoder targets
        lang_targets = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        if not train_vision_encoder:
            return lang_targets

        # Vision encoder targets (ViT attention)
        vision_targets = [
            "vision_model.encoder.layers.*.self_attn.q_proj",
            "vision_model.encoder.layers.*.self_attn.k_proj",
            "vision_model.encoder.layers.*.self_attn.v_proj",
            "vision_model.encoder.layers.*.self_attn.out_proj",
        ]
        return lang_targets + vision_targets

    # ── train ────────────────────────────────────────────────────────────────

    @staticmethod
    def train(
        model: Any,
        processor: Any,
        dataset: Any,
        *,
        method: str = "qlora",
        output_dir: str = "./output",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 2e-4,
        warmup_ratio: float = 0.03,
        lr_scheduler_type: str = "cosine",
        save_steps: int = 100,
        logging_steps: int = 10,
        max_seq_length: int = 2048,
        fp16: bool = False,
        bf16: bool = True,
        # Vision-specific
        image_processor_key: str = "pixel_values",
        # Preference training (DPO/ORPO/SimPO/KTO)
        beta: float = 0.1,
        # Remote
        remote_hyperparameters: Optional[Dict] = None,
        **kwargs,
    ) -> Union[None, VisionRemoteJob]:
        """
        Fine-tune the model on the given dataset.

        For LOCAL mode: runs training synchronously, saves to output_dir.
        For REMOTE mode: dispatches to langtrain-server, returns VisionRemoteJob.

        Args:
            dataset: HuggingFace Dataset, DatasetDict, or path to JSONL file.
                     For SFT: must have 'image' and 'text' (or 'conversations') columns.
                     For DPO/ORPO: must have 'image', 'prompt', 'chosen', 'rejected'.
                     For KTO: must have 'image', 'prompt', 'completion', 'label'.
        """
        if isinstance(model, _WrappedVisionRemoteModel):
            return FastVisionModel._train_remote(
                model, processor, dataset, method=method,
                output_dir=output_dir, num_train_epochs=num_train_epochs,
                per_device_train_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                learning_rate=learning_rate, warmup_ratio=warmup_ratio,
                lr_scheduler_type=lr_scheduler_type, max_seq_length=max_seq_length,
                beta=beta, extra=remote_hyperparameters or {},
            )

        return FastVisionModel._train_local(
            model, processor, dataset,
            method=method, output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate, warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            save_steps=save_steps, logging_steps=logging_steps,
            max_seq_length=max_seq_length, fp16=fp16, bf16=bf16,
            image_processor_key=image_processor_key, beta=beta,
            **kwargs,
        )

    @staticmethod
    def _train_remote(
        model: _WrappedVisionRemoteModel,
        processor: Any,
        dataset: Any,
        *,
        method: str,
        output_dir: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        warmup_ratio: float,
        lr_scheduler_type: str,
        max_seq_length: int,
        beta: float,
        extra: Dict,
    ) -> VisionRemoteJob:
        import json
        import tempfile

        client = model._client

        # Serialize dataset to JSONL for upload
        dataset_id = FastVisionModel._upload_dataset(client, dataset)

        hyperparameters = {
            **model._hyperparameters,
            "method": method,
            "num_train_epochs": num_train_epochs,
            "per_device_train_batch_size": per_device_train_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "warmup_ratio": warmup_ratio,
            "lr_scheduler_type": lr_scheduler_type,
            "max_seq_length": max_seq_length,
            "beta": beta,
            **model._lora_config,
            **extra,
        }

        payload = {
            "model_id": model.model_id,
            "dataset_id": dataset_id,
            "method": method,
            "hyperparameters": hyperparameters,
            "modality": "vision",
        }

        response = client.create_job(payload)
        job_id = response["id"]
        logger.info(f"[Langvision] Remote job created: {job_id}")
        return VisionRemoteJob(job_id, client)

    @staticmethod
    def _upload_dataset(client: LangvisionServerClient, dataset: Any) -> str:
        import json
        import tempfile

        if isinstance(dataset, str):
            # Already a file path
            return client.upload_dataset(dataset)

        # Convert HF Dataset to JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name
            if hasattr(dataset, "__iter__") and hasattr(dataset, "__len__"):
                for row in dataset:
                    # Serialize non-serializable fields (PIL images) as metadata
                    serializable = {}
                    for k, v in row.items():
                        try:
                            json.dumps(v)
                            serializable[k] = v
                        except (TypeError, ValueError):
                            serializable[k] = str(type(v).__name__)
                    f.write(json.dumps(serializable) + "\n")
            else:
                raise ValueError("dataset must be a HuggingFace Dataset or path to a JSONL file")

        try:
            return client.upload_dataset(tmp_path)
        finally:
            os.unlink(tmp_path)

    @staticmethod
    def _train_local(
        model: Any,
        processor: Any,
        dataset: Any,
        *,
        method: str,
        output_dir: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        learning_rate: float,
        warmup_ratio: float,
        lr_scheduler_type: str,
        save_steps: int,
        logging_steps: int,
        max_seq_length: int,
        fp16: bool,
        bf16: bool,
        image_processor_key: str,
        beta: float,
        **kwargs,
    ):
        from transformers import TrainingArguments

        inner = model._model if isinstance(model, _WrappedVisionLocalModel) else model

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            save_steps=save_steps,
            logging_steps=logging_steps,
            fp16=fp16,
            bf16=bf16,
            remove_unused_columns=False,
            **{k: v for k, v in kwargs.items() if not k.startswith("_")},
        )

        if method in ("dpo",):
            return FastVisionModel._train_dpo(inner, processor, dataset, training_args, beta)
        elif method in ("orpo",):
            return FastVisionModel._train_orpo(inner, processor, dataset, training_args, beta)
        elif method in ("simpo",):
            return FastVisionModel._train_simpo(inner, processor, dataset, training_args, beta)
        elif method in ("kto",):
            return FastVisionModel._train_kto(inner, processor, dataset, training_args, beta)
        elif method in ("rlhf",):
            return FastVisionModel._train_rlhf(inner, processor, dataset, training_args)
        elif method in ("grpo",):
            return FastVisionModel._train_grpo(inner, processor, dataset, training_args)
        else:
            # SFT / LoRA / QLoRA / DoRA / prefix / IA³ — all use SFT trainer
            return FastVisionModel._train_sft(inner, processor, dataset, training_args, max_seq_length)

    @staticmethod
    def _train_sft(model, processor, dataset, training_args, max_seq_length: int):
        TrainerCls = _make_vision_sft_trainer_cls()
        trainer = TrainerCls(
            model=model,
            args=training_args,
            train_dataset=dataset if not hasattr(dataset, "get") else dataset.get("train", dataset),
            tokenizer=processor,
        )
        trainer.train()
        return trainer

    @staticmethod
    def _train_dpo(model, processor, dataset, training_args, beta: float):
        try:
            from trl import DPOTrainer, DPOConfig
            dpo_args = DPOConfig(
                **training_args.to_dict(),
                beta=beta,
            )
            trainer = DPOTrainer(
                model=model,
                args=dpo_args,
                train_dataset=dataset if not hasattr(dataset, "get") else dataset.get("train", dataset),
                tokenizer=processor,
            )
        except ImportError:
            raise ImportError("DPO training requires `trl>=0.7.0`. Install with: pip install trl")
        trainer.train()
        return trainer

    @staticmethod
    def _train_orpo(model, processor, dataset, training_args, beta: float):
        try:
            from trl import ORPOTrainer, ORPOConfig
            orpo_args = ORPOConfig(
                **training_args.to_dict(),
                beta=beta,
            )
            trainer = ORPOTrainer(
                model=model,
                args=orpo_args,
                train_dataset=dataset if not hasattr(dataset, "get") else dataset.get("train", dataset),
                tokenizer=processor,
            )
        except ImportError:
            raise ImportError("ORPO training requires `trl>=0.8.0`. Install with: pip install trl")
        trainer.train()
        return trainer

    @staticmethod
    def _train_simpo(model, processor, dataset, training_args, beta: float):
        try:
            from trl import CPOTrainer, CPOConfig
            simpo_args = CPOConfig(
                **training_args.to_dict(),
                beta=beta,
                loss_type="simpo",
            )
            trainer = CPOTrainer(
                model=model,
                args=simpo_args,
                train_dataset=dataset if not hasattr(dataset, "get") else dataset.get("train", dataset),
                tokenizer=processor,
            )
        except ImportError:
            raise ImportError("SimPO training requires `trl>=0.9.0`. Install with: pip install trl")
        trainer.train()
        return trainer

    @staticmethod
    def _train_kto(model, processor, dataset, training_args, beta: float):
        try:
            from trl import KTOTrainer, KTOConfig
            kto_args = KTOConfig(
                **training_args.to_dict(),
                beta=beta,
            )
            trainer = KTOTrainer(
                model=model,
                args=kto_args,
                train_dataset=dataset if not hasattr(dataset, "get") else dataset.get("train", dataset),
                tokenizer=processor,
            )
        except ImportError:
            raise ImportError("KTO training requires `trl>=0.9.0`. Install with: pip install trl")
        trainer.train()
        return trainer

    @staticmethod
    def _train_rlhf(model, processor, dataset, training_args):
        try:
            from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
            ppo_config = PPOConfig(
                output_dir=training_args.output_dir,
                learning_rate=training_args.learning_rate,
                batch_size=training_args.per_device_train_batch_size,
            )
            policy = AutoModelForCausalLMWithValueHead.from_pretrained(model.config.name_or_path)
            trainer = PPOTrainer(config=ppo_config, model=policy, tokenizer=processor)
            logger.info("[Langvision] RLHF/PPO trainer initialized — implement reward loop in callback")
        except ImportError:
            raise ImportError("RLHF training requires `trl>=0.7.0`. Install with: pip install trl")
        return trainer

    @staticmethod
    def _train_grpo(model, processor, dataset, training_args):
        try:
            from trl import GRPOTrainer, GRPOConfig
            grpo_config = GRPOConfig(
                **training_args.to_dict(),
            )
            trainer = GRPOTrainer(
                model=model,
                args=grpo_config,
                train_dataset=dataset if not hasattr(dataset, "get") else dataset.get("train", dataset),
                tokenizer=processor,
            )
        except ImportError:
            raise ImportError("GRPO training requires `trl>=0.12.0`. Install with: pip install trl")
        trainer.train()
        return trainer

    # ── generate ─────────────────────────────────────────────────────────────

    @staticmethod
    def generate(
        model: Any,
        processor: Any,
        images: Any,
        prompt: str,
        *,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
        **kwargs,
    ) -> str:
        """
        Run vision inference on a single image + text prompt.

        Args:
            images: PIL.Image or list of PIL.Images
            prompt: Text prompt / instruction

        Returns:
            Generated text string
        """
        if isinstance(model, _WrappedVisionRemoteModel):
            raise ValueError(
                "generate() is not supported in remote mode. "
                "Deploy the model via the langtrain-server deployment API."
            )

        import torch

        inner = model._model if isinstance(model, _WrappedVisionLocalModel) else model

        # Handle both list and single image
        if not isinstance(images, list):
            images = [images]

        inputs = processor(images=images, text=prompt, return_tensors="pt").to(inner.device)

        with torch.no_grad():
            output_ids = inner.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                **kwargs,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_len:]
        return processor.decode(generated_ids[0], skip_special_tokens=True)

    # ── save_pretrained ──────────────────────────────────────────────────────

    @staticmethod
    def save_pretrained(
        model: Any,
        output_dir: str,
        *,
        processor: Any = None,
        merge_before_save: bool = False,
        push_to_hub: bool = False,
        hub_model_id: Optional[str] = None,
        token: Optional[str] = None,
    ):
        """
        Save the fine-tuned model (and optionally processor) to disk.

        Args:
            merge_before_save: Merge LoRA weights into the base model before saving.
                               Produces a standalone model (no PEFT library needed for inference).
            push_to_hub: Push to HuggingFace Hub after saving.
        """
        if isinstance(model, _WrappedVisionRemoteModel):
            raise ValueError("Use job.download() to retrieve a remotely trained model.")

        inner = model._model if isinstance(model, _WrappedVisionLocalModel) else model

        os.makedirs(output_dir, exist_ok=True)

        if merge_before_save:
            try:
                inner = inner.merge_and_unload()
                logger.info("[Langvision] LoRA weights merged into base model")
            except AttributeError:
                logger.info("[Langvision] Model has no LoRA adapter — saving as-is")

        inner.save_pretrained(output_dir)

        if processor is not None:
            processor.save_pretrained(output_dir)

        if push_to_hub and hub_model_id:
            inner.push_to_hub(hub_model_id, token=token)
            if processor is not None:
                processor.push_to_hub(hub_model_id, token=token)
            logger.info(f"[Langvision] Model pushed to Hub: {hub_model_id}")

    # ── for_inference ────────────────────────────────────────────────────────

    @staticmethod
    def for_inference(model: Any) -> Any:
        """
        Switch the model to inference mode.
        Disables dropout, gradient computation, etc.
        """
        if isinstance(model, _WrappedVisionRemoteModel):
            return model
        inner = model._model if isinstance(model, _WrappedVisionLocalModel) else model
        inner.eval()
        try:
            for param in inner.parameters():
                param.requires_grad_(False)
        except Exception:
            pass
        return model
