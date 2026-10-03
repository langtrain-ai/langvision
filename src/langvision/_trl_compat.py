"""
Work with both older and current TRL releases.

TRL renamed things that langtune relies on:
  - trainers take ``processing_class=`` instead of ``tokenizer=`` (0.12+)
  - SFTConfig's ``max_seq_length`` became ``max_length``
  - ORPO and CPO (used for SimPO) moved to ``trl.experimental`` (1.x)
These helpers pick whichever the installed version expects.
Shared with langtune (langtune/_trl_compat.py); keep the two in step.
"""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import logging
from typing import Any, Tuple

logger = logging.getLogger(__name__)

_RENAMED_FIELDS = {"max_seq_length": "max_length"}


def load_trainer(name: str) -> Tuple[type, type]:
    """Return (Trainer, Config) for "SFT", "DPO", "KTO", "ORPO" or "CPO"."""
    for module in ("trl", f"trl.experimental.{name.lower()}"):
        try:
            mod = importlib.import_module(module)
            return getattr(mod, f"{name}Trainer"), getattr(mod, f"{name}Config")
        except (ImportError, AttributeError):
            continue
    raise ImportError(f"Your TRL version has no {name}Trainer. Run: pip install -U trl")


def make_config(config_cls: type, **kwargs: Any) -> Any:
    """Build a TRL config, renaming or dropping arguments this TRL version doesn't know."""
    fields = {f.name for f in dataclasses.fields(config_cls)}
    out = {}
    for key, value in kwargs.items():
        if key not in fields and _RENAMED_FIELDS.get(key) in fields:
            key = _RENAMED_FIELDS[key]
        if key in fields:
            out[key] = value
        else:
            logger.debug("[Langvision] %s has no %r; ignoring it", config_cls.__name__, key)
    return config_cls(**out)


def make_trainer(trainer_cls: type, tokenizer: Any, **kwargs: Any) -> Any:
    """Build a TRL trainer, passing the tokenizer under the name this TRL version expects."""
    params = inspect.signature(trainer_cls.__init__).parameters
    key = "processing_class" if "processing_class" in params else "tokenizer"
    return trainer_cls(**kwargs, **{key: tokenizer})
