"""Named configurations of :class:`MoiraiXModule` / :class:`MoiraiXForecast`.

The two original model families are presets of the unified model:

- ``"decoder"`` — causal next-token decoder (GPT-style; the former ``moiraic``):
  ``causal=True, mask_inputs=False, predict_next=True``.
- ``"encoder"`` — masked-reconstruction encoder (BERT-style; the former ``moiraie``):
  ``mask_inputs=True, predict_next=False``; ``causal`` is read from the checkpoint
  and defaults to ``False`` (``True`` for the causal masked-reconstruction runs).

New checkpoints serialize every flag in ``config.json`` and load directly with
``MoiraiXModule.from_pretrained``. ``load_module(path, preset=...)`` exists for
checkpoints trained before the flags existed (e.g. ``moiraic_training_11``,
``moiraie_training_7``): it pins the preset's flags and, like the old preset
classes, ignores any later-added config keys (they fall back to their defaults).
"""

import inspect
import json
import os
from typing import Any, Optional

from huggingface_hub import hf_hub_download

from .module import MoiraiXModule

PRESETS: dict[str, dict[str, Any]] = {
    "decoder": dict(causal=True, mask_inputs=False, predict_next=True),
    "encoder": dict(mask_inputs=True, predict_next=False),
}

# Flags whose value comes from the checkpoint if present, else this default.
PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "decoder": {},
    "encoder": dict(causal=False),
}

# Decode defaults of the former MoiraicForecast / MoiraieForecast.
FORECAST_DEFAULTS: dict[str, dict[str, Any]] = {
    "decoder": dict(ar_method="branch"),
    "encoder": dict(use_cache=False, ar_method=None),
}

# Config keys the pre-flag preset modules understood; anything else in an old
# config.json was swallowed by their ``**kwargs`` and left at its default.
_PRESET_KEYS = {
    "d_model",
    "d_ff",
    "num_layers",
    "patch_size",
    "max_seq_len",
    "attn_dropout_p",
    "dropout_p",
    "scaling",
    "num_predict_token",
    "quantile_levels",
    "min_scale",
    "train_scale_full_observed",
    "causal",
    "mask_inputs",
    "predict_next",
}


def _read_config(path: str) -> dict:
    if os.path.isdir(path):
        config_file = os.path.join(path, "config.json")
    else:
        config_file = hf_hub_download(repo_id=path, filename="config.json")
    with open(config_file) as f:
        return json.load(f)


def load_module(
    path: str, preset: Optional[str] = None, **kwargs: Any
) -> MoiraiXModule:
    """Load a :class:`MoiraiXModule` checkpoint, optionally under a named preset."""
    if preset is None:
        return MoiraiXModule.from_pretrained(path, **kwargs)
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; expected one of {list(PRESETS)}")

    config = _read_config(path)
    overrides = {
        name: param.default
        for name, param in inspect.signature(MoiraiXModule.__init__).parameters.items()
        if name not in _PRESET_KEYS and param.default is not inspect.Parameter.empty
    }
    overrides.update({k: config.get(k, v) for k, v in PRESET_DEFAULTS[preset].items()})
    overrides.update(PRESETS[preset])
    overrides.update(kwargs)
    return MoiraiXModule.from_pretrained(path, **overrides)
