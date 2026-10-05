"""Loading pre-flag checkpoints (former moiraic / moiraie) through presets."""

import json

import pytest
import torch

from moiraix.model import MoiraiXModule, load_module

BASE_CONFIG = dict(
    d_model=64,
    d_ff=64,
    num_layers=1,
    patch_size=4,
    max_seq_len=32,
    attn_dropout_p=0.0,
    dropout_p=0.0,
    min_scale=0.1,
)


def save_old_checkpoint(tmp_path, config, **flags):
    """Write a checkpoint whose config.json lacks the unified flags."""
    torch.manual_seed(0)
    MoiraiXModule(**BASE_CONFIG, **flags).save_pretrained(tmp_path)
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


def test_decoder_preset(tmp_path):
    path = save_old_checkpoint(
        tmp_path, BASE_CONFIG, causal=True, mask_inputs=False, predict_next=True
    )
    m = load_module(path, "decoder")
    assert (m.causal, m.mask_inputs, m.predict_next) == (True, False, True)
    assert not hasattr(m, "mask_encoding") or m.mask_encoding is None


def test_encoder_preset_reads_causal_from_config(tmp_path):
    path = save_old_checkpoint(tmp_path, BASE_CONFIG)
    m = load_module(path, "encoder")
    assert (m.causal, m.mask_inputs, m.predict_next) == (False, True, False)

    causal_dir = tmp_path / "causal"
    causal_dir.mkdir()
    path = save_old_checkpoint(causal_dir, {**BASE_CONFIG, "causal": True}, causal=True)
    assert load_module(path, "encoder").causal is True


def test_preset_ignores_later_config_keys(tmp_path):
    # The old preset classes swallowed unknown kwargs, so a stray newer key
    # must fall back to its default rather than change behavior.
    path = save_old_checkpoint(tmp_path, {**BASE_CONFIG, "arcsinh_norm": True})
    assert load_module(path, "encoder").arcsinh_norm is False
    assert load_module(path).arcsinh_norm is True


def test_unknown_preset(tmp_path):
    path = save_old_checkpoint(tmp_path, BASE_CONFIG)
    with pytest.raises(ValueError):
        load_module(path, "nope")
