"""Long-horizon quantile spread head (TimesFM 2.5's continuous quantile head,
ported per .claude/HANDOFF_SPREAD_HEAD.md). ``num_spread_token=0`` (default)
must leave every existing surface byte-identical."""

import json

import numpy as np
import torch

from moiraix.loss.packed import PackedQuantileSpreadLoss
from moiraix.model import MoiraiXForecast, MoiraiXModule, MoiraiXPretrain


def make_module(**kwargs):
    torch.manual_seed(0)
    return MoiraiXModule(
        d_model=64,
        d_ff=64,
        num_layers=1,
        patch_size=4,
        max_seq_len=32,
        attn_dropout_p=0.0,
        dropout_p=0.0,
        **kwargs,
    )


def make_inputs(batch=2, seq=8, patch=4):
    g = torch.Generator().manual_seed(1)
    target = torch.randn(batch, seq, patch, generator=g) * 5 + 1
    observed_mask = torch.ones(batch, seq, patch, dtype=torch.bool)
    sample_id = torch.ones(batch, seq, dtype=torch.long)
    time_id = torch.arange(seq).expand(batch, seq).contiguous()
    variate_id = torch.zeros(batch, seq, dtype=torch.long)
    prediction_mask = torch.zeros(batch, seq, dtype=torch.bool)
    prediction_mask[:, -2:] = True
    return target, observed_mask, sample_id, time_id, variate_id, prediction_mask


def test_no_head_by_default():
    m = make_module()
    assert m.num_spread_token == 0
    assert not hasattr(m, "out_proj_spread")
    inputs = make_inputs()
    try:
        m(*inputs, training_mode=False, return_spread=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("expected return_spread=True to raise without a spread head")


def test_state_dict_keys_unchanged():
    m0 = make_module()
    m_spread = make_module(num_spread_token=4)
    added = set(m_spread.state_dict()) - set(m0.state_dict())
    expected = {k for k in m_spread.state_dict() if k.startswith("out_proj_spread.")}
    assert added == expected
    assert added, "expected the spread head to add new state-dict keys"
    assert set(m0.state_dict()) <= set(m_spread.state_dict())


def test_spread_shape_and_gather():
    m = make_module(num_spread_token=4).eval()
    inputs = make_inputs(batch=2, seq=8, patch=4)
    with torch.no_grad():
        dense = m(*inputs, training_mode=False, return_spread=True)
        assert isinstance(dense, tuple) and len(dense) == 2
        _, spread_dense = dense

        Q = m.num_quantiles
        P = m.patch_size
        assert spread_dense.shape == (2, 8, 4 * Q * P)

        positions = torch.tensor([0, 1, 2]).expand(2, 3)
        _, spread_sparse = m(
            *inputs, training_mode=False, return_spread=True, spread_positions=positions
        )
        assert spread_sparse.shape == (2, 3, 4 * Q * P)
        torch.testing.assert_close(spread_sparse, spread_dense[:, :3])


def test_graft_preserves_pretrained_weights(tmp_path):
    """Mode A (HANDOFF §7.A.1/§8 test 8): grafting a spread head onto a
    checkpoint saved without one must add exactly the new head's keys and
    leave every inherited tensor byte-identical -- this is the check that
    catches a silent ``strict=False`` mis-load."""
    base = make_module()
    base.save_pretrained(tmp_path)

    grafted = MoiraiXModule.from_pretrained(str(tmp_path), num_spread_token=2)

    base_sd, grafted_sd = base.state_dict(), grafted.state_dict()
    added = set(grafted_sd) - set(base_sd)
    expected = {k for k in grafted_sd if k.startswith("out_proj_spread.")}
    assert added == expected
    assert not (set(base_sd) - set(grafted_sd)), "graft dropped pretrained keys"
    for k in base_sd:
        torch.testing.assert_close(grafted_sd[k], base_sd[k], rtol=0, atol=0)


def test_graft_config_roundtrip(tmp_path):
    """Mode A (HANDOFF §8 test 9): re-saving a grafted module must persist
    ``num_spread_token`` in config.json, or a finetuned checkpoint would
    silently evaluate with the head disabled."""
    base = make_module()
    base.save_pretrained(tmp_path)
    grafted = MoiraiXModule.from_pretrained(str(tmp_path), num_spread_token=2)
    grafted.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    assert config["num_spread_token"] == 2


def test_freeze_pattern_spread_head():
    """Mode A (HANDOFF §7.A.2/§8 test 10): ``freeze_pattern='spread_head'``
    unlocks only the spread head; ``configure_optimizers`` param groups must
    total exactly its element count; two training steps must not move any
    non-spread tensor."""
    module = make_module(num_spread_token=2)
    predict_next = module.predict_next

    pretrain = MoiraiXPretrain(
        module=module,
        min_patches=1,
        min_mask_ratio=0.5,
        max_mask_ratio=0.5,
        max_dim=8,
        num_training_steps=10,
        num_warmup_steps=1,
        spread_loss_func=PackedQuantileSpreadLoss(shift=predict_next),
        num_spread_positions=2,
        freeze_pattern="spread_head",
        train_spread_only=True,
    )

    for pn, p in pretrain.module.named_parameters():
        expected = pn.startswith("out_proj_spread.")
        assert p.requires_grad == expected, pn

    optim = pretrain.configure_optimizers()
    optim_numel = sum(
        p.numel() for group in optim["optimizer"].param_groups for p in group["params"]
    )
    spread_numel = sum(p.numel() for p in module.out_proj_spread.parameters())
    assert optim_numel == spread_numel

    before = {
        k: v.clone()
        for k, v in pretrain.module.state_dict().items()
        if not k.startswith("out_proj_spread.")
    }

    batch = {
        "target": torch.randn(2, 8, 4),
        "observed_mask": torch.ones(2, 8, 4, dtype=torch.bool),
        "sample_id": torch.ones(2, 8, dtype=torch.long),
        "time_id": torch.arange(8).expand(2, 8).contiguous(),
        "variate_id": torch.zeros(2, 8, dtype=torch.long),
        "prediction_mask": torch.zeros(2, 8, dtype=torch.bool),
    }
    batch["prediction_mask"][:, -2:] = True

    optimizer = torch.optim.SGD(
        [p for p in pretrain.module.parameters() if p.requires_grad], lr=0.1
    )
    for _ in range(2):
        optimizer.zero_grad()
        loss = pretrain.training_step(batch, 0)
        loss.backward()
        optimizer.step()

    after = pretrain.module.state_dict()
    for k, v in before.items():
        torch.testing.assert_close(after[k], v, msg=f"non-spread tensor {k} moved")


def _make_forecast_pair(
    num_spread_token, prediction_length, context_length=32, patch_size=4, **fc_kwargs
):
    """A causal decoder (predict_next=True) with branch AR, wrapping the
    *same* module weights, once with the spread head applied and once
    without -- isolating the overlay's effect from anything else."""
    torch.manual_seed(0)
    module = MoiraiXModule(
        d_model=64,
        d_ff=64,
        num_layers=1,
        patch_size=patch_size,
        max_seq_len=128,
        attn_dropout_p=0.0,
        dropout_p=0.0,
        causal=True,
        mask_inputs=False,
        predict_next=True,
        num_predict_token=1,
        num_spread_token=num_spread_token,
    ).eval()
    common = dict(
        prediction_length=prediction_length,
        target_dim=1,
        feat_dynamic_real_dim=0,
        past_feat_dynamic_real_dim=0,
        context_length=context_length,
        module=module,
        ar_method="branch",
        **fc_kwargs,
    )
    fc_spread = MoiraiXForecast(**common, use_spread_head=True, spread_fix_crossing=False)
    fc_nospread = MoiraiXForecast(**common, use_spread_head=False)
    return fc_spread, fc_nospread


def _predict(fc, context_length, seed=0):
    rng = np.random.default_rng(seed)
    past = [rng.standard_normal(context_length).astype("float32")]
    return fc.predict(past)


def test_median_is_preserved():
    """HANDOFF §8 test 4: the spread overlay is identically 0 at the median
    quantile by construction, so enabling the head must not move the median
    forecast at all (run with spread_fix_crossing=False -- sorting is a
    no-op on a monotone head but we don't want it masking a real bug)."""
    fc_spread, fc_nospread = _make_forecast_pair(num_spread_token=4, prediction_length=64)
    out_spread = _predict(fc_spread, context_length=32)
    out_nospread = _predict(fc_nospread, context_length=32)

    median_idx = fc_spread.module.num_quantiles // 2
    np.testing.assert_array_equal(
        out_spread[:, median_idx], out_nospread[:, median_idx]
    )
    # sanity: the head is actually doing something to the non-median channels
    assert not np.allclose(out_spread, out_nospread)


def test_horizon_beyond_spread_falls_back():
    """HANDOFF §8 test 5: horizon tokens beyond the spread head's reach
    (``num_spread_token`` patches) must fall back to the untouched AR
    quantiles, while the tokens within reach are spread-adjusted."""
    patch_size = 4
    num_spread_token = 4  # covers 16 steps
    prediction_length = 64  # 16 patches -- well beyond the spread head's reach
    fc_spread, fc_nospread = _make_forecast_pair(
        num_spread_token=num_spread_token,
        prediction_length=prediction_length,
        patch_size=patch_size,
    )
    out_spread = _predict(fc_spread, context_length=32)
    out_nospread = _predict(fc_nospread, context_length=32)

    tail_start = num_spread_token * patch_size
    np.testing.assert_array_equal(
        out_spread[:, :, tail_start:], out_nospread[:, :, tail_start:]
    )
    assert not np.allclose(
        out_spread[:, :, :tail_start], out_nospread[:, :, :tail_start]
    )
