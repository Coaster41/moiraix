"""arcsinh-after-zscore normalization (exp-014): the input/loss space is
arcsinh((x - loc)/scale) and inference inverts with sinh before
de-normalizing, so forecasts stay in data space."""

import torch

from moiraix.model import MoiraiXModule


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


def test_scaled_target_is_arcsinh_of_zscore():
    m = make_module().eval()
    inputs = make_inputs()
    with torch.no_grad():
        _, st_plain = m(*inputs, training_mode=True)
        m.arcsinh_norm = True
        _, st_arcsinh = m(*inputs, training_mode=True)
    torch.testing.assert_close(st_arcsinh, torch.asinh(st_plain))


def test_inference_inverts_with_sinh_back_to_data_space():
    m = make_module(arcsinh_norm=True).eval()
    (
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
    ) = make_inputs()
    with torch.no_grad():
        preds, _ = m(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            training_mode=True,
        )
        out = m(
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
            prediction_mask,
            training_mode=False,
        )
        # Same context-only stats the forward pass uses in eval mode.
        loc, scale = m.scaler(
            target,
            observed_mask * ~prediction_mask.unsqueeze(-1),
            sample_id,
            variate_id,
        )
    torch.testing.assert_close(out, torch.sinh(preds) * scale + loc)


def test_flag_defaults_off_and_off_path_unchanged():
    m = make_module()
    assert m.arcsinh_norm is False
    m = m.eval()
    inputs = make_inputs()
    with torch.no_grad():
        preds, _ = m(*inputs, training_mode=True)
        out = m(*inputs, training_mode=False)
        loc, scale = m.scaler(
            inputs[0],
            inputs[1] * ~inputs[5].unsqueeze(-1),
            inputs[2],
            inputs[4],
        )
    torch.testing.assert_close(out, preds * scale + loc)
