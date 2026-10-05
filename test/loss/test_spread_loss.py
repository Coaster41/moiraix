import pytest
import torch

from moiraix.loss.packed.quantile import PackedQuantileMTPLoss, PackedQuantileSpreadLoss


@pytest.mark.parametrize("shift", [False, True])
def test_spread_loss_matches_mtp_dense(shift):
    """Sparse PackedQuantileSpreadLoss must agree with the trusted dense
    PackedQuantileMTPLoss when evaluated at every valid anchor position with
    nst == npt (HANDOFF_SPREAD_HEAD §8 test 6 -- the correctness anchor)."""
    torch.manual_seed(0)
    B, seq_len, nst, Q, P = 1, 10, 3, 2, 2
    quantile_levels = (0.3, 0.7)

    sample_id = torch.ones(B, seq_len, dtype=torch.long)
    variate_id = torch.zeros(B, seq_len, dtype=torch.long)
    prediction_mask = torch.ones(B, seq_len, dtype=torch.bool)
    observed_mask = torch.ones(B, seq_len, P, dtype=torch.bool)
    target = torch.randn(B, seq_len, P)
    pred_dense = torch.randn(B, seq_len, nst * Q * P)

    mtp_loss = PackedQuantileMTPLoss(quantile_levels=quantile_levels, shift=shift)
    dense = mtp_loss(
        pred=pred_dense,
        target=target,
        prediction_mask=prediction_mask,
        observed_mask=observed_mask,
        sample_id=sample_id,
        variate_id=variate_id,
    )

    W = seq_len - nst + 1
    off = 1 if shift else 0
    positions = torch.arange(W - off).expand(B, W - off)
    spread_loss = PackedQuantileSpreadLoss(quantile_levels=quantile_levels, shift=shift)
    sparse = spread_loss(
        pred=pred_dense[:, : W - off, :],
        target=target,
        positions=positions,
        prediction_mask=prediction_mask,
        observed_mask=observed_mask,
        sample_id=sample_id,
        variate_id=variate_id,
    )

    torch.testing.assert_close(sparse, dense)


def test_spread_loss_ignores_cross_series_windows():
    """A window straddling a packing boundary must contribute zero -- only the
    offsets that share the anchor's (sample, variate) id count (HANDOFF §8
    test 7; same rationale as FIX B in HANDOFF_MOIRAI2_PARITY.md)."""
    nst, P = 3, 1
    quantile_levels = (0.5,)

    sample_id = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2]])
    variate_id = torch.zeros_like(sample_id)
    prediction_mask = torch.ones_like(sample_id, dtype=torch.bool)
    observed_mask = torch.ones(1, 8, P, dtype=torch.bool)

    target = torch.zeros(1, 8, P)
    target[0, 2, 0] = 1.0
    target[0, 4, 0] = 3.0
    target[0, 5, 0] = 5.0
    target[0, 6, 0] = 7.0

    positions = torch.tensor([[2, 4]])
    # anchor=2 straddles the sample boundary (window covers positions 2,3,4);
    # anchor=4 lies entirely within sample 2 (window covers positions 4,5,6).
    pred = torch.tensor([[[3.0, 100.0, 100.0], [3.0, 7.0, 7.0]]])

    loss_fn = PackedQuantileSpreadLoss(quantile_levels=quantile_levels, shift=False)
    result = loss_fn(
        pred=pred,
        target=target,
        positions=positions,
        prediction_mask=prediction_mask,
        observed_mask=observed_mask,
        sample_id=sample_id,
        variate_id=variate_id,
    )

    # anchor=2: only tok0 (position 2 itself) is valid -> |3-1|*0.5 = 1.0
    # anchor=4: all 3 toks valid -> |3-3|*0.5 + |7-5|*0.5 + |7-7|*0.5 = 1.0
    # total = 2.0 over 4 active entries (1 + 3) -> mean 0.5
    torch.testing.assert_close(result, torch.tensor(0.5))
