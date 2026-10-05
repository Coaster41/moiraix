"""Robust epoch aggregates for the pretrain metrics.

The batch mean produced by ``self.log(on_epoch=True)`` is hijacked by a
sub-percent tail of batches (``scale`` floors at ``sqrt(min_scale)``, so a
near-constant context followed by a burst in the prediction window gives a
normalized target of order 1e4-1e5). These tests pin the median / trimmed-mean
aggregates that make the curve readable, and pin that the training objective
itself is untouched.
"""

import torch

from moiraix.loss.packed import PackedNMAELoss
from moiraix.model import MoiraiXModule, MoiraiXPretrain

# The measured shape of the pathology on a real 30M run: median 0.36, mean
# 20.1, max 2.4e5, with ~0.9% of batches carrying ~90% of the mean.
BULK = [0.3 + 0.001 * i for i in range(991)]
TAIL = [1e4, 2e4, 3e4, 5e4, 8e4, 1e5, 1.5e5, 2e5, 2.4e5]


def make_pretrain(**kwargs):
    torch.manual_seed(0)
    module = MoiraiXModule(
        d_model=64,
        d_ff=64,
        num_layers=1,
        patch_size=4,
        max_seq_len=32,
        attn_dropout_p=0.0,
        dropout_p=0.0,
    )
    defaults = dict(
        module=module,
        min_patches=1,
        min_mask_ratio=0.5,
        max_mask_ratio=0.5,
        max_dim=8,
        num_training_steps=10,
        num_warmup_steps=1,
    )
    defaults.update(kwargs)
    return MoiraiXPretrain(**defaults)


def capture(pretrain):
    """Record what would be logged, since self.log is a no-op with no Trainer."""
    logged = {}

    def fake_log(name, value, **kwargs):
        logged[name] = float(value)

    pretrain.log = fake_log
    return logged


def make_batch(batch=2, seq=8, patch=4):
    g = torch.Generator().manual_seed(1)
    prediction_mask = torch.zeros(batch, seq, dtype=torch.bool)
    prediction_mask[:, -2:] = True
    return dict(
        target=torch.randn(batch, seq, patch, generator=g) * 5 + 1,
        observed_mask=torch.ones(batch, seq, patch, dtype=torch.bool),
        sample_id=torch.ones(batch, seq, dtype=torch.long),
        time_id=torch.arange(seq).expand(batch, seq).contiguous(),
        variate_id=torch.zeros(batch, seq, dtype=torch.long),
        prediction_mask=prediction_mask,
    )


# --------------------------------------------------------------------------- #
# The aggregates themselves                                                   #
# --------------------------------------------------------------------------- #


def test_trimmed_mean_removes_the_tail_that_hijacks_the_mean():
    pretrain = make_pretrain()
    logged = capture(pretrain)
    values = [torch.tensor(v) for v in BULK + TAIL]

    pretrain._log_robust_epoch({"train/L": values})

    plain_mean = sum(BULK + TAIL) / len(BULK + TAIL)
    assert plain_mean > 100, "fixture should reproduce a hijacked mean"

    # frac=0.01 of 1000 drops 10 values: all 9 of TAIL plus the largest of BULK
    expected_trimmed = sum(sorted(BULK)[:-1]) / (len(BULK) - 1)
    assert logged["train/L_epoch_median"] < 1.0
    assert abs(logged["train/L_epoch_trimmed"] - expected_trimmed) < 1e-3
    # the whole point: the robust aggregates sit with the bulk, not the mean
    assert logged["train/L_epoch_trimmed"] < plain_mean / 100


def test_median_is_insensitive_to_the_tail_size():
    pretrain = make_pretrain()
    logged_small = capture(pretrain)
    pretrain._log_robust_epoch({"m": [torch.tensor(v) for v in BULK + TAIL]})
    with_tail = logged_small["m_epoch_median"]

    pretrain2 = make_pretrain()
    logged_none = capture(pretrain2)
    pretrain2._log_robust_epoch({"m": [torch.tensor(v) for v in BULK]})
    without_tail = logged_none["m_epoch_median"]

    assert abs(with_tail - without_tail) < 0.01


def test_trimmed_mean_is_skipped_when_nothing_would_be_dropped():
    """A short validation loader has fewer than 1/frac batches, so no trim is
    possible and reporting a 'trimmed' value equal to the mean would be a lie."""
    pretrain = make_pretrain(robust_epoch_trim_frac=0.01)
    logged = capture(pretrain)
    pretrain._log_robust_epoch({"val/L": [torch.tensor(v) for v in BULK[:10]]})
    assert "val/L_epoch_median" in logged
    assert "val/L_epoch_trimmed" not in logged


def test_trim_frac_zero_logs_median_only():
    pretrain = make_pretrain(robust_epoch_trim_frac=0.0)
    logged = capture(pretrain)
    pretrain._log_robust_epoch({"x": [torch.tensor(v) for v in BULK + TAIL]})
    assert "x_epoch_median" in logged
    assert "x_epoch_trimmed" not in logged


def test_empty_buffer_logs_nothing():
    pretrain = make_pretrain()
    logged = capture(pretrain)
    pretrain._log_robust_epoch({"x": []})
    assert logged == {}


def test_larger_trim_frac_drops_more():
    pretrain = make_pretrain(robust_epoch_trim_frac=0.5)
    logged = capture(pretrain)
    pretrain._log_robust_epoch({"x": [torch.tensor(v) for v in BULK + TAIL]})
    # dropping the top half must land below the median of the full set
    assert logged["x_epoch_trimmed"] < logged["x_epoch_median"]


# --------------------------------------------------------------------------- #
# Wiring through the steps and hooks                                          #
# --------------------------------------------------------------------------- #


def test_training_step_buffers_and_epoch_end_clears():
    pretrain = make_pretrain()
    logged = capture(pretrain)
    batch = make_batch()

    for _ in range(3):
        pretrain.training_step(batch, 0)
    assert len(pretrain._robust_train) == 3

    pretrain.on_train_epoch_end()
    name = f"train/{pretrain.hparams.loss_func.__class__.__name__}_epoch_median"
    assert name in logged
    assert pretrain._robust_train == [], "buffer must not leak across epochs"


def test_validation_step_buffers_loss_and_metric():
    # val_metric defaults to None; the configs set it, so pass one explicitly
    # to cover the per-metric collection branch.
    pretrain = make_pretrain(val_metric=PackedNMAELoss())
    logged = capture(pretrain)
    pretrain.validation_step(make_batch(), 0)

    keys = set(pretrain._robust_val)
    assert f"val/{pretrain.hparams.loss_func.__class__.__name__}" in keys
    assert "val/PackedNMAELoss" in keys, keys

    pretrain.on_validation_epoch_end()
    assert pretrain._robust_val == {}
    assert any(k.endswith("_epoch_median") for k in logged), logged


def test_buffered_value_matches_the_returned_loss():
    """The aggregate must describe the loss actually optimized, not a copy that
    could drift from it."""
    pretrain = make_pretrain()
    capture(pretrain)
    loss = pretrain.training_step(make_batch(), 0)
    assert len(pretrain._robust_train) == 1
    assert torch.equal(pretrain._robust_train[0], loss.detach())


def test_buffered_values_are_detached():
    """Holding graph references for a whole epoch would retain every batch's
    activations."""
    pretrain = make_pretrain()
    capture(pretrain)
    pretrain.training_step(make_batch(), 0)
    assert not pretrain._robust_train[0].requires_grad
    assert pretrain._robust_train[0].grad_fn is None


def test_objective_is_unchanged():
    """Adding the aggregates must not perturb the optimized loss at all."""
    a = make_pretrain()
    capture(a)
    loss_a = a.training_step(make_batch(), 0)

    b = make_pretrain(robust_epoch_trim_frac=0.0)
    capture(b)
    loss_b = b.training_step(make_batch(), 0)

    assert torch.equal(loss_a, loss_b)
    assert loss_a.requires_grad, "training_step must still return a live graph"


def test_hparam_default_is_one_percent():
    assert make_pretrain().hparams.robust_epoch_trim_frac == 0.01


# --------------------------------------------------------------------------- #
# Under a real Trainer                                                        #
# --------------------------------------------------------------------------- #


def test_keys_reach_the_logger_under_a_real_trainer():
    """The monkeypatched tests above pin the arithmetic; this one pins that
    Lightning calls the hooks and accepts ``self.log`` from an epoch-end hook
    (logging outside *_step is restricted, so this is not a given)."""
    import lightning as L
    from torch.utils.data import DataLoader, Dataset

    batch = make_batch()

    class Repeat(Dataset):
        def __len__(self):
            return 8

        def __getitem__(self, i):
            return batch

    loader = DataLoader(Repeat(), batch_size=1, collate_fn=lambda xs: xs[0])
    # log_on_step=True mirrors every real pretrain config, and is what makes
    # Lightning suffix the mean with "_epoch" -- so the robust keys sit
    # directly beside it as "_epoch_median" / "_epoch_trimmed".
    pretrain = make_pretrain(
        val_metric=PackedNMAELoss(),
        robust_epoch_trim_frac=0.25,
        log_on_step=True,
    )
    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(pretrain, train_dataloaders=loader, val_dataloaders=loader)

    logged = set(trainer.logged_metrics)
    loss_name = pretrain.hparams.loss_func.__class__.__name__
    assert f"train/{loss_name}_epoch_median" in logged, logged
    assert f"train/{loss_name}_epoch_trimmed" in logged, logged
    assert f"val/{loss_name}_epoch_median" in logged, logged
    assert "val/PackedNMAELoss_epoch_median" in logged, logged
    # the pre-existing mean and step series must still be there alongside
    assert f"train/{loss_name}_epoch" in logged, logged
    assert f"train/{loss_name}_step" in logged, logged
