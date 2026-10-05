"""Graft a long-horizon quantile-spread head onto an existing MoiraiX
checkpoint (HANDOFF_SPREAD_HEAD.md §7.A.1).

``PyTorchModelHubMixin.from_pretrained`` defaults to ``strict=False``, so a
checkpoint missing ``out_proj_spread.*`` keys loads silently -- no error, no
warning. That's what makes grafting work, and it's also what would let a
typo'd key name or a stale ``config.json`` ship a randomly-initialized head
into a training run without a single symptom. Run this verifier before every
Mode-A training job (see moiraix_spread_headonly.yaml).

Usage:
    python scripts/graft_spread_head.py CKPT_DIR --num-spread-token 16
"""

import argparse

import torch

from moiraix.model import MoiraiXModule


def graft(ckpt: str, num_spread_token: int, **overrides) -> MoiraiXModule:
    base = MoiraiXModule.from_pretrained(ckpt)  # nst=0 reference
    assert base.num_spread_token == 0, "checkpoint already has a spread head"
    new = MoiraiXModule.from_pretrained(
        ckpt, num_spread_token=num_spread_token, **overrides
    )
    base_sd, new_sd = base.state_dict(), new.state_dict()

    added = set(new_sd) - set(base_sd)
    expected = {k for k in new_sd if k.startswith("out_proj_spread.")}
    assert added == expected, f"unexpected new keys: {added ^ expected}"
    assert not (set(base_sd) - set(new_sd)), "graft dropped pretrained keys"

    # every inherited tensor must be byte-identical -- proves the silent load worked
    for k in base_sd:
        torch.testing.assert_close(new_sd[k], base_sd[k], rtol=0, atol=0)

    print(
        f"grafted {len(expected)} new tensors "
        f"({sum(new_sd[k].numel() for k in expected):,} params); "
        f"{len(base_sd)} pretrained tensors verified identical"
    )
    return new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ckpt", help="path or HF hub id of the source checkpoint")
    parser.add_argument("--num-spread-token", type=int, default=16)
    parser.add_argument(
        "--save-to",
        default=None,
        help="if set, save_pretrained the grafted module here after verification",
    )
    args = parser.parse_args()

    module = graft(
        args.ckpt,
        args.num_spread_token,
        dropout_p=0.0,
        attn_dropout_p=0.0,
    )
    if args.save_to is not None:
        module.save_pretrained(args.save_to)
        print(f"saved grafted module to {args.save_to}")
