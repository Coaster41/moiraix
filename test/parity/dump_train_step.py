"""Deterministic training-path parity: same config, same data draws, one
training_step + backward on CPU. Dumps batch, loss and gradient hashes.

The full cli.train run isn't run-to-run reproducible (masking RNG inside
multi-worker data loading), so this checks the training code in-process instead:
data builders -> transforms -> PackCollate -> MoiraiXPretrain -> loss -> grads.

    cd /srv/disk00/ctadler/uni2ts  && uni2ts/venv/bin/python  <this> --pkg uni2ts  --out a.pt
    cd /srv/disk00/ctadler/moiraix && moiraix/venv/bin/python <this> --pkg moiraix --out b.pt
"""

import argparse
import hashlib
import importlib
import os
import random

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

MODELS = ["moiraix_encoder", "moiraix_decoder", "moiraix_decoder_spread"]
DATA = "gift_eval_pretrain_trackb_p30_nosde_gift"
N_ITEMS = 48


def digest(t):
    return hashlib.sha256(
        t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def run(pkg, model_name):
    importlib.import_module(f"{pkg}.common.hydra_util")  # registers resolvers
    with initialize_config_dir(
        os.path.abspath("cli/conf/pretrain"), version_base="1.1"
    ):
        cfg = compose(
            "default",
            overrides=[
                f"model={model_name}",
                f"data={DATA}",
                "run_name=x",
                "val_data=gift_eval_test",
                "hydra.job.name=x",
            ],
            return_hydra_config=True,
        )
    seed(0)
    model = instantiate(cfg.model, _convert_="all")
    dataset = instantiate(cfg.data).load_dataset(model.train_transform_map)
    collate = instantiate(cfg.train_dataloader.collate_fn, _convert_="all")

    seed(1)
    idx = np.random.randint(0, len(dataset), size=N_ITEMS)
    items = [dataset[int(i)] for i in idx]
    batch = collate(items)

    seed(2)
    model.train()
    loss = model.training_step(batch, 0)
    loss.backward()
    return {
        "batch": {
            k: digest(v) for k, v in batch.items() if isinstance(v, torch.Tensor)
        },
        "loss": float(loss),
        "grads": {
            n: digest(p.grad) for n, p in model.named_parameters() if p.grad is not None
        },
        "init": {n: digest(p) for n, p in model.named_parameters()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkg", required=True, choices=["uni2ts", "moiraix"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    torch.use_deterministic_algorithms(True)
    res = {}
    for m in MODELS:
        res[m] = run(args.pkg, m)
        print(m, res[m]["loss"], len(res[m]["grads"]), flush=True)
    torch.save(res, args.out)


if __name__ == "__main__":
    main()
