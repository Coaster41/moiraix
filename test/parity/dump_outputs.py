"""Dump checkpoint weights + forecasts so the moiraix package can be compared
bit-for-bit against the uni2ts package it was split from.

Run once per venv (same GPU), then compare_outputs.py:
    /srv/disk00/ctadler/uni2ts/venv/bin/python  test/parity/dump_outputs.py --backend uni2ts  --out a.pt
    /srv/disk00/ctadler/moiraix/venv/bin/python test/parity/dump_outputs.py --backend moiraix --out b.pt
"""

import argparse
import hashlib
import time

import numpy as np
import torch

import datasets

U = "/srv/disk00/ctadler/uni2ts/outputs/pretrain/"
CHECKPOINTS = {
    # name: (path, kind) -- kind picks the uni2ts preset class / moiraix preset
    "moiraic_base_11": (
        U + "moiraic/gift_eval_pretrain_weighted/moiraic_training_11",
        "moiraic",
    ),
    "moiraie_base_7": (
        U + "moiraie/gift_eval_pretrain_weighted/moiraie_training_7",
        "moiraie",
    ),
    "moiraie_causal_0": (
        U + "moiraie_causal/gift_eval_pretrain_weighted/moiraie_training_0",
        "moiraie",
    ),
    "moiraix_enc_cpm_arcsinh_0": (
        U
        + "moiraix/gift_eval_pretrain_trackb_p30_nosde_gift/moiraix_enc_cpm_arcsinh_0",
        "moiraix",
    ),
    "moiraix_enc_cpm_gift_1": (
        U + "moiraix/gift_eval_pretrain_trackb_p30_nosde_gift/moiraix_enc_cpm_gift_1",
        "moiraix",
    ),
    "moiraix_dec_ntp_arcsinh_0": (
        U
        + "moiraix/gift_eval_pretrain_trackb_p30_nosde_gift/moiraix_dec_ntp_arcsinh_0",
        "moiraix",
    ),
    "moiraix_dec_spread_scratch50_0": (
        U
        + "moiraix_decoder_spread/gift_eval_pretrain_trackb_p30_nosde_gift/moiraix_dec_spread_scratch50_0",
        "moiraix",
    ),
}
SERIES = [
    "ett1/H",
    "electricity/D",
    "kdd_cup_2018_with_missing/H",
    "m4_monthly",
    "us_births/D",
]
TEST_DIR = "/srv/disk00/ctadler/uni2ts/datasets/giftevaltest/"
CONTEXT = 1000
HORIZON = 160  # 10 patches at p16: forces AR for every decode method


def load_series():
    out = []
    for name in SERIES:
        ds = datasets.load_from_disk(TEST_DIR + name)
        for i in range(min(4, len(ds))):
            t = np.asarray(ds[i]["target"], dtype=np.float32)
            for row in t.reshape(-1, t.shape[-1])[:2]:  # univariate, like gift-eval
                if len(row) > HORIZON + 16:
                    out.append(row[:-HORIZON, None])  # (time, tgt=1)
    return out


def get_api(backend):
    if backend == "uni2ts":
        from uni2ts.model.moiraic import MoiraicForecast, MoiraicModule
        from uni2ts.model.moiraie import MoiraieForecast, MoiraieModule
        from uni2ts.model.moiraix import MoiraiXForecast, MoiraiXModule

        classes = {
            "moiraic": (MoiraicModule, MoiraicForecast),
            "moiraie": (MoiraieModule, MoiraieForecast),
            "moiraix": (MoiraiXModule, MoiraiXForecast),
        }

        def load(path, kind):
            return classes[kind][0].from_pretrained(path)

        def forecast(module, kind, **kw):
            return classes[kind][1](module=module, **kw)

    else:
        from moiraix.model import FORECAST_DEFAULTS, MoiraiXForecast, load_module

        presets = {"moiraic": "decoder", "moiraie": "encoder", "moiraix": None}

        def load(path, kind):
            return load_module(path, presets[kind])

        def forecast(module, kind, **kw):
            preset = presets[kind]
            defaults = dict(FORECAST_DEFAULTS[preset]) if preset else {}
            defaults.update(kw)
            return MoiraiXForecast(module=module, **defaults)

    return load, forecast


def tensor_digest(sd):
    return {
        k: hashlib.sha256(
            v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        for k, v in sd.items()
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", required=True, choices=["uni2ts", "moiraix"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    load, make_forecast = get_api(args.backend)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    series = load_series()
    results = {"backend": args.backend, "series": len(series), "ckpts": {}}

    for name, (path, kind) in CHECKPOINTS.items():
        module = load(path + "/HF_checkpoints/last", kind)
        flags = {
            k: getattr(module, k, None)
            for k in (
                "causal",
                "mask_inputs",
                "predict_next",
                "num_predict_token",
                "arcsinh_norm",
                "num_spread_token",
            )
        }
        entry = {
            "flags": flags,
            "weights": tensor_digest(module.state_dict()),
            "forecasts": {},
        }
        # A plain-moiraix next-token model has no default AR method (raises by design).
        methods = [] if kind == "moiraix" and module.predict_next else [("default", {})]
        if module.predict_next:
            methods += [
                (m, {"ar_method": m}) for m in ("naive", "branch", "trajectory")
            ]
        else:
            methods += [(m, {"ar_method": m}) for m in ("naive", "branch")]
            methods += [("single_pass", {"ar_method": None, "single_pass_horizon": -1})]
        for label, kw in methods:
            fc = make_forecast(
                module,
                kind,
                prediction_length=HORIZON,
                context_length=CONTEXT,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
                **kw,
            ).to(device)
            torch.manual_seed(0)
            np.random.seed(0)
            t0 = time.time()
            with torch.no_grad():
                pred = fc.predict(series)
            entry["forecasts"][label] = np.asarray(pred)
            print(
                f"{name:32s} {label:12s} {entry['forecasts'][label].shape} {time.time() - t0:.1f}s",
                flush=True,
            )
        results["ckpts"][name] = entry
        module.to("cpu")

    torch.save(results, args.out)


if __name__ == "__main__":
    main()
