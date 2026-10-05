# MoiraiX

A unified Moirai-architecture time-series foundation model. Each of these is a
constructor flag:
- causal or bidirectional attention
- masked or raw inputs
- next-token or masked-reconstruction objective
- multi-token prediction

Any combination of these can be trained, and it can be decoded single-pass or
autoregressively (naive, branch or trajectory sampling, with a KV cache for causal
models).

The code is derived from [Salesforce uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
(Apache-2.0, see `LICENSE.txt`).

## Install

```shell
virtualenv -p python3.10 venv
venv/bin/pip install -r requirements-lock.txt   # exact versions the checkpoints were built with
venv/bin/pip install -e '.[dev]'
```

Create a `.env` that points at the datasets:
```
LOTSA_V1_PATH=/path/to/giftevalpretrain
GIFT_EVAL_PRETRAIN_PATH=/path/to/giftevalpretrain
GIFT_EVAL_TEST_PATH=/path/to/giftevaltest
```

## Usage

```python
from moiraix.model import MoiraiXForecast, MoiraiXModule, load_module, FORECAST_DEFAULTS

module = MoiraiXModule.from_pretrained("outputs/pretrain/.../HF_checkpoints/last")
forecast = MoiraiXForecast(
    module=module, prediction_length=64, context_length=4000,
    target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
)
predictor = forecast.create_predictor(batch_size=128, device="cuda")

# Checkpoints from before the unified flags (former moiraic / moiraie):
module = load_module(path, preset="decoder")   # or "encoder"
forecast = MoiraiXForecast(module=module, **FORECAST_DEFAULTS["decoder"], ...)
```

Training:
```shell
python -m cli.train -cp conf/pretrain run_name=my_run model=moiraix_encoder \
  data=gift_eval_pretrain_trackb_p30_nosde_gift val_data=gift_eval_test
```

See `.claude/DESIGN.md` for the flags and how decoding works.
