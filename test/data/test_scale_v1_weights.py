#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Pin down the Track C data-SCALE mixture (``gift_eval_pretrain_scale_v1``) and
the ``synth_*_v1`` generators that feed it.

Context: the ~400M ``moiraix_enc_cpm_gift_300m_0`` run tied the 30M model. The
audit found the mixture, not the model, was the binding constraint -- 80 epochs
consumed ~1.34T observations of which only ~50B were unique (~27x repetition),
because the cap-based weighting made ERA5+CMIP6 (82% of the on-disk corpus)
effectively unreachable. ``scale_v1`` raises climate 0.15% -> 12%, adds RMISC
and the v1 synthetic corpora, and halves the 812x-recycled GIFT split.

The share formula these tests rely on is derived in test_synthetic_weights.py::
``share_d = (num_ts_d * weight_d) / sum_i (num_ts_i * weight_i)``.

Tests that need the built corpora are skipped when they are absent, so this file
passes on a checkout without the 1.1TB dataset tree.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf

from moiraix.common import hydra_util  # noqa: F401  registers cls_getattr
from moiraix.common.env import env
from moiraix.data.builder.lotsa_v1.synth_scale import (
    DEFAULT_NUM_SERIES,
    GENERATORS,
    MAX_LEN,
    MIN_LEN,
)
from moiraix.data.builder.lotsa_v1.synth_trackb import _series_seed

CONF_DIR = Path(__file__).resolve().parents[2] / "cli/conf/pretrain/data"
BASE_CFG = CONF_DIR / "gift_eval_pretrain_trackb_p30_nosde_gift.yaml"
SCALE_CFG = CONF_DIR / "gift_eval_pretrain_scale_v1.yaml"

#  Targets passed to scripts/gen_scale_v1_config.py. The tolerance is loose
#  because block budgets are split by rules (unique obs / num_ts**0.5) whose
#  rounding lands a few tenths of a percent off the request.
TARGET_SHARES = {
    "climate": 0.12,
    "rmisc": 0.08,
    "synth": 0.27,
    "tsmixup": 0.08,
    "gift": 0.05,
    "core_real": 0.40,
}
SHARE_TOL = 0.005

CLIMATE_PREFIXES = ("era5_", "cmip6_")
SYNTH_DATASETS = {
    "chronos_kernel_synth_1m",
    "synth_stress_v0",
    "synth_backbone_v0",
    "synth_sarima2_v0",
    "synth_backbone_v1",
    "synth_stress_v1",
    "synth_sarima2_v1",
    "synth_mixed_v1",
}
V1_DATASETS = tuple(DEFAULT_NUM_SERIES)


def _block_of(dataset: str) -> str:
    if dataset.startswith(CLIMATE_PREFIXES):
        return "climate"
    if dataset.startswith("rmisc_"):
        return "rmisc"
    if dataset in SYNTH_DATASETS:
        return "synth"
    if dataset == "chronos_tsmixup_10m":
        return "tsmixup"
    if dataset.startswith("gift_eval_train_split"):
        return "gift"
    return "core_real"


def _weights(cfg_path: Path) -> dict[str, float]:
    """Flatten a mixture config to {dataset: weight}."""
    cfg = OmegaConf.load(cfg_path)
    out = {}
    for entry in cfg["_args_"]:
        wm = OmegaConf.to_container(entry.get("weight_map") or {})
        for ds in list(entry["datasets"]):
            out[ds] = float(wm.get(ds, 1.0))
    return out


def _num_ts(dataset: str) -> int:
    import json

    for info in sorted((env.LOTSA_V1_PATH / dataset).rglob("dataset_info.json")):
        for split in (json.loads(info.read_text()).get("splits") or {}).values():
            if split.get("num_examples"):
                return int(split["num_examples"])
    raise FileNotFoundError(dataset)


requires_corpus = pytest.mark.skipif(
    not SCALE_CFG.exists() or not (env.LOTSA_V1_PATH / "synth_backbone_v1").exists(),
    reason="scale_v1 config or the v1 corpora are not built on this machine",
)


# ===========================================================================
# 1. The generators
# ===========================================================================
@pytest.mark.parametrize("dataset", list(GENERATORS))
def test_v1_series_are_well_formed(dataset):
    """Every draw has the requested length, is finite, and is non-degenerate.

    A constant target is worthless to train on: the packed scaler floors at
    min_scale and the quantile loss then sees no signal. ``gen_mixed`` redraws
    rather than emit one (the intermittent branch can go all-zero at low rates).
    """
    gen = GENERATORS[dataset]
    for i in range(200):
        rng = np.random.default_rng(_series_seed(20260811, dataset, i))
        length = int(rng.integers(MIN_LEN, MAX_LEN + 1))
        values = gen(rng, length)
        assert len(values) == length
        assert values.dtype == np.float32
        finite = values[np.isfinite(values)]
        assert finite.size > 0, f"{dataset}[{i}] is all-NaN"
        assert finite.std() > 1e-6, f"{dataset}[{i}] is constant"


def test_v1_series_are_long_enough_to_fill_a_packed_sequence():
    """The point of v1: a packed sequence is 512 patches x 16 = 8192 obs, and
    v0's 2048..4096 could not fill one. Regressing MIN_LEN would silently undo
    most of the unique-token gain."""
    assert MIN_LEN >= 4096 and MAX_LEN >= 8192


@pytest.mark.parametrize("dataset", V1_DATASETS)
def test_v1_draws_are_disjoint_from_v0(dataset):
    """``_series_seed`` keys on the dataset name, so v1 index i is an
    independent draw from v0 index i -- v1 is new data, not a re-emission."""
    v0 = dataset.replace("_v1", "_v0")
    assert _series_seed(20260811, dataset, 7) != _series_seed(20260811, v0, 7)


def test_v1_series_are_deterministic():
    """Same (seed, dataset, index) => same series, independent of sharding."""
    for dataset, gen in GENERATORS.items():
        draws = []
        for _ in range(2):
            rng = np.random.default_rng(_series_seed(20260811, dataset, 3))
            draws.append(gen(rng, int(rng.integers(MIN_LEN, MAX_LEN + 1))))
        np.testing.assert_array_equal(draws[0], draws[1])


# ===========================================================================
# 2. The mixture
# ===========================================================================
@requires_corpus
def test_scale_v1_block_shares_hit_their_targets():
    """The headline claim of the config: climate 12%, rmisc 8%, gift down to 5%."""
    weights = _weights(SCALE_CFG)
    eff = {ds: math.ceil(_num_ts(ds) * w) for ds, w in weights.items()}
    total = sum(eff.values())
    shares: dict[str, float] = {}
    for ds, e in eff.items():
        shares[_block_of(ds)] = shares.get(_block_of(ds), 0.0) + e / total
    assert set(shares) == set(TARGET_SHARES)
    for block, target in TARGET_SHARES.items():
        assert shares[block] == pytest.approx(
            target, abs=SHARE_TOL
        ), f"{block}: {shares[block]:.3%} vs target {target:.1%}"


@requires_corpus
def test_scale_v1_unthrottles_the_climate_reserve():
    """ERA5+CMIP6 go from ~0.15% (unreachable) to ~12%, an ~80x lift. This is
    the single change the whole rerun rests on."""
    base, scale = _weights(BASE_CFG), _weights(SCALE_CFG)
    climate = [d for d in scale if d.startswith(CLIMATE_PREFIXES)]
    assert len(climate) == 63  # 30 ERA5 years + 33 CMIP6 pentads

    def share(weights):
        eff = {ds: math.ceil(_num_ts(ds) * w) for ds, w in weights.items()}
        return sum(eff[d] for d in climate) / sum(eff.values())

    assert share(base) < 0.005
    assert share(scale) > 0.10


@requires_corpus
def test_scale_v1_preserves_within_block_proportions():
    """core_real / climate / tsmixup / gift are rescaled by ONE factor per
    block, so the validated relative shape of the real mixture survives. If a
    future edit re-tunes individual real datasets, this catches it."""
    base, scale = _weights(BASE_CFG), _weights(SCALE_CFG)
    for block in ("core_real", "climate"):
        ratios = [
            scale[d] / base[d]
            for d in base
            if _block_of(d) == block and d in scale and base[d] > 0
        ]
        assert len(ratios) > 5
        assert max(ratios) / min(ratios) == pytest.approx(
            1.0, abs=1e-6
        ), f"{block} weights were not rescaled uniformly"


@requires_corpus
def test_scale_v1_cuts_the_most_repeated_component():
    """gift_eval_train_split held 10% of sampling on 0.165B unique observations
    -- 812x repetition over an 80-epoch run, an overfitting magnet at 400M. It
    stays in the mix (FIX G) but at half the share."""
    base, scale = _weights(BASE_CFG), _weights(SCALE_CFG)
    assert scale["gift_eval_train_split"] < base["gift_eval_train_split"]


@requires_corpus
def test_scale_v1_adds_only_and_leaves_the_base_config_intact():
    """Additive by construction: every dataset the base mixture drew from is
    still drawn from, and the base file itself is untouched."""
    base, scale = _weights(BASE_CFG), _weights(SCALE_CFG)
    assert set(base) <= set(scale)
    assert all(w > 0 for w in scale.values())
    for ds in V1_DATASETS:
        assert ds in scale and ds not in base


@requires_corpus
def test_scale_v1_raises_the_unique_token_pool():
    """The number the rerun exists to move: unique observations actually
    touched over a 100-epoch run, and hence the repetition factor."""

    def touched(weights, epochs=100):
        seen = 4000 * 64 * 8 * 8192 * epochs  # micro-batches x bs x ranks x obs/seq
        eff = {ds: math.ceil(_num_ts(ds) * w) for ds, w in weights.items()}
        total = sum(eff.values())
        obs = {
            ds: sum(f.stat().st_size for f in (env.LOTSA_V1_PATH / ds).rglob("*.arrow"))
            / 4
            for ds in weights
        }
        return sum(min(eff[ds] / total * seen, obs[ds]) for ds in weights)

    assert touched(_weights(SCALE_CFG)) > 2.5 * touched(_weights(BASE_CFG))
