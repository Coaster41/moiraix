#  Track C: data-SCALE synthetic corpora (.claude/HANDOFF_SCALE_DATA.md).
#
#  Motivation. The 400M `moiraix_enc_cpm_gift_300m_0` run tied the 30M model.
#  Auditing the mix showed why: 80 epochs consume ~1.34T observations but only
#  ~50B of them are UNIQUE (~27x repetition), because Moirai's cap-based
#  weighting pins every dataset at ~126k effective windows. The pure-synthetic
#  block was the worst offender per token -- 2.8M short series (~6.6B obs) held
#  at a 27% sampling share, i.e. ~60x repetition.
#
#  This module adds `_v1` corpora that scale the VALIDATED Track B generators
#  (exp-006 picked `nosde`, so SDE is deliberately not scaled) along two axes:
#
#    1. count  -- ~3.4M series total vs 2.8M
#    2. LENGTH -- 4096..16384 vs Track B's 2048..4096. A packed training
#       sequence is max_seq_len(512) x patch(16) = 8192 observations, so v0
#       series could not even fill one; v1 series both fill it and expose many
#       more distinct crops per series. This also matches the ctx-4000 eval.
#
#  Together: ~6.6B -> ~35B unique synthetic observations (~140GB on disk).
#
#  Plus ONE new family, `synth_mixed_v1`, covering the four regions the
#  generator survey (.claude/Synth_data_background.md) flags as UNCOVERED by
#  every existing family -- the Chronos-2-style support-coverage bet:
#
#    intermittent -- count / Croston-style zero-inflated demand (integer, sparse)
#    longmem      -- long-memory fractional Gaussian noise + fractional trends
#    calendar     -- day-of-week / hour-of-day / annual / holiday effects
#    stochvol     -- log-volatility AR(1) => unconditionally heavy-tailed
#
#  ADDITIVE ONLY: nothing here reads, writes, or reweights a v0 dataset. New
#  Arrow dirs (`synth_*_v1`), new builders, new seeds -- the v1 draws are
#  disjoint from v0 because `_series_seed` keys on the dataset name.
#
#  Build (all four, one SLURM job):
#      sbatch slurm/scripts/build_synth_scale_v1.sh
#  or individually:
#      python -m moiraix.data.builder.lotsa_v1.synth_scale \
#          --dataset synth_backbone_v1 --num-series 1500000

import argparse
import os
import shutil
from collections import defaultdict
from functools import partial
from typing import Optional

import numpy as np
from scipy.signal import lfilter

import datasets
from moiraix.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder
from .synth_stress import CATEGORIES, CATEGORY_PROBS
from .synth_stress import GENERATORS as STRESS_GENERATORS
from .synth_stress import LOTSA_FEATURES, _log_uniform
from .synth_trackb import (
    _observation_layer,
    _series_seed,
    _trend,
    gen_backbone,
    gen_sarima2,
)

#  Long enough to fill a packed sequence (512 patches x 16 = 8192 obs) and to
#  yield many distinct crops per series. v0 used 2048..4096.
MIN_LEN, MAX_LEN = 4096, 16384

#  Default corpus sizes, chosen to land pure-synthetic unique observations at
#  ~35B (~5.3x v0) for ~140GB of disk. Mean length ~10240.
DEFAULT_NUM_SERIES = {
    "synth_backbone_v1": 1_500_000,
    "synth_stress_v1": 800_000,
    "synth_sarima2_v1": 500_000,
    "synth_mixed_v1": 600_000,
}


# ---------------------------------------------------------------------------
# scaled re-draws of the validated Track A/B families
# ---------------------------------------------------------------------------


def gen_stress(rng: np.random.Generator, T: int) -> np.ndarray:
    """synth_stress_v0's per-series draw, with T supplied by the caller."""
    category = str(rng.choice(CATEGORIES, p=CATEGORY_PROBS))
    return STRESS_GENERATORS[category](rng, T)


# ---------------------------------------------------------------------------
# family 4 (new) -- survey-uncovered regions
# ---------------------------------------------------------------------------


def _fgn(rng: np.random.Generator, T: int, H: float) -> np.ndarray:
    """Long-memory fractional Gaussian noise by spectral synthesis.

    Shapes white noise in the Fourier domain by |f|^(0.5-H): H=0.5 recovers
    white noise, H->1 gives strong positive long-range dependence, H<0.5 gives
    anti-persistence. O(T log T), unlike the O(T^2) Cholesky route.
    """
    n = 1 << int(np.ceil(np.log2(max(T, 2)))) + 1  # pad to avoid wrap-around
    freqs = np.fft.rfftfreq(n)
    scale = np.empty_like(freqs)
    scale[0] = 0.0
    scale[1:] = freqs[1:] ** (0.5 - H)
    spec = np.fft.rfft(rng.standard_normal(n)) * scale
    x = np.fft.irfft(spec, n=n)[:T]
    std = x.std()
    return (x / std if std > 1e-8 else x).astype(np.float32)


def _gen_longmem(rng: np.random.Generator, T: int) -> np.ndarray:
    """Long-memory / multi-scale: fGn, optionally integrated and/or trended."""
    H = rng.uniform(0.55, 0.95) if rng.random() < 0.8 else rng.uniform(0.1, 0.45)
    y = _fgn(rng, T, H)
    if rng.random() < 0.35:  # integrate to fBm (non-stationary long memory)
        y = np.cumsum(y).astype(np.float32)
        y /= max(float(y.std()), 1e-6)
    if rng.random() < 0.4:
        y = y + _trend(rng, T, force=True)
    if rng.random() < 0.3:  # superpose a second, faster-mixing scale
        y = y + rng.uniform(0.2, 1.0) * _fgn(rng, T, rng.uniform(0.5, 0.7))
    return _observation_layer(rng, y.astype(np.float32))


def _gen_intermittent(rng: np.random.Generator, T: int) -> np.ndarray:
    """Count / intermittent demand: sparse integer events, Croston-style.

    Covers the low-count, zero-inflated regime (retail SKUs, error counts,
    rare-event telemetry) that every Gaussian family misses.
    """
    kind = rng.random()
    #  slowly varying intensity so the process is non-homogeneous
    base = _log_uniform(rng, 0.02, 6.0)
    lam = np.full(T, base, dtype=np.float64)
    if rng.random() < 0.7:  # seasonal intensity
        P = _log_uniform(rng, 6.0, 400.0)
        lam *= np.exp(
            rng.uniform(0.2, 1.2)
            * np.sin(2 * np.pi * np.arange(T) / P + rng.uniform(0, 2 * np.pi))
        )
    if rng.random() < 0.4:  # drifting intensity
        lam *= np.exp(rng.uniform(-1.5, 1.5) * np.linspace(0.0, 1.0, T))

    if kind < 0.45:  # Poisson counts
        y = rng.poisson(lam).astype(np.float32)
    elif kind < 0.8:  # over-dispersed (negative binomial via gamma mixture)
        r = _log_uniform(rng, 0.3, 8.0)
        y = rng.poisson(rng.gamma(r, lam / r)).astype(np.float32)
    else:  # Croston: geometric inter-arrivals x lognormal sizes
        p = float(np.clip(base / (1.0 + base), 0.01, 0.6))
        y = np.zeros(T, dtype=np.float32)
        n_ev = max(1, rng.binomial(T, p))
        pos = rng.choice(T, size=min(n_ev, T), replace=False)
        y[pos] = rng.lognormal(rng.uniform(-1, 2), rng.uniform(0.3, 1.2), len(pos))
        y = np.round(y).astype(np.float32) if rng.random() < 0.5 else y
    if rng.random() < 0.25:  # occasional censoring at a stock cap
        #  Guard: at very low rates the 0.9-0.99 quantile is itself 0, and
        #  clipping there would flatten the whole series to a constant.
        cap = float(np.quantile(y, rng.uniform(0.9, 0.99)))
        if cap > 0:
            y = np.minimum(y, cap)
    return y.astype(np.float32)


#  (period, amplitude-scale) of the calendar cycles, in steps, for freq H and D.
_CAL_CYCLES = {
    "H": ((24.0, 1.0), (168.0, 0.7), (8766.0, 0.5)),
    "D": ((7.0, 1.0), (365.25, 0.7)),
}


def _gen_calendar(rng: np.random.Generator, T: int, freq: str) -> np.ndarray:
    """Calendar structure: hour-of-day / day-of-week / annual + holiday spikes.

    Uses explicit per-phase profiles rather than a single sinusoid, so weekday
    /weekend and business-hour discontinuities are real steps, not smooth waves.
    """
    t = np.arange(T)
    y = np.zeros(T, dtype=np.float64)
    for period, amp_scale in _CAL_CYCLES[freq]:
        if rng.random() > 0.85:
            continue
        p_int = int(round(period))
        amp = amp_scale * _log_uniform(rng, 0.3, 2.5)
        if p_int <= 400:  # short cycle -> arbitrary per-phase profile (steps ok)
            profile = rng.normal(0.0, 1.0, p_int)
            if rng.random() < 0.5:  # smooth some profiles (diurnal shapes)
                profile = np.convolve(
                    np.r_[profile, profile, profile], np.ones(3) / 3, "same"
                )[p_int : 2 * p_int]
            y += amp * profile[t % p_int]
        else:  # annual -> harmonic
            y += amp * np.sin(2 * np.pi * t / period + rng.uniform(0, 2 * np.pi))
    #  holidays: a handful of fixed phases per year with a decaying shoulder
    year = 8766.0 if freq == "H" else 365.25
    if rng.random() < 0.7 and T > year / 4:
        for _ in range(int(rng.integers(2, 9))):
            phase = rng.uniform(0, year)
            mag = rng.uniform(1.0, 6.0) * float(rng.choice([-1.0, 1.0]))
            width = max(1.0, rng.uniform(0.002, 0.02) * year)
            d = np.abs(((t - phase + year / 2) % year) - year / 2)
            y += mag * np.exp(-0.5 * (d / width) ** 2)
    y = y + _trend(rng, T) * rng.uniform(0.5, 3.0)
    if rng.random() < 0.4:  # multiplicative calendar on a positive level
        y = (2.0 + y - y.min()) * (1.0 + 0.3 * _fgn(rng, T, 0.7))
    return _observation_layer(rng, y.astype(np.float32))


def _gen_stochvol(rng: np.random.Generator, T: int) -> np.ndarray:
    """Stochastic volatility: log-vol AR(1) => unconditionally heavy-tailed.

    The survey's other uncovered region. Vectorized as an AR(1) via lfilter
    rather than a GARCH recursion, which would be a Python loop per series.
    """
    phi = rng.uniform(0.9, 0.999)
    sig_h = rng.uniform(0.05, 0.4)
    h = lfilter([1.0], [1.0, -phi], sig_h * rng.standard_normal(T))
    h -= h.mean()
    vol = np.exp(np.clip(h, -6.0, 6.0))
    eps = (
        rng.standard_t(rng.uniform(3.0, 12.0), T)
        if rng.random() < 0.4
        else rng.standard_normal(T)
    )
    y = vol * eps
    if rng.random() < 0.5:  # integrate -> price-like random walk with vol clusters
        y = np.cumsum(y)
    y = y / max(float(np.std(y)), 1e-6)
    if rng.random() < 0.3:
        y = y + _trend(rng, T, force=True)
    return y.astype(np.float32)


_MIXED_KINDS = ("intermittent", "longmem", "calendar", "stochvol")
_MIXED_PROBS = (0.30, 0.25, 0.30, 0.15)


def _gen_mixed_once(rng: np.random.Generator, T: int) -> np.ndarray:
    kind = str(rng.choice(_MIXED_KINDS, p=_MIXED_PROBS))
    if kind == "intermittent":
        return _gen_intermittent(rng, T)
    if kind == "longmem":
        return _gen_longmem(rng, T)
    if kind == "calendar":
        return _gen_calendar(rng, T, "H" if rng.random() < 0.6 else "D")
    return _gen_stochvol(rng, T)


def gen_mixed(rng: np.random.Generator, T: int) -> np.ndarray:
    """One mixed-family series, redrawing if a draw comes out degenerate.

    The intermittent branch can legitimately produce an all-zero series at the
    low end of its rate range. A constant target is useless to train on (the
    scaler floors at min_scale and the quantile loss sees no signal), so redraw
    -- bounded, and still deterministic in `rng`.
    """
    for _ in range(5):
        y = _gen_mixed_once(rng, T)
        finite = y[np.isfinite(y)]
        if finite.size and float(finite.std()) > 1e-6:
            return y
    return _gen_stochvol(rng, T)  # always non-degenerate


GENERATORS = dict(
    synth_backbone_v1=gen_backbone,
    synth_stress_v1=gen_stress,
    synth_sarima2_v1=gen_sarima2,
    synth_mixed_v1=gen_mixed,
)


def _example_gen(shard_bounds):
    """HF from_generator worker; items are (dataset, global_seed, lo, hi)."""
    for dataset, global_seed, lo, hi in shard_bounds:
        gen = GENERATORS[dataset]
        tag = dataset.replace("synth_", "").replace("_v1", "")
        for i in range(lo, hi):
            #  _series_seed keys on the dataset name, so v1 draws are disjoint
            #  from v0's even at the same (seed, index).
            rng = np.random.default_rng(_series_seed(global_seed, dataset, i))
            T = int(rng.integers(MIN_LEN, MAX_LEN + 1))
            yield dict(
                item_id=f"synth_{tag}_v1_{i:07d}",
                start="2000-01-01",
                freq="H",
                target=gen(rng, T),
            )


class _SynthScaleMixin:
    """Shared build machinery (kept out of the LOTSADatasetBuilder hierarchy so
    the abstract dataset_list check only fires on the concrete classes)."""

    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(
        self,
        dataset: str,
        num_series: Optional[int] = None,
        seed: int = 20260811,
        num_proc: Optional[int] = None,
    ):
        assert dataset in self.dataset_list
        num_series = num_series or DEFAULT_NUM_SERIES[dataset]
        num_proc = num_proc or min(48, os.cpu_count())
        #  v1 series are ~3x longer than v0's, so shard finer to keep each
        #  worker's Arrow buffer small.
        bounds = np.linspace(0, num_series, num_proc * 8 + 1, dtype=int)
        shard_bounds = [
            (dataset, seed, int(lo), int(hi)) for lo, hi in zip(bounds[:-1], bounds[1:])
        ]
        cache_dir = str(self.storage_path / f"{dataset}_build_cache")
        hf_dataset = datasets.Dataset.from_generator(
            _example_gen,
            features=LOTSA_FEATURES,
            gen_kwargs=dict(shard_bounds=shard_bounds),
            num_proc=num_proc,
            cache_dir=cache_dir,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset, num_proc=num_proc)
        del hf_dataset
        shutil.rmtree(cache_dir, ignore_errors=True)
        print(f"[{dataset}] Saved {num_series} series to {self.storage_path / dataset}")


class SynthBackboneV1DatasetBuilder(_SynthScaleMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_backbone_v1"]


class SynthStressV1DatasetBuilder(_SynthScaleMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_stress_v1"]


class SynthSarima2V1DatasetBuilder(_SynthScaleMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_sarima2_v1"]


class SynthMixedV1DatasetBuilder(_SynthScaleMixin, LOTSADatasetBuilder):
    dataset_list = ["synth_mixed_v1"]


BUILDER_OF = dict(
    synth_backbone_v1=SynthBackboneV1DatasetBuilder,
    synth_stress_v1=SynthStressV1DatasetBuilder,
    synth_sarima2_v1=SynthSarima2V1DatasetBuilder,
    synth_mixed_v1=SynthMixedV1DatasetBuilder,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Track C scale-up synthetic corpus."
    )
    parser.add_argument("--dataset", required=True, choices=list(GENERATORS))
    parser.add_argument("--num-series", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--num-proc", type=int, default=None)
    parser.add_argument("--storage-path", default=None)
    args = parser.parse_args()

    from pathlib import Path

    kwargs = {}
    if args.storage_path is not None:
        kwargs["storage_path"] = Path(args.storage_path)
    BUILDER_OF[args.dataset](datasets=[args.dataset], **kwargs).build_dataset(
        dataset=args.dataset,
        num_series=args.num_series,
        seed=args.seed,
        num_proc=args.num_proc,
    )
