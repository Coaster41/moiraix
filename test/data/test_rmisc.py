"""Guard the RMISC overlap filter and the multivariate -> univariate conversion.

The overlap filter is the load-bearing part: RMISC repackages many of the same
sources as GIFT-Eval and LOTSA under different names, so a dir slipping back
into RMISC_DIRS is either test leakage (gift_test) or a silent doubling of a
dataset's sampling weight (lotsa_dup). These tests pin the filter's decisions so
a future edit has to be deliberate.
"""

import numpy as np
import pyarrow as pa
import pytest

from moiraix.data.builder.lotsa_v1.rmisc import (
    EXCLUDE,
    MIN_SERIES_LEN,
    PREFIX,
    RMISC_DATASETS,
    RMISC_DIRS,
    RMISCDatasetBuilder,
    _channel_columns,
    _group_bounds,
    _is_degenerate,
    _series_gen,
)

#  Dirs that MUST stay excluded, and why. These are the non-obvious aliases --
#  the ones a reader would not catch by eye when scanning RMISC's dir listing.
LEAKY_ALIASES = {
    "M4": "gift_test",
    "ETT": "gift_test",
    "ETTMulti": "gift_test",
    "BeutenbergWeather": "gift_test",  # == GIFT's jena_weather station
    "Mdense": "gift_test",  # == GIFT's M_DENSE
    "HierachicalSales": "gift_test",  # == GIFT's hierarchical_sales
    "LoopSeattleLA": "gift_test",  # == GIFT's LOOP_SEATTLE
    "KDDCup2018": "gift_test",
    "Bizitobs_application": "gift_test",
    "WeatherBench5-625deg": "lotsa_dup",  # ERA5 regridded; we have era5_*
    "AzurePublicDatasetV1": "lotsa_dup",  # == azure_vm_traces_2017
    "QtrafficSpeed": "lotsa_dup",  # == Q-TRAFFIC
    "LargeST": "lotsa_dup",  # == largest_2017..2021
    "Subseasonal": "lotsa_dup",
    "CSTSNormalTest": "rmisc_test",
    "Behavior-1k": "sim",
}


class TestOverlapFilter:
    @pytest.mark.parametrize("name,reason", sorted(LEAKY_ALIASES.items()))
    def test_leaky_dirs_stay_excluded(self, name, reason):
        assert name not in RMISC_DIRS, f"{name} would leak/duplicate ({reason})"
        assert EXCLUDE[name] == reason

    def test_kept_and_excluded_are_disjoint(self):
        assert not set(RMISC_DIRS) & set(EXCLUDE)

    def test_dataset_list_is_prefixed(self):
        assert RMISCDatasetBuilder.dataset_list == RMISC_DATASETS
        assert RMISC_DATASETS == [PREFIX + d for d in RMISC_DIRS]
        assert len(set(RMISC_DATASETS)) == len(RMISC_DATASETS)

    def test_no_gift_eval_test_dataset_survives(self):
        """Every GIFT-Eval test dataset must be absent from the kept dirs.

        Matched case-insensitively with separators stripped, so RMISC's spelling
        (Bizitobs_l2c_H, Mdense, HierachicalSales) can't sneak past a name that
        differs only in punctuation.
        """
        gift = (
            "m4 electricity solar hospital covid_deaths us_births saugeenday "
            "temperature_rain kdd_cup_2018 car_parts restaurant hierarchical_sales "
            "loop_seattle sz_taxi m_dense ett1 ett2 jena_weather "
            "bitbrains_fast_storage bitbrains_rnd bizitobs_application "
            "bizitobs_service bizitobs_l2c"
        ).split()
        canon = {d.lower().replace("_", "").replace("-", "") for d in RMISC_DIRS}
        for name in gift:
            assert name.lower().replace("_", "") not in canon, f"{name} kept"


class TestDegenerateChannelFilter:
    def test_keeps_real_signal(self):
        rng = np.random.default_rng(0)
        assert not _is_degenerate(rng.standard_normal(256).astype(np.float32))

    def test_drops_constant(self):
        assert _is_degenerate(np.full(256, 7.0, dtype=np.float32))

    def test_drops_binary_flag(self):
        # Rossmann `Open` / `Promo`
        flag = np.tile([0.0, 1.0], 128).astype(np.float32)
        assert _is_degenerate(flag)

    def test_drops_calendar_counter(self):
        # Rossmann `DayOfWeek`: an exact period-7 integer sawtooth
        assert _is_degenerate(np.tile(np.arange(1, 8), 40).astype(np.float32))

    def test_keeps_intermittent_counts(self):
        """Low-cardinality but non-periodic counts are real (intermittent demand)."""
        rng = np.random.default_rng(1)
        counts = rng.poisson(0.4, 512).astype(np.float32)
        assert not _is_degenerate(counts)

    def test_keeps_seasonal_float_series(self):
        """A seasonal float signal must not be mistaken for a calendar counter."""
        t = np.arange(512)
        x = (np.sin(2 * np.pi * t / 7) + 0.01 * np.arange(512)).astype(np.float32)
        assert not _is_degenerate(x)

    def test_drops_too_short(self):
        assert _is_degenerate(np.arange(MIN_SERIES_LEN - 1, dtype=np.float32))


def _wide_table() -> pa.Table:
    """Two source series interleaved, as RMISC stores them, plus junk columns."""
    n = 128
    rng = np.random.default_rng(2)
    hours = (np.arange(2 * n) % n).astype("timedelta64[h]").astype("timedelta64[s]")
    return pa.table(
        {
            "date": pa.array(np.datetime64("2020-01-01", "s") + hours),
            "load": pa.array(rng.standard_normal(2 * n)),
            "temp": pa.array(rng.standard_normal(2 * n)),
            "Open": pa.array(np.tile([0.0, 1.0], n)),  # binary flag -> dropped
            "SinceYear": pa.array(np.full(2 * n, 2011.0)),  # constant -> dropped
            "_original_filename": pa.array(["a"] * n + ["b"] * n).dictionary_encode(),
        }
    )


class TestConversion:
    def test_channel_columns_drop_timestamp_and_key(self):
        table = _wide_table()
        cols = _channel_columns({"timestamp": None}, table.schema)
        assert "date" not in cols and "_original_filename" not in cols
        assert {"load", "temp"} <= set(cols)

    def test_group_bounds_split_source_series(self):
        groups = _group_bounds(_wide_table())
        assert len(groups) == 2
        assert sorted(len(rows) for _, rows in groups) == [128, 128]

    def test_series_gen_flattens_to_univariate(self, tmp_path):
        import pyarrow.parquet as pq

        path = tmp_path / "part0.parquet"
        pq.write_table(_wide_table(), path)
        rows = list(
            _series_gen(
                [str(path)], "Toy", {"timestamp": "date", "freq": "h"}, MIN_SERIES_LEN
            )
        )
        # 2 source series x 2 real channels; Open/SinceYear/date excluded.
        assert len(rows) == 4
        assert {r["item_id"].rsplit("_", 1)[1] for r in rows} == {"load", "temp"}
        for r in rows:
            assert len(r["target"]) == 128
            assert r["freq"] == "h"
            assert np.asarray(r["target"]).dtype == np.float32


def _subsecond_table(step_ms: int = 599994) -> pa.Table:
    """A dir sampled off the second, as BTS is: 10-minutely but logged at 599994ms."""
    n = 128
    rng = np.random.default_rng(3)
    offsets = (np.arange(n) * step_ms).astype("timedelta64[ms]")
    return pa.table(
        {
            "date": pa.array(np.datetime64("2021-01-01T00:03:14.602", "ms") + offsets),
            "load": pa.array(rng.standard_normal(n)),
            "_original_filename": pa.array(["a"] * n).dictionary_encode(),
        }
    )


class TestSubSecondTimestamps:
    """BTS took the whole 106-dir build down at 10/106; both halves are pinned here."""

    def test_start_survives_the_lotsa_timestamp_cast(self, tmp_path):
        """`start` must fit timestamp[s]: Arrow refuses a lossy cast, it won't truncate.

        Asserted against the real LOTSA_FEATURES rather than just checking the
        floored value, since the schema is what actually rejected it:
        "Casting from timestamp[us] to timestamp[s] would lose data: 1609459394602000".
        """
        import pyarrow.parquet as pq

        import datasets
        from moiraix.data.builder.lotsa_v1.rmisc import LOTSA_FEATURES

        path = tmp_path / "part0.parquet"
        pq.write_table(_subsecond_table(), path)
        gen_kwargs = dict(
            files=[str(path)], name="BTSLike", meta={"timestamp": "date"}, min_len=64
        )
        ds = datasets.Dataset.from_generator(
            _series_gen,
            features=LOTSA_FEATURES,
            gen_kwargs=gen_kwargs,
            cache_dir=str(tmp_path / "cache"),
        )
        assert ds.num_rows == 1
        assert ds[0]["start"].microsecond == 0

    def test_jittered_freq_snaps_to_a_clean_alias(self, tmp_path):
        """Off-the-second sampling must not render literally as '599994ms'."""
        import pyarrow.parquet as pq

        path = tmp_path / "part0.parquet"
        pq.write_table(_subsecond_table(), path)
        rows = list(_series_gen([str(path)], "BTSLike", {"timestamp": "date"}, 64))
        assert rows[0]["freq"] == "10min"

    def test_genuine_subsecond_freq_is_left_exact(self, tmp_path):
        """The snap is only for >=1s: BCI's 4ms EEG sampling is real, not jitter."""
        import pyarrow.parquet as pq

        path = tmp_path / "part0.parquet"
        pq.write_table(_subsecond_table(step_ms=4), path)
        rows = list(_series_gen([str(path)], "BCILike", {"timestamp": "date"}, 64))
        assert rows[0]["freq"] == "4ms"
