#  RMISC (https://huggingface.co/datasets/nju-zhangsq/RMISC) as LOTSA-format
#  pretraining data.
#
#  RMISC ships 197 dataset dirs (816 GB of parquet). Each dir is a WIDE table:
#  columns are channels, rows are timesteps, and `_original_filename` separates
#  the independent series that were concatenated into one file. We keep only the
#  dirs that survive the overlap filter (see EXCLUDE below), flatten every
#  (series, channel) pair into its own univariate LOTSA row, and delete the raw
#  parquet as soon as a dir is converted -- peak disk stays at one dir's raw
#  parquet plus its Arrow output, not 816 GB.
#
#  OVERLAP FILTER (deliberately conservative, per EXCLUDE's four reason codes):
#    gift_test  -- the dir IS a GIFT-Eval test dataset. Training on it is direct
#                  test leakage. Includes the non-obvious ones: Beutenberg ==
#                  Jena weather station (gift `jena_weather`), Mdense ==
#                  `M_DENSE`, HierachicalSales == `hierarchical_sales`.
#    lotsa_dup  -- already present in the gift_eval pretrain corpus under
#                  another name, so including it would silently double its
#                  sampling weight. Includes derivative repackagings:
#                  WeatherBench5-625deg is ERA5 regridded to 5.625 deg (we
#                  already have era5_1989..2018), AzurePublicDatasetV1 is
#                  `azure_vm_traces_2017`, QtrafficSpeed is `Q-TRAFFIC`.
#    rmisc_test -- RMISC's own meta.json marks split == "test".
#    sim        -- physics-simulation telemetry, not real-world series.
#
#  Build (resumable; re-running skips dirs that already have a DONE marker):
#    python -m moiraix.data.builder.lotsa_v1.rmisc --num-proc 8
#  or via slurm: sbatch slurm/scripts/build_rmisc.sh

import argparse
import json
import os
import shutil
import traceback
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

import datasets
from moiraix.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder
from .synth_stress import LOTSA_FEATURES

REPO_ID = "nju-zhangsq/RMISC"
PREFIX = "rmisc_"  # LOTSA dataset name = PREFIX + RMISC dir name

MIN_SERIES_LEN = 64  # >= 2 patches at patch_size 32; drop shorter remnants
MIN_DISTINCT = 3  # drop constants and binary flags (see _is_degenerate)
MAX_CYCLE_PERIOD = 31  # longest exact integer cycle treated as a calendar counter
DEFAULT_FREQ = "h"  # when no timestamp column and meta.freq is unusable
DEFAULT_START = "2000-01-01"

#  Column names that are timestamps rather than channels. meta.json's own
#  `timestamp` field is unreliable (e.g. ETT declares null but has a `date`
#  column), so we union it with this exact-name blacklist (case-insensitive).
TIMESTAMP_NAMES = frozenset(
    x.lower()
    for x in (
        "date",
        "time",
        "timestamp",
        "datetime",
        "date_time",
        "ds",
        "dt",
        "utc_timestamp",
        "cet_cest_timestamp",
        "measured_on",
        "trade_date",
        "valid_time",
        "index",
        "unnamed: 0",
    )
)

EXCLUDE: dict[str, str] = {
    #  --- GIFT-Eval TEST datasets: training on these is test leakage ---
    **{
        d: "gift_test"
        for d in (
            "M4",
            "Electricity",
            "ElectricityHourly",
            "SolarEnergy",
            "Solar10Minutes",
            "Solar4Seconds",
            "CovidDeaths",
            "Covid",
            "USBirths",
            "TemperatureRain",
            "KDDCup2018",
            "Restaurant",
            "HierachicalSales",
            "LoopSeattleLA",
            "Mdense",
            "ETT",
            "ETTMulti",
            "Weather",
            "WeatherTest",
            "BeutenbergWeather",
            "Bizitobs_application",
            "Bizitobs_l2c_H",
        )
    },
    #  --- already in the gift_eval pretrain corpus under another name ---
    **{
        d: "lotsa_dup"
        for d in (
            "AustralianElectricityDemand",
            "AzurePublicDatasetV1",
            "BDG2-Bear",
            "BDG2-Fox",
            "BDG2-Panther",
            "BDG2-Rat",
            "BeijingAirQuality",
            "BeijingSubway",
            "Bitcoin",
            "BuildingsBenchComAmy",
            "BuildingsBenchComTmy",
            "BuildingsBenchResAmy",
            "BuildingsBenchResTmy",
            "BuildingsBenchRealCSV",
            "Bull",
            "CMIP6-2000-PartI",
            "CMIP6-2000-PartII",
            "CMIP6-2000-PartIII",
            "CMIP6-2005-PartI",
            "CMIP6-2005-PartII",
            "CMIP6-2005-PartIII",
            "CMIP6-2010-PartI",
            "CMIP6-2010-PartII",
            "CMIP6-2010-PartIII",
            "Cif2016-12",
            "Cif2016-6",
            "CovidMobility",
            "ERA5HourlySingleLevels",
            "FavoritaSales",
            "FavoritaTransactions",
            "FavoritaTransactionsKnownOil",
            "FredMD",
            "GFC2012",
            "Hog",
            "Ideal",
            "LargeST",
            "LondonSmartMeters",
            "M3",
            "M5",
            "MelbournePedestrianCounts",
            "NN5Daily",
            "OikolabWeather",
            "PEMS-Bay-METRO-LA",
            "PEMSCalifornia",
            "ProEnFo",
            "Pvdaq",
            "QtrafficSpeed",
            "Rideshare",
            "ResidentialPower",
            "SHandHZMetro",
            "SmartMeterUK30m",
            "SmartMeterUK60m",
            "Subseasonal",
            "Sunspots",
            "TourismMonthly",
            "Traffic",
            "TrafficHourly",
            "VehicleTrips",
            "WeatherBench5-625deg",
            "WikiTrafficDaily",
            "WikiTrafficWeekly",
            "WindFarms",
            "WindPower4secs",
        )
    },
    #  --- RMISC's own held-out split (meta.json split == "test") ---
    **{
        d: "rmisc_test"
        for d in (
            "CSTSNonnormalTest",
            "CSTSNormalTest",
            "Darts",
            "Illness",
            "KaggleTS",
        )
    },
    #  --- physics-simulation telemetry, not real-world time series ---
    "Behavior-1k": "sim",
}

#  The 106 RMISC dirs that survive EXCLUDE. Frozen so `dataset_list` is static
#  (configs resolve it via ${cls_getattr:${._target_},dataset_list}) and so a
#  silent upstream repo change can't quietly widen what we train on.
RMISC_DIRS: tuple[str, ...] = (
    "ACSF1",
    "AMarketChina",
    "AMarketChinaKnownOpen",
    "AliCar",
    "ApplianceEnergy",
    "AzurePublicDatasetV2",
    "BCI_Competetion_IV_1",
    "BCI_Competetion_IV_2a",
    "BCI_Competetion_IV_2b",
    "BTS",
    "BatteryRUL",
    "BooksPerPerson",
    "BoschCNC",
    "BrainInvadersBi2014b",
    "BritainCoal",
    "CSE-CIC-IDS2018",
    "CSI500",
    "CSTSNonnormalTrain",
    "CSTSNormalTrain",
    "Car",
    "CausalEffects",
    "ChengduTaxi",
    "ChinaMinuteStock",
    "CinCECGTorso",
    "Computers",
    "Cryptocurrency",
    "CryptocurrencyKnownOpen",
    "Dominick",
    "EMG4Gestures",
    "ERCOT",
    "EbayServer",
    "EigenWorms",
    "ExchangeRate",
    "FordA",
    "FrothFlotation",
    "Gait",
    "GasPipeline",
    "GasSensorDynamic",
    "GasSensorTemperature",
    "GlobalClimateChange",
    "HAR70Plus",
    "HARTH",
    "HetergeneousHAR",
    "HouseholdPower",
    "HungarianChickenpoxCases",
    "IndoorLocalisation",
    "InlineSkate",
    "KeplerLightCurves",
    "LBNL",
    "MZVAV",
    "MetroTraffic",
    "Metropt3",
    "MiniApp",
    "MotionSense",
    "MotorTemperature",
    "NAB",
    "NIFTYStock",
    "NIFTYStockKnownOpen",
    "OPSD",
    "OPSD-Household",
    "OPSD-PV-Wind",
    "OPSD-When2Heat",
    "OccupancyDetection",
    "OilWell",
    "PAMAP2",
    "PM25FiveCities",
    "PUMP",
    "Rebound",
    "Rohlik_orders_1D",
    "Rohlik_orders_1W",
    "RoomOccupancy",
    "Rossmann_1D",
    "Rossmann_1W",
    "SP500",
    "SP500KnownOpen",
    "SWAT",
    "Satellite",
    "ServerMachineDataset",
    "ShellHackathon",
    "SmartMeterAus30m",
    "SmartMeterAus60m",
    "SmellSensor",
    "StarLightCurves",
    "StockFactorsCleaned",
    "StockMarketData",
    "T-Drive",
    "TetuanPowerConsumption",
    "Tigge",
    "TimeMMD",
    "TushareETFDaily",
    "TushareIndexDaily",
    "TushareStockDaily",
    "TushareStockDailyMetrics",
    "TushareStockWeekly",
    "UK-DALE",
    "UKEconomy",
    "USAirPollution",
    "WADI",
    "WISDM_V1",
    "WISDM_V2",
    "WISDM_V3",
    "WeeklyFuelPricesItaly",
    "WeeklyRoadFuelPrices",
    "WindElec",
    "Worms",
    "XiamenAirQuality",
)

RMISC_DATASETS: list[str] = [PREFIX + d for d in RMISC_DIRS]


def _normalize_freq(freq) -> Optional[str]:
    """Coerce a meta.json `freq` entry into a pandas offset alias, or None."""
    if isinstance(freq, (list, tuple)):
        freq = freq[0] if freq else None
    if not isinstance(freq, str) or not freq:
        return None
    try:
        pd.tseries.frequencies.to_offset(freq)
    except (ValueError, TypeError):
        return None
    return freq


def _timestamp_columns(meta: dict, schema: pa.Schema) -> set[str]:
    declared = meta.get("timestamp")
    if isinstance(declared, str):
        declared = [declared]
    elif not isinstance(declared, (list, tuple)):
        declared = []
    return {
        name
        for name in schema.names
        if name in declared or name.lower() in TIMESTAMP_NAMES
    }


def _channel_columns(meta: dict, schema: pa.Schema) -> list[str]:
    """Numeric channel columns: everything that isn't a timestamp or the group key."""
    drop = _timestamp_columns(meta, schema) | {"_original_filename"}
    return [
        f.name
        for f in schema
        if f.name not in drop
        and (pa.types.is_floating(f.type) or pa.types.is_integer(f.type))
    ]


def _group_bounds(table: pa.Table) -> list[tuple[int, np.ndarray]]:
    """Row indices for each independent source series in a wide RMISC table.

    Series are delimited by `_original_filename`. Rows of one series are not
    guaranteed contiguous, so we stable-argsort by the dictionary code: that
    preserves within-series time order while grouping the rows.
    """
    if "_original_filename" not in table.schema.names:
        return [(0, np.arange(table.num_rows))]
    col = table.column("_original_filename").combine_chunks()
    if not pa.types.is_dictionary(col.type):
        col = pc.dictionary_encode(col)
    codes = col.indices.to_numpy(zero_copy_only=False).astype(np.int64)
    order = np.argsort(codes, kind="stable")
    sorted_codes = codes[order]
    edges = np.flatnonzero(np.diff(sorted_codes)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges, [len(order)]])
    return [
        (int(sorted_codes[lo]), order[lo:hi])
        for lo, hi in zip(starts, stops)
        if hi > lo
    ]


def _timestamp_index(table: pa.Table, ts_cols: set[str]):
    """Materialize the first usable timestamp column as a datetime64 array, once.

    Returned to the caller so per-series starts are a numpy lookup rather than a
    pyarrow `take` per group -- see _series_gen.
    """
    for name in table.schema.names:
        if name not in ts_cols:
            continue
        try:
            ts = pd.to_datetime(table.column(name).to_pandas(), errors="coerce")
            if ts.notna().sum() < 2:
                continue
            if getattr(ts.dtype, "tz", None) is not None:
                ts = ts.dt.tz_localize(None)
            return ts.to_numpy()
        except (ValueError, TypeError, pa.ArrowInvalid, OverflowError):
            continue
    return None


def _resolve_freq(table: pa.Table, ts_cols: set[str], meta: dict) -> str:
    """Infer freq from the timestamps themselves; meta.freq is only a fallback.

    meta.json's freq is ambiguous across dirs (`"m"` is minute in some, month in
    others), so the actual spacing of the timestamp column wins where one exists.
    """
    for name in table.schema.names:
        if name not in ts_cols:
            continue
        try:
            head = table.column(name).slice(0, 256).to_pandas()
            ts = pd.to_datetime(head, errors="coerce").dropna()
            if len(ts) < 3:
                continue
            delta = pd.Series(ts.values).diff().dropna()
            delta = delta[delta > pd.Timedelta(0)]
            if delta.empty:
                continue
            median = delta.median()
            #  Jittered timestamps give a median that is only near-round: BTS is
            #  10-minutely but logs at 599994ms, which to_offset renders literally
            #  as '599994ms'. Snap to the second for anything sampled at 1s or
            #  slower; sub-second dirs (BCI's 4ms EEG) are left exact.
            if median >= pd.Timedelta("1s"):
                median = median.round("s")
            offset = pd.tseries.frequencies.to_offset(median)
            if offset is not None:
                return offset.freqstr
        except (ValueError, TypeError, pa.ArrowInvalid, OverflowError):
            continue
    return _normalize_freq(meta.get("freq")) or DEFAULT_FREQ


def _is_degenerate(values: np.ndarray) -> bool:
    """True for channels that are not forecastable series.

    RMISC's wide tables mix real signals with exogenous junk: Rossmann carries
    `Open`/`Promo` (binary flags), `CompetitionOpenSinceYear` (constant per
    store) and `DayOfWeek` (a period-7 sawtooth) alongside real `Sales`. We
    cannot filter these by meta.json's targets/covariates split -- ETT's
    "covariates" (HUFL, MUFL) and ApplianceEnergy's (T1..T9) are genuine
    sensor series -- nor by column name, since Rossmann's binary `Open` and
    Tushare's `open` price collide. So the test is statistical:

      * fewer than MIN_DISTINCT distinct values  -> constant or binary flag
      * an exact integer cycle of period == its own cardinality -> a calendar
        counter. Real series are never periodic to the bit, so this cannot
        false-positive on, say, an intermittent-demand count.
    """
    finite = values[np.isfinite(values)]
    if finite.size < MIN_SERIES_LEN:
        return True
    distinct = np.unique(finite)
    if distinct.size < MIN_DISTINCT:
        return True
    period = distinct.size
    if (
        period <= MAX_CYCLE_PERIOD
        and values.size > period
        and np.array_equal(distinct, np.floor(distinct))
        and np.array_equal(values[:-period], values[period:])
    ):
        return True
    return False


def _series_gen(files: list[str], name: str, meta: dict, min_len: int):
    """Yield one LOTSA row per (source series, channel) pair.

    Columns are materialized to numpy ONCE per file and then sliced per group.
    A pyarrow `take` per (group, channel) instead would be 663k takes over a
    13M-row ChunkedArray for AzurePublicDatasetV2 alone, which does not finish.
    """
    default_start = np.datetime64(DEFAULT_START, "s")
    for path in files:
        table = pq.read_table(path)
        channels = _channel_columns(meta, table.schema)
        if not channels:
            continue
        ts_cols = _timestamp_columns(meta, table.schema)
        freq = _resolve_freq(table, ts_cols, meta)
        stamps = _timestamp_index(table, ts_cols)
        part = Path(path).stem

        columns = {
            channel: table.column(channel)
            .combine_chunks()
            .to_numpy(zero_copy_only=False)
            .astype(np.float32)
            for channel in channels
        }
        groups = _group_bounds(table)
        del table

        for code, rows in groups:
            start = default_start
            if stamps is not None:
                first = stamps[rows[0]]
                if not np.isnat(first):
                    start = first
            #  LOTSA_FEATURES declares `start` as timestamp[s], and Arrow refuses a
            #  lossy cast rather than truncating: BTS is sampled sub-second, so its
            #  raw start (…:14.602) fails with "Casting from timestamp[us] to
            #  timestamp[s] would lose data". Floor before handing it over.
            start_ts = pd.Timestamp(start).floor("s")
            for channel, column in columns.items():
                values = column[rows]
                if len(values) < min_len or _is_degenerate(values):
                    continue
                yield dict(
                    item_id=f"{PREFIX}{name}_{part}_{code}_{channel}",
                    start=start_ts,
                    freq=freq,
                    target=values,
                )


def _download_dir(name: str, raw_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    snapshot_download(
        REPO_ID,
        repo_type="dataset",
        local_dir=str(raw_root),
        allow_patterns=[f"{name}/*.parquet", f"{name}/meta.json"],
        max_workers=8,
    )
    return raw_root / name


class RMISCDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = RMISC_DATASETS
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(
        self,
        dataset: str,
        num_proc: Optional[int] = None,
        min_len: int = MIN_SERIES_LEN,
        keep_raw: bool = False,
        raw_root: Optional[Path] = None,
    ) -> dict:
        assert dataset.startswith(PREFIX), f"{dataset} is not an RMISC dataset"
        name = dataset[len(PREFIX) :]
        assert name in RMISC_DIRS, f"{name} is not a kept RMISC dir"

        num_proc = num_proc or min(8, os.cpu_count())
        out_path = self.storage_path / dataset
        raw_root = raw_root or self.storage_path / "_rmisc_raw"
        cache_dir = self.storage_path / f"{dataset}_build_cache"

        raw_dir = _download_dir(name, raw_root)
        meta = json.loads((raw_dir / "meta.json").read_text())
        files = sorted(str(p) for p in raw_dir.glob("*.parquet"))
        assert files, f"no parquet downloaded for {name}"

        hf_dataset = datasets.Dataset.from_generator(
            _series_gen,
            features=LOTSA_FEATURES,
            gen_kwargs=dict(files=files, name=name, meta=meta, min_len=min_len),
            num_proc=min(num_proc, len(files)),
            cache_dir=str(cache_dir),
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(out_path)
        num_series = len(hf_dataset)
        del hf_dataset

        shutil.rmtree(cache_dir, ignore_errors=True)
        if not keep_raw:
            shutil.rmtree(raw_dir, ignore_errors=True)

        stats = dict(
            dataset=dataset,
            rmisc_dir=name,
            num_series=num_series,
            domain=meta.get("domain"),
            quality=meta.get("quality"),
            num_source_series=meta.get("num_series"),
        )
        (out_path / "rmisc_manifest.json").write_text(json.dumps(stats, indent=1))
        print(f"[{dataset}] {num_series} univariate series -> {out_path}", flush=True)
        return stats


def _hub_cache_dir() -> Path:
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE) / f"datasets--{REPO_ID.replace('/', '--')}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the RMISC LOTSA-format corpus (download -> univariate -> "
        "delete raw, one dir at a time; resumable)."
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="LOTSA names to build (default: all kept RMISC dirs)",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument("--min-len", type=int, default=MIN_SERIES_LEN)
    parser.add_argument("--storage-path", default=None)
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep the downloaded parquet (default: delete after each dir)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild dirs that already have output",
    )
    args = parser.parse_args()

    kwargs = {}
    if args.storage_path is not None:
        kwargs["storage_path"] = Path(args.storage_path)

    targets = args.datasets or RMISC_DATASETS
    builder = RMISCDatasetBuilder(datasets=targets, **kwargs)
    storage = builder.storage_path

    summary, failed = [], []
    for i, ds in enumerate(targets, 1):
        done_marker = storage / ds / "rmisc_manifest.json"
        if done_marker.exists() and not args.force:
            summary.append(json.loads(done_marker.read_text()))
            print(f"[{ds}] ({i}/{len(targets)}) already built, skipping", flush=True)
            continue
        print(f"[{ds}] ({i}/{len(targets)}) building...", flush=True)
        shutil.rmtree(storage / ds, ignore_errors=True)
        #  One malformed dir must not cost the other 105: a single unhandled
        #  ArrowInvalid on BTS took the whole run down at 10/106. Record it, drop
        #  its partial output so it cannot be mistaken for built, and carry on --
        #  the config generator already emits only the datasets that exist.
        try:
            summary.append(
                builder.build_dataset(
                    ds,
                    num_proc=args.num_proc,
                    min_len=args.min_len,
                    keep_raw=args.keep_raw,
                )
            )
        except Exception as exc:  # noqa: BLE001 -- any failure is per-dir, not fatal
            failed.append(dict(dataset=ds, error=f"{type(exc).__name__}: {exc}"))
            print(f"[{ds}] FAILED, skipping: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc()
            shutil.rmtree(storage / ds, ignore_errors=True)
            shutil.rmtree(storage / f"{ds}_build_cache", ignore_errors=True)
        finally:
            #  The hub cache keeps a second copy of every blob it downloaded; drop it
            #  so peak disk stays at one dir's worth of raw parquet -- on the failure
            #  path too, or a crash-loop of big dirs would fill the disk.
            if not args.keep_raw:
                shutil.rmtree(
                    storage / "_rmisc_raw" / ds[len(PREFIX) :], ignore_errors=True
                )
                shutil.rmtree(_hub_cache_dir(), ignore_errors=True)

    if not args.keep_raw:
        shutil.rmtree(storage / "_rmisc_raw", ignore_errors=True)

    total = sum(s["num_series"] for s in summary)
    (storage / "rmisc_build_summary.json").write_text(
        json.dumps(
            dict(
                total_series=total,
                num_datasets=len(summary),
                datasets=summary,
                failed=failed,
            ),
            indent=1,
        )
    )
    print(
        f"\nRMISC build complete: {total} univariate series across {len(summary)} datasets"
    )
    if failed:
        print(f"{len(failed)} dirs FAILED and were skipped:")
        for f in failed:
            print(f"  {f['dataset']}: {f['error']}")
