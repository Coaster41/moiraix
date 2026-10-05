"""Compare two GIFT-Eval all_results.csv files metric-by-metric (ignores the
model-name column). Prints per-metric max relative diff and the geomean WQL."""

import sys

import numpy as np
import pandas as pd

WQL = "eval_metrics/mean_weighted_sum_quantile_loss"


def load(path):
    df = pd.read_csv(path).drop(columns=["model"]).set_index("dataset").sort_index()
    return df.select_dtypes("number")


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    if set(a.index) != set(b.index):
        print("dataset sets differ:", sorted(set(a.index) ^ set(b.index)))
    common = a.index.intersection(b.index)
    a, b = a.loc[common], b.loc[common]
    identical = a.equals(b)
    for col in a.columns:
        rel = (a[col] - b[col]).abs() / a[col].abs().clip(lower=1e-12)
        print(f"{col:50s} max rel diff {np.nanmax(rel.values):.3e}")
    gm = lambda s: float(np.exp(np.log(s).mean()))
    print(f"rows={len(common)}  geomean WQL: {gm(a[WQL]):.6f} vs {gm(b[WQL]):.6f}")
    print("ALL IDENTICAL" if identical else "NOT BIT-IDENTICAL")


if __name__ == "__main__":
    main(*sys.argv[1:3])
