"""Compare two dump_outputs.py results: weights must hash-match and forecasts
must be bit-identical (reports max abs diff otherwise)."""

import sys

import numpy as np
import torch


def main(a_path, b_path):
    a = torch.load(a_path, weights_only=False)
    b = torch.load(b_path, weights_only=False)
    ok = True
    for name, ea in a["ckpts"].items():
        eb = b["ckpts"][name]
        same_w = ea["weights"] == eb["weights"]
        same_f = ea["flags"] == eb["flags"]
        ok &= same_w and same_f
        flags = "identical" if same_f else f"DIFFER {ea['flags']} vs {eb['flags']}"
        print(
            f"{name}: weights {'identical' if same_w else 'DIFFER'} "
            f"({len(ea['weights'])} tensors), flags {flags}"
        )
        for label, fa in ea["forecasts"].items():
            fb = eb["forecasts"][label]
            if fa.shape != fb.shape:
                ok = False
                print(f"    {label:12s} SHAPE {fa.shape} vs {fb.shape}")
                continue
            exact = np.array_equal(fa, fb, equal_nan=True)
            ok &= exact
            diff = np.nanmax(np.abs(fa - fb)) if not exact else 0.0
            print(
                f"    {label:12s} {'bit-identical' if exact else f'max|diff|={diff:.3e}'}  {fa.shape}"
            )
    print("ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
