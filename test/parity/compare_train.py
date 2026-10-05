"""Compare per-step scalars logged by two cli.train runs (TensorBoard events)."""

import glob
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def scalars(run_dir):
    acc = EventAccumulator(
        glob.glob(f"{run_dir}/logs/**/events.out.tfevents.*", recursive=True)[0]
    )
    acc.Reload()
    return {
        tag: [(e.step, e.value) for e in acc.Scalars(tag)]
        for tag in acc.Tags()["scalars"]
    }


def main(a_dir, b_dir):
    a, b = scalars(a_dir), scalars(b_dir)
    ok = set(a) == set(b)
    if not ok:
        print("tag sets differ:", set(a) ^ set(b))
    for tag in sorted(set(a) & set(b)):
        same = a[tag] == b[tag]
        ok &= same
        diff = max((abs(x[1] - y[1]) for x, y in zip(a[tag], b[tag])), default=0.0)
        print(
            f"{tag:55s} n={len(a[tag]):3d} {'identical' if same else f'max|diff|={diff:.3e}'}"
        )
    print("ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
