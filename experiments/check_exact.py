"""
Two checks before any experiment is trusted.

1. The gradient recording does not change training: for a few (size, seed)
   pairs, train_add with and without the recording optimiser gives identical
   loss, accuracy, RQI and embedding trajectories (exact equality).
2. The released setting reproduces the A2 runs (results/a2_reference/runs_baseline.csv,
   copied from the A2 repository): same authors' iter_train, iter_rqi and final RQI.

Usage: python experiments/check_exact.py [--n-a2 12]
"""

import argparse
import contextlib
import csv
import io
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from logged_training import _record_gradients, run_logged, train_add  # noqa: E402


def plain_and_recorded(n, seed, steps):
    torch.set_num_threads(1)
    with contextlib.redirect_stdout(io.StringIO()):
        a = train_add(train_num=n, seed=seed, steps=steps, eff_steps=1)
        with _record_gradients():
            b = train_add(train_num=n, seed=seed, steps=steps, eff_steps=1)
    return a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a2", type=int, default=12, help="A2 runs to compare")
    a = ap.parse_args()

    print("Check 1: recording does not change training")
    keys = ["loss_train", "acc_train", "acc_test", "rqi", "repr_nn"]
    for n, seed in [(25, 0), (40, 3), (54, 7)]:
        x, y = plain_and_recorded(n, seed, steps=2000)
        same = all(np.array_equal(np.asarray(x[k]), np.asarray(y[k])) for k in keys)
        print(f"  n={n:2d} seed={seed}: identical = {same}")
        assert same

    print(f"Check 2: first {a.n_a2} A2 baseline runs reproduce")
    ref = os.path.join(ROOT, "results", "a2_reference", "runs_baseline.csv")
    rows = list(csv.DictReader(open(ref)))[: a.n_a2]
    for r in rows:
        n, seed = int(r["n_train"]), int(r["seed"])
        s, _ = run_logged(n, seed)
        ok = (s["authors_iter_train"] == int(r["iter_train"])
              and s["authors_iter_rqi"] == int(r["iter_rqi"])
              and abs(s["final_rqi"] - float(r["final_rqi"])) < 1e-12)
        print(f"  n={n:2d} seed={seed:2d}: iter_train {s['authors_iter_train']} vs "
              f"{r['iter_train']}, iter_rqi {s['authors_iter_rqi']} vs {r['iter_rqi']}"
              f" -> {'OK' if ok else 'MISMATCH'}")
        assert ok
    print("All checks passed.")


if __name__ == "__main__":
    main()
