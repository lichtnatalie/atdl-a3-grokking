"""
One configuration of the A3 sweep: 11 training set sizes x 20 seeds, as in A2,
with per-step logging (src/logged_training.py).

Released setting (authors' produce_figure_data.py): p = 10, sizes 5, 10, ..., 50, 54,
10^4 steps, AdamW, embedding lr 1e-3, decoder lr 1e-4, no weight decay, MSE.

Outputs
  results/<tag>/runs.csv              one row per run (events and final values)
  results/<tag>/traces/n<NN>_s<SS>.npz  curves every 10 steps
Rows are appended run by run, and finished (size, seed) pairs are skipped on a
restart, so an interrupted sweep can simply be started again.

Example
  python experiments/run_sweep.py --tag released
  python experiments/run_sweep.py --tag dec_lr_1e-3 --eta-dec 1e-3
  python experiments/run_sweep.py --tag emb_lr_1e-2 --eta-reprs 1e-2
"""

import argparse
import csv
import os
import sys
import time
from multiprocessing import get_context

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = list(range(5, 51, 5)) + [54]


def one_run(args):
    n, seed, cfg, out = args
    sys.path.insert(0, os.path.join(ROOT, "src"))
    from logged_training import run_logged
    t = time.time()
    s, tr = run_logged(n, seed, steps=cfg["steps"], eta_reprs=cfg["eta_reprs"],
                       eta_dec=cfg["eta_dec"], wd_dec=cfg["wd_dec"])
    np.savez_compressed(os.path.join(out, "traces", f"n{n:02d}_s{seed:02d}.npz"), **tr)
    s["seconds"] = round(time.time() - t, 1)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--eta-reprs", type=float, default=1e-3)  # authors' default
    ap.add_argument("--eta-dec", type=float, default=1e-4)    # authors' default
    ap.add_argument("--wd-dec", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    cfg = dict(steps=a.steps, eta_reprs=a.eta_reprs, eta_dec=a.eta_dec, wd_dec=a.wd_dec)
    out = os.path.join(ROOT, "results", a.tag)
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    path = os.path.join(out, "runs.csv")

    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add((int(r["n_train"]), int(r["seed"])))
    jobs = [(n, s, cfg, out) for s in range(a.seeds) for n in SIZES if (n, s) not in done]
    print(f"[{a.tag}] {len(done)} done, {len(jobs)} to run, config {cfg}", flush=True)

    with get_context("spawn").Pool(a.workers) as pool:
        for k, s in enumerate(pool.imap_unordered(one_run, jobs), 1):
            new = not os.path.exists(path)
            with open(path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(s))
                if new:
                    w.writeheader()
                w.writerow(s)
            print(f"[{a.tag}] {k}/{len(jobs)} n={s['n_train']} seed={s['seed']} "
                  f"rqi>{0.95}: {s['reached_rqi']} train90 at {s['iter_train90']} "
                  f"({s['seconds']} s)", flush=True)


if __name__ == "__main__":
    main()
