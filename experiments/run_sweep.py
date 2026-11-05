"""
One configuration of the A3 sweep: 11 training set sizes x 20 seeds, with
per-step logging (src/logged_training.py).

Released setting (authors' produce_figure_data.py): p = 10, sizes 5, 10, ..., 50, 54,
10^4 steps, AdamW, embedding lr 1e-3, decoder lr 1e-4, no weight decay, MSE.

Outputs (under --out-root, default results/)
  <out-root>/<tag>/config.json           the configuration of this sweep
  <out-root>/<tag>/runs.csv              one row per run (events and final values)
  <out-root>/<tag>/traces/n<NN>_s<SS>.npz  curves every 10 steps

Resuming. Rows are appended run by run, and finished (size, seed) pairs are
skipped when the same command is started again. Before resuming, the stored
configuration (config.json, and the configuration columns of runs.csv) must
match the requested one exactly; otherwise the script stops, so that runs with
different settings are never mixed under one tag.

Fresh reproduction. The repository already contains the finished results, so
re-running with the default output root only resumes (and skips everything).
To regenerate the experiments from scratch, use a new output root:
  python experiments/run_sweep.py --tag released --out-root results_fresh

Examples
  python experiments/run_sweep.py --tag released
  python experiments/run_sweep.py --tag dec_lr_1e-3 --eta-dec 1e-3
  python experiments/run_sweep.py --tag emb_lr_1e-2 --eta-reprs 1e-2
"""

import argparse
import csv
import json
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


def check_config(out, cfg, path):
    """Stop if this tag already holds runs with a different configuration."""
    cfg_path = os.path.join(out, "config.json")
    if os.path.exists(cfg_path):
        stored = json.load(open(cfg_path))
        if stored != cfg:
            sys.exit(f"[stop] {cfg_path} holds {stored}, but this call asks for {cfg}. "
                     "Use a different --tag or --out-root.")
    if os.path.exists(path):
        # results from before config.json existed: check the columns of runs.csv
        for r in csv.DictReader(open(path)):
            row = {k: (int(float(r[k])) if k == "steps" else float(r[k])) for k in cfg}
            if row != cfg:
                sys.exit(f"[stop] {path} contains runs with {row}, but this call asks "
                         f"for {cfg}. Use a different --tag or --out-root.")
    json.dump(cfg, open(cfg_path, "w"), indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-root", default=os.path.join(ROOT, "results"),
                    help="folder that holds one subfolder per tag (default: results/)")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--eta-reprs", type=float, default=1e-3)  # authors' default
    ap.add_argument("--eta-dec", type=float, default=1e-4)    # authors' default
    ap.add_argument("--wd-dec", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()

    cfg = dict(steps=a.steps, eta_reprs=a.eta_reprs, eta_dec=a.eta_dec, wd_dec=a.wd_dec)
    out = os.path.join(a.out_root, a.tag)
    os.makedirs(os.path.join(out, "traces"), exist_ok=True)
    path = os.path.join(out, "runs.csv")
    check_config(out, cfg, path)

    done = set()
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            done.add((int(r["n_train"]), int(r["seed"])))
    jobs = [(n, s, cfg, out) for s in range(a.seeds) for n in SIZES if (n, s) not in done]
    print(f"[{a.tag}] {out}: {len(done)} done, {len(jobs)} to run, config {cfg}", flush=True)
    if not jobs:
        return

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
