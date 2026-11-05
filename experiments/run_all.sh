#!/usr/bin/env bash
# All A3 configurations, in priority order. Each sweep is resumable.
# About 55 min per configuration on 2 CPU cores (about 30 s per run).
#
# The repository already contains the finished results in results/, so running
# this script as it is only resumes and skips every finished run. For a fresh
# reproduction, write to a new folder:
#   OUT_ROOT=results_fresh experiments/run_all.sh
#   python experiments/analyse.py --results-root results_fresh
set -e
cd "$(dirname "$0")/.."
OUT_ROOT="${OUT_ROOT:-results}"
python experiments/run_sweep.py --out-root "$OUT_ROOT" --tag released                      # embedding lr 1e-3, decoder lr 1e-4
python experiments/run_sweep.py --out-root "$OUT_ROOT" --tag dec_lr_1e-3 --eta-dec 1e-3    # faster decoder
python experiments/run_sweep.py --out-root "$OUT_ROOT" --tag dec_lr_1e-5 --eta-dec 1e-5    # slower decoder
python experiments/run_sweep.py --out-root "$OUT_ROOT" --tag emb_lr_3e-3 --eta-reprs 3e-3  # faster embeddings (H3)
python experiments/run_sweep.py --out-root "$OUT_ROOT" --tag emb_lr_1e-2 --eta-reprs 1e-2  # much faster embeddings (H3)
