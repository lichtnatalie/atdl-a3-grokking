#!/usr/bin/env bash
# All A3 configurations, in priority order. Each sweep is resumable.
# About 55 min per configuration on 2 CPU cores (about 30 s per run).
set -e
cd "$(dirname "$0")/.."
python experiments/run_sweep.py --tag released                      # embedding lr 1e-3, decoder lr 1e-4
python experiments/run_sweep.py --tag dec_lr_1e-3 --eta-dec 1e-3    # faster decoder
python experiments/run_sweep.py --tag dec_lr_1e-5 --eta-dec 1e-5    # slower decoder
python experiments/run_sweep.py --tag emb_lr_3e-3 --eta-reprs 3e-3  # faster embeddings (H3)
python experiments/run_sweep.py --tag emb_lr_1e-2 --eta-reprs 1e-2  # much faster embeddings (H3)
