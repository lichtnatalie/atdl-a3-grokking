# ATDL Assignment 3: representation learning vs. decoder overfitting

Advanced Topics in Deep Learning, University of Copenhagen, 2026

Most relevant paper: Z. Liu, O. Kitouni, N. Nolte, E. J. Michaud, M. Tegmark, M. Williams,
*Towards Understanding Grokking: An Effective Theory of Representation Learning*,
NeurIPS 2022. [arXiv:2205.10343](https://arxiv.org/abs/2205.10343)

## Question

Liu et al. explain their phase diagram as a *competition between representation
learning and decoder overfitting* (Sec. 4.1), shown with phase diagrams over
hyperparameters. This repository examines the competition inside individual runs
of the regression toy model (addition, p = 10) without weight decay:

- **H1, no further improvement:** once the decoder has fit the training set, the
  representation (RQI) stops improving. Embedding movement and gradient norms are
  recorded as diagnostics. In failed runs the embeddings keep moving back and forth,
  but without net progress, and the RQI usually stops improving before the fit.
- **H2, the race:** among runs whose training set determines the linear
  representation (λ_3 > 0), the RQI at the moment the decoder fits the training set
  separates runs that reach RQI > 0.95 from runs that do not. This describes the
  trajectories, most successful runs had already passed 0.95 by then.
- **H3, intervention:** a faster embedding learning rate helps, as a slower decoder does.

**Definitions.**
- "The decoder fits the training set" is the first step with *90% training
  accuracy*, the paper's own threshold (Sec. 4.1). 100% is kept as a robustness check.
- "The representation has formed" is the first step with *RQI > 0.95*.
- Runs that never reach an event within 10^4 steps are treated as right censored.

## Layout

```
src/logged_training.py       runs the authors' train_add unmodified and records gradient
                             norms, embedding movement, accuracy and RQI per step
src/effective_theory.py      λ_3 of a training set (spectrum of H, eq. 8)
experiments/check_exact.py   (1) the recording does not change training (exact equality),
                             (2) the released setting reproduces earlier runs
experiments/run_sweep.py     one configuration: 11 training set sizes x 20 seeds, resumable:
                             refuses to resume a tag whose stored configuration differs
experiments/run_all.sh       all configurations (OUT_ROOT=... for a separate output folder)
experiments/analyse.py       all numbers and figures in the report (seed-level sign tests)
results/analysis/            summary.txt (all quoted numbers) and per-setting tables
figures/                     figures (PDF and PNG)
results/<tag>/runs.csv       one row per run - results/<tag>/traces/*.npz: curves every 10 steps
results/<tag>/config.json    the configuration of that sweep
third_party/grokking-squared/  authors' code, unmodified (MIT), pinned at 0229df9 (2022-11-25)
```

## Configurations

| tag | embedding lr | decoder lr | purpose |
|---|---|---|---|
| `released` | 10^{-3} | 10^{-4} | the authors' setting; H1, H2 |
| `dec_lr_1e-3` | 10^{-3} | 10^{-3} | faster decoder |
| `dec_lr_1e-5` | 10^{-3} | 10^{-5} | slower decoder |
| `emb_lr_3e-3` | 3*10^{-3} | 10^{-4} | faster embeddings (H3) |
| `emb_lr_1e-2` | 10^{-2} | 10^{-4} | much faster embeddings (H3) |

All runs use AdamW, no weight decay, MSE loss, 10^4 steps, minibatch min(45, |D|) and
seeds 0-19. Runs with the same seed share their targets and initialisation, and have
nested training sets across sizes, so statistical tests use the **seed** as the unit.

## Reproduce

**Tested platform.** All training runs and the analysis were run on Ubuntu 24.04
(Linux, x86_64), Python 3.13.16, CPU only (2 cores), with the versions pinned in
`requirements.txt` (pip installed the Linux wheel `torch 2.14.1+cu130`; no GPU was used).
As far as I know, recent PyTorch releases no longer provide wheels for Intel
(x86_64) macOS, so the pinned `torch==2.14.1` will not install there; use Linux, or
an Apple-silicon Mac, where training has not been tested.

```bash
pip install -r requirements.txt        # pinned versions, Python 3.13
python experiments/check_exact.py      # ~3 min: logging leaves training unchanged
python experiments/analyse.py          # ~1 min: all numbers and figures from results/
```

**Results are included.** `results/` already contains all 1100 runs. Running
`experiments/run_all.sh` as it is therefore only resumes: it finds every run done
and skips it.

**Fresh reproduction.** To regenerate the experiments from scratch, write them to a
separate folder and analyse that folder:

```bash
OUT_ROOT=results_fresh experiments/run_all.sh                 # ~55 min per configuration on 2 CPU cores
python experiments/analyse.py --results-root results_fresh   # writes results_fresh/analysis and results_fresh/figures
```

Each sweep stores its configuration in `<out-root>/<tag>/config.json`. If a tag is
reused with a different learning rate, weight decay or step budget, `run_sweep.py`
stops instead of mixing configurations.

**Movement windows.** Embedding snapshots are stored every 100 steps (steps 99, 199,
...). Every movement window in `analyse.py` runs between two snapshots, and path
length and gradient norms are taken over exactly the same steps, the pre-fit window
ends at the last snapshot at or before the fit.
