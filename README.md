# ATDL Assignment 3: representation learning vs. decoder overfitting

**Individual submission** · Advanced Topics in Deep Learning, University of Copenhagen, 2026

Most relevant paper: Z. Liu, O. Kitouni, N. Nolte, E. J. Michaud, M. Tegmark, M. Williams,
*Towards Understanding Grokking: An Effective Theory of Representation Learning*,
NeurIPS 2022. [arXiv:2205.10343](https://arxiv.org/abs/2205.10343)

## Question

Liu et al. explain their phase diagram as a *competition between representation
learning and decoder overfitting* (Sec. 4.1), but show it only for averages over a
hyperparameter grid. This repository tests the explanation **run by run**, in the
regression toy model (addition, p = 10) without weight decay:

- **H1, freezing:** once the decoder fits the training set, the embeddings almost
  stop changing.
- **H2, the race:** among runs whose training set determines the linear
  representation (λ₃ > 0), the RQI reached at the moment the decoder fits the
  training set predicts whether the run reaches RQI > 0.95.
- **H3, intervention:** a faster embedding learning rate helps, as a slower decoder does.

**Definitions.**
- "The decoder fits the training set" is the first step with **90 % training
  accuracy**, the paper's own threshold (Sec. 4.1). 100 % is kept as a robustness check.
- "The representation has formed" is the first step with **RQI > 0.95**.
- Runs that never reach an event within 10⁴ steps are treated as right-censored.

## Layout

```
src/logged_training.py       runs the authors' train_add unmodified and records gradient
                             norms, embedding movement, accuracy and RQI per step
src/effective_theory.py      λ₃ of a training set (spectrum of H, eq. 8)
experiments/check_exact.py   (1) the recording does not change training (exact equality);
                             (2) the released setting reproduces earlier runs
experiments/run_sweep.py     one configuration: 11 training set sizes x 20 seeds, resumable
experiments/run_all.sh       all configurations
results/<tag>/runs.csv       one row per run; results/<tag>/traces/*.npz: curves every 10 steps
results/a2_reference/        reference runs of the released setting, used by check_exact.py
third_party/grokking-squared/  authors' code, unmodified (MIT), pinned at 0229df9 (2022-11-25)
```

## Configurations

| tag | embedding lr | decoder lr | purpose |
|---|---|---|---|
| `released` | 10⁻³ | 10⁻⁴ | the authors' setting; H1, H2 |
| `dec_lr_1e-3` | 10⁻³ | 10⁻³ | faster decoder |
| `dec_lr_1e-5` | 10⁻³ | 10⁻⁵ | slower decoder |
| `emb_lr_3e-3` | 3·10⁻³ | 10⁻⁴ | faster embeddings (H3) |
| `emb_lr_1e-2` | 10⁻² | 10⁻⁴ | much faster embeddings (H3) |

All runs use AdamW, no weight decay, MSE loss, 10⁴ steps, minibatch min(45, |D|) and
seeds 0–19. Runs with the same seed share their targets and initialisation, and have
nested training sets across sizes, so statistical tests use the **seed** as the unit.

## Reproduce

```bash
pip install -r requirements.txt
python experiments/check_exact.py      # ~3 min
experiments/run_all.sh                 # ~55 min per configuration on 2 CPU cores
```
