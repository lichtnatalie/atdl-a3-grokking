"""
Run the authors' toy-addition training (train_add, unmodified) and record what
happens to the embeddings around the moment the decoder fits the training set.

Why a wrapper and not a modified copy: the authors' train_add already computes
training accuracy and RQI at every step and returns the (raw and normalised)
embeddings at every step. The only quantity it does not expose is the gradient
on the embeddings. We get that by temporarily replacing torch.optim.AdamW with a
subclass that measures the gradient norm of its parameters and then calls the
normal AdamW step. Measuring a norm consumes no random numbers and changes no
tensor, so the training run is bit-for-bit the same as an unpatched run
(checked in experiments/check_exact.py).

train_add creates two optimisers, in this order:
  optimizer1 = AdamW({reprs}, lr=eta_reprs)        -> the embeddings
  optimizer2 = AdamW(model.parameters(), lr=eta_dec) -> the decoder
so the first recorded optimiser is the embeddings' one.

Event definitions (all at full step resolution):
  - decoder fits the training set: first step with training accuracy >= 90 %
    (the paper's threshold, Sec. 4.1; "memorization is also called overfitting").
    The 100 % step is kept as a robustness check.
  - representation formed: first step with RQI > 0.95 (as in A2 / Fig. 4(b)).
Runs that never reach an event within the step budget are right-censored.
"""

import contextlib
import io
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party", "grokking-squared", "toy"))
sys.path.insert(0, os.path.join(ROOT, "src"))

from train_add import train_add  # noqa: E402  (authors' code, unmodified)
from effective_theory import sample_train_set, spectrum  # noqa: E402

P = 10                 # number of symbols in the toy addition task
TRACE_EVERY = 10       # trace resolution in steps (events use every step)


class _RecordingAdamW(torch.optim.AdamW):
    """AdamW that stores the L2 norm of its gradients before each step."""

    instances = []  # creation order within one train_add call

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grad_norms = []
        _RecordingAdamW.instances.append(self)

    @torch.no_grad()
    def step(self, closure=None):
        sq = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    sq += float(torch.sum(p.grad.double() ** 2))
        self.grad_norms.append(sq ** 0.5)
        return super().step(closure)


@contextlib.contextmanager
def _record_gradients():
    original = torch.optim.AdamW
    _RecordingAdamW.instances = []
    torch.optim.AdamW = _RecordingAdamW
    try:
        yield _RecordingAdamW.instances
    finally:
        torch.optim.AdamW = original


def _first(mask):
    """Index of the first True entry, or -1 if there is none."""
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else -1


def _window_sum(x, k):
    n = len(x) // k * k
    return x[:n].reshape(-1, k).sum(axis=1)


def _window_mean(x, k):
    n = len(x) // k * k
    return x[:n].reshape(-1, k).mean(axis=1)


def run_logged(n_train, seed, steps=10_000, eta_reprs=1e-3, eta_dec=1e-4,
               wd_dec=0.0, threshold_rqi=0.95, threshold_P=0.01):
    """One training run. Returns (summary dict, trace dict)."""
    torch.set_num_threads(1)
    with _record_gradients() as opts, contextlib.redirect_stdout(io.StringIO()):
        d = train_add(train_num=n_train, seed=seed, steps=steps, eff_steps=1,
                      eta_reprs=eta_reprs, eta_dec=eta_dec,
                      weight_decay_dec=wd_dec,
                      threshold_rqi=threshold_rqi, threshold_P=threshold_P)
    assert len(opts) == 2, "expected one optimiser for embeddings, one for decoder"
    g_emb = np.asarray(opts[0].grad_norms)
    g_dec = np.asarray(opts[1].grad_norms)

    rqi = np.asarray(d["rqi"], dtype=float)
    acc_tr = np.asarray(d["acc_train"], dtype=float)
    acc_te = np.asarray(d["acc_test"], dtype=float)
    loss_tr = np.asarray(d["loss_train"], dtype=float)
    E = np.asarray(d["repr_nn"])[:, :, 0]               # raw embeddings, after each step
    E_norm = np.asarray(d["repr_normalized_nn"])[:, :, 0]  # (E - mean) / std, used by RQI

    # how far the embeddings move per step (normalised: the quantity RQI sees)
    speed_norm = np.r_[np.nan, np.linalg.norm(np.diff(E_norm, axis=0), axis=1)]
    speed_raw = np.r_[np.nan, np.linalg.norm(np.diff(E, axis=0), axis=1)]

    t90 = _first(acc_tr >= 0.9)
    t100 = _first(acc_tr >= 1.0)
    t_rqi = _first(rqi > threshold_rqi)

    eig, null_dim, _ = spectrum(sample_train_set(P, n_train, seed), P)
    assert null_dim == d["dof"], "training split differs from the authors' code"

    def at(t, x):
        return float(x[t]) if t >= 0 else np.nan

    summary = dict(
        n_train=n_train, seed=seed, fraction=n_train / d["all_num"],
        eta_reprs=eta_reprs, eta_dec=eta_dec, wd_dec=wd_dec, steps=steps,
        dof=int(d["dof"]), lambda3=float(eig[2]),
        reached_rqi=int(t_rqi >= 0), iter_rqi=t_rqi,
        reached_train90=int(t90 >= 0), iter_train90=t90,
        reached_train100=int(t100 >= 0), iter_train100=t100,
        rqi_at_train90=at(t90, rqi), rqi_at_train100=at(t100, rqi),
        max_rqi=float(rqi.max()), final_rqi=float(rqi[-1]),
        final_train_acc=float(acc_tr[-1]), final_test_acc=float(acc_te[-1]),
        ideal_test_acc=float(d["ideal_test_acc"]),
        # authors' own bookkeeping, kept for the cross-check against A2
        authors_iter_train=int(d["iter_train"]), authors_iter_rqi=int(d["iter_rqi"]),
    )

    k = TRACE_EVERY
    trace = dict(
        step=np.arange(0, len(rqi) // k * k, k) + k - 1,  # last step of each window
        rqi=rqi[k - 1::k][: len(rqi) // k].astype(np.float32),
        acc_train=acc_tr[k - 1::k][: len(rqi) // k].astype(np.float32),
        acc_test=acc_te[k - 1::k][: len(rqi) // k].astype(np.float32),
        loss_train=_window_mean(loss_tr, k).astype(np.float32),
        grad_emb=_window_mean(g_emb, k).astype(np.float32),
        grad_dec=_window_mean(g_dec, k).astype(np.float32),
        speed_norm=_window_sum(np.nan_to_num(speed_norm), k).astype(np.float32),
        speed_raw=_window_sum(np.nan_to_num(speed_raw), k).astype(np.float32),
        # normalised embeddings themselves, only every 100 steps (for figures)
        E_step=np.arange(99, len(rqi), 100),
        E_norm=E_norm[99::100].astype(np.float32),
    )
    return summary, trace
