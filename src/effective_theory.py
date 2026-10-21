"""
Training-free side of Liu et al. (2022), Sec. 3.2: what the effective theory
predicts from the training set alone, before any network is trained.

Setting (toy addition task, p symbols, 1-D embeddings E_0, ..., E_{p-1}):

  - Full dataset D_0: all unordered pairs (i, j) with 0 <= i <= j < p,
    so |D_0| = p(p+1)/2  (55 for p = 10). a+b and b+a count as one sample.
  - Training set D: a random subset of D_0 of size n; fraction r = n/|D_0|.
  - Parallelogram (i, j, m, n): two samples (i, j), (m, n) with i+j = m+n.
    P_0(D) is the set of parallelograms with BOTH samples in D.

Each parallelogram gives one linear constraint E_i + E_j - E_m - E_n = 0.
Stack them as rows of a matrix A_D (|P_0(D)| x p). The effective loss is

    l_0 = |A_D E|^2 / |P_0(D)|,     l_eff = l_0 / Z_0,     Z_0 = sum_k E_k^2

and l_eff = (1/2) E^T H E with   H = 2 A_D^T A_D / (|P_0(D)| Z_0)   (eq. 8).

Two directions are always free (eigenvalue 0): translation E_k = c, because
every row of A_D sums to 1+1-1-1 = 0, and scaling E_k = k, because
i+j-m-n = 0 on every parallelogram. So lambda_1 = lambda_2 = 0 always, and:

  - the linear representation is UNIQUE (up to translation and scaling)
    iff dim null(A_D) = 2  iff  lambda_3 > 0;
  - when lambda_3 > 0, the theory's grokking timescale is t_h = 1/lambda_3.

Fig. 4(a) of the paper plots P(dim null(A_D) = 2) against r.
"""

from itertools import combinations

import numpy as np


def full_dataset(p):
    """All unordered pairs (i, j), i <= j, in the authors' ordering."""
    return [(i, j) for i in range(p) for j in range(i, p)]


def parallelogram_rows(pairs, p):
    """One constraint row per pair of samples with equal sums.

    Returns an array of shape (#parallelograms, p). Trivially-zero rows can
    occur, e.g. (0,2) and (1,1): row = e_0 + e_2 - 2 e_1, which is fine;
    a row is only zero if the two samples are identical, which cannot happen
    because each unordered pair appears once.
    """
    rows = []
    for (i, j), (m, n) in combinations(pairs, 2):
        if i + j == m + n:
            row = np.zeros(p)
            row[i] += 1
            row[j] += 1
            row[m] -= 1
            row[n] -= 1
            rows.append(row)
    return np.array(rows).reshape(-1, p)


def spectrum(train_pairs, p, z0=None):
    """Eigenvalues of H (ascending) and the null-space dimension of A_D.

    z0 defaults to p: for normalised 1-D embeddings E~ = (E - mu)/sigma,
    sum_k E~_k^2 = p exactly.
    """
    if z0 is None:
        z0 = p
    A = parallelogram_rows(train_pairs, p)
    n_par = A.shape[0]
    if n_par == 0:  # no constraints at all: every direction is free
        return np.zeros(p), p, 0
    gram = A.T @ A
    eig = np.linalg.eigvalsh(gram)
    # rank via SVD is more robust than thresholding eigenvalues of A^T A
    null_dim = p - np.linalg.matrix_rank(A)
    H_eig = 2.0 * eig / (n_par * z0)
    return np.sort(H_eig), null_dim, n_par


def sample_train_set(p, n_train, seed, mimic_authors=True, output_dim=30):
    """Draw a training set of n_train samples.

    With mimic_authors=True this reproduces the exact split that
    train_add(seed=seed) in the authors' code draws: they seed numpy, then
    draw the (2p-1) x output_dim random target templates, then the split.
    Consuming the RNG in the same order gives identical training sets, which
    lets us cross-check against their code seed by seed.
    """
    rng_state = np.random.get_state()
    np.random.seed(seed)
    if mimic_authors:
        np.random.normal(0, 1, size=(2 * p - 1, output_dim))
    pairs = full_dataset(p)
    idx = np.random.choice(len(pairs), n_train, replace=False)
    np.random.set_state(rng_state)
    return [pairs[k] for k in idx]
