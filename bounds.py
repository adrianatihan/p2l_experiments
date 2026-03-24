"""
Generalization bound computation for P2L (Theorem 4.2).

Provides both the exact Ψ_{k,δ} bound and the legacy beta-function
approximation for comparison.
"""

import numpy as np
from scipy.special import gammaln, betainc
from scipy.special import logsumexp


def _log_binom(n, k):
    """Log of C(n, k) via gammaln."""
    if k < 0 or k > n:
        return -np.inf
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _psi_k_delta(eps, k, N, delta):
    """Exact Ψ_{k,δ}(ε) from Theorem 4.2 (log-space arithmetic)."""
    if eps <= 0.0:
        return 0.0
    if eps >= 1.0:
        return float("inf")

    log_1me = np.log(1.0 - eps)
    log_C_Nk = _log_binom(N, k)

    # First sum: m from k to N-1
    s1 = 0.0
    if k < N:
        ms = np.arange(k, N, dtype=np.float64)
        log_ratios = (gammaln(ms + 1) - gammaln(k + 1)
                      - gammaln(ms - k + 1) - log_C_Nk)
        log_powers = -(N - ms) * log_1me
        s1 = np.exp(logsumexp(log_ratios + log_powers))

    # Second sum: m from N+1 to 4N
    s2 = 0.0
    if 4 * N >= N + 1:
        ms = np.arange(N + 1, 4 * N + 1, dtype=np.float64)
        log_ratios = (gammaln(ms + 1) - gammaln(k + 1)
                      - gammaln(ms - k + 1) - log_C_Nk)
        log_powers = (ms - N) * log_1me
        s2 = np.exp(logsumexp(log_ratios + log_powers))

    return (delta / (2.0 * N)) * s1 + (delta / (6.0 * N)) * s2


def compute_generalization_bound(k, N, delta):
    """
    Exact risk bound ε̄(k, δ) via bisection on Ψ_{k,δ}.
    """
    if k >= N:
        return 1.0

    lo = k / N if N > 0 else 0.0
    hi = 1.0 - 1e-15

    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _psi_k_delta(mid, k, N, delta) > 1.0:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break

    return hi


def compute_generalization_bound_legacy(k, N, delta):
    """Beta-function approximation (for comparison)."""
    if k == N:
        return 1.0
    t1, t2 = 0.0, 1.0
    while t2 - t1 > 1e-10:
        t = (t1 + t2) / 2
        left = delta * betainc(k + 1, N - k, t)
        right = t * N * (betainc(k, N - k + 1, t) - betainc(k + 1, N - k, t))
        if left > right:
            t2 = t
        else:
            t1 = t
    return t2
