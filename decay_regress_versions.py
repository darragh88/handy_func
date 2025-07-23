
"""Decay regression implementations (original and v2 with per-feature decay).

This module contains two functions:

- decay_regress      : Original scalar-decay implementation.
- decay_regress_v2   : Extended version supporting per-feature decay_scales and
                       fixes so that passing identical decays reproduces v1.

Both functions expect `qarray`, `DT_DOUBLE`, and other utilities to be available in
your codebase. For convenience, minimal fallbacks are provided when qarray is not
installed so the code can run standalone for testing.

Author: (transcribed & refactored by ChatGPT)
"""

from __future__ import annotations

import numpy as np
from numbers import Real
from typing import Optional

# ---------------------------------------------------------------------
# Optional compatibility layer for qarray & DT_DOUBLE
# ---------------------------------------------------------------------
try:
    import qarray  # type: ignore
except ImportError:
    class _QArrayHelpers:
        @staticmethod
        def is_positive_definite(m):
            # Assumes symmetric matrix
            return np.all(np.linalg.eigvalsh(m) > 0)

        @staticmethod
        def geq(a, b):
            return np.greater_equal(a, b)

        @staticmethod
        def gt(a, b):
            return np.greater(a, b)

        @staticmethod
        def set_array(arr, mask, value):
            arr = np.array(arr, copy=True)
            arr[mask] = value
            return arr
    qarray = _QArrayHelpers()  # type: ignore

try:
    DT_DOUBLE  # type: ignore
except NameError:
    DT_DOUBLE = np.float64  # reasonable default

# =====================================================================
# Original implementation (v1)
# =====================================================================
def decay_regress(x: np.ndarray, y: np.ndarray, *, prior_mean: np.ndarray,
                  prior_covar: np.ndarray, decay_scale: float = np.Inf) -> np.ndarray:
    """
    Parallel rolling time-series regression with decay and a prior mean and
    prior covariance matrix.
    The N time-series are fit independently (in parallel), but share priors.
    References: https://en.wikipedia.org/wiki/Ordinary_least_squares,
                https://en.wikipedia.org/wiki/Weighted_least_squares,
                https://en.wikipedia.org/wiki/Bayesian_linear_regression (see Posterior
                distribution, mu_n, Precision Matrix).

    The hack is to estimate the residual variance directly and to use that to
    back out the Lambda matrix.
    The problem is that you need a beta to estimate the residual variance, so
    we do two passes, the first time with the prior beta.

    :param x: K x T x N numeric array with the second dimension as time. K variables. N time series.
               Note: x is not automatically augmented to contain an intercept term
    :param y: T x N array. Dependent variable
    :param prior_mean: K vector, prior for beta
    :param prior_covar: K x K matrix, covariance matrix for beta estimates
    :param decay_scale: number of time steps each to decay by e
    :return: array of betas of shape K x T x N
    """
    for arr in (x, y, prior_mean, prior_covar):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype.kind in 'fdi'
        assert not np.any(np.isinf(arr))

    assert x.ndim == 3
    K, T, N = x.shape
    assert y.shape == (T, N)
    assert prior_mean.shape == (K,)
    assert prior_covar.shape == (K, K)
    assert qarray.is_positive_definite(prior_covar)
    assert isinstance(decay_scale, Real)
    assert qarray.gt(decay_scale, 0)

    prior_precision = np.linalg.inv(prior_covar)
    decay = np.exp(-1.0 / decay_scale)

    # Weight NaN data points as zero
    bad_x = np.isnan(x)
    bad = bad_x.any(axis=0) | np.isnan(y)
    w = (~bad).astype(DT_DOUBLE)

    # Replace x and y with finite versions.
    x0 = qarray.set_array(x, bad_x, 0)
    y0 = qarray.set_array(y, bad, 0)

    # Keep track of moments across time.
    sw, swy2 = (np.zeros((N,), DT_DOUBLE) for _ in range(2))
    swx2 = np.zeros((K, K, N), DT_DOUBLE)
    swxy = np.zeros((K, N), DT_DOUBLE)

    beta = np.empty(x.shape, DT_DOUBLE)

    # template for filling in betas
    beta_init = qarray.set_array(np.empty((N, K), DT_DOUBLE), np.s_[...], prior_mean)

    def iterate_beta(beta_a: np.ndarray) -> np.ndarray:
        """
        Using the given beta, estimate the residual variance and use this to
        compute a posterior beta from the given prior
        :param beta_a: N x K
        :return: N x K
        """
        xy = swxy.T[:, None, :]   # N x 1 x K
        xx = swx2.T               # N x K x K
        bt = beta_a[:, :, None]   # N x K x 1
        btt = beta_a[:, None, :]  # N x 1 x K

        # Use this beta to estimate the residual variance
        with np.errstate(invalid='ignore'):
            res_var0 = (swy2 + (-2 * (xy @ bt) + btt @ xx @ bt)[:, 0, 0]) / sw

        # It should always be non-negative
        assert np.all(np.isnan(res_var0) | qarray.geq(res_var0, 0))
        good = np.nonzero(qarray.geq(res_var0, 0))[0]

        # We add a bit to prevent zero residual variance
        res_var = qarray.set_array(res_var0, np.isclose(res_var0, 0), 1e-5)

        # Compute precision matrix and recompute the beta with prior.
        lambda_ = res_var[:, None, None] * prior_precision   # N x K x K
        a = swx2.T + lambda_                                 # N x K x K
        b = swxy.T + lambda_ @ prior_mean                    # N x K
        return qarray.set_array(beta_a, good,
                                np.linalg.solve(a[good], b[good]))

    for t in range(T):
        # Update moments.
        sw   = sw   * decay + w[t]                          # N
        swy2 = swy2 * decay + w[t] * np.square(y0[t])       # N
        swxy = swxy * decay + w[t] * y0[t] * x0[:, t]       # K x N
        swx2 = swx2 * decay + w[t] * x0[:, t][:, None] * x0[:, t][None, :]  # K x K x N

        # Compute new posterior beta from old a couple of times.
        beta0 = iterate_beta(beta_init)
        beta1 = iterate_beta(beta0)
        beta[:, t] = beta1.T

    return beta


# =====================================================================
# Revised implementation (v2) with per-feature decay scales
# =====================================================================
def decay_regress_v2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    prior_mean: np.ndarray,
    prior_covar: np.ndarray,
    decay_scale: float = np.Inf,
    decay_scales: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Extended decay_regress that allows per-feature decay via `decay_scales`.

    If `decay_scales` is provided, it must be a length-K vector of positive values.
    When all entries are equal, the function collapses to the original behavior.

    The response-side accumulators (sw, swy2) decay by `decay_y`, which by default
    is tied to the single scalar (`decay_scale`) or to the first element of the
    per-feature vector when only `decay_scales` is provided.

    :return: array of betas of shape K x T x N
    """
    for arr in (x, y, prior_mean, prior_covar):
        assert isinstance(arr, np.ndarray)
        assert arr.dtype.kind in 'fdi'
        assert not np.any(np.isinf(arr))

    assert x.ndim == 3
    K, T, N = x.shape
    assert y.shape == (T, N)
    assert prior_mean.shape == (K,)
    assert prior_covar.shape == (K, K)
    assert qarray.is_positive_definite(prior_covar)

    # Decide how to decay.
    if decay_scales is None:
        # Only scalar decay.
        assert isinstance(decay_scale, Real) and qarray.gt(decay_scale, 0)
        decay_vec = np.full(K, np.exp(-1.0 / decay_scale), dtype=DT_DOUBLE)
        decay_y   = decay_vec[0]
    else:
        decay_scales = np.asarray(decay_scales, dtype=DT_DOUBLE)
        assert decay_scales.shape == (K,)
        assert np.all(qarray.gt(decay_scales, 0))
        decay_vec = np.exp(-1.0 / decay_scales).astype(DT_DOUBLE)
        # If user didn't provide a finite scalar decay, sync decay_y with vector.
        if np.isfinite(decay_scale) and qarray.gt(decay_scale, 0):
            decay_y = np.exp(-1.0 / decay_scale)
        else:
            decay_y = decay_vec[0]

    # Decide fast-path for swx2
    if np.allclose(decay_vec, decay_vec[0]):
        decay_outer = decay_vec[0]            # scalar fast path
        vec_is_scalar = True
    else:
        decay_outer = np.sqrt(decay_vec[:, None] * decay_vec[None, :])  # K x K
        vec_is_scalar = False

    prior_precision = np.linalg.inv(prior_covar)

    bad_x = np.isnan(x)
    bad = bad_x.any(axis=0) | np.isnan(y)
    w = (~bad).astype(DT_DOUBLE)

    x0 = qarray.set_array(x, bad_x, 0)
    y0 = qarray.set_array(y, bad, 0)

    sw, swy2 = (np.zeros(N, DT_DOUBLE) for _ in range(2))
    swx2 = np.zeros((K, K, N), dtype=DT_DOUBLE)
    swxy = np.zeros((K, N), dtype=DT_DOUBLE)

    beta = np.empty_like(x, dtype=DT_DOUBLE)
    beta_init = np.tile(prior_mean[:, None], (1, N)).T  # N x K

    def _iterate_beta(beta_a: np.ndarray) -> np.ndarray:
        xy  = swxy.T[:, None, :]   # N x 1 x K
        xx  = swx2.T               # N x K x K
        bt  = beta_a[:, :, None]   # N x K x 1
        btt = beta_a[:, None, :]   # N x 1 x K

        with np.errstate(invalid='ignore'):
            res_var0 = (swy2 + (-2 * (xy @ bt) + btt @ xx @ bt)[:, 0, 0]) / sw

        # Guard residual variance
        good = np.nonzero(qarray.geq(res_var0, 0))[0]
        # Replace non-positive or NaN with small epsilon
        res_var0 = np.where(np.isnan(res_var0), -1, res_var0)
        res_var  = np.where(res_var0 <= 0, 1e-5, res_var0).astype(DT_DOUBLE)

        lam = res_var[:, None, None] * prior_precision  # N x K x K
        A   = xx + lam
        b   = swxy.T + lam @ prior_mean

        beta_new = beta_a.copy()
        if good.size:
            beta_new[good] = np.linalg.solve(A[good], b[good])
        return beta_new

    for t in range(T):
        # decay previous moments
        sw   *= decay_y
        swy2 *= decay_y
        if vec_is_scalar:
            swxy *= decay_vec[0]
            swx2 *= decay_outer
        else:
            swxy *= decay_vec[:, None]          # K x N
            swx2 *= decay_outer[:, :, None]     # K x K x N

        # accumulate current observation
        wt = w[t]
        sw   += wt
        swy2 += wt * np.square(y0[t])
        swxy += wt * y0[t] * x0[:, t]
        swx2 += wt * x0[:, t][:, None] * x0[:, t][None, :]

        # two-pass variance / beta refinement
        beta0 = _iterate_beta(beta_init)
        beta1 = _iterate_beta(beta0)
        beta[:, t] = beta1.T

    return beta
