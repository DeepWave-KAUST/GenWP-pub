"""
Helpers for likelihood-based loss computation in the GenWP diffusion framework.

This module provides two core functions used by the variational lower bound
(VLB) computation in gaussian_diffusion.py:

  1. normal_kl(): KL divergence between two Gaussian distributions, used to
     compute L_{t-1} (the per-step VLB term for t > 0) when comparing the
     true posterior q(x_{t-1} | x_t, x_0) against the learned reverse process
     p_θ(x_{t-1} | x_t).

  2. discretized_gaussian_log_likelihood(): Discrete Gaussian negative
     log-likelihood, used to compute L_0 (the decoder loss at t = 0).

Note: In the default GenWP configuration, the training loss is MSE-based
(Eq. 9), not VLB-based. These functions are used only when learn_sigma=True
(learned variance with RESCALED_MSE loss) or use_kl=True (pure VLB training),
and also in the diagnostic calc_bpd_loop() for evaluating the full VLB.

Originally ported from Ho et al.'s diffusion codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import torch as th


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two diagonal Gaussian distributions:

        KL(N(mean1, exp(logvar1)) || N(mean2, exp(logvar2)))

    The formula (per element) is:
        0.5 * (-1 + logvar2 - logvar1 + exp(logvar1 - logvar2)
               + (mean1 - mean2)^2 · exp(-logvar2))

    In the diffusion framework, this is used to compute L_{t-1} (IDDPM Eq. 6):
    the KL divergence between the true posterior q(x_{t-1} | x_t, x_0) and
    the learned reverse process p_θ(x_{t-1} | x_t), both of which are Gaussian
    with known (or predicted) means and variances.

    All arguments support broadcasting, so batches can be compared against
    scalars (e.g., for the prior KL term where mean2=0, logvar2=0).

    Args:
        mean1:   Mean of the first Gaussian (e.g., true posterior mean).
        logvar1: Log-variance of the first Gaussian.
        mean2:   Mean of the second Gaussian (e.g., predicted mean).
        logvar2: Log-variance of the second Gaussian.

    Returns:
        Element-wise KL divergence tensor (same shape as the inputs after
        broadcasting), in nats.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, th.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Ensure log-variances are Tensors for th.exp() compatibility.
    # Broadcasting handles scalar-to-tensor promotion for addition/subtraction,
    # but th.exp() requires an actual Tensor.
    logvar1, logvar2 = [
        x if isinstance(x, th.Tensor) else th.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + th.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * th.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    Fast approximation of the standard normal CDF Φ(x) using a tanh-based
    formula (Abramowitz & Stegun style):

        Φ(x) ≈ 0.5 · (1 + tanh(√(2/π) · (x + 0.044715 · x³)))

    This avoids the cost of th.erf() while maintaining sufficient accuracy
    for the discretized log-likelihood computation.

    Args:
        x: Input tensor of any shape.

    Returns:
        Tensor of approximate CDF values in (0, 1), same shape as x.
    """
    return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a discretized Gaussian distribution.

    This function evaluates L_0, the decoder loss at diffusion step t = 0
    (IDDPM Eq. 5). It models the probability that a continuous Gaussian
    N(means, exp(2 · log_scales)) produces a discrete observation x by
    integrating the density over the bin [x - 1/255, x + 1/255]:

        p(x) = Φ((x + 1/255 - μ) / σ) - Φ((x - 1/255 - μ) / σ)

    with special handling at the boundaries x ≈ -1 and x ≈ +1 (the bin
    edges of the discretized range).

    Note: This function was originally designed for uint8 image data rescaled
    to [-1, 1]. For wavefield data (which takes arbitrary continuous values),
    this function is used only in VLB-based training or diagnostic evaluation,
    not in the default MSE-based training of GenWP.

    Args:
        x:          [B, C, H, W] Target values (assumed in [-1, 1] for images).
        means:      [B, C, H, W] Gaussian mean parameters μ.
        log_scales: [B, C, H, W] Gaussian log standard deviation parameters log(σ).

    Returns:
        [B, C, H, W] Log-probabilities in nats (element-wise).
    """
    assert x.shape == means.shape == log_scales.shape

    centered_x = x - means
    inv_stdv = th.exp(-log_scales)

    # Upper and lower edges of the discretization bin around x.
    # The bin width is 2/255, corresponding to the spacing between
    # adjacent uint8 values rescaled to [-1, 1].
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)

    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)

    # Log of the upper CDF (used for the left boundary x ≈ -1).
    log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
    # Log of (1 - lower CDF) (used for the right boundary x ≈ +1).
    log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))

    # CDF difference: probability mass within the bin [x - 1/255, x + 1/255].
    cdf_delta = cdf_plus - cdf_min

    # Handle boundary cases:
    #   - x ≈ -1 (left edge):  use log Φ(upper) directly (bin extends to -∞).
    #   - x ≈ +1 (right edge): use log(1 - Φ(lower)) directly (bin extends to +∞).
    #   - Interior:            use log(Φ(upper) - Φ(lower)).
    log_probs = th.where(
        x < -0.999,
        log_cdf_plus,
        th.where(x > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs