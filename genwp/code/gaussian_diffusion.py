"""
Gaussian diffusion utilities for the Generative Wave Propagator (GenWP).

This module implements the forward and reverse diffusion processes described in
Section II (Background: Generative Diffusion Models) and Section III (Conditional
Diffusion Model for Wavefield Propagation) of the manuscript.

The forward process (Eq. 6–7) progressively corrupts the clean target wavefield
u^{n+1} into Gaussian noise via a prescribed variance schedule {β_t}_{t=1}^{T}.
The reverse process (Eq. 8) is parameterized by the conditional denoiser f_θ,
which predicts the clean wavefield u^{n+1} from a noisy input x_t conditioned on
the recent wavefield history u^{n-4:n}, the velocity model v, and the wavefield
time-step index n.

This code adopts the x0-prediction parameterization (Section III.2, Eq. 9), where
the network directly predicts the clean target u^{n+1} rather than the injected
noise ε. This choice is motivated by the strong local coherence of seismic
wavefields and is naturally compatible with the one-step sampling strategy used
at inference time (Section IV, Eq. 13).

Originally ported from Ho et al.'s DDPM implementation:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py
"""

import enum
import math

import numpy as np
import torch as th

from .nn import mean_flat
from .losses import normal_kl, discretized_gaussian_log_likelihood


def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule by name.

    The beta schedule defines the variance {β_t}_{t=1}^{T} of the forward
    diffusion process (Eq. 6). In the manuscript, the diffusion process uses
    T = 1000 steps (Section III.I).

    Args:
        schedule_name: Either "linear" (Ho et al., 2020) or "cosine"
                       (Nichol & Dhariwal, 2021).
        num_diffusion_timesteps: Total number of diffusion steps T.

    Returns:
        A 1-D numpy array of beta values of length T.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al. (2020), scaled to remain similar
        # regardless of the total number of diffusion steps T.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given cumulative product function
    ᾱ(t) = Π_{i=1}^{t} (1 - β_i), defined continuously over t ∈ [0, 1].

    Args:
        num_diffusion_timesteps: The number of betas to produce (T).
        alpha_bar: A callable mapping t ∈ [0, 1] → ᾱ(t).
        max_beta:  Maximum allowable beta to prevent singularities.

    Returns:
        A numpy array of betas of length T.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class ModelMeanType(enum.Enum):
    """
    Specifies what the denoiser network f_θ predicts (Section III.2).

    In this work, START_X (x0-prediction) is used: the network directly
    outputs the clean target wavefield u^{n+1}, as motivated by the strong
    physical conditioning and compatibility with one-step sampling (Eq. 13).
    """
    PREVIOUS_X = enum.auto()  # Predict x_{t-1} (the posterior mean directly)
    START_X = enum.auto()     # Predict x_0 = u^{n+1} (x0-prediction, Eq. 9)
    EPSILON = enum.auto()     # Predict the noise ε (ε-prediction, Ho et al.)


class ModelVarType(enum.Enum):
    """
    Specifies how the model's output variance is determined.

    LEARNED_RANGE (Nichol & Dhariwal, 2021) interpolates between fixed lower
    and upper bounds on the variance. FIXED_SMALL and FIXED_LARGE use
    analytically derived variances from the posterior q(x_{t-1} | x_t, x_0).
    """
    LEARNED = enum.auto()        # Directly predict log-variance
    FIXED_SMALL = enum.auto()    # Use the posterior variance β̃_t (lower bound)
    FIXED_LARGE = enum.auto()    # Use β_t (upper bound)
    LEARNED_RANGE = enum.auto()  # Interpolate between FIXED_SMALL and FIXED_LARGE


class LossType(enum.Enum):
    """
    Specifies the training loss function.

    In this work, MSE is used: the denoising regression objective of Eq. 9,
    || u^{n+1} - f_θ(x_t, t, u^{n-4:n}, v, n) ||^2, computed per sample
    and then weighted by the causal weight ω(n) in train_util.py (Eq. 12).
    """
    MSE = enum.auto()           # Raw MSE loss (Eq. 9)
    RESCALED_MSE = enum.auto()  # MSE + rescaled VLB for learned variance
    KL = enum.auto()            # Variational lower bound (KL divergence)
    RESCALED_KL = enum.auto()   # Rescaled KL to estimate the full VLB

    def is_vb(self):
        return self == LossType.KL or self == LossType.RESCALED_KL


class GaussianDiffusion:
    """
    Core Gaussian diffusion utilities for training and sampling the GenWP model.

    Implements the forward noising process (Eq. 6–7), the reverse denoising
    process (Eq. 8), and the training loss computation (Eq. 9). All sampling
    and loss methods accept the conditional inputs (cond_prev, now_it) that
    represent the wavefield history u^{n-4:n} and the wavefield time-step
    index n, which are passed through to the denoiser f_θ.

    Args:
        betas:              1-D numpy array of β_t values for the forward process
                            (Eq. 6), of length T (e.g., T = 1000).
        model_mean_type:    What the network predicts: START_X for x0-prediction
                            (Section III.2), EPSILON for noise prediction, or
                            PREVIOUS_X for direct mean prediction.
        model_var_type:     How the output variance is determined (fixed or learned).
        loss_type:          Training loss type: MSE (Eq. 9), KL, or variants.
        rescale_timesteps:  If True, rescale diffusion time steps to [0, 1000]
                            before passing to the network.
    """

    def __init__(
        self,
        *,
        betas,
        model_mean_type,
        model_var_type,
        loss_type,
        rescale_timesteps=False,
    ):
        # ---- Model configuration ----
        # model_mean_type: determines the network's prediction target
        #   START_X  → x0-prediction (predicts clean u^{n+1}, adopted in this work)
        #   EPSILON  → ε-prediction  (predicts the injected noise)
        #   PREVIOUS_X → predicts the posterior mean x_{t-1} directly
        self.model_mean_type = model_mean_type
        # model_var_type: how the reverse-process variance is obtained
        #   FIXED_SMALL / FIXED_LARGE → analytically fixed (no learned variance)
        #   LEARNED / LEARNED_RANGE   → network also outputs variance parameters
        self.model_var_type = model_var_type
        self.loss_type = loss_type
        # Whether to rescale integer diffusion steps to floating-point [0, 1000]
        self.rescale_timesteps = rescale_timesteps

        # ---- Precompute diffusion schedule quantities (float64 for precision) ----
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        # Total number of diffusion time steps T (e.g., 1000 in Section III.I)
        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        # ᾱ_t = Π_{i=1}^{t} α_i  — cumulative product of (1 - β_i)
        # Used in the closed-form marginal q(x_t | x_0) of Eq. 7
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        # ᾱ_{t-1}: shifted by one step, with ᾱ_0 = 1 prepended
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        # ᾱ_{t+1}: shifted forward by one step, with 0 appended at the end
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # ---- Quantities for the forward process q(x_t | x_0), Eq. 7 ----
        # x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,  ε ~ N(0, I)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)             # √ᾱ_t
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)  # √(1 - ᾱ_t)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)       # 1/√ᾱ_t
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1) # √(1/ᾱ_t - 1)

        # ---- Quantities for the posterior q(x_{t-1} | x_t, x_0), Eqs. 10–11 of IDDPM ----
        # Posterior variance: β̃_t = β_t · (1 - ᾱ_{t-1}) / (1 - ᾱ_t)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clipped log-variance: posterior_variance is 0 at t=0, so we clip
        # the first entry to posterior_variance[1] for numerical stability.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # Posterior mean coefficients (IDDPM Eq. 11):
        #   μ̃_t(x_t, x_0) = coef1 · x_0 + coef2 · x_t
        # coef1 = β_t · √ᾱ_{t-1} / (1 - ᾱ_t)
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # coef2 = (1 - ᾱ_{t-1}) · √α_t / (1 - ᾱ_t)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_mean_variance(self, x_start, t):
        """
        Compute the mean and variance of the forward marginal q(x_t | x_0) (Eq. 7).

        Given the clean target x_0 = u^{n+1} and diffusion step t, returns the
        parameters of the Gaussian: x_t ~ N(√ᾱ_t · x_0, (1 - ᾱ_t) · I).

        Args:
            x_start: [B, C, H, W] Clean target wavefield u^{n+1}.
            t:       [B] Diffusion time step indices.

        Returns:
            Tuple of (mean, variance, log_variance), all of shape [B, C, H, W].
        """
        mean = (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = _extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = _extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from the forward process q(x_t | x_0) using the reparameterization
        trick (Eq. 7 in the manuscript):

            x_t = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε,   ε ~ N(0, I)

        In our context, x_0 = u^{n+1} is the clean target wavefield snapshot,
        and x_t is its noised version at diffusion step t. This noised input,
        together with the conditioning (u^{n-4:n}, v, n), is fed to the denoiser
        f_θ during training (Eq. 9).

        Args:
            x_start: [B, C, H, W] Clean target wavefield u^{n+1}.
            t:       [B] Diffusion time step indices.
            noise:   Optional pre-generated Gaussian noise ε.

        Returns:
            x_t: [B, C, H, W] Noised wavefield at diffusion step t.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0) (IDDPM Eqs. 10–11).

        This posterior is tractable because both q(x_t | x_0) and
        q(x_{t-1} | x_0) are Gaussian. The posterior mean is:
            μ̃_t = coef1 · x_0 + coef2 · x_t
        and the posterior variance is β̃_t.

        Args:
            x_start: [B, C, H, W] Clean data x_0 (= u^{n+1}, either ground-truth
                     or predicted by the network).
            x_t:     [B, C, H, W] Noised data at diffusion step t.
            t:       [B] Diffusion time step indices.

        Returns:
            Tuple of (posterior_mean, posterior_variance, posterior_log_variance_clipped).
        """
        assert x_start.shape == x_t.shape
        # Posterior mean: μ̃_t = coef1 · x_0 + coef2 · x_t
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        # Posterior variance: β̃_t
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        # Clipped log-variance for numerical stability
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
        self, model, x, cond_prev, now_it, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Apply the conditional denoiser f_θ to compute the reverse-process
        distribution p_θ(x_{t-1} | x_t) (Eq. 8), as well as the predicted
        clean wavefield x_0 = u^{n+1}.

        The denoiser receives four conditioning signals (Section V):
          - x:         the noisy diffusion input x_t
          - cond_prev: the 5-frame wavefield history u^{n-4:n} (spatial conditioning)
          - now_it:    the wavefield time-step index n (temporal embedding)
          - t:         the diffusion step t (noise-level embedding)

        Depending on model_mean_type:
          - START_X (x0-prediction, Section III.2): the network directly outputs
            the predicted clean wavefield pred_xstart = f_θ(x_t, t, u^{n-4:n}, v, n),
            and the posterior mean is derived from it via Eq. 11.
          - EPSILON: the network outputs the predicted noise ε, from which x_0 is
            recovered via Eq. 7, and the posterior mean follows from Eq. 11.
          - PREVIOUS_X: the network directly outputs the posterior mean.

        Args:
            model:          The conditional denoiser f_θ (U-Net).
            x:              [B, C, H, W] Noisy input x_t at diffusion step t.
            cond_prev:      [B, 5, H, W] Conditioning history u^{n-4:n}.
            now_it:         [B] Wavefield time-step index n.
            t:              [B] Diffusion time step indices.
            clip_denoised:  If True, clip the predicted x_0 to [-1, 1].
            denoised_fn:    Optional post-processing function for predicted x_0.
            model_kwargs:   Dict of extra keyword arguments (velocity model v, etc.).

        Returns:
            Dict with keys:
              'mean':         [B, C, H, W] Predicted posterior mean μ_θ(x_t, t).
              'variance':     [B, C, H, W] Predicted posterior variance.
              'log_variance': [B, C, H, W] Log of the predicted posterior variance.
              'pred_xstart':  [B, C, H, W] Predicted clean wavefield u^{n+1}.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, C = x.shape[:2]
        assert t.shape == (B,)

        # ---- Forward pass through the conditional denoiser f_θ ----
        # The model receives: x_t (noisy input), cond_prev (u^{n-4:n}),
        # now_it (snapshot index n), and scaled diffusion step t.
        # model_kwargs contains additional conditioning (velocity model v).
        model_output = model(x, cond_prev, now_it, self._scale_timesteps(t), **model_kwargs)

        # ================================================================
        # Compute the reverse-process VARIANCE
        # Two cases: learned variance or fixed variance.
        # ================================================================
        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            # When variance is learned, the network outputs 2C channels:
            # the first C channels predict the mean (or x_0 / ε), and the
            # second C channels predict variance-related values.
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)

            if self.model_var_type == ModelVarType.LEARNED:
                # Directly predict the log-variance.
                model_log_variance = model_var_values
                model_variance = th.exp(model_log_variance)
            else:
                # LEARNED_RANGE (Nichol & Dhariwal, 2021, IDDPM Eq. 15):
                # Interpolate between log(β̃_t) and log(β_t).
                # The network outputs a value in [-1, 1] that is mapped to
                # a fraction v ∈ [0, 1], and the log-variance is:
                #   log σ²_t = v · log(β_t) + (1 - v) · log(β̃_t)
                min_log = _extract_into_tensor(
                    self.posterior_log_variance_clipped, t, x.shape
                )
                max_log = _extract_into_tensor(np.log(self.betas), t, x.shape)
                # Map network output from [-1, 1] to [0, 1]
                frac = (model_var_values + 1) / 2
                model_log_variance = frac * max_log + (1 - frac) * min_log
                model_variance = th.exp(model_log_variance)
        else:
            # Fixed variance: no learned variance parameters.
            # FIXED_LARGE uses β_t (with β̃_1 at t=0 for a better decoder NLL).
            # FIXED_SMALL uses the posterior variance β̃_t.
            model_variance, model_log_variance = {
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            # Extract values at the specific diffusion step t for each sample.
            model_variance = _extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = _extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            """Optional post-processing of the predicted clean wavefield x_0."""
            if denoised_fn is not None:
                x = denoised_fn(x)
            # Note: clipping is disabled for wavefield data, which can take
            # arbitrary amplitude values (unlike natural images in [-1, 1]).
            return x

        # ================================================================
        # Compute the reverse-process MEAN and the predicted x_0
        # Three prediction modes corresponding to ModelMeanType.
        # ================================================================
        if self.model_mean_type == ModelMeanType.PREVIOUS_X:
            # Direct mean prediction: the model output IS the posterior mean.
            # We back-compute x_0 from the predicted mean using IDDPM Eq. 11.
            pred_xstart = process_xstart(
                self._predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output)
            )
            model_mean = model_output

        elif self.model_mean_type in [ModelMeanType.START_X, ModelMeanType.EPSILON]:
            if self.model_mean_type == ModelMeanType.START_X:
                # x0-prediction (Section III.2, Eq. 9): the network directly
                # outputs the predicted clean wavefield u^{n+1}.
                pred_xstart = process_xstart(model_output)
            else:
                # ε-prediction (Ho et al., 2020): the network outputs the
                # predicted noise ε. Recover x_0 via the reparameterization
                # of Eq. 7: x_0 = (x_t - √(1-ᾱ_t)·ε) / √ᾱ_t
                pred_xstart = process_xstart(
                    self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)
                )
            # Derive the posterior mean from the predicted x_0 via Eq. 11:
            #   μ̃_t = coef1 · pred_xstart + coef2 · x_t
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_xstart, x_t=x, t=t
            )
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (
            model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        )

        return {
            "mean": model_mean,             # Posterior mean μ_θ(x_t, t)
            "variance": model_variance,     # Posterior variance σ²_t
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,     # Predicted clean wavefield u^{n+1}
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):
        """
        Recover x_0 from the predicted noise ε using the forward-process
        reparameterization (Eq. 7):
            x_0 = (1/√ᾱ_t) · x_t - (√(1/ᾱ_t - 1)) · ε

        Args:
            x_t: [B, C, H, W] Noisy data at diffusion step t.
            t:   [B] Diffusion time step indices.
            eps: [B, C, H, W] Predicted noise.

        Returns:
            pred_xstart: [B, C, H, W] Recovered clean data x_0.
        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_xstart_from_xprev(self, x_t, t, xprev):
        """
        Recover x_0 from the predicted posterior mean x_{t-1} by inverting
        IDDPM Eq. 11:
            x_0 = (1/coef1) · xprev - (coef2/coef1) · x_t

        Args:
            x_t:   [B, C, H, W] Noisy data at diffusion step t.
            t:     [B] Diffusion time step indices.
            xprev: [B, C, H, W] Predicted posterior mean (= x_{t-1}).

        Returns:
            pred_xstart: [B, C, H, W] Recovered clean data x_0.
        """
        assert x_t.shape == xprev.shape
        return (
            _extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - _extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        """
        Back-compute the noise ε from x_t and the predicted x_0, by inverting
        Eq. 7:
            ε = (x_t / √ᾱ_t - x_0) / √(1/ᾱ_t - 1)

        This is used when the model predicts x_0 (or x_{t-1}) but we need
        the noise representation for DDIM sampling.

        Args:
            x_t:          [B, C, H, W] Noisy data at diffusion step t.
            t:            [B] Diffusion time step indices.
            pred_xstart:  [B, C, H, W] Predicted clean data x_0.

        Returns:
            eps: [B, C, H, W] The implied noise.
        """
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _scale_timesteps(self, t):
        """
        Optionally rescale integer diffusion time steps to floating-point
        values in [0, 1000] before passing them to the network's sinusoidal
        positional encoding.
        """
        if self.rescale_timesteps:
            return t.float() * (1000.0 / self.num_timesteps)
        return t

    # ====================================================================
    # Sampling methods (Sections IV and II)
    # ====================================================================

    def p_sample(
        self, model, x, cond_prev, now_it, t, clip_denoised=True, denoised_fn=None, model_kwargs=None
    ):
        """
        Sample x_{t-1} from the learned reverse process p_θ(x_{t-1} | x_t)
        at a single diffusion step (Eq. 8).

        Computes the posterior mean and variance via p_mean_variance(), then
        draws a sample: x_{t-1} = μ_θ + σ_t · z, where z ~ N(0, I).
        At t = 0, no noise is added (the final prediction is deterministic).

        Note: For wavefield simulation at inference time, this iterative reverse
        chain is replaced by the one-step sampling of Eq. 13, which directly
        feeds pure noise z ~ N(0, I) at t = T and obtains the predicted
        wavefield in a single forward pass.

        Args:
            model:      The conditional denoiser f_θ.
            x:          [B, C, H, W] Current noisy state x_t.
            cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n}.
            now_it:     [B] Wavefield time-step index n.
            t:          [B] Diffusion time step indices.
            (remaining args as in p_mean_variance)

        Returns:
            Dict with 'sample' (x_{t-1}) and 'pred_xstart' (predicted u^{n+1}).
        """
        # Compute posterior mean, variance, and predicted x_0.
        out = self.p_mean_variance(
            model,
            x,
            cond_prev, now_it,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        noise = th.randn_like(x)
        # Mask to suppress noise at t = 0 (final step is deterministic).
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        # x_{t-1} = μ_θ + σ_t · z, where σ_t = exp(0.5 · log σ²_t) = √σ²_t
        sample = out["mean"] + nonzero_mask * th.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def p_sample_loop(
        self,
        model,
        cond_prev, now_it,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generate samples by running the full DDPM reverse chain from t = T
        down to t = 0 (Eq. 8 iterated T times).

        This produces a complete denoised sample by iteratively applying
        p_sample() at each diffusion step. For wavefield simulation, the
        one-step inference of Eq. 13 is preferred; this full chain is
        provided for completeness and diagnostic purposes.

        Args:
            model:      The conditional denoiser f_θ.
            cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n}.
            now_it:     [B] Wavefield time-step index n.
            shape:      Output shape (B, C, H, W).
            noise:      Optional initial noise; if None, sampled from N(0, I).
            (remaining args as in p_sample)

        Returns:
            Tuple of (final_sample, intermediate_samples, intermediate_pred_xstart).
        """
        final = None
        for sample, image_all, pred_xstart in self.p_sample_loop_progressive(
            model,
            cond_prev, now_it,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final, image_all, pred_xstart = sample, image_all, pred_xstart
        return final["sample"], image_all, pred_xstart

    def p_sample_loop_progressive(
        self,
        model,
        cond_prev, now_it,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        Generator that yields intermediate samples at each reverse diffusion
        step, enabling visualization of the progressive denoising from pure
        noise to the predicted clean wavefield.

        Iterates from t = T-1 down to t = 0, calling p_sample() at each step.
        Intermediate samples and x_0 predictions are saved every 50 steps.

        Args: Same as p_sample_loop().

        Yields:
            Tuple of (out_dict, accumulated_samples, accumulated_pred_xstart).
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)

        # Reverse time indices: t = T-1, T-2, ..., 1, 0
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_all = []      # Stores intermediate samples every 50 steps
        pred_xstart = []    # Stores intermediate x_0 predictions every 50 steps
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    image,
                    cond_prev, now_it,
                    t,
                    clip_denoised=clip_denoised,
                    denoised_fn=denoised_fn,
                    model_kwargs=model_kwargs,
                )
                # Save intermediate results every 50 diffusion steps
                # for monitoring the denoising trajectory.
                if (i+1) % 50 == 0:
                    image_all.append(out["sample"])
                    pred_xstart.append(out["pred_xstart"])
                yield out, image_all, pred_xstart
                image = out["sample"]

    # ====================================================================
    # DDIM sampling (Denoising Diffusion Implicit Models)
    # ====================================================================

    def ddim_sample(
        self,
        model,
        x,
        cond_prev, now_it,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t-1} from the model using DDIM (Song et al., 2020).

        DDIM provides a deterministic (when η = 0) or stochastic (when η > 0)
        sampling trajectory that can skip diffusion steps for faster generation.
        The update rule is:
            x_{t-1} = √ᾱ_{t-1} · pred_xstart + √(1 - ᾱ_{t-1} - σ²) · ε + σ · z

        where ε is the predicted noise, σ = η · √((1-ᾱ_{t-1})/(1-ᾱ_t)) · √(1-ᾱ_t/ᾱ_{t-1}),
        and z ~ N(0, I). Setting η = 0 gives the deterministic DDIM trajectory.

        Args:
            model:      The conditional denoiser f_θ.
            x:          [B, C, H, W] Current state x_t.
            cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n}.
            now_it:     [B] Wavefield time-step index n.
            t:          [B] Diffusion time step indices.
            eta:        DDIM stochasticity parameter (0 = deterministic).
            (remaining args as in p_sample)

        Returns:
            Dict with 'sample' (x_{t-1}) and 'pred_xstart' (predicted u^{n+1}).
        """
        # Get the posterior mean/variance and the predicted x_0.
        out = self.p_mean_variance(
            model,
            x,
            cond_prev, now_it,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Back-compute the implied noise ε from the predicted x_0,
        # regardless of whether the model natively predicts x_0 or ε.
        eps = self._predict_eps_from_xstart(x, t, out["pred_xstart"])

        alpha_bar = _extract_into_tensor(self.alphas_cumprod, t, x.shape)
        alpha_bar_prev = _extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
        # DDIM noise scale σ (controlled by η)
        sigma = (
            eta
            * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
            * th.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        # DDIM mean prediction
        noise = th.randn_like(x)
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_prev)
            + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )
        # No noise at t = 0 (final step outputs the clean prediction directly).
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        sample = mean_pred + nonzero_mask * sigma * noise
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    def ddim_reverse_sample(
        self,
        model,
        x,
        cond_prev, now_it,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        eta=0.0,
    ):
        """
        Sample x_{t+1} from the model using the DDIM reverse ODE (encoding
        direction). This is the deterministic inversion of DDIM sampling,
        mapping clean data back to noise. Only valid for η = 0.

        Args: Same as ddim_sample().

        Returns:
            Dict with 'sample' (x_{t+1}) and 'pred_xstart'.
        """
        assert eta == 0.0, "Reverse ODE only for deterministic path"
        out = self.p_mean_variance(
            model,
            x,
            cond_prev, now_it,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )
        # Back-compute the noise ε from x_t and predicted x_0.
        eps = (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x.shape) * x
            - out["pred_xstart"]
        ) / _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x.shape)
        alpha_bar_next = _extract_into_tensor(self.alphas_cumprod_next, t, x.shape)

        # Reversed DDIM update: x_{t+1} = √ᾱ_{t+1} · pred_xstart + √(1 - ᾱ_{t+1}) · ε
        mean_pred = (
            out["pred_xstart"] * th.sqrt(alpha_bar_next)
            + th.sqrt(1 - alpha_bar_next) * eps
        )

        return {"sample": mean_pred, "pred_xstart": out["pred_xstart"]}

    def ddim_sample_loop(
        self,
        model,
        cond_prev, now_it,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generate samples by running the full DDIM reverse chain from t = T
        down to t = 0.

        Same interface as p_sample_loop() but uses DDIM updates.

        Returns:
            Tuple of (final_sample, intermediate_samples, intermediate_pred_xstart).
        """
        final = None
        for sample, image_all, pred_xstart in self.ddim_sample_loop_progressive(
            model,
            cond_prev, now_it,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            eta=eta,
        ):
            final, image_all, pred_xstart = sample, image_all, pred_xstart
        return final["sample"], image_all, pred_xstart

    def ddim_sample_loop_progressive(
        self,
        model,
        cond_prev, now_it,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
        eta=0.0,
    ):
        """
        Generator that yields intermediate samples at each DDIM reverse step,
        enabling visualization of the progressive denoising trajectory.

        Args: Same as ddim_sample_loop().

        Yields:
            Tuple of (out_dict, accumulated_samples, accumulated_pred_xstart).
        """
        if device is None:
            device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        if noise is not None:
            image = noise
        else:
            image = th.randn(*shape, device=device)
        # Reverse time indices: t = T-1, T-2, ..., 1, 0
        indices = list(range(self.num_timesteps))[::-1]

        if progress:
            from tqdm.auto import tqdm
            indices = tqdm(indices)

        image_all = []      # Stores intermediate samples every 50 steps
        pred_xstart = []    # Stores intermediate x_0 predictions every 50 steps
        for i in indices:
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.ddim_sample(
                        model,
                        image,
                        cond_prev, now_it,
                        t,
                        clip_denoised=clip_denoised,
                        denoised_fn=denoised_fn,
                        model_kwargs=model_kwargs,
                        eta=eta,
                )
            # Save intermediate results every 50 diffusion steps.
            if i % 50 == 0:
                image_all.append(out["sample"])
                pred_xstart.append(out["pred_xstart"])
            yield out, image_all, pred_xstart
            image = out["sample"]

    # ====================================================================
    # Variational lower bound (VLB) computation
    # ====================================================================

    def _vb_terms_bpd(
        self, model, x_start, x_t, cond_prev, now_it, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Compute a single term of the variational lower bound (VLB), measured
        in bits-per-dimension.

        For t > 0, computes the KL divergence between the true posterior
        q(x_{t-1} | x_t, x_0) and the learned reverse process p_θ(x_{t-1} | x_t)
        (IDDPM Eq. 6, corresponding to L_{t-1}).

        For t = 0, computes the negative log-likelihood of the data under the
        learned decoder distribution (IDDPM Eq. 5, corresponding to L_0).

        Args:
            model:      The denoiser (or a lambda returning frozen outputs).
            x_start:    [B, C, H, W] Clean target u^{n+1}.
            x_t:        [B, C, H, W] Noised data at diffusion step t.
            cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n}.
            now_it:     [B] Wavefield time-step index n.
            t:          [B] Diffusion time step indices.

        Returns:
            Dict with:
              'output':      [B] NLL (at t=0) or KL (at t>0) in bits.
              'pred_xstart': [B, C, H, W] Predicted x_0.
        """
        # True posterior q(x_{t-1} | x_t, x_0) computed from ground-truth x_0.
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        # Learned reverse process p_θ(x_{t-1} | x_t), predicted from x_t.
        out = self.p_mean_variance(
            model, x_t, cond_prev, now_it, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )

        # KL divergence between the true and predicted Gaussian posteriors
        # (IDDPM Eq. 6: L_{t-1} for t > 0).
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)  # Convert from nats to bits

        # Discrete Gaussian negative log-likelihood for the decoder
        # (IDDPM Eq. 5: L_0 for t = 0).
        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # Use decoder NLL at t = 0, KL divergence at t > 0.
        output = th.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}

    # ====================================================================
    # Training loss computation (Section III, Eq. 9)
    # ====================================================================

    def training_losses(self, model, x_start, cond_prev, now_it, t, model_kwargs=None, noise=None):
        """
        Compute the per-sample training loss for a single diffusion step
        (the base loss of Eq. 9 before causal weighting).

        This is the core loss function:
            L_base = || u^{n+1} - f_θ(x_t, t, u^{n-4:n}, v, n) ||^2

        The causal weight ω(n) from Eq. 12 is applied externally in
        train_util.py's forward_backward() method.

        Steps:
          1. Sample the noised target: x_t = √ᾱ_t · u^{n+1} + √(1-ᾱ_t) · ε (Eq. 7)
          2. Run the denoiser: model_output = f_θ(x_t, t, u^{n-4:n}, v, n)
          3. Compute the per-sample MSE between the prediction and the target.

        Args:
            model:        The conditional denoiser f_θ (U-Net).
            x_start:      [B, 1, H, W] Clean target wavefield u^{n+1} (= x_0).
            cond_prev:    [B, 5, H, W] Conditioning history u^{n-4:n} (5 recent
                          snapshots stacked channel-wise).
            now_it:       [B] Wavefield time-step index n, passed to the network's
                          sinusoidal positional embedding.
            t:            [B] Sampled diffusion time step indices t ~ Uniform(1, T).
            model_kwargs: Dict with additional conditioning (velocity model v, etc.).
            noise:        Optional pre-generated noise ε; if None, sampled from N(0, I).

        Returns:
            Dict with:
              'mse':  [B] Per-sample root-MSE ||target - model_output||_{RMSE}
                      (square root of the spatial-mean squared error).
              'loss': [B] Same as 'mse' (or 'mse' + 'vb' if variance is learned).
              'vb':   [B] (Optional) VLB term for learned-variance models.
        """
        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = th.randn_like(x_start)

        # ---- Step 1: Forward noising (Eq. 7) ----
        # x_t = √ᾱ_t · u^{n+1} + √(1 - ᾱ_t) · ε
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            # ---- KL-based loss (variational lower bound) ----
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                x_start=x_start,
                x_t=x_t,
                cond_prev=cond_prev,
                now_it=now_it,
                t=t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                # Scale to estimate the full VLB across all T steps.
                terms["loss"] *= self.num_timesteps

        elif self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # ---- MSE-based loss (Eq. 9) ----

            # Step 2: Forward pass through the conditional denoiser f_θ.
            # The model receives: x_t (noisy input), cond_prev (u^{n-4:n}),
            # now_it (snapshot index n), and the scaled diffusion step t.
            model_output = model(x_t, cond_prev, now_it, self._scale_timesteps(t), **model_kwargs)

            # Handle learned variance: split the output into mean and variance channels.
            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, C = x_t.shape[:2]
                assert model_output.shape == (B, C * 2, *x_t.shape[2:])
                model_output, model_var_values = th.split(model_output, C, dim=1)
                # Freeze the mean prediction so the VLB gradient only affects
                # the variance parameters (prevents variance learning from
                # interfering with the mean prediction).
                frozen_out = th.cat([model_output.detach(), model_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r=frozen_out: r,
                    x_start=x_start,
                    x_t=x_t,
                    cond_prev=cond_prev,
                    now_it=now_it,
                    t=t,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE:
                    # Scale down the VLB term so it doesn't dominate the MSE.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # ---- Step 3: Determine the regression target ----
            # Depending on model_mean_type, the target is:
            #   START_X  → x_start = u^{n+1}  (x0-prediction, Section III.2)
            #   EPSILON  → noise = ε           (ε-prediction, Ho et al.)
            #   PREVIOUS_X → posterior mean μ̃_t (direct mean prediction)
            target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(
                    x_start=x_start, x_t=x_t, t=t
                )[0],
                ModelMeanType.START_X: x_start,
                ModelMeanType.EPSILON: noise,
            }[self.model_mean_type]
            assert model_output.shape == target.shape == x_start.shape

            # ---- Per-sample root-MSE loss (Eq. 9) ----
            # Computes √(mean((target - prediction)²)) over spatial dims [C, H, W].
            # This is the base denoising loss before causal weighting (Eq. 12).
            terms["mse"] = th.sqrt(th.mean((target - model_output) ** 2, dim=[1, 2, 3]))

            # If variance is learned, add the VLB term to the total loss.
            if "vb" in terms:
                terms["loss"] = terms["mse"] + terms["vb"]
            else:
                terms["loss"] = terms["mse"]
        else:
            raise NotImplementedError(self.loss_type)

        return terms

    def _prior_bpd(self, x_start):
        """
        Compute the prior KL term of the variational lower bound:
        KL(q(x_T | x_0) || N(0, I)), measured in bits-per-dimension.

        This term depends only on the forward process and cannot be optimized.

        Args:
            x_start: [B, C, H, W] Clean data x_0.

        Returns:
            [B] KL values in bits, one per batch element.
        """
        batch_size = x_start.shape[0]
        t = th.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(
            mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0
        )
        return mean_flat(kl_prior) / np.log(2.0)

    def calc_bpd_loop(self, model, x_start, cond_prev, now_it, clip_denoised=True, model_kwargs=None):
        """
        Compute the entire variational lower bound by evaluating the VLB
        term at every diffusion step t = T-1, ..., 0, plus the prior term.

        This is primarily a diagnostic tool for evaluating model quality.

        Args:
            model:      The conditional denoiser f_θ.
            x_start:    [B, C, H, W] Clean target u^{n+1}.
            cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n}.
            now_it:     [B] Wavefield time-step index n.

        Returns:
            Dict with:
              'total_bpd':  [B] Total VLB per batch element (bits-per-dim).
              'prior_bpd':  [B] Prior KL term.
              'vb':         [B, T] Per-step VLB terms.
              'xstart_mse': [B, T] Per-step x_0 MSE.
              'mse':        [B, T] Per-step noise MSE.
        """
        device = x_start.device
        batch_size = x_start.shape[0]

        vb = []
        xstart_mse = []
        mse = []
        for t in list(range(self.num_timesteps))[::-1]:
            t_batch = th.tensor([t] * batch_size, device=device)
            noise = th.randn_like(x_start)
            x_t = self.q_sample(x_start=x_start, t=t_batch, noise=noise)
            # Compute VLB term at the current diffusion step.
            with th.no_grad():
                out = self._vb_terms_bpd(
                    model,
                    x_start=x_start,
                    x_t=x_t,
                    cond_prev=cond_prev,
                    now_it=now_it,
                    t=t_batch,
                    clip_denoised=clip_denoised,
                    model_kwargs=model_kwargs,
                )
            vb.append(out["output"])
            xstart_mse.append(mean_flat((out["pred_xstart"] - x_start) ** 2))
            eps = self._predict_eps_from_xstart(x_t, t_batch, out["pred_xstart"])
            mse.append(mean_flat((eps - noise) ** 2))

        vb = th.stack(vb, dim=1)
        xstart_mse = th.stack(xstart_mse, dim=1)
        mse = th.stack(mse, dim=1)

        prior_bpd = self._prior_bpd(x_start)
        total_bpd = vb.sum(dim=1) + prior_bpd
        return {
            "total_bpd": total_bpd,
            "prior_bpd": prior_bpd,
            "vb": vb,
            "xstart_mse": xstart_mse,
            "mse": mse,
        }


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices and
    broadcast to match a target shape.

    This utility is used throughout to index into precomputed diffusion
    schedule arrays (e.g., √ᾱ_t, √(1-ᾱ_t)) at the specific diffusion
    steps t for each sample in the batch.

    Args:
        arr:             1-D numpy array of precomputed values (length T).
        timesteps:       [B] Tensor of integer indices into arr.
        broadcast_shape: Target shape (B, C, H, W) for broadcasting.

    Returns:
        Tensor of shape broadcast_shape with the indexed values.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)