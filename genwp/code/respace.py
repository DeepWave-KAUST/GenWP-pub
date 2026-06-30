"""
Timestep respacing utilities for the GenWP diffusion framework.

This module enables accelerated sampling by using a subset of the original
T diffusion steps. Instead of iterating through all T = 1000 steps during
the reverse process, the diffusion schedule can be "respaced" to use only
a selected subset of steps (e.g., 2 steps for DDIM-2, 100 evenly spaced
steps, or custom multi-section strides).

The key idea is that the cumulative product ᾱ_t at the selected steps is
preserved from the original schedule, so the marginal distributions q(x_t | x_0)
at the retained steps are identical to those in the full schedule. Only the
transition betas between consecutive retained steps are recomputed to maintain
this invariant.

For GenWP inference (Section IV, Eq. 13), one-step sampling is used: the
denoiser directly predicts u^{n+1} from pure noise z ~ N(0, I) in a single
forward pass, so the number of reverse diffusion steps is effectively 1.
The respacing mechanism here supports this by allowing the diffusion object
to operate with an arbitrary subset of its original timesteps.

Components:
  - space_timesteps(): Selects which original timesteps to retain, given a
    desired step count per section or a DDIM-style uniform stride.
  - SpacedDiffusion: A GaussianDiffusion subclass that recomputes the beta
    schedule for the retained timesteps and wraps the model to map from
    sequential indices back to original timestep values.
  - _WrappedModel: A model wrapper that translates the sequential indices
    used internally by SpacedDiffusion back to the original diffusion
    timestep values before passing them to the U-Net.
"""

import numpy as np
import torch as th

from .gaussian_diffusion import GaussianDiffusion


def space_timesteps(num_timesteps, section_counts):
    """
    Select a subset of timesteps from the original diffusion process by
    evenly striding within equally-sized sections of the full schedule.

    The original T timesteps are divided into len(section_counts) equal
    sections, and within each section, the specified number of timesteps
    are selected with uniform stride.

    Example: If T = 300 and section_counts = [10, 15, 20], then:
      - Steps 0–99 are strided to select 10 steps
      - Steps 100–199 are strided to select 15 steps
      - Steps 200–299 are strided to select 20 steps
    yielding 45 total retained steps.

    Special case: If section_counts is a string starting with "ddim" (e.g.,
    "ddim2"), the DDIM-style fixed uniform stride is used, selecting N
    evenly spaced steps from the full schedule. For GenWP one-step inference
    (Eq. 13), "ddim1" or "ddim2" would retain only 1 or 2 steps.

    Args:
        num_timesteps:  Total number of diffusion steps T in the original
                        schedule (e.g., 1000).
        section_counts: Either:
                          - A list of integers specifying the step count per
                            section (e.g., [10, 15, 20]).
                          - A comma-separated string of integers (e.g., "10,15,20").
                          - A string "ddimN" where N is the desired total number
                            of evenly spaced steps.

    Returns:
        A set of integer timestep indices from the original schedule to retain.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            # DDIM-style uniform stride: find an integer stride that yields
            # exactly the desired number of steps.
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]

    # Divide T timesteps into len(section_counts) equal sections.
    # If T is not evenly divisible, the first (T % len) sections get one
    # extra step each.
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        # Compute the fractional stride to select section_count evenly
        # spaced steps from a section of 'size' steps.
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process that operates on a subset of the original T timesteps.

    This subclass recomputes the beta schedule so that the cumulative products
    ᾱ_t at the retained timesteps match those of the original full schedule.
    This ensures that q(x_t | x_0) at any retained step is identical to the
    original process — only the transitions between consecutive retained
    steps are modified.

    The model is wrapped by _WrappedModel to translate between the sequential
    indices (0, 1, ..., len(use_timesteps)-1) used internally and the original
    timestep values that the U-Net expects for its sinusoidal embedding.

    For GenWP, this mechanism supports:
      - Full T-step DDPM sampling (use_timesteps = {0, 1, ..., T-1})
      - Accelerated DDIM sampling with fewer steps (e.g., "ddim50" for 50 steps)
      - One-step inference (Eq. 13) where the reverse chain is collapsed to a
        single forward pass of f_θ

    Args:
        use_timesteps: A collection (set or list) of integer timestep indices
                       from the original schedule to retain.
        **kwargs:      Arguments forwarded to GaussianDiffusion.__init__(),
                       including 'betas', 'model_mean_type', 'model_var_type',
                       'loss_type', and 'rescale_timesteps'.
    """

    def __init__(self, use_timesteps, **kwargs):
        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []              # Maps sequential index → original timestep
        self.original_num_steps = len(kwargs["betas"])  # Original T

        # Create a temporary full diffusion process to access ᾱ_t values.
        base_diffusion = GaussianDiffusion(**kwargs)
        last_alpha_cumprod = 1.0

        # Recompute betas for the retained timesteps.
        # For each retained step i with cumulative product ᾱ_i, the new beta is:
        #   β'_i = 1 - ᾱ_i / ᾱ_{prev_retained}
        # This ensures that the marginal q(x_t | x_0) is unchanged at each
        # retained step, even though intermediate steps have been skipped.
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)

        # Replace the original betas with the recomputed ones and initialize
        # the parent GaussianDiffusion with the new (shorter) schedule.
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):
        """
        Override to wrap the model with _WrappedModel before computing the
        reverse-process mean and variance. The wrapper translates sequential
        indices back to original timestep values for the U-Net's embedding.
        """
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):
        """
        Override to wrap the model with _WrappedModel before computing the
        training loss. Ensures the U-Net receives original timestep values
        even when the diffusion object uses sequential indices internally.
        """
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def _wrap_model(self, model):
        """
        Wrap the model with _WrappedModel if not already wrapped.
        The wrapper handles the timestep index mapping and optional rescaling.
        """
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        """
        Identity function: timestep scaling is handled by _WrappedModel,
        so SpacedDiffusion's _scale_timesteps is a no-op.
        """
        return t


class _WrappedModel:
    """
    A wrapper that translates sequential diffusion step indices (used
    internally by SpacedDiffusion) back to the original timestep values
    before passing them to the U-Net's sinusoidal embedding.

    SpacedDiffusion operates with sequential indices {0, 1, ..., K-1} where
    K = len(use_timesteps). The U-Net, however, expects timestep values from
    the original T-step schedule (e.g., 0, 50, 100, ..., 950 for 20-step
    DDIM with T=1000). This wrapper performs the lookup:

        original_ts = timestep_map[sequential_ts]

    and optionally rescales to [0, 1000] if rescale_timesteps is True.

    Args:
        model:              The underlying U-Net denoiser f_θ.
        timestep_map:       List mapping sequential index → original timestep.
        rescale_timesteps:  If True, rescale original timesteps to [0, 1000].
        original_num_steps: Original number of diffusion steps T (for rescaling).
    """

    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, cond_prev, now_it, ts, **kwargs):
        """
        Forward pass with timestep remapping.

        Translates the sequential indices ts (used by SpacedDiffusion) to
        the original timestep values via the timestep_map lookup table,
        optionally rescales them, and calls the underlying U-Net.

        Args:
            x:         [B, C, H, W] Noisy diffusion input x_t.
            cond_prev: [B, snap_steps+1, H, W] Spatial conditioning (u^{n-4:n}, v).
            now_it:    [B] Wavefield time-step index n.
            ts:        [B] Sequential diffusion step indices (0 to K-1).
            **kwargs:  Additional model kwargs (e.g., class labels).

        Returns:
            Model output (predicted u^{n+1} or noise, depending on configuration).
        """
        # Build a lookup tensor from the timestep_map list.
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)

        # Map sequential indices → original timestep values.
        new_ts = map_tensor[ts]

        # Optionally rescale to [0, 1000] for the U-Net's sinusoidal embedding.
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)

        return self.model(x, cond_prev, now_it, new_ts, **kwargs)