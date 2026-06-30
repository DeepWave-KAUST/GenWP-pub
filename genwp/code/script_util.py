"""
Factory utilities for constructing the GenWP model and diffusion process.

This module provides default hyperparameters and factory functions that
instantiate the conditional denoiser f_θ (UNetModel) and the Gaussian
diffusion framework (SpacedDiffusion) with configurations matching the
manuscript (Sections III–V):

  - Diffusion: T = 1000 steps, cosine noise schedule, x0-prediction
    (predict_xstart=True), fixed variance, MSE loss (Eq. 9).
  - U-Net: 4 resolution stages with channel multipliers (1, 2, 4, 8),
    base channels = 64 → feature dims [64, 128, 256, 512], 2 ResBlocks
    per level, self-attention at downsample factors {16, 32} (the two
    coarsest levels), FiLM conditioning (use_scale_shift_norm=True).

The module also provides argparse helpers for command-line configuration.
"""

import argparse
import inspect
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps
from .unet import UNetModel

# Number of classes for optional class-conditional generation.
# Not used in wavefield propagation (class_cond=False by default).
NUM_CLASSES = 1000


def model_and_diffusion_defaults():
    """
    Return a dict of default hyperparameters for training the GenWP model.

    These defaults correspond to the configuration described in the manuscript:

    Model (Section V):
      - in_channels=1:              Single-channel acoustic wavefield u^{n+1}.
      - num_channels=64:            Base channel count (feature dims: 64→128→256→512).
      - out_channels=1:             Predicted single-channel wavefield.
      - channel_mult=(1,2,4,8):     Four resolution stages.
      - num_res_blocks=2:           Two residual blocks per encoder level.
      - num_heads=4:                Four attention heads.
      - attention_resolutions=(16,32): Self-attention at the two coarsest levels.
      - dropout=0.0:                No dropout.
      - use_scale_shift_norm=True:  FiLM (scale+shift) conditioning in ResBlocks.
      - use_checkpoint=False:       No gradient checkpointing by default.

    Diffusion (Section III):
      - diffusion_steps=1000:       T = 1000 diffusion steps (Section III.I).
      - noise_schedule="cosine":    Cosine beta schedule (Nichol & Dhariwal, 2021).
      - predict_xstart=True:        x0-prediction parameterization (Section III.2).
      - learn_sigma=False:          Fixed variance (not learned).
      - sigma_small=False:          Use β_t (FIXED_LARGE) rather than β̃_t.
      - rescale_timesteps=True:     Rescale t to [0, 1000] for the network.
      - use_kl=False:               Use MSE loss (Eq. 9), not KL.
      - timestep_respacing="ddim2": DDIM-style respacing for fast sampling.
    """
    return dict(
        in_channels=1,
        num_channels=64,
        out_channels=1,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions=(16, 32),
        dropout=0.0,
        learn_sigma=False,
        sigma_small=False,
        class_cond=False,
        diffusion_steps=1000,
        noise_schedule="cosine",
        timestep_respacing="",
        use_kl=False,
        predict_xstart=True,
        rescale_timesteps=True,
        rescale_learned_sigmas=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )


def create_model_and_diffusion(
    class_cond,
    learn_sigma,
    sigma_small,
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    num_heads,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    diffusion_steps,
    noise_schedule,
    timestep_respacing,
    use_kl,
    predict_xstart,
    rescale_timesteps,
    rescale_learned_sigmas,
    use_checkpoint,
    use_scale_shift_norm,
    snap_steps,
):
    """
    Create both the U-Net denoiser f_θ and the Gaussian diffusion process.

    This is the top-level factory function called by training and inference
    scripts. It delegates to create_model() for the network and
    create_gaussian_diffusion() for the diffusion framework.

    Args:
        class_cond:             If True, enable class-conditional generation
                                (not used for wavefield propagation).
        learn_sigma:            If True, the network also predicts variance
                                parameters (LEARNED_RANGE). If False, variance
                                is fixed (Section III).
        sigma_small:            If True and learn_sigma=False, use the smaller
                                posterior variance β̃_t (FIXED_SMALL); otherwise
                                use β_t (FIXED_LARGE).
        in_channels:            Input channels of x_t (1 for acoustic wavefields).
        num_channels:           Base channel count of the U-Net (e.g., 64).
        out_channels:           Output channels (1 for acoustic wavefields).
        channel_mult:           Tuple of channel multipliers per U-Net level
                                (e.g., (1, 2, 4, 8) → 64, 128, 256, 512).
        num_res_blocks:         Number of ResBlocks per encoder level.
        num_heads:              Number of attention heads.
        num_heads_upsample:     Number of attention heads in the decoder
                                (-1 means same as num_heads).
        attention_resolutions:  Set of downsample factors at which self-attention
                                is applied (e.g., (16, 32) for the two coarsest
                                levels, Section V).
        dropout:                Dropout rate in ResBlocks.
        diffusion_steps:        Total number of diffusion steps T (e.g., 1000).
        noise_schedule:         Name of the beta schedule ("linear" or "cosine").
        timestep_respacing:     Respacing string for faster sampling (e.g.,
                                "ddim2" for 2-step DDIM). Empty string or [steps]
                                for no respacing.
        use_kl:                 If True, use the KL-based VLB loss instead of MSE.
        predict_xstart:         If True, use x0-prediction (Section III.2);
                                if False, use ε-prediction.
        rescale_timesteps:      If True, rescale diffusion steps to [0, 1000]
                                before passing to the network.
        rescale_learned_sigmas: If True and learn_sigma=True, use RESCALED_MSE
                                loss (MSE + scaled VLB).
        use_checkpoint:         If True, use gradient checkpointing in ResBlocks.
        use_scale_shift_norm:   If True, use FiLM (scale+shift) conditioning;
                                otherwise use additive conditioning.
        snap_steps:             Number of conditioning wavefield snapshots in the
                                spatial conditioning input. Together with the
                                velocity model, forms a (snap_steps+1)-channel
                                input to the convolutional stem (Section V).

    Returns:
        Tuple of (model, diffusion):
          - model:     UNetModel instance (the conditional denoiser f_θ).
          - diffusion: SpacedDiffusion instance (the diffusion process with
                       optional timestep respacing for faster sampling).
    """
    model = create_model(
        in_channels=in_channels,
        num_channels=num_channels,
        out_channels=out_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        snap_steps=snap_steps,
    )
    diffusion = create_gaussian_diffusion(
        steps=diffusion_steps,
        learn_sigma=learn_sigma,
        sigma_small=sigma_small,
        noise_schedule=noise_schedule,
        use_kl=use_kl,
        predict_xstart=predict_xstart,
        rescale_timesteps=rescale_timesteps,
        rescale_learned_sigmas=rescale_learned_sigmas,
        timestep_respacing=timestep_respacing,
    )
    return model, diffusion


def create_model(
    in_channels,
    num_channels,
    out_channels,
    channel_mult,
    num_res_blocks,
    learn_sigma,
    class_cond,
    use_checkpoint,
    attention_resolutions,
    num_heads,
    num_heads_upsample,
    use_scale_shift_norm,
    dropout,
    snap_steps,
):
    """
    Instantiate the U-Net conditional denoiser f_θ (Section V).

    Creates a UNetModel with the specified architecture hyperparameters.
    The output channels are doubled if learn_sigma=True, since the network
    then predicts both the mean and variance parameters.

    Args:
        in_channels:           Input channels of x_t (1 for acoustic).
        num_channels:          Base channel count (model_channels).
        out_channels:          Output channels (1 for acoustic).
        channel_mult:          Channel multipliers per U-Net level.
        num_res_blocks:        ResBlocks per encoder level.
        learn_sigma:           If True, double the output channels for
                               joint mean + variance prediction.
        class_cond:            If True, enable class-conditional embedding.
        use_checkpoint:        If True, use gradient checkpointing.
        attention_resolutions: Downsample factors at which to apply attention.
        num_heads:             Number of attention heads in the encoder.
        num_heads_upsample:    Number of attention heads in the decoder.
        use_scale_shift_norm:  If True, use FiLM conditioning in ResBlocks.
        dropout:               Dropout rate.
        snap_steps:            Number of conditioning wavefield snapshots
                               (determines the input channels of the
                               convolutional stem: snap_steps + 1).

    Returns:
        UNetModel instance.
    """
    return UNetModel(
        in_channels=in_channels,
        model_channels=num_channels,
        out_channels=out_channels,
        num_res_blocks=num_res_blocks,
        attention_resolutions=attention_resolutions,
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        num_heads=num_heads,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        snap_steps=snap_steps,
    )


def create_gaussian_diffusion(
    *,
    steps=1000,
    learn_sigma=False,
    sigma_small=False,
    noise_schedule="linear",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
    timestep_respacing="",
):
    """
    Instantiate the Gaussian diffusion framework (Section II and III).

    Creates a SpacedDiffusion object that wraps GaussianDiffusion with
    optional timestep respacing for accelerated sampling. The key
    configuration choices for GenWP are:

      - x0-prediction (predict_xstart=True): the network directly predicts
        the clean target u^{n+1} (Section III.2, Eq. 9).
      - MSE loss: the base denoising regression objective (Eq. 9), which is
        then weighted by the causal weight ω(n) in train_util.py (Eq. 12).
      - Cosine noise schedule: provides better behavior across noise levels
        than the linear schedule (Nichol & Dhariwal, 2021).

    Args:
        steps:                  Total number of diffusion steps T (e.g., 1000).
        learn_sigma:            If True, the network predicts variance parameters
                                (LEARNED_RANGE); loss includes a VLB term.
        sigma_small:            If True and not learn_sigma, use the posterior
                                variance β̃_t (FIXED_SMALL); otherwise use β_t
                                (FIXED_LARGE).
        noise_schedule:         Beta schedule name: "linear" or "cosine".
        use_kl:                 If True, use KL-based VLB loss (RESCALED_KL).
        predict_xstart:         If True, use x0-prediction (ModelMeanType.START_X);
                                if False, use ε-prediction (ModelMeanType.EPSILON).
        rescale_timesteps:      If True, rescale integer diffusion steps to
                                floating-point [0, 1000] for the network.
        rescale_learned_sigmas: If True and learn_sigma, use RESCALED_MSE loss.
        timestep_respacing:     String controlling timestep respacing for faster
                                sampling (e.g., "ddim2" for 2-step DDIM, "100"
                                for 100 uniformly spaced steps). Empty string
                                or omitted means no respacing (use all T steps).

    Returns:
        SpacedDiffusion instance with the configured schedule, loss, and
        prediction parameterization.
    """
    # Compute the beta schedule {β_t}_{t=1}^{T} (Eq. 6).
    betas = gd.get_named_beta_schedule(noise_schedule, steps)

    # ---- Determine the training loss type ----
    if use_kl:
        # KL-based variational lower bound, rescaled to estimate the full VLB.
        loss_type = gd.LossType.RESCALED_KL
    elif rescale_learned_sigmas:
        # MSE + rescaled VLB for learned variance (Nichol & Dhariwal, 2021).
        loss_type = gd.LossType.RESCALED_MSE
    else:
        # Standard MSE loss (Eq. 9): the default for GenWP.
        # || u^{n+1} - f_θ(x_t, t, u^{n-4:n}, v, n) ||^2
        loss_type = gd.LossType.MSE

    # If no respacing is specified, use all T diffusion steps.
    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        # ---- Prediction parameterization (Section III.2) ----
        # START_X: x0-prediction, the network directly outputs u^{n+1}.
        # EPSILON: ε-prediction, the network outputs the injected noise.
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        # ---- Variance type ----
        # FIXED_LARGE: use β_t (default when learn_sigma=False, sigma_small=False).
        # FIXED_SMALL: use posterior variance β̃_t.
        # LEARNED_RANGE: interpolate between β̃_t and β_t (when learn_sigma=True).
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def add_dict_to_argparser(parser, default_dict):
    """
    Add each key-value pair from a defaults dict as a command-line argument
    to an argparse.ArgumentParser. Automatically infers the argument type
    from the default value (with special handling for booleans via str2bool).

    Args:
        parser:       An argparse.ArgumentParser instance.
        default_dict: Dict mapping argument names to default values.
    """
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    """
    Extract a subset of attributes from an argparse.Namespace into a dict.

    Args:
        args: An argparse.Namespace object.
        keys: Iterable of attribute names to extract.

    Returns:
        Dict mapping each key to its value from args.
    """
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    Parse a string as a boolean value for argparse.

    Accepts: "yes", "true", "t", "y", "1" → True
             "no", "false", "f", "n", "0" → False

    Ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")