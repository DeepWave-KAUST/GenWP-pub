"""
Inference script for the Generative Wave Propagator (GenWP).

This script implements the recursive one-step sampling procedure described
in Section IV (Eq. 13) of the manuscript. Starting from the initial wavefield
u^0 (source wavelet at t = 0 s), the trained conditional denoiser f_θ
recursively generates subsequent wavefield snapshots:

    û^{n+1} = f_θ(z, t = T, u^{n-4:n}, v, n+1),    z ~ N(0, I)

At each step, a fresh draw of isotropic Gaussian noise z is supplied as the
noisy input, and the diffusion-step embedding is fixed to t = T (the maximum
noise level). Because the x0-prediction parameterization (Section III.2) and
the strong physical conditioning make the conditional distribution
p_θ(u^{n+1} | u^{n-4:n}, v, n) nearly deterministic, the network directly
outputs the predicted clean wavefield in a single forward pass — no iterative
reverse diffusion chain is needed.

A sliding buffer of the 5 most recent snapshots (u^{n-4:n}) is maintained
and updated after each prediction, with zero padding at the beginning of
the sequence (Section III.1). The velocity model v is provided as part of
the spatial conditioning throughout the entire simulation.

The script loops over a list of test velocity models (Overthrust, SEG/EAGE,
Marmousi — Sections III.II–III.IV) and saves the predicted wavefield
snapshots and accuracy metrics to .mat files.

Usage:
    python sample.py --model_path ./checkpoints/trained_model.pt --batch_size 1
"""

import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from code.datasets import normalizer_vel, denormalizer_vel
import scipy.io as sio
from code import logger
from code.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import time


def main():
    # ---- Parse arguments and set up ----
    args = create_argparser().parse_args()
    device = th.device('cuda')
    logger.configure()

    # Output directory for predicted wavefields and metrics.
    dir_output = f'./output/'
    os.makedirs(dir_output, exist_ok=True)

    # ---- Create the model and diffusion process ----
    logger.log("creating model and diffusion...")
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **params,
        snap_steps=args.snap_steps,  # Number of conditioning history frames (5)
    )

    # Load the trained model checkpoint (EMA-smoothed parameters).
    model.load_state_dict(
        th.load(f'{args.model_path}{(train_step):06d}.pt', map_location=device)
    )
    model.to(device=device)
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    # Optional class-conditional kwargs (not used for wavefield propagation).
    model_kwargs = {}
    if args.class_cond:
        classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes

    # ---- Select sampling strategy ----
    # For diagnostic purposes, the full DDPM or DDIM reverse chain can be used.
    # However, in practice, GenWP uses one-step sampling (Eq. 13), implemented
    # below via a direct call to model(z, cond_prev, now_it, t) instead of
    # iterating through the reverse chain.
    sample_fn = (
        diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
    )

    criterion = th.nn.L1Loss()  # L1 (MAE) metric for accuracy evaluation

    # ---- Diffusion step embedding fixed to t = T (Eq. 13) ----
    # For one-step sampling, the diffusion-step embedding is set to the maximum
    # noise level t = T. The value 0.999 corresponds to the rescaled timestep
    # (t/T ≈ 1.0). Under x0-prediction, the network output is interpreted
    # directly as the clean target u^{n+1} (Section III.2).
    t = th.tensor([0.999] * args.batch_size, device=device)

    shot_id = 1     # Source shot index for testing
    it_gap = 1      # Temporal stride between consecutive conditioning frames

    # ---- Test models (Sections III.II–III.IV) ----
    # Overthrust and SEG/EAGE: in-distribution tests
    # Marmousi: out-of-distribution test (not in training set)
    md_list = ['Overthrust', 'SEGEAGE', 'Marmousi']

 
    for md in md_list:
        print(f'Sampling start for {md} with batch size {args.batch_size}')

        # ---- Load test data ----
        # Each .mat file contains:
        #   - 'vp':    [nz, nx] Velocity model
        #   - 'snaps': [nt_full, nz, nx] FD reference wavefield snapshots
        dict = sio.loadmat(f'../dataset/test/{md}/shot{shot_id}.mat')
        vp = dict['vp']
        snaps = dict['snaps']

        # Subsample to match the training snapshot interval (every 5th FD step
        # if the FD output has finer temporal sampling than Δt = 0.01 s).
        snaps = snaps[::5]
        nt = snaps.shape[0]  # Number of wavefield snapshots (e.g., 101)

        # Normalize velocity to [-1, 1] for network input.
        vp = normalizer_vel(vp)
        nz, nx = vp.shape

        # ---- Initialize inference tensors ----
        # z: Fresh isotropic Gaussian noise, supplied as the "noisy input" at
        # each recursive step (Eq. 13). The noise contributes residual
        # stochasticity, while the actual prediction is dominated by the
        # conditioning channels (u^{n-4:n}, v, n).
        z = th.randn(args.batch_size, args.out_channels, nz, nx, device=device)

        # Velocity model: [B, 1, nz, nx], replicated across the batch.
        vp = th.tensor(vp, dtype=th.float32).unsqueeze(0).unsqueeze(1).to(device=device)
        vp = vp.repeat(args.batch_size, 1, 1, 1)

        # FD reference snapshots: [B, nt, nz, nx] for comparison.
        snaps = th.tensor(snaps, dtype=th.float32).unsqueeze(0).to(device=device)
        snaps = snaps.repeat(args.batch_size, 1, 1, 1)

        # ---- Initialize the sliding conditioning buffer (Section III.1) ----
        # prev_snaps: [B, snap_steps, nz, nx] — the 5 most recent snapshots.
        # At the start, only the initial wavefield u^0 (source wavelet at
        # t = 0 s) is available; all preceding entries are zero, reflecting
        # the physical fact that the wavefield is identically zero prior to
        # source excitation.
        prev_snaps = th.zeros((args.batch_size, args.snap_steps, nz, nx),
                              dtype=th.float32, device=device)
        prev_snaps[:, 0] = snaps[:, 0]  # u^0 = source wavelet

        # Storage for the full predicted snapshot sequence.
        pred_snaps = th.zeros_like(snaps)
        pred_snaps[:, 0] = snaps[:, 0]  # Copy the initial condition

        # ---- Recursive one-step inference loop (Section IV, Eq. 13) ----
        # For n = 0, 1, ..., N-1:
        #   1. Assemble conditioning: cond_prev = [u^{n-4:n} ∥ v]
        #   2. Compute û^{n+1} = f_θ(z, t=T, u^{n-4:n}, v, n+1) in one forward pass
        #   3. Append û^{n+1} to the sliding buffer for the next step
        start = time.time()
        for it in range(1, nt):
            print(f'sampling time index {it}')

            # Normalized wavefield time-step index n for the sinusoidal
            # positional embedding (multiplied by 0.001 as in dataset.py).
            now_it = th.tensor(np.array([it*0.001], dtype=np.float32),
                               dtype=th.float32, device=device)

            # Assemble the spatial conditioning tensor:
            # [B, snap_steps + 1, nz, nx] = [prev_snaps ∥ vp]
            # This is the same format as during training (Section V).
            cond_prev = th.cat([prev_snaps, vp], dim=1)

            # ---- One-step sampling (Eq. 13) ----
            # Directly call f_θ with noise z at t = T. The strong physical
            # conditioning makes the prediction nearly deterministic, so a
            # single forward pass suffices (Section IV).
            with th.no_grad():
                sample = model(z, cond_prev, now_it, t)

            # Store the predicted snapshot û^{n+1}.
            pred_snaps[:, it:it+1] = sample[:, :1]

            # ---- Update the sliding buffer ----
            # Shift the conditioning history: drop the oldest frame (index -1)
            # and prepend the new prediction as the most recent frame (index 0).
            # This mirrors the recursive structure: each step consumes the
            # previous prediction as part of its conditioning input (Section IV).
            prev_snaps = th.cat([sample[:, :1], prev_snaps[:, :-1]], dim=1)

        end = time.time()
        print(f'time cost {end - start} s')

        # ---- Evaluate accuracy ----
        # Compute the mean absolute error (MAE) between the predicted and
        # FD reference wavefield sequences (Eq. 14 in Section III.V).
        with th.no_grad():
            accs = criterion(pred_snaps, snaps)

        # ---- Save results ----
        # Output .mat file containing the predicted wavefields and MAE metric
        # for further analysis (e.g., snapshot comparison, shot-gather
        # extraction, trace-by-trace evaluation).
        fn = f'{dir_output}{md}_shot{shot_id}_batch{args.batch_size}_out.mat'
        sio.savemat(fn,
                {'predict':pred_snaps.squeeze().cpu().numpy(), 'accs': accs.item()})

    logger.log("sampling complete")


def create_argparser():
    """
    Build the argument parser with default inference hyperparameters.

    Inference defaults:
      - clip_denoised: True — clip predicted x_0 to [-1, 1] (disabled
                       internally for wavefield data).
      - use_ddim:      True — selects DDIM sampling (though one-step
                       inference via direct model call is used in practice).
      - batch_size:    1 — single-shot inference by default.
      - model_path:    Path to the trained model checkpoint.
      - snap_steps:    5 — conditioning history length |u^{n-4:n}| = 5,
                       must match the training configuration.

    These are merged with model_and_diffusion_defaults() to provide the
    full set of model architecture and diffusion schedule parameters.

    Returns:
        An argparse.ArgumentParser with all inference arguments.
    """
    defaults = dict(
        clip_denoised=True,
        use_ddim=True,
        batch_size=1,
        model_path="./checkpoints/trained_model.pt",
        snap_steps=5,   # Must match training: 5 preceding snapshots
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()