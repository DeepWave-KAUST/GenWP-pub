"""
Training script for the Generative Wave Propagator (GenWP).

This is the top-level entry point that orchestrates the full training pipeline
described in the manuscript (Sections III–V):

  1. Parses command-line arguments for model, diffusion, and training
     hyperparameters.
  2. Creates the conditional denoiser f_θ (UNetModel, Section V) and the
     Gaussian diffusion process (SpacedDiffusion, Section II).
  3. Sets up the data loader yielding (u^{n+1}, u^{n-4:n}, v, n) training
     pairs from HDF5 shard files (Section III.I).
  4. Launches the TrainLoop, which iteratively trains f_θ with the causal
     time-weighted MSE loss (Eq. 12) using AdamW optimization.

Default training configuration (Section III.I):
  - Batch size:        36
  - Learning rate:     1 × 10^{-4} (fixed, no annealing)
  - Optimizer:         AdamW with β = (0.9, 0.999)
  - EMA decay:         0.999
  - Diffusion steps:   T = 1000 (cosine schedule)
  - Conditioning:      5 preceding wavefield snapshots (snap_steps=5)
  - Training duration: 550,000 iterations (~95 hours on a single A100 GPU)

Usage:
    python train.py --data_dir ../dataset/train/ --batch_size 36 --lr 1e-4
"""

import argparse
from code import logger
from code.datasets import load_data
from code.resample import create_named_schedule_sampler
from code.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from code.train_util import TrainLoop
import torch as th


def main():
    # ---- Parse command-line arguments ----
    args = create_argparser().parse_args()
    logger.configure()

    # ---- Set compute device ----
    device = th.device('cuda')

    # ---- Create the model and diffusion process ----
    # The model is the conditional denoiser f_θ (U-Net, Section V).
    # The diffusion object manages the forward noising schedule {β_t}
    # and the training loss computation (Eq. 9).
    logger.log("creating model and diffusion...")
    params = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(
        **params,
        snap_steps=args.snap_steps,  # Number of conditioning history frames (5)
    )

    # Optional: load a pretrained checkpoint for fine-tuning or resuming.
    # pretrained_dict = th.load('./checkpoints/pretrained_model.pt', map_location=device)
    # model.load_state_dict(pretrained_dict)

    model.to(device)

    # ---- Create the diffusion timestep sampler ----
    # Default is "uniform": t ~ Uniform(1, T), as described in Section III.2.
    # The causal weighting (Eq. 12) operates on the wavefield index n, not
    # on the diffusion step t, so t is always sampled uniformly.
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    # ---- Set up the training data loader ----
    # Returns an infinite generator yielding:
    #   (u^{n+1}, u^{n-4:n} ∥ v, n_norm, n_int, cond_dict)
    # from sharded HDF5 files containing wavefield snapshots and velocity
    # models (Section III.I).
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        snap_steps=args.snap_steps,
        batch_size=args.batch_size,
        device=device,
        class_cond=args.class_cond,
    )

    # ---- Launch the training loop ----
    # The TrainLoop handles:
    #   - Forward/backward passes with the causal time-weighted loss (Eq. 12)
    #   - AdamW optimization with lr = 1e-4 (Section III.I)
    #   - EMA parameter updates (decay = 0.999) for stable inference
    #   - Periodic logging and checkpoint saving
    #   - The CausalWeightManager for adaptive per-snapshot weighting (Eq. 10–11)
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    """
    Build the argument parser with default training hyperparameters.

    Training defaults (Section III.I of the manuscript):
      - data_dir:           Path to the training HDF5 shard directory.
      - schedule_sampler:   "uniform" — diffusion step t ~ Uniform(1, T).
      - lr:                 1 × 10^{-4}, fixed learning rate (no annealing).
      - weight_decay:       0.0 (no L2 regularization).
      - lr_anneal_steps:    0 (no learning rate annealing).
      - batch_size:         36 (Section III.I).
      - ema_rate:           0.999 — EMA decay for stable inference weights.
      - log_interval:       100 — log metrics every 100 steps.
      - save_interval:      10000 — save checkpoints every 10,000 steps.
      - resume_checkpoint:  "" — no checkpoint to resume from by default.
      - use_fp16:           False — use float32 training by default.
      - fp16_scale_growth:  1e-3 — loss scale increment per step (if fp16).
      - snap_steps:         5 — number of preceding wavefield snapshots in the
                            conditioning history u^{n-4:n} (Section III.1).

    These defaults are merged with model_and_diffusion_defaults() which
    provides the U-Net architecture and diffusion schedule hyperparameters.

    Returns:
        An argparse.ArgumentParser with all training arguments registered.
    """
    defaults = dict(
        data_dir="../dataset/train/",
        schedule_sampler="uniform",       # Uniform sampling of diffusion step t
        lr=1e-4,                          # Fixed learning rate (Section III.I)
        weight_decay=0.0,                 # No L2 regularization
        lr_anneal_steps=0,                # No LR annealing (0 = constant LR)
        batch_size=36,                    # Batch size (Section III.I)
        ema_rate="0.999",                 # EMA decay rate for model parameters
        log_interval=100,                 # Log metrics every 100 steps
        save_interval=10000,              # Save checkpoints every 10K steps
        resume_checkpoint="",             # Path to resume checkpoint (empty = fresh start)
        use_fp16=False,                   # Mixed-precision training flag
        fp16_scale_growth=1e-3,           # fp16 loss scale growth rate
        snap_steps=5,                     # Conditioning history length |u^{n-4:n}| = 5
    )
    # Merge with model and diffusion defaults (architecture, schedule, loss, etc.).
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()