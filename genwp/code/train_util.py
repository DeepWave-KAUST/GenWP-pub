"""
Training loop for the Generative Wave Propagator (GenWP).

This module implements the main training procedure described in Sections III and IV
of the manuscript. The core idea is to train a conditional diffusion model that learns
a one-step wavefield propagator P_{Δt}: (u^{n-4:n}, v) → u^{n+1}, advancing the
seismic wavefield by a physical time increment Δt = 0.01 s (10× the FD solver's
stability-limited time step).

Key components:
  - TrainLoop: orchestrates the training loop, including forward/backward passes,
    optimizer steps, EMA parameter updates, and checkpoint management.
  - CausalWeightManager: implements the causal time-weighted loss (Section III.3,
    Eqs. 10–12), which adaptively weights per-snapshot losses using exponential
    moving averages of historical prediction errors to respect the causal structure
    of wave propagation and suppress recursive error accumulation.
"""

import copy
import functools
import os

import blobfile as bf
import numpy as np
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import logger
from .fp16_util import (
    make_master_params,
    master_params_to_model_params,
    model_grads_to_master_grads,
    unflatten_master_params,
    zero_grad,
)
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import time

# Initial value for the log2 loss scale used in mixed-precision (fp16) training.
# The scale quickly climbs to ~20-21 within the first ~1K training steps.
INITIAL_LOG_LOSS_SCALE = 20.0

# Directory to save model checkpoints (EMA snapshots and optimizer states).
dir_checkpoints = './checkpoints/'
os.makedirs(dir_checkpoints, exist_ok=True)

class TrainLoop:
    """
    Main training loop for the conditional diffusion wavefield propagator.

    At each iteration, the loop:
      1. Draws a minibatch of (u^{n+1}, u^{n-4:n}, n, v) tuples from the dataset.
      2. Samples diffusion time steps t ~ Uniform(1, T) via the schedule sampler.
      3. Computes the causal weight ω(n) for each sample from the CausalWeightManager
         (Eq. 11), which suppresses the contribution of later snapshots until earlier
         ones have been well learned.
      4. Evaluates the x0-prediction denoising loss (Eq. 9) weighted by ω(n) to form
         the full training objective (Eq. 12).
      5. Updates the CausalWeightManager's EMA error buffer L_ema (Eq. 10) with the
         per-sample losses from the current minibatch.
      6. Performs an optimizer step and updates EMA copies of the model parameters.
    """

    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        num_physical_steps=101, epsilon=1e-1, delta=0.99, ema_decay=0.9999,
    ):
        """
        Initialize the training loop.

        Args:
            model:              The conditional denoiser f_θ (U-Net, Section V of the
                                manuscript) that predicts the clean wavefield u^{n+1}
                                from its noised version x_t, conditioned on (u^{n-4:n}, v, n).
            diffusion:          The Gaussian diffusion process object managing the forward
                                noising schedule {β_t} and the training loss computation
                                (Eq. 9).
            data:               An iterator yielding training minibatches. Each call to
                                next(data) returns:
                                  - batch_now_snap:   [B, 1, H, W]  target snapshot u^{n+1}
                                  - batch_cond_prev:  [B, 5, H, W]  conditioning history u^{n-4:n}
                                  - batch_now_it:     [B]            wavefield time-step index n
                                                                     (for sinusoidal embedding)
                                  - batch_now_it_idx: [B]            integer snapshot index
                                                                     (for causal weight lookup)
                                  - cond:             dict           model_kwargs containing the
                                                                     velocity model v, etc.
            batch_size:         Number of samples per minibatch.
            lr:                 Learning rate for the AdamW optimizer.
            ema_rate:           Exponential moving average decay rate(s) for maintaining
                                smoothed copies of the model parameters, used for stable
                                inference (e.g., 0.999 as in Section III.I).
            log_interval:       Log training metrics every this many steps.
            save_interval:      Save model checkpoints every this many steps.
            resume_checkpoint:  Path to a checkpoint file to resume training from.
            use_fp16:           Whether to use mixed-precision (float16) training.
            fp16_scale_growth:  Increment for the log2 loss scale per step in fp16 mode.
            schedule_sampler:   Sampler for diffusion time steps t; defaults to
                                UniformSampler (t ~ Uniform(1, T)).
            weight_decay:       L2 regularization coefficient for AdamW.
            lr_anneal_steps:    If > 0, linearly anneal the learning rate to zero over
                                this many steps.
            num_physical_steps: Total number of wavefield physical time steps N_t in one
                                simulation (e.g., 101 snapshots for 0–1 s at Δt = 0.01 s).
                                Used by the CausalWeightManager.
            epsilon:            Causal decay coefficient ε in Eq. 11, controlling how
                                rapidly the weight decays with cumulative preceding error.
            delta:              Saturation threshold δ in Eq. 11; weights above δ are
                                clamped to 1.0. Training is considered converged once
                                min_n ω(n) ≥ δ.
            ema_decay:          EMA smoothing coefficient γ in Eq. 10 for tracking
                                per-snapshot prediction errors L_ema[n].
        """
        self.model = model
        self.device = next(model.parameters()).device
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.lr = lr
        # Parse EMA rate(s): can be a single float or a comma-separated string.
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval        # Steps between metric logging
        self.save_interval = save_interval      # Steps between checkpoint saves
        self.resume_checkpoint = resume_checkpoint  # Path to resume checkpoint
        self.use_fp16 = use_fp16                # Mixed-precision training flag
        self.fp16_scale_growth = fp16_scale_growth  # fp16 loss scale growth rate
        # Diffusion time step sampler: defaults to Uniform(1, T) as in the manuscript
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0           # Current training step (relative to resume)
        self.resume_step = 0    # Step number from which training was resumed

        self.global_batch = self.batch_size

        self.model_params = list(self.model.parameters())
        self.master_params = self.model_params
        self.lg_loss_scale = INITIAL_LOG_LOSS_SCALE
        self.sync_cuda = th.cuda.is_available()

        # Load model weights from a checkpoint if resuming training.
        self._load_and_sync_parameters()
        if self.use_fp16:
            self._setup_fp16()

        # Initialize AdamW optimizer with β1=0.9, β2=0.999 (Section III.I).
        self.opt = AdamW(self.master_params, lr=self.lr, weight_decay=self.weight_decay, betas=(0.9, 0.999))
        if self.resume_step:
            # Restore optimizer state and EMA parameters from checkpoint.
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            # Initialize EMA parameters as copies of the current model parameters.
            self.ema_params = [
                copy.deepcopy(self.master_params) for _ in range(len(self.ema_rate))
            ]

        # ----------------------------------------------------------------
        # Causal Weight Manager (Section III.3, Eqs. 10–12)
        # Maintains per-snapshot EMA error estimates L_ema[n] and computes
        # the causal weights ω(n) that enforce a training order aligned
        # with the physical direction of wave propagation.
        # ----------------------------------------------------------------
        self.cwm = CausalWeightManager(
            num_physical_steps=num_physical_steps,
            epsilon=epsilon,
            delta=delta,
            ema_decay=ema_decay,
            device=self.device
        )

    def _load_and_sync_parameters(self):
        """
        Load model parameters from a checkpoint file if one is available.
        Also parses the resume step number from the checkpoint filename.
        """
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            self.model.load_state_dict(th.load(resume_checkpoint, map_location=self.device))

    def _load_ema_parameters(self, rate):
        """
        Load EMA-smoothed model parameters from a checkpoint file.

        Args:
            rate: The EMA decay rate identifying which EMA checkpoint to load.

        Returns:
            A list of parameter tensors representing the EMA-smoothed model state.
        """
        ema_params = copy.deepcopy(self.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load_state_dict(
                ema_checkpoint, map_location=self.device
            )
            ema_params = self._state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        """
        Load the AdamW optimizer state from a checkpoint file to resume training
        with consistent momentum and adaptive learning rate estimates.
        """
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load_state_dict(
                opt_checkpoint, map_location=self.device
            )
            self.opt.load_state_dict(state_dict)

    def _setup_fp16(self):
        """
        Set up mixed-precision (fp16) training by creating float32 master copies
        of the model parameters and converting the model itself to float16.
        """
        self.master_params = make_master_params(self.model_params)
        self.model.convert_to_fp16()

    def run_loop(self):
        """
        Execute the main training loop.

        Iterates until lr_anneal_steps is reached (or indefinitely if
        lr_anneal_steps == 0). Each iteration:
          1. Fetches the next minibatch from the data iterator.
          2. Runs one training step (forward + backward + optimizer update).
          3. Periodically logs metrics and saves checkpoints.

        The data iterator yields five items per batch:
          - batch_now_snap:   target wavefield snapshot u^{n+1}
          - batch_cond_prev:  5-frame conditioning history u^{n-4:n}
          - batch_now_it:     wavefield time-step index n (for network embedding)
          - batch_now_it_idx: integer snapshot index (for causal weight lookup)
          - cond:             model kwargs (velocity model v, etc.)
        """
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            # Fetch the next training minibatch from the data loader.
            batch_now_snap, batch_cond_prev, batch_now_it, batch_now_it_idx, cond = next(self.data)

            self.run_step(batch_now_snap, batch_cond_prev, batch_now_it, batch_now_it_idx, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1

        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch_now_snap, batch_cond_prev, batch_now_it, batch_now_it_idx, cond):
        """
        Execute a single training step: forward pass, backward pass, and
        optimizer update.

        Args:
            batch_now_snap:   [B, 1, H, W] Target wavefield snapshot u^{n+1}.
            batch_cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n} (5 most
                              recent snapshots stacked channel-wise).
            batch_now_it:     [B] Wavefield time-step index n, passed to the
                              network's sinusoidal positional embedding.
            batch_now_it_idx: [B] Integer snapshot index, used to look up and
                              update the causal weight ω(n).
            cond:             dict of additional model kwargs (velocity model v).
        """
        self.forward_backward(batch_now_snap, batch_cond_prev, batch_now_it, batch_now_it_idx, cond)
        if self.use_fp16:
            self.optimize_fp16()
        else:
            self.optimize_normal()

        self.log_step()

    def forward_backward(self, batch_now_snap, batch_cond_prev, batch_now_it, batch_now_it_idx, cond):
        """
        Compute the causal time-weighted training loss (Eq. 12) and backpropagate.

        This method implements the core of the training objective:
          L = E_{n, t, u^{n+1}, ε} [ ω(n) · || u^{n+1} - f_θ(x_t, t, u^{n-4:n}, v, n) ||^2 ]

        Steps:
          1. Zero all parameter gradients.
          2. Move data to the compute device (GPU).
          3. Sample diffusion time steps t ~ Uniform(1, T).
          4. Retrieve causal weights ω(n) from the CausalWeightManager (Eq. 11).
             These are computed from the accumulated EMA error buffer and do NOT
             depend on the current minibatch — they reflect training history.
          5. Compute the per-sample x0-prediction MSE loss via the diffusion object.
          6. Weight each sample's loss by ω(n) and backpropagate.
          7. Update the EMA error buffer L_ema[n] (Eq. 10) with the current batch's
             per-sample errors. Only the snapshot indices present in the current
             minibatch are updated; all other entries remain unchanged.

        Args:
            batch_now_snap:   [B, 1, H, W] Target snapshot u^{n+1}.
            batch_cond_prev:  [B, 5, H, W] Conditioning history u^{n-4:n}.
            batch_now_it:     [B] Wavefield time-step index n (for embedding).
            batch_now_it_idx: [B] Integer snapshot index (for causal weight lookup).
            cond:             dict of model kwargs (velocity model v, etc.).
        """
        zero_grad(self.model_params)

        # Transfer all batch tensors to the compute device (e.g., GPU).
        batch_now_snap = batch_now_snap.to(self.device)
        batch_cond_prev = batch_cond_prev.to(self.device)
        batch_now_it = batch_now_it.to(self.device)
        batch_now_it_idx = batch_now_it_idx.to(self.device)    # Integer index for causal weights

        # Sample diffusion time steps t ~ Uniform(1, T) for each sample in the batch.
        # 'weights' here are the importance weights from the schedule sampler (distinct
        # from the causal weights ω(n)); for UniformSampler they are all ones.
        t, weights = self.schedule_sampler.sample(batch_now_snap.shape[0], self.device)

        # ------------------------------------------------------------------
        # Retrieve causal weights ω(n) from the EMA error state table (Eq. 11).
        #   cumsum[i] = Σ_{k=0}^{i-1} L_ema[k]   (cumulative preceding error)
        #   ω(i)      = exp(-ε · cumsum[i])        (causal decay)
        # All estimates come from historical accumulation across past training
        # iterations and are independent of the current minibatch.
        # ------------------------------------------------------------------
        causal_weights = self.cwm.get_weights()                # [N_t] weights for all snapshots
        sample_weights = causal_weights[batch_now_it_idx]      # [B]   per-sample ω(n)

        # Build a partial function for the diffusion training loss computation.
        # This calls the diffusion object's training_losses method, which:
        #   1. Noises the target u^{n+1} via x_t = √ᾱ_t · u^{n+1} + √(1-ᾱ_t) · ε (Eq. 7)
        #   2. Runs the denoiser f_θ(x_t, t, u^{n-4:n}, v, n) to predict u^{n+1}
        #   3. Returns the per-sample MSE: || u^{n+1} - f_θ(...) ||^2  (Eq. 9)
        compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                batch_now_snap,       # x_0 = u^{n+1}, the clean target
                batch_cond_prev,      # u^{n-4:n}, 5-frame conditioning history
                batch_now_it,         # Wavefield time-step index n (for embedding)
                t,                    # Diffusion time step t (sampled)
                model_kwargs=cond,    # Additional conditioning (velocity model v)
        )

        losses = compute_losses()

        # If using a loss-aware sampler (e.g., for importance sampling of diffusion
        # time steps t), update it with the per-sample losses from this batch.
        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
            )

        # ---------------------------------------------------------------
        # Compute the full causal time-weighted loss (Eq. 12):
        #   L = (1/B) Σ_b  ω(n_b) · || u^{n+1}_b - f_θ(x_t, t, ...) ||^2
        # The causal weight ω(n) is treated as a constant w.r.t. model
        # parameters (stop-gradient), since it is derived from the
        # stop-gradient EMA buffer L_ema.
        # ---------------------------------------------------------------
        loss = (losses["mse"] * sample_weights).mean()

        # Log per-quartile loss statistics for monitoring training dynamics.
        log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
        )

        # Backpropagate. In fp16 mode, scale the loss to prevent gradient underflow.
        if self.use_fp16:
            loss_scale = 2 ** self.lg_loss_scale
            (loss * loss_scale).backward()
        else:
            loss.backward()

        # ------------------------------------------------------------------
        # Update the EMA error buffer L_ema[n] (Eq. 10):
        #   L_ema^{(i+1)}[n] = γ · L_ema^{(i)}[n] + (1 - γ) · ||u^{n+1} - f_θ(...)||^2
        # Only the snapshot indices that appear in the current minibatch are
        # updated; the remaining entries of L_ema are left unchanged. This
        # selective update decouples the running estimate from the stochasticity
        # of minibatch sampling and lets L_ema accumulate information across
        # the entire training history (Section III.3).
        # ------------------------------------------------------------------
        self.cwm.update(batch_now_it_idx, losses["mse"].detach())

    def optimize_fp16(self):
        """
        Perform a single optimizer step under mixed-precision (fp16) training.

        Handles NaN detection in gradients, loss scale adjustment, gradient
        transfer from fp16 model params to fp32 master params, and EMA updates.
        """
        # Check for NaN/Inf gradients; if found, reduce the loss scale and skip.
        if any(not th.isfinite(p.grad).all() for p in self.model_params):
            self.lg_loss_scale -= 1
            logger.log(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            return

        # Copy gradients from fp16 model parameters to fp32 master parameters.
        model_grads_to_master_grads(self.model_params, self.master_params)
        # Undo the loss scaling applied during backward().
        self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        # Update EMA copies of model parameters for stable inference.
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)
        # Sync the fp16 model parameters with the updated fp32 master parameters.
        master_params_to_model_params(self.model_params, self.master_params)
        # Gradually increase the loss scale for fp16 stability.
        self.lg_loss_scale += self.fp16_scale_growth

    def optimize_normal(self):
        """
        Perform a single optimizer step under standard (fp32) training.

        Logs gradient norm, optionally anneals the learning rate, steps the
        AdamW optimizer, and updates EMA copies of the model parameters.
        """
        self._log_grad_norm()
        self._anneal_lr()
        self.opt.step()
        # Update EMA copies of model parameters (decay rate 0.999, Section III.I).
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.master_params, rate=rate)

    def _log_grad_norm(self):
        """
        Compute and log the L2 norm of the gradient vector across all model
        parameters. Useful for monitoring training stability.
        """
        sqsum = 0.0
        for p in self.master_params:
            sqsum += (p.grad ** 2).sum().item()
        logger.logkv_mean("grad_norm", np.sqrt(sqsum))

    def _anneal_lr(self):
        """
        Linearly anneal the learning rate from the initial value to zero over
        lr_anneal_steps. Does nothing if lr_anneal_steps == 0.
        """
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """Log the current training step count and cumulative samples processed."""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)
        if self.use_fp16:
            logger.logkv("lg_loss_scale", self.lg_loss_scale)

    def save(self):
        """
        Save model checkpoints, including EMA-smoothed parameter snapshots.

        Saves one checkpoint per EMA rate. Filenames follow the convention:
          - ema_{rate}_{step:06d}.pt   for EMA-smoothed parameters
          - model{step:06d}.pt         for the raw model (if rate == 0)
        """
        def save_checkpoint(rate, params):
            state_dict = self._master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                 filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join(dir_checkpoints, filename), "wb") as f:
                th.save(state_dict, f)

        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

    def _master_params_to_state_dict(self, master_params):
        """
        Convert the list of master (fp32) parameter tensors back into a
        state_dict compatible with model.load_state_dict().
        Handles the unflattening needed for fp16 training.
        """
        if self.use_fp16:
            master_params = unflatten_master_params(
                self.model.parameters(), master_params
            )
        state_dict = self.model.state_dict()
        for i, (name, _value) in enumerate(self.model.named_parameters()):
            assert name in state_dict
            state_dict[name] = master_params[i]
        return state_dict

    def _state_dict_to_master_params(self, state_dict):
        """
        Convert a state_dict into a list of master parameter tensors,
        suitable for use as master_params in fp16 training.
        """
        params = [state_dict[name] for name, _ in self.model.named_parameters()]
        if self.use_fp16:
            return make_master_params(params)
        else:
            return params


# ============================================================================
# Causal Weight Manager (Section III.3, Eqs. 10–12)
# ============================================================================
class CausalWeightManager:
    """
    Manages per-snapshot EMA error estimates and computes causal weights ω(n)
    for the training objective (Eq. 12).

    The causal time-weighted loss aligns the training trajectory with the physical
    direction of wave propagation. It maintains a running EMA estimate L_ema[n] of
    the prediction error at each wavefield time-step index n (Eq. 10). These
    estimates are accumulated across all training iterations, with only the entries
    corresponding to snapshot indices in the current minibatch being updated. This
    selective update decouples the running estimates from minibatch stochasticity
    and lets L_ema aggregate information over the entire training history.

    The causal weight for snapshot n is:
        ω(n) = exp(-ε · Σ_{k=0}^{n-1} L_ema[k])         (Eq. 11)
    When the cumulative error at preceding snapshots is large, ω(n) is small and
    snapshot n contributes negligibly to the gradient. As training progresses and
    early entries of L_ema shrink, ω(n) grows, and progressively later snapshots
    are admitted into the active training set.
    """

    def __init__(self, num_physical_steps, epsilon=1e-4, delta=0.99,
                 ema_decay=0.99, device='cuda'):
        """
        Args:
            num_physical_steps: Total number of wavefield physical time steps N_t
                                (e.g., 101 for 0–1 s at Δt = 0.01 s).
            epsilon:            Causal decay coefficient ε in Eq. 11. Controls
                                how rapidly ω(n) decays with cumulative preceding error.
            delta:              Saturation threshold δ in Eq. 11. Weights ≥ δ are
                                clamped to exactly 1.0 to prevent numerical drift near
                                saturation and to provide a clean convergence indicator.
            ema_decay:          EMA smoothing coefficient γ in Eq. 10 (denoted γ in the
                                manuscript). Higher values produce smoother estimates.
            device:             Compute device ('cuda' or 'cpu').
        """
        self.Nt        = num_physical_steps
        self.epsilon   = epsilon
        self.delta     = delta
        self.ema_decay = ema_decay
        self.device    = device

        # Initialize L_ema to all ones (Eq. 10 initialization), encoding the prior
        # assumption that all snapshots are equally untrained at the start. With
        # L_ema[k] = 1 for all k, the cumulative sum grows linearly with n, so
        # ω(n) = exp(-ε · n) decays exponentially — ensuring that late-time
        # snapshots contribute negligibly at the beginning of training.
        self.L_ema = th.ones(num_physical_steps, device=device)

    @th.no_grad()
    def update(self, t_indices, errors):
        """
        Update the EMA error estimates L_ema[n] using per-sample losses from
        the current minibatch (Eq. 10):

            L_ema^{(i+1)}[n] = γ · L_ema^{(i)}[n] + (1 - γ) · ||u^{n+1} - f_θ(...)||^2

        Only entries whose snapshot index appears in the current minibatch are
        updated; all other entries are left unchanged.

        Args:
            t_indices: [B] Integer wavefield time-step indices from the current batch.
            errors:    [B] Corresponding per-sample prediction errors (detached from
                       the computation graph to prevent gradient flow into L_ema).
        """
        for t, e in zip(t_indices, errors):
            t = t.item()
            self.L_ema[t] = (self.ema_decay * self.L_ema[t]
                             + (1 - self.ema_decay) * e)

    @th.no_grad()
    def get_weights(self):
        """
        Compute the causal weights ω(n) for all physical time steps based on
        the current EMA error state table (Eq. 11).

        The weight for snapshot n is:
            cumsum[n] = Σ_{k=0}^{n-1} L_ema[k]
            ω(n) = exp(-ε · cumsum[n])

        Note: cumsum[0] = 0, so ω(0) = 1 always — the first snapshot (source
        injection) is always fully weighted.

        Returns:
            weights: [N_t] Causal weight for each physical time step. Values are
                     in (0, 1], with ω(n) clamped to 1.0 when ω(n) ≥ δ.
        """
        # Compute the exclusive cumulative sum: cumsum[n] = Σ_{k=0}^{n-1} L_ema[k].
        # cumsum[0] = 0 by construction (no preceding error for the first snapshot).
        cumsum = th.zeros(self.Nt, device=self.device)
        cumsum[1:] = th.cumsum(self.L_ema[:-1], dim=0)

        # Apply the exponential causal decay (Eq. 11).
        weights = th.exp(-self.epsilon * cumsum)

        # Saturation threshold: clamp weights ≥ δ to exactly 1.0. This prevents
        # numerical drift near saturation and provides a clean convergence indicator:
        # training is regarded as having propagated through the entire wavefield
        # sequence once min_n ω(n) ≥ δ.
        weights = th.where(
            weights >= self.delta,
            th.ones_like(weights),
            weights
        )
        return weights

    def is_converged(self):
        """
        Check whether training has converged in the causal sense: all physical
        time steps have been admitted into the active training set (min_n ω(n) ≥ δ).
        """
        return self.get_weights().min().item() >= self.delta


def parse_resume_step_from_filename(filename):
    """
    Extract the training step number from a checkpoint filename.

    Expected format: path/to/modelNNNNNN.pt, where NNNNNN is the zero-padded
    step count. Returns 0 if the filename does not match this pattern.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def parse_dataname_from_filename(filename):
    """
    Extract a dataset identifier from a checkpoint filename.

    Expected format: path containing 'gaussian5' followed by the dataset name.
    Returns 0 if the filename does not match this pattern.
    """
    split = filename.split("gaussian5")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return split1
    except ValueError:
        return 0


def get_blob_logdir():
    """
    Return the directory for storing training logs. Defaults to the logger's
    current directory, but can be overridden via the DIFFUSION_BLOB_LOGDIR
    environment variable.
    """
    return os.environ.get("DIFFUSION_BLOB_LOGDIR", logger.get_dir())


def find_resume_checkpoint():
    """
    Attempt to automatically discover the latest checkpoint on blob storage.

    Returns None by default; override this function on infrastructure where
    automatic checkpoint discovery is available.
    """
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    """
    Locate the EMA checkpoint file corresponding to a given main checkpoint,
    step number, and EMA decay rate.

    Args:
        main_checkpoint: Path to the main model checkpoint.
        step:            Training step number.
        rate:            EMA decay rate.

    Returns:
        The path to the EMA checkpoint if it exists, otherwise None.
    """
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    """
    Log training loss statistics, including per-quartile breakdowns by
    diffusion time step t.

    For each loss key, logs the mean value and also the mean within each
    of four quartiles of the diffusion schedule (t divided into four equal
    ranges over [0, T]). This helps monitor whether the denoiser performs
    uniformly across noise levels or struggles at specific scales.

    Args:
        diffusion: The diffusion object (used to determine the total number
                   of diffusion time steps T for quartile computation).
        ts:        [B] Tensor of diffusion time steps for the current batch.
        losses:    dict mapping loss names to [B] tensors of per-sample values.
    """
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)