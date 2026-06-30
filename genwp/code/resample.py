"""
Diffusion timestep sampling strategies for the GenWP training loop.

This module provides samplers that determine how diffusion time steps
t ∈ {1, ..., T} are drawn during training. The choice of sampler affects
the distribution of noise levels seen by the denoiser f_θ and can reduce
the variance of the training objective (Eq. 9).

In the default GenWP configuration, UniformSampler is used: diffusion
steps are drawn uniformly at random from {1, ..., T}, as stated in
Section III.2 ("samples drawn at random noise levels t ~ Uniform(1, T)").
This uniform sampling is combined with the causal time-weighted loss
(Section III.3, Eq. 12), which operates on the wavefield time-step index
n — a separate axis from the diffusion time step t.

An alternative LossSecondMomentResampler is also provided, which performs
importance sampling of t based on the second moment of historical per-step
losses, spending more training compute on noise levels where the denoiser
struggles. This sampler is not used in the default GenWP setup but is
available for experimentation.

Originally ported from the improved DDPM codebase (Nichol & Dhariwal, 2021).
"""

from abc import ABC, abstractmethod

import numpy as np
import torch as th
import torch.distributed as dist


def create_named_schedule_sampler(name, diffusion):
    """
    Factory function to create a ScheduleSampler by name.

    Args:
        name:      Sampler name. Supported values:
                     "uniform"             — UniformSampler (default for GenWP)
                     "loss-second-moment"  — LossSecondMomentResampler
        diffusion: The GaussianDiffusion (or SpacedDiffusion) object, used
                   to determine the total number of diffusion steps T.

    Returns:
        A ScheduleSampler instance.
    """
    if name == "uniform":
        return UniformSampler(diffusion)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(diffusion)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    Abstract base class for diffusion timestep samplers.

    A ScheduleSampler defines a distribution over diffusion time steps
    t ∈ {0, 1, ..., T-1} from which training samples are drawn. The default
    behavior is unbiased importance sampling: each sampled time step is
    accompanied by a weight w(t) = 1 / (T · p(t)) that corrects for the
    non-uniform sampling distribution, ensuring the expected value of the
    weighted loss equals the uniformly-averaged loss.

    Subclasses must implement weights(), which returns a positive (unnormalized)
    weight for each diffusion step. Subclasses may also override sample() to
    change the reweighting behavior.
    """

    @abstractmethod
    def weights(self):
        """
        Return a numpy array of positive (unnormalized) sampling weights,
        one per diffusion step.

        Higher weights increase the probability that the corresponding
        diffusion step is sampled during training.
        """

    def sample(self, batch_size, device):
        """
        Draw a batch of diffusion time steps via importance sampling.

        Each time step is drawn with probability proportional to its weight,
        and the corresponding importance-sampling correction weight is
        computed as w(t) = 1 / (T · p(t)), ensuring unbiased estimation of
        the uniformly-averaged training loss.

        Args:
            batch_size: Number of time steps to sample (= batch size B).
            device:     Torch device for the output tensors.

        Returns:
            Tuple of (timesteps, weights):
              - timesteps: [B] LongTensor of sampled diffusion step indices.
              - weights:   [B] FloatTensor of importance-sampling correction
                           weights. For UniformSampler, these are all ones.
        """
        w = self.weights()
        p = w / np.sum(w)  # Normalize to a proper probability distribution.
        # Draw B indices from {0, ..., T-1} with probabilities p.
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = th.from_numpy(indices_np).long().to(device)
        # Importance-sampling correction: w(t) = 1 / (T · p(t)).
        weights_np = 1 / (len(p) * p[indices_np])
        weights = th.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    """
    Uniform diffusion timestep sampler: t ~ Uniform(0, T-1).

    All diffusion steps have equal sampling probability (1/T), and the
    importance-sampling weights are all ones. This is the default sampler
    for GenWP, consistent with the standard DDPM training procedure where
    diffusion steps are drawn uniformly (Section III.2).

    Note: The causal time-weighting in GenWP (Section III.3, Eq. 12) operates
    on the wavefield time-step index n, NOT on the diffusion step t. The
    diffusion step t is always sampled uniformly.

    Args:
        diffusion: The diffusion object (provides num_timesteps = T).
    """

    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """
    Abstract base class for samplers that adapt their distribution based on
    observed training losses.

    Subclasses maintain a history of per-step losses and reweight the sampling
    distribution to focus training compute on diffusion steps where the
    denoiser has higher error. This can reduce the variance of the training
    objective at the cost of introducing bias (corrected by importance weights).

    This class provides update_with_local_losses(), which handles multi-GPU
    synchronization via all_gather before delegating to the subclass-specific
    update_with_all_losses().
    """

    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the sampling distribution using losses from the current rank.

        In distributed training, this method synchronizes losses across all
        ranks via all_gather, ensuring all workers maintain identical
        reweighting state. After synchronization, it delegates to
        update_with_all_losses() with the globally aggregated data.

        Args:
            local_ts:     [B] IntTensor of diffusion time step indices from
                          the current rank's minibatch.
            local_losses: [B] FloatTensor of corresponding per-sample losses
                          (detached from the computation graph).
        """
        batch_sizes = [
            th.tensor([0], dtype=th.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            th.tensor([len(local_ts)], dtype=th.int32, device=local_ts.device),
        )

        # Pad all_gather buffers to the maximum batch size across ranks.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [th.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [th.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)

        # Unpad and flatten the gathered batches.
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the sampling distribution using globally aggregated losses.

        This method is called identically on all ranks (after all_gather
        synchronization), so it must be deterministic to maintain consistent
        state across workers.

        Args:
            ts:     List of int diffusion time step indices.
            losses: List of float losses, one per time step.
        """


class LossSecondMomentResampler(LossAwareSampler):
    """
    Importance sampler that draws diffusion steps proportionally to the
    square root of their second moment of historical losses.

    Steps with higher average squared loss are sampled more frequently,
    focusing training compute on noise levels where the denoiser struggles
    most. A small uniform probability floor (uniform_prob) ensures that all
    steps are visited at least occasionally.

    The sampler maintains a circular buffer of the most recent
    history_per_term losses for each diffusion step. Until every step has
    accumulated history_per_term observations (the "warm-up" phase), the
    sampler falls back to uniform sampling.

    This sampler is NOT used in the default GenWP configuration (which uses
    UniformSampler for t and causal weighting for n), but is available for
    experimentation with alternative training strategies.

    Args:
        diffusion:        The diffusion object (provides num_timesteps = T).
        history_per_term: Number of recent losses to store per diffusion step
                          (circular buffer depth).
        uniform_prob:     Minimum probability floor per step, ensuring all
                          steps are visited (default: 0.001).
    """

    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        # Circular buffer: [T, history_per_term] storing recent losses per step.
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        # Count of losses accumulated so far for each step (up to history_per_term).
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        """
        Compute sampling weights proportional to the RMS (root-mean-square)
        of historical losses at each diffusion step.

        During warm-up (before all steps have history_per_term observations),
        returns uniform weights.

        Returns:
            [T] numpy array of positive sampling weights.
        """
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        # RMS of the loss history for each diffusion step.
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        # Mix with a uniform floor to ensure all steps are visited.
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        """
        Record new losses into the per-step circular buffer.

        Once a step's buffer is full (history_per_term entries), new losses
        shift out the oldest entry (FIFO behavior).

        Args:
            ts:     List of int diffusion time step indices.
            losses: List of float losses, one per time step.
        """
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Buffer full: shift left and insert the new loss at the end.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                # Buffer not yet full: append at the current count position.
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        """
        Check whether all diffusion steps have accumulated a full history
        of losses (history_per_term observations each). Until warm-up is
        complete, the sampler uses uniform weights.
        """
        return (self._loss_counts == self.history_per_term).all()