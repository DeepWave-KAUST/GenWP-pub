"""
U-Net architecture for the conditional denoiser f_θ of the Generative Wave
Propagator (GenWP).

This module implements the network described in Section V of the manuscript.
The architecture is a U-Net adapted from the improved DDPM codebase (Nichol &
Dhariwal, 2021), comprising four resolution stages with feature dimensions
[model_channels, 2×, 4×, 8×] (e.g., 64, 128, 256, 512), arranged as an
encoder–bottleneck–decoder with skip connections between matching stages.

Four conditioning signals are injected into every residual block through
complementary mechanisms (Section V):

  1. Noisy diffusion input x_t:
       Concatenated with the spatial conditioning feature map at the U-Net
       entry after passing through a shallow convolutional stem.

  2. Diffusion step t:
       Mapped to a sinusoidal positional encoding → 2-layer MLP → embedding
       vector. Modulates the main residual branch via a FiLM-style (Feature-wise
       Linear Modulation) affine transformation (scale + shift), controlling
       the noise-level dependence of the denoiser.

  3. Spatial conditioning (u^{n-4:n}, v):
       The 5 most recent wavefield snapshots and the velocity model are stacked
       channel-wise into a (snap_steps + 1)-channel input, processed by a
       shallow convolutional stem into a feature map at the base channel width.
       At each resolution stage, this feature map is adaptively downsampled to
       match the current spatial resolution and fused with the residual output
       via channel-wise concatenation followed by a 3×3 convolution.

  4. Wavefield time-step index n:
       Mapped to a sinusoidal positional encoding → 2-layer MLP → embedding
       vector. Modulates the spatial conditioning feature map via a FiLM-style
       affine transformation, allowing the propagation operator to adapt to
       time-dependent characteristics of the wavefield sequence.

The final output projection is zero-initialized so that the network begins
training as the identity mapping on its main residual path, stabilizing the
early optimization dynamics (Section V).
"""

from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    checkpoint,
)


class PositionalEncod(nn.Module):
    """
    Fourier positional encoding for scalar inputs.

    Produces a feature vector by concatenating sin and cos of the input
    at geometrically spaced frequencies: [sin(π·2^k · x), cos(π·2^k · x)]
    for k = 0, ..., PosEnc-1, appended to the original input.

    Args:
        PosEnc: Number of frequency bands (each produces a sin and cos pair).
        device: Compute device.
    """

    def __init__(self, PosEnc=2, device='cuda'):
        super().__init__()
        self.PEnc = PosEnc
        # Precompute frequency multipliers: π · 2^k for k = 0, ..., PosEnc-1
        self.k_pi_sx = (th.tensor(np.pi)*(2**th.arange(self.PEnc))).reshape(-1, self.PEnc).to(device)
        self.k_pi_sx = self.k_pi_sx.T

    def forward(self, input):
        """
        Args:
            input: [B, D] Tensor; the first column is encoded.

        Returns:
            [B, D + 2*PosEnc] Tensor with positional features appended.
        """
        tmpsx = th.cat([th.sin(self.k_pi_sx*input[:,0]), th.cos( self.k_pi_sx*input[:,0])], axis=0)
        return th.cat([input, tmpsx.T],-1)


class TimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for scalar time indices, following the
    Transformer convention (Vaswani et al., 2017).

    Used for both the diffusion step t and the wavefield time-step index n
    (Section V). Each scalar index is mapped to a dim-dimensional vector via
    interleaved sin/cos of geometrically spaced frequencies.

    Args:
        dim:   Embedding dimension (must be even).
        scale: Linear scale applied to the input before encoding. For the
               diffusion step t this may be set to 1000/T to normalize; for
               the wavefield index n this is typically 1.0.

    Input:  x of shape [B]
    Output: embedding of shape [B, dim]
    """

    def __init__(self, dim, scale=1.0):
        super().__init__()
        assert dim % 2 == 0
        self.dim = dim
        self.scale = scale

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        # Frequency spacing: exp(-k · log(10000) / (dim/2)) for k = 0, ..., dim/2 - 1
        emb = math.log(10000) / half_dim
        emb = th.exp(th.arange(half_dim, device=device) * -emb)
        # Outer product: [B] × [dim/2] → [B, dim/2]
        emb = th.outer(x * self.scale, emb)
        # Concatenate sin and cos: [B, dim]
        emb = th.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TimestepBlock(nn.Module):
    """
    Abstract base class for modules whose forward() method accepts the
    diffusion step embedding (time_emb), the spatial conditioning feature map
    (cond_emb), and the wavefield time-step index embedding (now_it_emb)
    as additional arguments beyond the main input x.
    """

    @abstractmethod
    def forward(self, x, time_emb, cond_emb, now_it_emb):
        """
        Apply the module to `x` given conditioning embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that forwards the conditioning embeddings (time_emb,
    cond_emb, now_it_emb) to children that support them (i.e., TimestepBlock
    subclasses), and calls plain forward(x) on all other layers.

    This allows mixing standard layers (e.g., Downsample, AttentionBlock) with
    conditioning-aware layers (e.g., ResBlock) in a single nn.Sequential.
    """

    def forward(self, x, time_emb, cond_emb, now_it_emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, time_emb, cond_emb, now_it_emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    Spatial upsampling by a factor of 2 (nearest-neighbor interpolation),
    optionally followed by a 3×3 convolution.

    Used in the decoder path of the U-Net to progressively restore spatial
    resolution.

    Args:
        channels:  Number of input/output channels.
        use_conv:  If True, apply a learned 3×3 convolution after upsampling.
        dims:      Spatial dimensionality (1D, 2D, or 3D).
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    Spatial downsampling by a factor of 2, using either a strided convolution
    or average pooling.

    Used in the encoder path of the U-Net to progressively reduce spatial
    resolution.

    Args:
        channels:  Number of input/output channels.
        use_conv:  If True, use a learned strided 3×3 convolution; otherwise
                   use average pooling.
        dims:      Spatial dimensionality (1D, 2D, or 3D).
    """

    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    Residual block with dual FiLM conditioning, as described in Section V.

    Each residual block performs the following:

    1. Main residual branch (modulated by diffusion step t):
       - GroupNorm → SiLU → 3×3 Conv → [FiLM by t] → GroupNorm → SiLU →
         Dropout → 3×3 Conv (zero-initialized)
       - The diffusion step embedding of t generates scale and shift parameters
         via a FiLM-style affine transformation, controlling the noise-level
         dependence of the denoiser (Section V).

    2. Spatial conditioning injection (modulated by wavefield index n):
       - The spatial conditioning feature map (from the convolutional stem
         processing u^{n-4:n} and v) is adaptively downsampled to match the
         current resolution stage.
       - A 3×3 convolution projects it to the block's channel dimension.
       - The wavefield time-step index embedding of n generates scale and shift
         parameters via an analogous FiLM operation, acting as a temporal-context
         signal (Section V).
       - The modulated conditioning is processed through GroupNorm → SiLU →
         Dropout → zero-initialized 3×3 Conv.
       - The result is concatenated channel-wise with the residual output and
         fused via GroupNorm → SiLU → 3×3 Conv.

    This factorization reflects the distinct roles of the two scalar indices:
    t acts on the diffusion variable as a noise-level signal, while n acts on
    the physical conditioning as a temporal-context signal (Section V).

    Args:
        channels:               Number of input channels.
        time_emb_channels:      Dimension of the diffusion step t embedding.
        dropout:                Dropout probability.
        out_channels:           Number of output channels (defaults to input channels).
        cond_channels:          Number of channels in the spatial conditioning feature
                                map. If None, spatial conditioning injection is disabled.
        it_embed_dim:           Dimension of the wavefield time-step index n embedding.
        use_conv:               If True, use 3×3 conv for the skip connection when
                                changing channel count; otherwise use 1×1 conv.
        use_scale_shift_norm:   If True, use FiLM (scale + shift) modulation;
                                otherwise use additive injection.
        dims:                   Spatial dimensionality (1D, 2D, or 3D).
        use_checkpoint:         If True, use gradient checkpointing to reduce memory.
    """

    def __init__(
        self,
        channels,
        time_emb_channels,
        dropout,
        out_channels=None,
        cond_channels=None,
        it_embed_dim=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.time_emb_channels = time_emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.cond_channels = cond_channels
        self.it_embed_dim = it_embed_dim
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        # ---- Main residual branch: input layers ----
        # GroupNorm → SiLU → 3×3 Conv
        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # ---- Diffusion step t embedding projection ----
        # Projects the t embedding to scale+shift parameters (if FiLM) or
        # additive bias. This is the noise-level signal that modulates the
        # main residual branch (Section V).
        self.time_emb_layers = nn.Sequential(
            SiLU(),
            linear(
                time_emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )

        # ---- Main residual branch: output layers ----
        # GroupNorm → SiLU → Dropout → zero-initialized 3×3 Conv
        # Zero initialization ensures the network starts as the identity
        # mapping on its main residual path (Section V).
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        # ---- Skip connection ----
        # Identity if channels match; otherwise a 1×1 or 3×3 conv.
        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

        # ---- Spatial conditioning injection (Section V) ----
        # Only created when cond_channels is specified, i.e., when the block
        # receives the spatial conditioning feature map from (u^{n-4:n}, v).
        if cond_channels is not None:
            # Project the spatial conditioning to the block's channel dimension.
            # This includes adaptive downsampling (done in forward) and a
            # 3×3 convolution.
            self.cond_emb_conv = nn.Sequential(
                normalization(self.cond_channels),
                SiLU(),
                conv_nd(dims, self.cond_channels, self.out_channels, 3, padding=1),
            )

            # Wavefield time-step index n embedding projection.
            # Projects the n embedding to scale+shift parameters for FiLM
            # modulation of the spatial conditioning. This is the temporal-
            # context signal (Section V).
            self.it_emb_layers = nn.Sequential(
                SiLU(),
                linear(
                    it_embed_dim,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )

            # Post-modulation processing of the spatial conditioning:
            # GroupNorm → SiLU → Dropout → zero-initialized 3×3 Conv.
            self.cond_fuse = nn.Sequential(
                normalization(self.out_channels),
                SiLU(),
                nn.Dropout(p=dropout),
                zero_module(
                    conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
                ),
            )

            # Final fusion: concatenate the residual output and the modulated
            # spatial conditioning channel-wise, then reduce back to
            # out_channels via GroupNorm → SiLU → 3×3 Conv (Section V).
            self.allcond_fuse = nn.Sequential(
                normalization(2 * self.out_channels),
                SiLU(),
                conv_nd(dims, 2 * self.out_channels, self.out_channels, 3, padding=1),
            )

    def forward(self, x, time_emb, cond_emb=None, now_it_emb=None):
        """
        Apply the residual block with dual FiLM conditioning.

        Args:
            x:           [B, C, H, W] Input feature map.
            time_emb:    [B, time_emb_channels] Diffusion step t embedding.
            cond_emb:    [B, cond_channels, H0, W0] Spatial conditioning feature
                         map from the convolutional stem (u^{n-4:n}, v). May be
                         at a different resolution than x and will be adaptively
                         downsampled. None if conditioning is disabled.
            now_it_emb:  [B, it_embed_dim] Wavefield time-step index n embedding.
                         None if conditioning is disabled.

        Returns:
            [B, out_channels, H, W] Output feature map.
        """
        return checkpoint(
            self._forward, (x, time_emb, cond_emb, now_it_emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, time_emb, cond_emb=None, now_it_emb=None):
        # ---- Main residual branch with diffusion step t modulation ----
        # Step 1: Input layers (GroupNorm → SiLU → 3×3 Conv)
        h = self.in_layers(x)

        # Step 2: Project the diffusion step t embedding to spatial dimensions.
        time_emb_out = self.time_emb_layers(time_emb).type(h.dtype)
        while len(time_emb_out.shape) < len(h.shape):
            time_emb_out = time_emb_out[..., None]  # [B, C] → [B, C, 1, 1]

        # Step 3: Apply t modulation via FiLM (scale + shift) or addition.
        if self.use_scale_shift_norm:
            # FiLM: split embedding into scale γ and shift β, then apply
            # h = (GroupNorm(h) · (1 + γ)) + β
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(time_emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            # Additive: h = h + time_emb_out, then output layers
            h = h + time_emb_out
            h = self.out_layers(h)

        # Step 4: Residual connection
        out = self.skip_connection(x) + h

        # ---- Spatial conditioning injection with n modulation (Section V) ----
        if self.cond_channels is not None and cond_emb is not None:
            B, C, H, W = out.shape

            # Step 5: Adaptively downsample the spatial conditioning feature map
            # to match the current resolution stage of the U-Net.
            cond_ds = F.adaptive_avg_pool2d(cond_emb, output_size=(H, W))

            # Step 6: Project to the block's channel dimension via 3×3 conv.
            cond_ds = self.cond_emb_conv(cond_ds)

            # Step 7: Project the wavefield time-step index n embedding.
            it_emb_out = self.it_emb_layers(now_it_emb).type(h.dtype)
            while len(it_emb_out.shape) < len(out.shape):
                it_emb_out = it_emb_out[..., None]  # [B, C] → [B, C, 1, 1]

            # Step 8: Apply n modulation via FiLM or addition on the spatial
            # conditioning. This is the temporal-context signal (Section V):
            # n acts on the physical conditioning, while t acts on the
            # diffusion variable.
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.cond_fuse[0], self.cond_fuse[1:]
                scale, shift = th.chunk(it_emb_out, 2, dim=1)
                cond_ds = out_norm(cond_ds) * (1 + scale) + shift
                cond_ds = out_rest(cond_ds)
            else:
                cond_ds = cond_ds + it_emb_out
                cond_ds = self.cond_fuse(cond_ds)

            # Step 9: Fuse the residual output with the modulated spatial
            # conditioning via channel-wise concatenation + 3×3 conv.
            fused = th.cat([out, cond_ds], dim=1)       # [B, 2*C, H, W]
            out = self.allcond_fuse(fused)              # [B, C, H, W]

        return out


class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block for capturing long-range spatial coherence
    in the wavefield (Section V).

    Self-attention layers are inserted at the two coarsest resolutions of the
    U-Net to model global spatial dependencies that local convolutions cannot
    capture, such as wavefront geometry across the full domain.

    Args:
        channels:       Number of input/output channels.
        num_heads:      Number of attention heads.
        use_checkpoint: If True, use gradient checkpointing to save memory.
    """

    def __init__(self, channels, num_heads=4, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        # Joint QKV projection: maps C channels to 3C channels.
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        # Zero-initialized output projection for residual learning.
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        # Flatten spatial dimensions: [B, C, H, W] → [B, C, H*W]
        x = x.reshape(b, c, -1)
        # Compute Q, K, V via joint linear projection after group normalization.
        qkv = self.qkv(self.norm(x))
        # Split into multi-head format: [B*num_heads, C/num_heads, H*W]
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        # Apply scaled dot-product attention.
        h = self.attention(qkv)
        # Reshape back: [B, C, H*W]
        h = h.reshape(b, -1, h.shape[-1])
        # Zero-initialized output projection + residual connection.
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    Scaled dot-product attention operating on pre-computed Q, K, V tensors.

    Computes: Attention(Q, K, V) = softmax(Q^T K / √d) V
    where d is the per-head channel dimension.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        Args:
            qkv: [B*num_heads, 3*(C/num_heads), N] Concatenated Q, K, V
                 where N = H*W is the number of spatial positions.

        Returns:
            [B*num_heads, C/num_heads, N] Attention output.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        # Scale factor: 1/√(√d) applied to both Q and K for fp16 stability,
        # equivalent to 1/√d when multiplied together.
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        FLOPs counter for the `thop` profiling package.

        Counts the two matrix multiplications in attention:
        (1) Q^T K  and  (2) attention_weights × V.
        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class LinearAttention(nn.Module):
    """
    Memory-efficient linearized attention using the kernel trick.

    Replaces softmax(QK^T) with φ(Q)φ(K)^T where φ(x) = elu(x) + 1,
    reducing complexity from O(N²) to O(N·C). Useful for high-resolution
    feature maps where standard attention is prohibitively expensive.

    Not used at the default configuration (attention only at the two coarsest
    resolutions), but available for experimentation.
    """

    def __init__(self):
        super().__init__()
        self.eps = 1e-6

    def forward(self, Q, K, V):
        """
        Args:
            Q, K, V: [B, N, C] Query, Key, Value tensors.

        Returns:
            [B, N, C] Attention output.
        """
        phi = lambda x: F.elu(x) + 1
        Q_phi = phi(Q)  # [B, N, C]
        K_phi = phi(K)  # [B, N, C]
        # KV accumulator: [B, C, C] — avoids materializing the N×N attention matrix.
        KV = th.einsum('bnc,bnd->bcd', K_phi, V)
        # Normalization denominator
        denom = th.einsum('bnc,bnc->bn', Q_phi, K_phi.sum(dim=1, keepdim=True).expand_as(Q_phi))
        Z = 1.0 / (denom + self.eps)
        # Compute output: Q_phi @ KV, then normalize
        out = th.einsum('bnc,bcd->bnd', Q_phi, KV)
        out = out * Z.unsqueeze(-1)
        return out


class UNetModel(nn.Module):
    """
    The full U-Net conditional denoiser f_θ (Section V).

    This network predicts the clean wavefield u^{n+1} from its noised version
    x_t, conditioned on:
      - The 5-frame wavefield history u^{n-4:n} and velocity model v
        (stacked as a (snap_steps+1)-channel spatial input)
      - The diffusion step t (sinusoidal embedding → MLP)
      - The wavefield time-step index n (sinusoidal embedding → MLP)

    Architecture:
      - Encoder: num_res_blocks ResBlocks per level, with downsampling between
        levels. Self-attention is inserted at resolutions in attention_resolutions.
      - Bottleneck: ResBlock → AttentionBlock → ResBlock at the coarsest level.
      - Decoder: (num_res_blocks + 1) ResBlocks per level with skip connections
        from the encoder, and upsampling between levels.
      - Output: GroupNorm → SiLU → zero-initialized 3×3 Conv.

    Args:
        in_channels:           Channels of the noisy input x_t (1 for acoustic).
        model_channels:        Base channel count (e.g., 64). Feature dims at
                               each level are model_channels × channel_mult[level].
        out_channels:          Channels of the predicted u^{n+1} (1 for acoustic).
        num_res_blocks:        Number of ResBlocks per encoder level.
        attention_resolutions: Set of downsample factors at which self-attention
                               is applied (e.g., {4, 8} for the two coarsest levels).
        dropout:               Dropout probability in ResBlocks.
        channel_mult:          Tuple of channel multipliers per level (e.g.,
                               (1, 2, 4, 8) gives 64, 128, 256, 512).
        conv_resample:         If True, use learned convolutions for up/downsampling.
        dims:                  Spatial dimensionality (2 for 2D wavefields).
        time_emb_scale:        Scale for the diffusion step t encoding.
        num_classes:           If specified, enables class-conditional generation.
        use_checkpoint:        If True, use gradient checkpointing to save memory.
        num_heads:             Number of attention heads.
        num_heads_upsample:    Number of attention heads in the decoder (defaults
                               to num_heads).
        use_scale_shift_norm:  If True, use FiLM (scale + shift) modulation in
                               ResBlocks; otherwise use additive injection.
        snap_steps:            Number of conditioning wavefield snapshots (e.g., 10
                               for 5 snapshots × 2 or the actual history length).
                               Together with the velocity model, forms a
                               (snap_steps + 1)-channel spatial conditioning input.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        time_emb_scale=1.0,
        num_classes=None,
        use_checkpoint=False,
        num_heads=4,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        snap_steps=10,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        # Padding size to ensure spatial dimensions are divisible by 2^num_levels.
        self.padder_size = 2 ** len(channel_mult)
        self.snap_steps = snap_steps

        # ================================================================
        # Embedding networks (Section V)
        # ================================================================

        # ---- Diffusion step t embedding ----
        # Sinusoidal positional encoding → 2-layer MLP (dim → 4×dim → 4×dim).
        # Produces a vector of dimension time_embed_dim = 4 × model_channels
        # (e.g., 256 when model_channels = 64).
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(model_channels, time_emb_scale),
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # ---- Wavefield time-step index n embedding ----
        # Sinusoidal positional encoding → 2-layer MLP, same architecture as
        # the diffusion step embedding but with scale = 1.0.
        # This allows the propagation operator to adapt to time-dependent
        # characteristics of the wavefield sequence (Section V).
        it_embed_dim = model_channels * 4
        self.it_embed = nn.Sequential(
            TimeEmbedding(model_channels, 1.0),
            linear(model_channels, it_embed_dim),
            SiLU(),
            linear(it_embed_dim, it_embed_dim),
        )

        # ---- Spatial conditioning stem (Section V) ----
        # Processes the (snap_steps + 1)-channel input, which comprises the
        # snap_steps most recent wavefield snapshots (u^{n-4:n}) and the
        # velocity model v, stacked channel-wise. The stem produces a feature
        # map at the base channel width (model_channels), which is then
        # concatenated with x_t at the U-Net entry and also injected into
        # every ResBlock at every resolution stage.
        self.cond_embed = nn.Sequential(
            conv_nd(dims, self.snap_steps + 1, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
            conv_nd(dims, model_channels, model_channels, 3, padding=1),
            normalization(model_channels),
            SiLU(),
        )

        # Optional class-conditional embedding (not used for wavefield propagation).
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # ================================================================
        # U-Net entry: concatenate x_t with spatial conditioning, then project.
        # Input channels = in_channels (x_t) + model_channels (cond stem output).
        # ================================================================
        self.inp = conv_nd(dims, in_channels + model_channels, model_channels, 3, padding=1)

        # ================================================================
        # Encoder path (Section V)
        # ================================================================
        self.downs = nn.ModuleList([])
        encoder_channels = [model_channels]  # Track channel counts for skip connections
        ch = model_channels
        ds = 1  # Current downsample factor
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        cond_channels=model_channels,
                        it_embed_dim=it_embed_dim,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # Insert self-attention at specified resolutions (e.g., the
                # two coarsest levels, Section V).
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                self.downs.append(TimestepEmbedSequential(*layers))
                encoder_channels.append(ch)
            # Downsample between levels (except at the last level).
            if level != len(channel_mult) - 1:
                self.downs.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                encoder_channels.append(ch)
                ds *= 2

        # ================================================================
        # Bottleneck (Section V): ResBlock → AttentionBlock → ResBlock
        # at the coarsest resolution.
        # ================================================================
        self.middle = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_channels=model_channels,
                it_embed_dim=it_embed_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                cond_channels=model_channels,
                it_embed_dim=it_embed_dim,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # ================================================================
        # Decoder path (Section V)
        # Mirror of the encoder with skip connections from matching levels.
        # Each level has (num_res_blocks + 1) ResBlocks to accommodate the
        # skip connection from the corresponding encoder level.
        # ================================================================
        self.ups = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        # Input channels: current ch + skip connection channels
                        ch + encoder_channels.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        cond_channels=model_channels,
                        it_embed_dim=it_embed_dim,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                # Insert self-attention at the same resolutions as the encoder.
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                        )
                    )
                # Upsample at the last ResBlock of each level (except level 0).
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.ups.append(TimestepEmbedSequential(*layers))

        # ================================================================
        # Output projection (Section V)
        # GroupNorm → SiLU → zero-initialized 3×3 Conv.
        # Zero initialization ensures the network begins training as the
        # identity mapping on its main residual path.
        # ================================================================
        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """Convert the encoder, bottleneck, and decoder to float16."""
        self.downs.apply(convert_module_to_f16)
        self.middle.apply(convert_module_to_f16)
        self.ups.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """Convert the encoder, bottleneck, and decoder to float32."""
        self.downs.apply(convert_module_to_f32)
        self.middle.apply(convert_module_to_f32)
        self.ups.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        """Get the dtype used by the encoder/bottleneck/decoder (fp16 or fp32)."""
        return next(self.downs.parameters()).dtype

    def forward(self, inp, cond_prev, now_it, timesteps, y=None):
        """
        Forward pass of the conditional denoiser f_θ (Section V).

        Implements the mapping:
            f_θ(x_t, t, u^{n-4:n}, v, n) → predicted u^{n+1}

        Processing steps:
          1. Pad input and conditioning to be divisible by 2^num_levels.
          2. Process the spatial conditioning (u^{n-4:n}, v) through the
             shallow convolutional stem → cond_prev feature map.
          3. Concatenate x_t with the cond_prev feature map at the U-Net entry.
          4. Compute sinusoidal embeddings for t (time_emb) and n (it_emb).
          5. Pass through encoder → bottleneck → decoder with skip connections.
             At every ResBlock, t modulates the main branch via FiLM, and the
             spatial conditioning is injected and modulated by n via FiLM.
          6. Apply the output projection and crop to the original spatial size.

        Args:
            inp:        [B, in_channels, H, W] Noisy diffusion input x_t.
            cond_prev:  [B, snap_steps+1, H, W] Spatial conditioning: the
                        snap_steps most recent wavefield snapshots u^{n-4:n}
                        and the velocity model v, stacked channel-wise.
            now_it:     [B] Wavefield time-step index n (float, for embedding).
            timesteps:  [B] Diffusion step t (integer or float).
            y:          [B] Optional class labels (not used for wavefields).

        Returns:
            [B, out_channels, H, W] Predicted clean wavefield u^{n+1}.
        """
        b, c, h, w = inp.shape

        # Step 1: Pad inputs to ensure spatial dims are divisible by padder_size.
        inp = self.check_image_size(inp)
        cond_prev = self.check_image_size(cond_prev)

        # Step 2: Process spatial conditioning through the convolutional stem.
        # (u^{n-4:n}, v) → feature map at base channel width.
        cond_prev = self.cond_embed(cond_prev)

        # Step 3: Concatenate x_t with the spatial conditioning feature map.
        # This provides a direct, full-resolution physical context alongside
        # the diffusion input (Section V).
        x = th.cat([inp, cond_prev], dim=1)

        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"

        # Step 4: Compute embedding vectors for t and n.
        time_emb = self.time_embed(timesteps)   # [B, time_embed_dim]
        it_emb = self.it_embed(now_it)           # [B, it_embed_dim]

        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            time_emb = time_emb + self.label_emb(y)

        # Step 5a: Encoder path — collect skip connections.
        skips = []
        x = x.type(self.inner_dtype)
        x = self.inp(x)        # Project concatenated input to model_channels.
        skips.append(x)

        for module in self.downs:
            x = module(x, time_emb, cond_prev, it_emb)
            skips.append(x)

        # Step 5b: Bottleneck.
        x = self.middle(x, time_emb, cond_prev, it_emb)

        # Step 5c: Decoder path — consume skip connections in reverse order.
        for module in self.ups:
            cat_in = th.cat([x, skips.pop()], dim=1)
            x = module(cat_in, time_emb, cond_prev, it_emb)

        # Step 6: Output projection and crop to original spatial size.
        x = x.type(inp.dtype)
        x = self.out(x)
        return x[:, :, :h, :w]

    def check_image_size(self, x):
        """
        Pad the input tensor so that its spatial dimensions are divisible by
        padder_size = 2^num_levels. Uses replicate padding to avoid boundary
        artifacts in the wavefield.

        Args:
            x: [B, C, H, W] Input tensor.

        Returns:
            Padded tensor with H and W divisible by padder_size.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), mode='replicate')
        return x