"""
fd_torch.py
===========

PyTorch implementation of the finite-difference (FD) acoustic wave-equation
solver used to generate the wavefield snapshots for the generative wave
propagator dataset.

This script is part of the dataset-generation workflow of the paper. It uses a
10th-order staggered-grid finite-difference scheme to simulate acoustic
wavefields for a given velocity model and a list of source positions. The
simulated pressure snapshots are saved and later used as training/testing data
for the conditional diffusion-based wave propagator.

Main components
---------------
1. FD10StaggeredConv2d
   Implements the 10th-order staggered finite-difference spatial derivatives
   using fixed convolution kernels. The convolution weights are registered as
   buffers because they are numerical FD coefficients rather than learnable
   neural-network parameters.

2. build_spml_torch_exact
   Builds the split perfectly matched layer (SPML) damping profiles along the
   model boundaries. These damping arrays suppress artificial boundary
   reflections in the finite-difference simulation.

3. set_wavelet_torch
   Generates the Ricker source wavelet used to inject seismic energy into the
   model.

4. acoustic_iso_torch
   Main acoustic finite-difference solver. It propagates the pressure and
   particle-velocity fields in time, extracts the physical model region after
   removing the PML and stencil padding, and saves the simulated snapshots.

Notes
-----
- The implementation is designed for data generation, not for training a neural
  network. Therefore, `run_fd` is wrapped with `torch.no_grad()` to avoid
  storing unnecessary computational graphs.
- The generated snapshots correspond to the FD reference wavefields used to
  train and evaluate the generative wave propagator.
- `time_downsample` controls the interval at which snapshots are retained from
  the original FD time stepping. For example, if dt = 0.001 s and
  time_downsample = 10, the saved wavefield interval is 0.01 s.
"""

import os
import math
import numpy as np
import scipy.io as sio

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. FD10 staggered finite difference using PyTorch conv2d
# ============================================================

class FD10StaggeredConv2d(nn.Module):
    """
    Fixed 10th-order staggered-grid spatial-derivative operator.

    The original FD solver computes spatial derivatives on a staggered grid.
    Here, the same operation is implemented with `torch.nn.functional.conv2d`.
    This is convenient because convolution is highly optimized on GPUs and can
    be applied to the full 2-D wavefield at once.

    Input shape:
        f: [B, 1, nz, nx], where B is the batch dimension. In this solver B=1.

    Output shape:
        df: [B, 1, nz, nx]

    Mapping to the original CUDA implementation:
        stagger = 0 -> forward derivative
        stagger = 1 -> backward derivative

    Important:
        The FD coefficients are stored as buffers rather than parameters,
        because they are fixed numerical-stencil coefficients and should not be
        optimized by PyTorch.
    """

    def __init__(self, dx, dz, ext_cells=5, device="cuda", dtype=torch.float32):
        super().__init__()

        self.dx = dx
        self.dz = dz
        self.ext_cells = ext_cells

        c1 = 1.21124268E+00
        c2 = -8.97216797E-02
        c3 = 1.38427734E-02
        c4 = -1.76565988E-03
        c5 = 1.18679470E-04

        coeff = torch.tensor([c1, c2, c3, c4, c5], device=device, dtype=dtype)

        # forward, stagger = 0:
        # c1*(f[j+1]-f[j]) + c2*(f[j+2]-f[j-1]) + ...
        k_forward = torch.zeros(10, device=device, dtype=dtype)

        k_forward[0] = -coeff[4]  # j-4
        k_forward[1] = -coeff[3]  # j-3
        k_forward[2] = -coeff[2]  # j-2
        k_forward[3] = -coeff[1]  # j-1
        k_forward[4] = -coeff[0]  # j
        k_forward[5] =  coeff[0]  # j+1
        k_forward[6] =  coeff[1]  # j+2
        k_forward[7] =  coeff[2]  # j+3
        k_forward[8] =  coeff[3]  # j+4
        k_forward[9] =  coeff[4]  # j+5

        # backward, stagger = 1:
        # c1*(f[j]-f[j-1]) + c2*(f[j+1]-f[j-2]) + ...
        k_backward = torch.zeros(10, device=device, dtype=dtype)

        k_backward[0] = -coeff[4]  # j-5
        k_backward[1] = -coeff[3]  # j-4
        k_backward[2] = -coeff[2]  # j-3
        k_backward[3] = -coeff[1]  # j-2
        k_backward[4] = -coeff[0]  # j-1
        k_backward[5] =  coeff[0]  # j
        k_backward[6] =  coeff[1]  # j+1
        k_backward[7] =  coeff[2]  # j+2
        k_backward[8] =  coeff[3]  # j+3
        k_backward[9] =  coeff[4]  # j+4

        self.register_buffer("kx_forward", k_forward.view(1, 1, 1, 10) / dx)
        self.register_buffer("kx_backward", k_backward.view(1, 1, 1, 10) / dx)

        self.register_buffer("kz_forward", k_forward.view(1, 1, 10, 1) / dz)
        self.register_buffer("kz_backward", k_backward.view(1, 1, 10, 1) / dz)

    def derx_forward_(self, f, df):
        f_pad = F.pad(f, pad=(4, 5, 0, 0), mode="replicate")
        tmp = F.conv2d(f_pad, self.kx_forward)
        self._copy_inner(tmp, df)

    def derx_backward_(self, f, df):
        f_pad = F.pad(f, pad=(5, 4, 0, 0), mode="replicate")
        tmp = F.conv2d(f_pad, self.kx_backward)
        self._copy_inner(tmp, df)

    def derz_forward_(self, f, df):
        f_pad = F.pad(f, pad=(0, 0, 4, 5), mode="replicate")
        tmp = F.conv2d(f_pad, self.kz_forward)
        self._copy_inner(tmp, df)

    def derz_backward_(self, f, df):
        f_pad = F.pad(f, pad=(0, 0, 5, 4), mode="replicate")
        tmp = F.conv2d(f_pad, self.kz_backward)
        self._copy_inner(tmp, df)

    def _copy_inner(self, tmp, df):
        """
        Copy only the valid interior derivative values.

        The 10th-order FD stencil requires five extra cells on each side.
        Therefore, the derivative is only physically meaningful in the inner
        region. The outer `ext_cells` region is kept at zero and acts as stencil
        padding.
        """
        e = self.ext_cells
        df.zero_()

        if e == 0:
            df.copy_(tmp)
        else:
            df[:, :, e:-e, e:-e].copy_(tmp[:, :, e:-e, e:-e])


# ============================================================
# 2. SPML damping
# ============================================================

def build_spml_torch_exact(
    dx,
    dz,
    n_pml,
    R,
    Vmax,
    nz,
    nx,
    device="cuda",
    dtype=torch.float32,
):
    """
    Build SPML damping profiles for the x and z directions.

    The damping coefficients gradually increase toward the model boundaries.
    During time stepping, they attenuate outgoing waves and reduce artificial
    reflections from the computational-domain edges.

    Parameters
    ----------
    dx, dz : float
        Spatial sampling intervals in the x and z directions.
    n_pml : int
        Number of PML cells around the physical model.
    R : float
        Target reflection coefficient of the absorbing boundary.
    Vmax : float
        Maximum velocity in the model, used to scale the damping strength.
    nz, nx : int
        Size of the PML-extended model.

    Returns
    -------
    ddx, ddz : torch.Tensor
        Damping arrays with shape [nz, nx]. These arrays include PML padding,
        but do not include the additional FD-stencil padding (`ext_cells`).
    """

    ddx = torch.zeros((nz, nx), device=device, dtype=dtype)
    ddz = torch.zeros((nz, nx), device=device, dtype=dtype)

    i = torch.arange(nz, device=device).view(nz, 1).expand(nz, nx)
    j = torch.arange(nx, device=device).view(1, nx).expand(nz, nx)

    coef_x = -math.log(R) * 3.0 * Vmax / (2.0 * (dx * n_pml) ** 2)
    coef_z = -math.log(R) * 3.0 * Vmax / (2.0 * (dz * n_pml) ** 2)

    # top-left
    mask = (i >= 0) & (i < n_pml) & (j >= 0) & (j < n_pml)
    x = n_pml - j
    z = n_pml - i
    ddx = torch.where(mask, coef_x * x ** 2, ddx)
    ddz = torch.where(mask, coef_z * z ** 2, ddz)

    # top-right
    mask = (i >= 0) & (i < n_pml) & (j > nx - n_pml) & (j < nx)
    x = j - (nx - n_pml)
    z = n_pml - i
    ddx = torch.where(mask, coef_x * x ** 2, ddx)
    ddz = torch.where(mask, coef_z * z ** 2, ddz)

    # bottom-left
    mask = (i >= nz - n_pml) & (i < nz) & (j >= 0) & (j < n_pml)
    x = n_pml - j
    z = i - (nz - n_pml)
    ddx = torch.where(mask, coef_x * x ** 2, ddx)
    ddz = torch.where(mask, coef_z * z ** 2, ddz)

    # bottom-right
    mask = (i >= nz - n_pml) & (i < nz) & (j >= nx - n_pml) & (j < nx)
    x = j - (nx - n_pml)
    z = i - (nz - n_pml)
    ddx = torch.where(mask, coef_x * x ** 2, ddx)
    ddz = torch.where(mask, coef_z * z ** 2, ddz)

    # top-middle
    mask = (i < n_pml) & (j >= n_pml) & (j < nx - n_pml + 1)
    z = n_pml - i
    ddx = torch.where(mask, torch.zeros_like(ddx), ddx)
    ddz = torch.where(mask, coef_z * z ** 2, ddz)

    # bottom-middle
    mask = (i >= nz - n_pml) & (i < nz) & (j >= n_pml) & (j < nx - n_pml)
    z = i - (nz - n_pml)
    ddx = torch.where(mask, torch.zeros_like(ddx), ddx)
    ddz = torch.where(mask, coef_z * z ** 2, ddz)

    # left-middle
    mask = (i >= n_pml) & (i < nz - n_pml) & (j < n_pml)
    x = n_pml - j
    ddx = torch.where(mask, coef_x * x ** 2, ddx)
    ddz = torch.where(mask, torch.zeros_like(ddz), ddz)

    # right-middle
    mask = (i >= n_pml) & (i < nz - n_pml) & (j >= nx - n_pml) & (j < nx)
    x = j - (nx - n_pml)
    ddx = torch.where(mask, coef_x * x ** 2, ddx)
    ddz = torch.where(mask, torch.zeros_like(ddz), ddz)

    return ddx, ddz


# ============================================================
# 3. Ricker wavelet
# ============================================================

def set_wavelet_torch(freq, dt, nt, device="cuda", dtype=torch.float32):
    """
    Generate a Ricker wavelet source time function.

    This wavelet is injected into the pressure-stress update at each time step.
    The implementation follows the original code:

        src[i] = (1 - 2 * (pi * f * i * dt)^2)
                 * exp(-(pi * f * i * dt)^2)

    Parameters
    ----------
    freq : float
        Dominant frequency of the Ricker wavelet.
    dt : float
        Temporal sampling interval of the FD solver.
    nt : int
        Number of FD time steps.
    """
    t = torch.arange(nt, device=device, dtype=dtype) * dt
    x = torch.pi * freq * t
    src = (1.0 - 2.0 * x ** 2) * torch.exp(-x ** 2)
    return src


# ============================================================
# 4. Main solver
# ============================================================

class acoustic_iso_torch:
    """
    2-D acoustic finite-difference solver implemented in PyTorch.

    This class generates pressure wavefield snapshots for a given velocity
    model. The solver uses a first-order velocity-stress acoustic formulation:
    particle velocities are updated from pressure gradients, while stress
    components are updated from velocity divergence and source injection.

    In the dataset-generation workflow, each call to `run_fd` simulates several
    source positions for one velocity model and saves the corresponding pressure
    snapshots, velocity model, and source location to `.mat` files.
    """

    def __init__(
        self,
        dx,
        dz,
        dt,
        nt,
        loc_x_list,
        freq,
        n_pml,
        vel,
        rho,
        device="cuda",
        dtype=torch.float32,
    ):
        self.list_of_methods = {
            "FD4": 2,
            "FD6": 3,
            "FD8": 4,
            "FD10": 5,
            "FD12": 6,
            "FD14": 7,
            "FD16": 8,
        }

        self.dx = dx
        self.dz = dz
        self.dt = dt
        self.nt = nt
        self.loc_x_list = loc_x_list
        self.loc_z = 1
        self.ns = len(loc_x_list)
        self.freq = freq
        self.n_pml = n_pml

        self.vel = np.asarray(vel, dtype=np.float32)
        self.rho_np = np.asarray(rho, dtype=np.float32)
        self.K_np = self.rho_np * self.vel * self.vel

        self.R = 1.0e-6
        self.Vmax = float(np.max(self.vel))

        self.nz, self.nx = self.vel.shape

        self.device = device
        self.dtype = dtype

        self.forwrd = 0
        self.bckwrd = 1

    def set_method(self, method):
        """
        Select the finite-difference stencil order.

        At present, only FD10 is implemented in the PyTorch-convolution version.
        The value `ext_cells` indicates how many extra cells are required by the
        stencil on each side of the computational domain.
        """
        self.method = method
        self.ext_cells = self.list_of_methods.get(method)

        if self.ext_cells is None:
            raise ValueError(f"Unknown finite-difference method: {method}")

        if method != "FD10":
            raise NotImplementedError(
                "This PyTorch-conv version currently implements FD10 only."
            )

    def extend_velocity(self):
        """
        Same as original:
            self.rho = np.pad(self.rho, pad_width=self.n_pml, mode='edge')
            self.K   = np.pad(self.K,   pad_width=self.n_pml, mode='edge')
        """
        rho = np.pad(self.rho_np, pad_width=self.n_pml, mode="edge")
        K = np.pad(self.K_np, pad_width=self.n_pml, mode="edge")

        return rho, K

    def extend_wave_array(self):
        """
        Allocate all tensors.

        Important:
        - rho, K, source, ddx, ddz include n_pml padding only.
        - pressure, Vx, Vz, sigmaxx, sigmazz further include ext_cells padding.
        """

        rho, K = self.extend_velocity()

        self.rho = torch.as_tensor(rho, device=self.device, dtype=self.dtype)
        self.K = torch.as_tensor(K, device=self.device, dtype=self.dtype)

        # source/ddx/ddz shape: [nz + 2*n_pml, nx + 2*n_pml]
        self.source = torch.zeros_like(self.K)
        self.ddx = torch.zeros_like(self.K)
        self.ddz = torch.zeros_like(self.K)

        pressure_np = np.zeros_like(K, dtype=np.float32)
        pressure_np = np.pad(
            pressure_np,
            pad_width=self.ext_cells,
            mode="edge",
        )

        shape_ext = pressure_np.shape

        # wavefield shape: [1, 1, nz_ext, nx_ext]
        self.pressure = torch.zeros(shape_ext, device=self.device, dtype=self.dtype)[None, None]

        self.sigmaxx = torch.zeros_like(self.pressure)
        self.sigmazz = torch.zeros_like(self.pressure)
        self.sigmaxx0 = torch.zeros_like(self.pressure)
        self.sigmazz0 = torch.zeros_like(self.pressure)

        self.Vx = torch.zeros_like(self.pressure)
        self.Vz = torch.zeros_like(self.pressure)
        self.Vx0 = torch.zeros_like(self.pressure)
        self.Vz0 = torch.zeros_like(self.pressure)

        self.dpressure_Vx = torch.zeros_like(self.pressure)
        self.dpressure_Vz = torch.zeros_like(self.pressure)
        self.dVx = torch.zeros_like(self.pressure)
        self.dVz = torch.zeros_like(self.pressure)

        nz_ext0, nx_ext0 = self.K.shape

        self.grid_i = torch.arange(
            nz_ext0,
            device=self.device,
            dtype=self.dtype,
        ).view(nz_ext0, 1)

        self.grid_j = torch.arange(
            nx_ext0,
            device=self.device,
            dtype=self.dtype,
        ).view(1, nx_ext0)

    def build_boundary(self):
        """Construct the SPML damping arrays after velocity/PML extension."""
        nz_ext0, nx_ext0 = self.K.shape

        self.ddx, self.ddz = build_spml_torch_exact(
            dx=self.dx,
            dz=self.dz,
            n_pml=self.n_pml,
            R=self.R,
            Vmax=self.Vmax,
            nz=nz_ext0,
            nx=nx_ext0,
            device=self.device,
            dtype=self.dtype,
        )

    def set_source_loc_torch(self, loc_x):
        """
        PyTorch version of source_loc.py/set_source_loc.

        Original:
            source[i,j] = exp(-0.2*((n_pml + loc_z_ini - i)^2
                                  +(n_pml + loc_x_ini - j)^2))
        """

        src_z = self.n_pml + self.loc_z
        src_x = self.n_pml + loc_x

        self.source.copy_(
            torch.exp(
                -0.2 * (
                    (src_z - self.grid_i) ** 2
                    + (src_x - self.grid_j) ** 2
                )
            )
        )

    def reset_wavefields(self):
        """
        PyTorch replacement of repeated zero_matrix_cuda calls.
        """

        self.pressure.zero_()

        self.sigmaxx.zero_()
        self.sigmazz.zero_()
        self.sigmaxx0.zero_()
        self.sigmazz0.zero_()

        self.Vx.zero_()
        self.Vz.zero_()
        self.Vx0.zero_()
        self.Vz0.zero_()

        self.dpressure_Vx.zero_()
        self.dpressure_Vz.zero_()
        self.dVx.zero_()
        self.dVz.zero_()

    def acoustic_iso_stress_torch(self, it, src):
        """
        Update the acoustic stress components at one FD time step.

        The update uses the velocity derivatives `dVx` and `dVz`, the bulk
        modulus `K = rho * vp^2`, the SPML damping arrays, and the source
        wavelet value at time index `it`. The source is added to both stress
        components so that their sum forms the pressure field.
        """

        e = self.ext_cells
        slc = (slice(None), slice(None), slice(e, -e), slice(e, -e))

        ddx = self.ddx[None, None, :, :]
        ddz = self.ddz[None, None, :, :]
        K = self.K[None, None, :, :]
        source = self.source[None, None, :, :]

        sigmaxx0_inner = self.sigmaxx0[slc]
        sigmazz0_inner = self.sigmazz0[slc]

        dVx_inner = self.dVx[slc]
        dVz_inner = self.dVz[slc]

        self.sigmaxx[slc].copy_(
            (
                (1.0 - 0.5 * self.dt * ddx) * sigmaxx0_inner
                - K * self.dt * dVx_inner
            )
            / (1.0 + 0.5 * self.dt * ddx)
            + source * src[it]
        )

        self.sigmazz[slc].copy_(
            (
                (1.0 - 0.5 * self.dt * ddz) * sigmazz0_inner
                - K * self.dt * dVz_inner
            )
            / (1.0 + 0.5 * self.dt * ddz)
            + source * src[it]
        )

    def acoustic_iso_stress_plus_torch(self):
        """
        Recover pressure from the two stress components.

        For the acoustic formulation used here, the scalar pressure field is
        represented as the sum of the normal stress components.
        """

        self.pressure.copy_(self.sigmaxx + self.sigmazz)

    def acoustic_velocity_torch(self):
        """
        Update particle velocities from pressure gradients.

        The x- and z-components of particle velocity are damped by the SPML
        profiles and driven by the corresponding pressure derivatives.
        """

        e = self.ext_cells
        slc = (slice(None), slice(None), slice(e, -e), slice(e, -e))

        ddx = self.ddx[None, None, :, :]
        ddz = self.ddz[None, None, :, :]
        rho = self.rho[None, None, :, :]

        Vx0_inner = self.Vx0[slc]
        Vz0_inner = self.Vz0[slc]

        dpressure_Vx_inner = self.dpressure_Vx[slc]
        dpressure_Vz_inner = self.dpressure_Vz[slc]

        self.Vx[slc].copy_(
            (
                (1.0 - 0.5 * self.dt * ddx) * Vx0_inner
                - self.dt / rho * dpressure_Vx_inner
            )
            / (1.0 + 0.5 * self.dt * ddx)
        )

        self.Vz[slc].copy_(
            (
                (1.0 - 0.5 * self.dt * ddz) * Vz0_inner
                - self.dt / rho * dpressure_Vz_inner
            )
            / (1.0 + 0.5 * self.dt * ddz)
        )

    def extract_matrix_torch(self):
        """
        PyTorch version of output.py/extract_matrix.

        Original:
            matrix[i,j] = matrix1[i + pad_width, j + pad_width]

        In run_fd:
            pad_width = self.ext_cells + self.n_pml
        """

        pad_width = self.ext_cells + self.n_pml

        return self.pressure[
            0,
            0,
            pad_width:pad_width + self.nz,
            pad_width:pad_width + self.nx,
        ]

    @torch.no_grad()
    def run_fd(self, data_id, data_folder, md=None, time_downsample=2):
        """
        Run finite-difference modeling and save wavefield snapshots.

        Parameters
        ----------
        data_id : int
            Index of the current velocity model. It is used only for naming the
            output file.
        data_folder : str
            Directory where the generated `.mat` files will be saved.
        md : str or None
            Model/data prefix used in the output filename.
        time_downsample : int
            Snapshot saving interval. The solver computes all `nt` FD time
            steps, but only every `time_downsample`-th snapshot is saved.
            For example, with dt = 0.001 s and time_downsample = 10, the saved
            snapshots have a physical interval of 0.01 s, matching the training
            interval used by the generative wave propagator.
        """

        self.extend_wave_array()
        self.build_boundary()

        src = set_wavelet_torch(
            freq=self.freq,
            dt=self.dt,
            nt=self.nt,
            device=self.device,
            dtype=self.dtype,
        )

        fd = FD10StaggeredConv2d(
            dx=self.dx,
            dz=self.dz,
            ext_cells=self.ext_cells,
            device=self.device,
            dtype=self.dtype,
        ).to(self.device)

        for ishot, loc_x in enumerate(self.loc_x_list):
            print(
                f"+++++++++++++ This is: {ishot + 1} shot "
                f"loc {loc_x} seismic modeling +++++++++++++"
            )

            self.set_source_loc_torch(loc_x)
            self.reset_wavefields()

            # Store the pressure snapshot at every FD time step before temporal
            # downsampling. Shape: [nt, nz, nx] in the physical model region.
            allsnap = torch.zeros(
                self.nt,
                self.nz,
                self.nx,
                device=self.device,
                dtype=self.dtype,
            )

            for it in range(self.nt):
                # ----------------------------------------------------
                # 1. dVx/dx and dVz/dz, backward staggered derivative
                # ----------------------------------------------------
                fd.derx_backward_(self.Vx, self.dVx)
                fd.derz_backward_(self.Vz, self.dVz)

                # ----------------------------------------------------
                # 2. stress update
                # ----------------------------------------------------
                self.acoustic_iso_stress_torch(it, src)

                # ----------------------------------------------------
                # 3. pressure = sigmaxx + sigmazz
                # ----------------------------------------------------
                self.acoustic_iso_stress_plus_torch()

                # ----------------------------------------------------
                # 4. pressure derivative, forward staggered derivative
                # ----------------------------------------------------
                fd.derx_forward_(self.pressure, self.dpressure_Vx)
                fd.derz_forward_(self.pressure, self.dpressure_Vz)

                # ----------------------------------------------------
                # 5. velocity update
                # ----------------------------------------------------
                self.acoustic_velocity_torch()

                # ----------------------------------------------------
                # 6. copy current fields to previous-time fields
                # ----------------------------------------------------
                self.sigmaxx0.copy_(self.sigmaxx)
                self.sigmazz0.copy_(self.sigmazz)

                self.Vx0.copy_(self.Vx)
                self.Vz0.copy_(self.Vz)

                # Remove PML and FD-stencil padding, then save the physical
                # pressure wavefield for the current time step.
                allsnap[it].copy_(
                    self.extract_matrix_torch()
                )

            # Move snapshots back to CPU and keep only the selected time steps.
            # The resulting array is the reference wavefield sequence used by
            # the diffusion-based wave propagator.
            allsnap_np = allsnap.detach().cpu().numpy()
            allsnap_np = allsnap_np[::time_downsample]

            # Save the generated sample. Each file contains:
            #   snaps : pressure snapshots after temporal downsampling
            #   vp    : velocity model used for simulation
            #   loc_x : source x-location
            np.savez_compressed(
                f"{data_folder}/{md}_vel{data_id + 1}_shot{ishot + 1}.npz",
                snaps=allsnap_np,
                vp=self.vel,
                loc_x=loc_x,
            )

            # sio.savemat(
                # f"{data_folder}/{md}_vel{data_id + 1}_shot{ishot + 1}.mat",
                # {
                    # "snaps": allsnap_np,
                    # "vp": self.vel,
                    # "loc_x": loc_x,
                # },
            # )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()