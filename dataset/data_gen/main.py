# main.py
"""
Main script for generating finite-difference wavefield datasets.

This script is used to generate the training wavefield snapshots for the
generative wave propagator. For each velocity model, several source positions
are randomly selected along the surface. A PyTorch-based 10th-order staggered-grid
finite-difference acoustic solver is then used to simulate wave propagation.

The generated data are saved as .mat files, where each file contains:
    - snaps: wavefield snapshots after temporal downsampling
    - vp:    the velocity model used for simulation
    - loc_x: source location along the horizontal direction

The generated wavefield snapshots are later used to train the conditional
diffusion-based wave propagator, which learns to recursively predict the next
wavefield snapshot from previous snapshots and the velocity model.
"""

import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io as sio
from scipy.ndimage import zoom

from fd_torch import acoustic_iso_torch


def generate_random_integers(nx, count, min_difference=10):
    """
    Generate random source locations along the horizontal direction.

    Parameters
    ----------
    nx : int
        Maximum horizontal grid index.
    count : int
        Number of random source locations to generate.
    min_difference : int, optional
        Minimum grid-point distance between any two source locations.
        This avoids placing two sources too close to each other.

    Returns
    -------
    numbers : list of int
        Randomly generated source locations.

    Notes
    -----
    The source positions are randomly distributed along the surface, which is
    consistent with the data-generation setting described in the manuscript.
    """

    numbers = []

    # Keep sampling until the required number of source locations is obtained.
    while len(numbers) < count:
        num = random.randint(0, nx)

        # Accept the new source only if it is sufficiently far from all
        # previously selected source locations.
        if all(abs(num - existing) >= min_difference for existing in numbers):
            numbers.append(num)

    return numbers


def main():
    """
    Generate wavefield snapshots for multiple benchmark velocity-model datasets.

    The workflow is:

    1. Define finite-difference modeling parameters.
    2. Loop over different velocity-model families.
    3. Load each velocity model and resize it to 128 x 128.
    4. Randomly select several source locations.
    5. Run the PyTorch-based FD10 acoustic solver.
    6. Save the simulated wavefield snapshots as .mat files.

    The generated dataset is not stored directly in the GitHub repository due
    to its large size. Instead, this script allows users to reproduce the
    training data.
    """

    # ============================================================
    # 1. Basic finite-difference and data-generation parameters
    # ============================================================

    # Use GPU if available; otherwise fall back to CPU.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Spatial sampling intervals in the horizontal and vertical directions.
    # In the manuscript, both are set to 10 m.
    dx = 10.0
    dz = 10.0

    # Temporal sampling interval of the underlying FD solver.
    # The FD solver uses dt = 0.001 s, while the saved snapshots are further
    # downsampled in time.
    dt = 0.001

    # Number of shots simulated for each velocity model.
    nshot = 5

    # Target model size used for wavefield simulation.
    # The original velocity patches are resized to 128 x 128.
    nx = 128
    nz = 128

    # Constant-density acoustic assumption.
    # rho is set to 1 everywhere, so the bulk modulus is K = rho * vp^2.
    rho = np.ones((nz, nx), dtype=np.float32)

    # Number of FD time steps.
    # With dt = 0.001 s and nt = 1001, the total recording time is 1.0 s.
    nt = 1001

    # Dominant frequency of the Ricker source wavelet.
    freq = 15.0

    # Number of PML cells used to absorb outgoing waves at the boundaries.
    n_pml = 50

    # Save one snapshot every `time_downsample` FD time steps.
    # With dt = 0.001 s and time_downsample = 10, the saved snapshot interval
    # is 0.01 s, giving 101 snapshots from 1001 FD time samples.
    time_downsample = 10

    # Folder used to save generated training data.
    data_folder = "../train/"
    os.makedirs(data_folder, exist_ok=True)

    # Number of velocity models used from each benchmark dataset.
    # These datasets jointly provide diverse geological structures for training.
    md_dict = {
        "sigsbee": 83,
        "seam2d": 83,
        "otway": 83,
        "hess": 83,
        "bp1994": 83,
        "bp2004": 83,
        "bp2007": 83,
        "SEGEAGE": 2430,
        "SEAMArid": 2430,
        "Overthrust": 2430,
    }

    # ============================================================
    # 2. Loop over all velocity-model families and model samples
    # ============================================================

    for md, ndata in md_dict.items():
        for data_id in range(ndata):
            print(
                f"--------------- Simulating for {md} "
                f"data {data_id + 1} ---------------"
            )

            # ============================================================
            # 3. Load and resize the velocity model
            # ============================================================

            # Load the velocity model from the preprocessed velocity-model folder.
            # The expected file name is, for example:
            #     ../velocity_model/Overthrust_1.npz
            data = np.load(f"../velocity_model/{md}_{data_id + 1}.npz")
            vp = data["vp"]

            # Resize the velocity model to 128 x 128.
            # Here zoom=(0.5, 0.5) assumes that the original patch size is
            # 256 x 256. Bilinear interpolation is used with order=1.
            resized_vp = zoom(vp, zoom=(0.5, 0.5), order=1)

            # ============================================================
            # 4. Random source locations
            # ============================================================

            # Randomly generate source locations along the surface.
            # The locations are sorted only for convenient data organization.
            loc_x_list = sorted(
                generate_random_integers(
                    nx=nx - 1,
                    count=nshot,
                    min_difference=10,
                )
            )

            # ============================================================
            # 5. Run PyTorch-conv acoustic finite-difference solver
            # ============================================================

            # Initialize the acoustic finite-difference solver.
            # The solver internally pads the velocity model using PML cells
            # and simulates pressure wavefield propagation.
            solver = acoustic_iso_torch(
                dx=dx,
                dz=dz,
                dt=dt,
                nt=nt,
                loc_x_list=loc_x_list,
                freq=freq,
                n_pml=n_pml,
                vel=resized_vp,
                rho=rho,
                device=device,
            )

            # Use the 10th-order staggered-grid finite-difference method.
            solver.set_method("FD10")

            # Run finite-difference modeling for all source locations.
            # For each shot, this function saves one .mat file containing:
            #     snaps: [nt_downsampled, nz, nx]
            #     vp:    [nz, nx]
            #     loc_x: source location
            solver.run_fd(
                data_id=data_id,
                data_folder=data_folder,
                md=md,
                time_downsample=time_downsample,
            )

    print("Simulation finished.")
    print(f"Results saved in: {data_folder}")


if __name__ == "__main__":
    main()