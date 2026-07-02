<div align="center">

<h1><strong>Generative Wave Propagator</strong></h1>

<h4>Shijun Cheng and Tariq Alkhalifah</h3>

<h4><em>DeepWave Consortium, King Abdullah University of Science and Technology (KAUST)</em></h4>

<p><em>Corresponding author: Shijun Cheng (<a href="mailto:sjcheng.academic@gmail.com">sjcheng.academic@gmail.com</a>)</em></p>

</div>

## Overview

We introduce a conditional diffusion-based wavefield propagator that advances seismic wavefields recursively from one time step to the next. The model is conditioned on a short history of recent wavefield snapshots, the velocity model, and the wavefield time-step index, and is trained with a causal time-weighted loss that aligns the training trajectory with the physical direction of wave propagation to suppress recursive error accumulation. During inference, the strong physical conditioning allows each wavefield snapshot to be generated in a single forward pass, bypassing the costly iterative reverse sampling of standard diffusion models. The learned propagator operates at a physical time step ten times larger than the finite-difference stability limit, achieving an end-to-end speedup of 2.17× over a GPU-accelerated tenth-order staggered-grid FD solver under matched hardware conditions.

## Project structure

This repository is organized as follows:

```text
.
├── dataset/
│   ├── data_gen/
│   │   ├── fd_torch.py
│   │   ├── main.py
│   │   └── npz2h5.py
│   └── test/
│       ├── Marmousi/
│       │   ├── shot1.mat
│       │   └── shot2.mat
│       ├── Overthrust/
│       │   ├── shot1.mat
│       │   └── shot2.mat
│       └── SEGEAGE/
│           ├── shot1.mat
│           └── shot2.mat
│
├── genwp/
│   └── Python library containing the implementation of the generative
│       wave propagator, including model definitions, training, sampling,
│       and utility functions.
│
├── environment.yml
├── install_env.sh
├── pyproject.toml
├── LICENSE
└── README.md
```

### `genwp/`

The `genwp/` folder contains the core implementation of the proposed generative wave propagator, including the conditional diffusion model, training pipeline, sampling/inference code, and supporting utilities.

### `dataset/`

The `dataset/` folder contains data-generation scripts and small-scale test data.

- `dataset/data_gen/fd_torch.py`: PyTorch implementation of the 10th-order staggered-grid finite-difference acoustic solver used to generate wavefield snapshots.
- `dataset/data_gen/main.py`: main script for generating finite-difference training data from velocity models.
- `dataset/data_gen/npz2h5.py`: script for converting generated `.npz` files to `.h5` files for faster loading during training.
- `dataset/test/`: small-scale test data for the Marmousi, Overthrust, and SEG/EAGE models. Each model contains two source positions, stored as `shot1.mat` and `shot2.mat`.

## Dataset generation

The full training dataset is not included in this repository because of its large storage size. Instead, we provide scripts and resources to reproduce the dataset.

### Velocity models

The velocity models used to generate the wavefield training data are publicly available through our open-source Zenodo dataset:

> **Zenodo**: [https://doi.org/10.5281/zenodo.19790506](https://doi.org/10.5281/zenodo.19790506)

These velocity models can also be found in the supplementary files of our previous repository [DiffVMB-pub](https://github.com/DeepWave-KAUST/DiffVMB-pub). The dataset comprises 7871 velocity patches (256 × 256) extracted from a collection of industry benchmark models, including BP1994, BP2004, Hess, Otway, Sigsbee, SEG/EAGE, Overthrust, and SEAM Arid.

### Generating wavefield snapshots

After downloading the velocity models, generate the training wavefields by running:

```bash
cd dataset/data_gen
python main.py
```

Depending on the chosen output format in `fd_torch.py`, the generated data can be saved as `.npz` or `.mat` files.

If the generated data are saved in `.npz` format, they can be converted to `.h5` format for faster training-time loading by running:

```bash
python npz2h5.py
```

The HDF5 format is recommended for large-scale training because it supports compressed storage and efficient random-access data loading.

## Supplementary files

To ensure reproducibility, we provide our trained diffusion model.

* **Trained model**: Download our trained diffusion model from [Google Drive](https://drive.google.com/file/d/17hMBkiTQ4G8WpyeJvSjAtlAxqr0K73kU/view?usp=drive_link). After downloading, extract the contents and place the checkpoint file in the `./checkpoints/` directory.

## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. Activate the environment by typing:
```
conda activate genwp
```

After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

## Running code :page_facing_up:
When you have downloaded the supplementary files and have installed the environment, you can run the training and sampling code.
For training, you can directly run:
```
python train.py
```

When you test the performance of our trained diffusion model, you can use the test data we provide and directly run:
```
python sample.py
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GeForce A100 GPU. Different environment
configurations may be required for different combinations of workstation and GPU. If your graphics card does not support large batch size training, please reduce the configuration value of args (`batch_size`) in the `genwp/train.py` file.

## Acknowledgements
This implementation is motivated from the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open source code.

## Cite us
Cheng and Alkhalifah. (2026) Generative wave propagator.