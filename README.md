Reproducible material for **DW0126: Generative wave propagator - Shijun Cheng and Tariq Alkhalifah**

[Click here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0126?csf=1&web=1&e=ghL7SP) to access the Project Report. Authentication to the _Restricted Area_ filespace is required.

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

The full training dataset is not included in this repository because of its large storage size. Instead, we provide scripts to reproduce the dataset.

The velocity models used to generate the wavefield dataset can be found in the `dataset.zip` file in the **DiffVMB2D** folder of the _Restricted Area_ filespace on [here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0100?csf=1&web=1&e=QZGuad). Please download and extract `dataset.zip` before running the wavefield-generation scripts.

To generate the training dataset, run:

```bash
cd dataset/data_gen
python main.py
```

Depending on the chosen output format in `fd_torch.py`, the generated data can be saved as `.npz` or `.mat` files.

If the generated data are saved in `.npz` format, they can be converted to `.h5` format for faster training-time loading by running:

```bash
python npz2h5.py
```

The HDF5 format is recommended for large-scale training because it supports compressed storage and efficient data access.

## Supplementary files
To ensure reproducibility, we provide the the data set for training and inference stages and our trainined diffusion model.

* **Dataset**:
Download the training and testing data set [here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0126?csf=1&web=1&e=ghL7SP). Then, use `unzip` to extract the contents.

* **Trained model**:
Download our trained diffusion model [here](https://kaust.sharepoint.com/:f:/r/sites/M365_Deepwave_Documents/Shared%20Documents/Restricted%20Area/REPORTS/DW0126?csf=1&web=1&e=ghL7SP). Then, use `unzip` to extract the contents.

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
For traning, you can directly run:
```
python train.py
```

When you test the performance of our trained diffusion model, you can use the test data we provide and directly run:
```
python sample.py
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce A100 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU. If your graphics card does not large batch size training, please reduce the configuration value of args (`batch_size`) in the `genwp/train.py` file.

## Acknowledgements
This implementation is motivated from the paper [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2102.09672) and the code adapted from their [repository](https://github.com/openai/improved-diffusion). We are grateful for their open source code.

## Cite us 
DW0126 - Cheng and Alkhalifah. (2026) Generative wave propagator.
