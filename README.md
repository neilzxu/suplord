# Code for [Dynamic Algorithms for Online Multiple Testing](https://arxiv.org/abs/2010.13953)
## Setup

Setup tried with [conda 4.8.3](https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html) and [python 3.8.3](https://www.python.org/downloads/).
```
conda env create -f environment.yml --name <env name>`
conda activate <env name>
pip install -r requirements.txt
```

## Producing figures in the paper

This section describes how to run the simulations/experiments and generate plots for the relevant figures in the paper.

### Numerical simulations

To run the numerical simulations in the paper and create the associated plots, run:
`python src/main.py --processes <# of processes allowed for process pool>`
As a result, `figures/` will contain all experiment plots (and some extra ones).

`fetch_figures.sh` denotes the relations between the figures in the paper and the images produced. Run `fetch_figures.sh` to copy only the plots that appear in the paper to `final_figures/`.

### Experiments on IMPC dataset

Tables 2 and 3 in the paper contain the results of an experiment on the IMPC dataset. This experiment can be reproduced as follows:

1. Download the IMPC data by running `./get_real_data.sh`. The data will be in `real_data/`.
2. Run `python src/impc.py`.

The results are output to console.

### Plot for tradeoffs in the choice of *a*

`python src/w_b_tradeoff.py` produces the plot in Figure 6 in the location `final_figures/w0_b0.png`.


## Code structure

We will briefly describe how the code in the repo is structured. In addition to this overview, some modules also has individual documentation.

- `alg/`: Contains code for all the online multiple testing algorithms, and related functions. Look here for implementations of each algorithm.
- `analysis/`: Code describing what types of plots are generated from list of results of experiments. The functions in here are called on the results of each experiment to create the figures for the paper.
- `exp/`: Primary module for creating and running experiments. Contains high level functions that can take dictionary specifications and run an experiment (create the appropriate datasets, choose algorithms for comparison, etc.) based on those specifications. Also contains code for specifying styles of plots created for each experiment.
- `data.py`: Specifies how artificial data for simulations is parameterized.
- `impc.py`: Script for running IMPC experiment.
- `main.py`: Script for running most experiments.
- `metrics.py`: Functions specifying how to compute different metrics of interest for trial results.
- `plot.py`: Functions for plotting data.
- `result.py`: Specification of how to format results of experiments i.e. the `Result` class.
- `utils.py`: Utility functions.

## Citation
```
@inproceedings{xu_dynamic_2021,
  title = {Dynamic {{Algorithms}} for {{Online Multiple Testing}}},
  author = {Xu, Ziyu and Ramdas, Aaditya},
  year = {2021},
  booktitle = {Mathematical and Scientific Machine Learning}
}
```

