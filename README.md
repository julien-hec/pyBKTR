# pyBKTR

## Intro
This project is a *Python* implementation of the BKTR algorithm presented by Mengying Lei, Aurélie Labbe & Lijun Sun (2023).
The article presenting the algorithm can be found [here](https://arxiv.org/abs/2109.00046).

BKTR stands for Scalable Spatiotemporally Varying Coefficient Modelling with Bayesian Kernelized Tensor Regression.
It allows to model spatiotemporally varying coefficients using a Bayesian framework.
We implemented the algorithm and more in a *Python* package that uses [PyTorch](https://pytorch.org/) as a tensor operation backend.

For information, an alternative *R* implementation of the algorithm can be found [here](https://github.com/julien-hec/BKTR). The *Python* implementation is synchronized with this repository and development is done in parallel. The synchronization of features will be done at a subrevision level (x.y.0).

An article presenting the *R* package in details is currently in preparation and should be available soon. The documentation for the *R* package is a work in progress and will be made available in the coming weeks.

## Installation

### Install from PyPI
The package is available on PyPI and can be installed using pip:
```bash
pip install pyBKTR
```

### Install from source
To install the package from source and use the latest release, you can clone the repository and install it using pip and the repository url:
```bash
pip install git+https://github.com/julien-hec/pyBKTR.git
```

### Notes
If you obtain an error message when installing the package, it may be due to the installation of the `torch` package.
The `torch` package is a dependency of the `pyBKTR` package and there is a good chance that the error comes from the installation of `torch`.
Because of its ability to perform tensor operations on the GPU, it can sometimes be more complicated to install than other R packages.
Some of the most common problems are related to the CUDA version, the GPU driver version, that can be incompatible with a specific version of `torch`.
We provide some guidance for the installation of `torch` below.

#### Installing torch alone
A simple way to see if `pyBKTR` installation problems come from the `torch` installation is to try to install torch alone first:
```bash
pip install torch
```

#### More information on torch installation
If you obtain an error message, we encourage you to refer to the [pytorch installation documentation](https://pytorch.org/get-started/locally/).
This documentation provides a lot of information on how to install `torch` on different platforms and with different configurations.
It allows for example to install `torch` with or without GPU support if you have GPU drivers issues.


#### Getting started with Colab
If you want to get started quickly with `pyBKTR` on Google Colab, you can use the following examples
- BKTR on Light BIXI Data with CPU ([Colab](https://colab.research.google.com/drive/13er8x9GfD4IsERb1WbbkTsk3UEpOEqdK?usp=sharing) & [GitHub](https://github.com/julien-hec/bktr-examples/blob/main/BKTR-installations/Python_BKTR_CPU.ipynb))
- BKTR on Light BIXI Data with GPU ([Colab](https://colab.research.google.com/drive/1eXlGv1gATiJbhvwsa_ro2h5o1aD38sxa?usp=sharing) & [GitHub](https://github.com/julien-hec/bktr-examples/blob/main/BKTR-installations/Python_BKTR_GPU.ipynb))


## Simple Example
To verify that everything is running smooth you can try to run a BKTR regression on the BIXI data presented in the package. (The data is already preloaded in the package in the `BixiData` class)

The following code will run a BKTR regression using sensible defaults on the BIXI data and print a summary of the results. To use a subset of the BIXI dataset as a simple example, we can also use the `is_light` argument during the `BixiData` initialization  to only load a subset of the data (25 stations and 50 days).

```python
from pyBKTR.bktr import BKTRRegressor
from pyBKTR.examples.bixi import BixiData

bixi_data = BixiData(is_light=True)
bktr_regressor = BKTRRegressor(
    data_df=bixi_data.data_df,
    spatial_positions_df=bixi_data.spatial_positions_df,
    temporal_positions_df=bixi_data.temporal_positions_df,
    burn_in_iter=200,
    sampling_iter=200
)
bktr_regressor.mcmc_sampling()
print(bktr_regressor.summary)
```

## Contributing
Contributions are welcome. Do not hesitate to open an issue or a pull request if you encounter any problem or have any suggestion.


## Dev Notes
### Dev environment setup
If you wish to contribute to this project, we strongly recommend you to use the precommit setup created in the project. To get started, simply follow these instructions.

First, install the project locally with the development resources. If you use zsh, you might need to put single quotes around the path `'.[dev]'`
```bash
pip install .[dev]
```

Then, install the git hook scripts.
```bash
pre-commit install
```

Finally, everything should work fine if when run the pre-commit hooks.
```bash
pre-commit run --all-files
```

### Documentation Generation
You should already have the dev environment setup

Pandoc needs to be installed on your local machine, follow the instructions in the following link
https://pandoc.org/installing.html


From the docs folder run the following line to regenerate the static doc
```bash
sphinx-apidoc -f -o . .
```
then
```bash
make html
```

### Publish to PyPI
First build the package locally
```bash
python3 -m pip install --upgrade build
python3 -m build
```

Then upload to PyPI
```bash
python3 -m pip install --upgrade twine
twine upload dist/*
```
Using the proper credentials, the package should be uploaded to PyPI and be available for download.
