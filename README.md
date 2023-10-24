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

## Simple Example
To verify that everything is running smooth you can try to run a BKTR regression on the BIXI data presented in the package. (The data is already preloaded in the package in the `BixiData` class)

The following code will run a BKTR regression using sensible defaults on the BIXI data and print a summary of the results.
```python
from pyBKTR.bktr import BKTRRegressor
from pyBKTR.examples.bixi import BixiData

bixi_data = BixiData()
bktr_regressor = BKTRRegressor(
    data_df=bixi_data.data_df,
    spatial_positions_df=bixi_data.spatial_positions_df,
    temporal_positions_df=bixi_data.temporal_positions_df,
    burn_in_iter=5,
    sampling_iter=10
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
sphinx-apidoc -f -o . ../pyBKTR
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
