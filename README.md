# pyBKTR
This project is a python implementation of the BKTR algorithm (by Mengying Lei, Aurelie Labbe, Lijun Sun).

BKTR stands for Scalable Spatiotemporally Varying Coefficient Modelling with **Bayesian Kernelized Tensor Regression**.

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
