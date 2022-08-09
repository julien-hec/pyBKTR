# pyBKTR
This project is a python implementation of the BKTR algorithm (by Mengying Lei, Aurelie Labbe, Lijun Sun).

BKTR stands for Scalable Spatiotemporally Varying Coefficient Modelling with **Bayesian Kernelized Tensor Regression**.

### Documentation Generation
From the docs folder run the following line to regenerate the static doc
```bash
sphinx-apidoc -f -o . ../pyBKTR
```
then
```bash
make html
```
