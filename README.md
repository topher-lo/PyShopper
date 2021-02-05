# PyShopper
PyShopper is a Python implementation of Shopper, a probablistic model of shopping baskets, from [the paper](https://arxiv.org/abs/1711.03560 "Arxiv paper"):
+ Francisco J. R. Ruiz, Susan Athey, David M. Blei. SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements. ArXiv 1711.03560. 2017.

[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![Generic badge](https://img.shields.io/badge/version-v0.01-4B8BBE.svg)]()
[![Open In nbviewer](https://warehouse-camo.ingress.cmh1.psfhosted.org/b76644f44625d8876b279659d108c1e5334fd8b3/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f76696577253230696e2d6e627669657765722d6f72616e6765)](https://nbviewer.jupyter.org/github/topher-lo/PyShopper/blob/main/example.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/topher-lo/PyShopper)


## Project Status
- This mini-project is under active development. 
- The code is currently NOT usable. I've specified the Shopper model in PyMC3 but it is not optimized.
- PyShopper code has only been run on a dataset with 100 observed trips (~300 observations). 
- Inference via NUTS MCMC sampling seems to converge in this very limited sample. 
- Sampling, however, is very slow and takes over 5 hours to complete on my T590 ThinkPad.

## Background
The goals of this mini-project were to:
- Push the boundaries of my understanding of PyMC3
- Replicate results from an economic paper with a Bayesian model
- Implement a Bayesian model that does not have any pre-existing solution in Python (as of January 2021)

## Install
PyShopper depends on the following packages:
- `pandas`
- `numpy`
- `pymc3 >= 3.10`
- `arviz`
- `scikit-learn`
- `theano`

## A quick example
```python
from pyshopper import shopper

# Load data
X_train = shopper.load_data('data/train.tsv',
                            'data/prices.tsv')

# Create Shopper instance
model = shopper.Shopper(X_train)

# Fit model
res = model.fit(draws=1000, random_seed=42)

# Return trace plot
res.trace_plot()
```

## To Do
- Implement baseline Shopper without seasonality effects on a limited dataset (Expected completion: 2/7/2021)
- Run baseline Shopper with MCMC sampling on simulated data
- Implement variational inference for Shopper using PyMC3's ADVI API
- Run baseline Shopper with ADVI on simulated data
- Optimise memory usage and speed

## Roadmap
- Implement baseline Shopper without seasonality effects (In Progress)
- Add random sampling of unordered baskets (TBD)
- Add seasonality effects (TBD)
- Add thinking ahead procedure (TBD)
