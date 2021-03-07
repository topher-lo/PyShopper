# PyShopper
[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs)
[![Open In nbviewer](https://warehouse-camo.ingress.cmh1.psfhosted.org/b76644f44625d8876b279659d108c1e5334fd8b3/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f76696577253230696e2d6e627669657765722d6f72616e6765)](https://nbviewer.jupyter.org/github/topher-lo/PyShopper/blob/main/example.ipynb)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/topher-lo/PyShopper)
[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/)

`PyShopper` is a Python implementation of Shopper, a probablistic model of shopping baskets, from [the paper](https://arxiv.org/abs/1711.03560 "Arxiv paper"):
+ Francisco J. R. Ruiz, Susan Athey, David M. Blei. SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements. ArXiv 1711.03560. 2017.

## Project Status
- This mini-project is under active development
- Inference is NOT optimised
- Implementation is NOT tested
- An example is provided in the Jupyter notebook `example.ipynb`
- `PyShopper` code has been tested on a very limited dataset (2 items, 10 sessions, ~2300 observations)

## Background
The goals of this mini-project were to:
- Push the boundaries of my understanding of PyMC3
- Replicate results from an economic paper with a Bayesian model
- Implement a Bayesian model that does not have any pre-existing solution in Python (as of January 2021)

## Install
`PyShopper` has been tested with Python 3.8 and depends on the following packages:
- `arviz`
- `numpy`
- `pandas`
- `pymc3 >= 3.11`
- `seaborn`
- `scikit-learn`
- `theano`

To use `PyShopper`, you must first clone this repo:
```
git clone git@github.com:topher-lo/PyShopper.git
cd [..path/to/repo]
```
Then install its dependencies using either `pip`:
```
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
```
Or `conda`:
```
conda env create -f environment.yml
```

## :rocket: A quick example
```python
from pyshopper import shopper

# Load data
# Note: this dataset is unlikely to large to fit in memory
# Consider limiting the number of items and users (see example.ipynb)
X_train = shopper.load_data('data/train.tsv',
                            'data/prices.tsv')

# Create Shopper instance
model = shopper.Shopper(X_train)

# Fit model using variational inference
res = model.fit(N=10000, method='ADVI', random_seed=42)

# Return ELBO trace plot
res.elbow_plot()

# Return summary of common posterior statistics
# Note: we must draw a sample from the 
# approximated posterior distribution
res.summary(draws=1000)
```

## Roadmap
- Implement baseline Shopper without seasonality effects on a limited dataset (Complete)
- Implement basket sub-sampling and one-vs-each approximation to softmax (TBD)
- Add seasonality effects (TBD)
- Add thinking ahead procedure (TBD)
