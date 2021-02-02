# PyShopper
PyShopper is a Python implementation of Shopper, a probablistic model of shopping baskets, from [the paper](https://arxiv.org/abs/1711.03560 "Arxiv paper"):
+ Francisco J. R. Ruiz, Susan Athey, David M. Blei. SHOPPER: A Probabilistic Model of Consumer Choice with Substitutes and Complements. ArXiv 1711.03560. 2017.

## Background
The goals of this mini-project were to:
- Push the boundaries of my understanding of PyMC3
- Replicate results from an economic paper with a Bayesian model
- Implement a Bayesian model that does not have any pre-existing solution in Python (as of January 2021)

## A quick example
```python
from pyshopper import shopper

# Load data
data = shopper.load_data('data/train.tsv',
                         'data/prices.tsv')

# Create Shopper instance
model = shopper.Shopper(data)

# Fit model
res = model.fit(draws=10000, random_seed=42)
```

## Roadmap
- Implement baseline Shopper without seasonality effects (TBD)
- Add random sampling of unordered baskets (TBD)
- Add seasonality effects (TBD)
- Add thinking ahead procedure (TBD)
