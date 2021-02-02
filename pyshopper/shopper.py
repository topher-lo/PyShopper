"""This module contains the `Shopper` class that implements
Shopper, a probablistic model of shopping baskets, from the paper:

+ Francisco J. R. Ruiz, Susan Athey, David M. Blei. SHOPPER:
A Probabilistic Model of Consumer Choice with Substitutes and Complements.
ArXiv 1711.03560. 2017.
"""

import arviz as az
import logging
import numpy as np
import theano
import pandas as pd
import pymc3 as pm

import theano.tensor as tt

from sklearn import preprocessing


# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global variables
DATA_PATH = 'data/train.tsv'
PRICES_PATH = 'data/prices.tsv'


def load_data(data_path: str = DATA_PATH, prices_path: str = PRICES_PATH):
    """Extract, transforms, and loads data for Shopper.

    Args:
        data_path (str): 
            Shopping trips data.
            Path to .tsv file with four columns of data (without header)
            in the order of user_id, item_id, session_id, and quantity.
            Each row represents one trip.

        prices_path (str): 
            Prices data.
            Path to .tsv file with three columns of data (without header)
            in the order of item_id, session_id, and price.
            Each row represents the price per item per session.

    Returns:
        Joined Pandas DataFrame of shopping trips data and prices data.
    """
    trips = pd.read_csv(data_path,
                        header=None,
                        names=['user_id', 'item_id', 'session_id', 'quantity'],
                        sep='\t')
    prices = pd.read_csv(prices_path,
                         header=None,
                         names=['item_id', 'session_id', 'price'],
                         sep='\t')
    data = trips.join(prices, on=['item_id', 'session_id'])
    return data


class Shopper:
    """Shopper implementation.

    Let T = number of trips; U = number of users;
    C = number of items; and W = number of weeks.

    Note: model currently only supports ordered baskets.

    Attributes:
        data (Pandas DataFrame): 
            Observed trips data (number of trips by 4).
            DataFrame with columns: user_id, item_id, session_id, and price.

        model (PyMC3 Model): 
            Shopper model.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 K: int = 50,
                 price_dim: int = 10,
                 rho_var: float = 1,
                 alpha_var: float = 1,
                 lambda_var: float = 1,
                 theta_var: float = 1,
                 delta_var: float = 0.01,
                 mu_var: float = 0.01,
                 gamma_rate: float = 1000,
                 gamma_shape: float = 100,
                 beta_rate: float = 1000,
                 beta_shape: float = 100):
        """Intialises Shopper instance.

        Args:
            data (Pandas DataFrame): 
                Observed trips data (number of trips by 4).
                DataFrame with columns: user_id, item_id, session_id, and price.

            K (int): 
                Number of latent factors for alpha_c, rho_c, and theta_u;
                defaults to 50.

            price_dim (int): 
                Number of latent factors for price vectors gamma_u and beta_c;
                defaults to 10.

            rho_var (float): 
                Prior variance over rho_c; defaults to 1.

            alpha_var (float): 
                Prior variance over alpha_c; defaults to 1.

            theta_var (float): 
                Prior variance over theta_u; defaults to 1.

            lambda_var (float): 
                Prior variance over lambda_c; defaults to 1.

            delta_var (float): 
                Prior variance over delta_w; defaults to 0.01.

            mu_var (float): 
                Prior variance over mu_c; defaults to 0.01.

            gamma_rate (float): 
                Prior rate over gamma_u; defaults to 1000.

            gamma_shape (float): 
                Prior shape over gamma_u; defaults to 100.

            beta_rate (float): 
                Prior rate over beta_c; defaults to 1000.

            beta_shape (float): 
                Prior shape over beta_c; defaults to 100.
        """
        # Set data
        self.data = data
        # Number of observations
        N_obs = len(data)
        # Order
        order = data.groupby(['user_id', 'session_id'])['item_id']\
                 .cumcount()
        # Scaling factor
        data.loc[:, 'sf'] = order.apply(lambda x: 1 / x if x > 0 else 0)
        # Number of items
        C = data['item_id'].nunique()
        # Number of users
        U = data['user_id'].nunique()
        # Trips (user_id, session_id)
        trips = X.set_index(['user_id', 'session_id'])\
                 .index\
                 .to_flat_index()\
                 .to_frame()\
                 .astype(str)
        trips_idx = preprocessing.LabelEncoder()\
                                 .fit_transform(trips)
        # Items
        items = X['item_id']
        items_idx = preprocessing.LabelEncoder().fit_transform(items)
        # Users
        users = X['user_id']
        users_idx = preprocessing.LabelEncoder().fit_transform(users)

        logging.info('Building the Shopper model...')
        with pm.Model() as shopper:
            # Latent variables

            # per item interaction coefficients
            rho_c = pm.Normal('rho_c',
                              mu=0,
                              sigma=rho_var,
                              shape=(C, K))
            # per item attributes
            alpha_c = pm.Normal('alpha_c',
                                mu=0,
                                sigma=alpha_var,
                                shape=(C, K))
            # per user preferences
            theta_u = pm.Normal('theta_u',
                                mu=0,
                                sigma=theta_var,
                                shape=(U, K))
            # per item popularities
            lambda_c = pm.Normal('lambda_c',
                                 mu=0,
                                 sigma=lambda_var,
                                 shape=T)
            # per user price sensitivities
            gamma_u = pm.Gamma('gamma_u',
                               beta=gamma_rate,
                               alpha=gamma_shape,
                               shape=(U, price_dim))
            # per item price sensitivities
            beta_c = pm.Gamma('beta_c',
                              beta=beta_rate,
                              alpha=beta_shape,
                              shape=(C, price_dim))

            # Baseline utility per item per user:
            # Item popularity + Consumer Preferences - Price Effects
            # Note: variation comes from customer index and item prices
            psi_tc = lambda_c[items_idx] +\
                theta_u[users_idx]*alpha_c[items_idx] -\
                gamma_u[users_idx]*beta_c[items_idx]*np.log(data['price'])

            # sum^{i-1}_j [alpha_{y_tj}]
            def basket_items_attr(idx, alpha_c, order):
                # If first item in basket
                if order[idx] == 0:
                    # No price-attributes interaction effects
                    phi_ti = 0
                else:
                    phi_ti += alpha_c[idx - 1]
                return phi_ti

            phi_0 = theano.shared(0)  # phi_ti initial value
            phi_ti, updates = theano.scan(fn=basket_items_attr,
                                          sequences=[tt.arange(N_obs),
                                                     alpha_c,
                                                     order],
                                          outputs_info=phi_0,
                                          n_steps=N_obs)

            # Mean utility per basket per item
            Psi_tci = psi_tc + rho_c[items_idx]*phi_ti*data['sf']

            # Softmax likelihood p(y_ti = c | y_t0, y_t1, ..., y_ti-1)
            p = pm.Deterministic('p', tt.nnet.softmax(Psi_tci))
            labels = preprocessing.LabelBinarizer()\
                                  .fit_transform(data['item_id'])
            y = pm.Categorical('y', p=p, observed=np.bool8(labels))

        logging.info("Done building the Shopper model.")
        # Set shopper to model attribute
        self.model = shopper

    def fit(self,
            draws,
            step=None,
            random_seed=42,
            return_inferencedata=True):
        """Estimate parameters using Bayesian inference.
        Returns ShopperResults instance.

        Args:
            draws (int): 
                Number of draws.

            step (function or iterable of functions): 
                A step function or collection of functions;
                defaults None (which uses the NUTS step method).

            random_seed (int): 
                Random seed; defaults to 42.

            return_inferencedata (bool): 
                If True, returns arviz.InferenceData object.
                Otherwise, returns MultiTrace.InferenceData object.
                Defaults to True.

        Methods supported:

        - MCMC -- Monte Carlo Markov Chains

        """
        model = self.model
        with model:
            res = pm.sample(draws=draws,
                            step=step,
                            random_seed=random_seed,
                            return_inferencedata=return_inferencedata)
        return ShopperResults(res)


class ShopperResults:
    """Results class for a fitted Shopper model.

    Attributes:
        res (arviz.InferenceData or MultiTrace.InferenceData): 
            PyMC3 sampling results object.
    """
    def __init__(self, res):
        self.res = res

    def trace_plot(self):
        """Returns trace plots.

        Requires the Shopper model to be fitted with
        MCMC sampling.
        """
        return az.plot_trace(self.res)

    def rhat(self):
        """Returns the Gelman-Rubin statistic.

        Requires the Shopper model to be fitted with
        MCMC sampling.
        """
        return az.summary(self.res)

    def energy_plot(self):
        """Returns energy plot to check for convergence.
        Commonly used for high-dimensional models where it
        is too cumbersome to examine all parameter's traces.

        Requires the Shopper model to be fitted with
        MCMC sampling.
        """
        return az.plot_energy(self.res)

    def predict(self, X):
        """Returns predicted probabilities for
        samples in X.
        """
        pass

    def score(self, X, y):
        """Returns the mean accuracy on the given test data
        and labels.
        """
        pass


if __name__ == "__main__":
    pass
