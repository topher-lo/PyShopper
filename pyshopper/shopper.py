"""This module contains the `Shopper` class that implements
Shopper, a probablistic model of shopping baskets, from the paper:

+ Francisco J. R. Ruiz, Susan Athey, David M. Blei. SHOPPER:
A Probabilistic Model of Consumer Choice with Substitutes and Complements.
ArXiv 1711.03560. 2017.
"""

import arviz as az
import copy
import logging
import numpy as np
import theano
import pandas as pd
import pymc3 as pm
import seaborn as sns

import theano.tensor as tt

from sklearn import preprocessing
from matplotlib import pyplot as plt

from pymc3.variational.callbacks import CheckParametersConvergence


# Logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Set seaborn default theme
sns.set_theme()

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
    data = pd.merge(trips, prices, on=['item_id', 'session_id'])
    return data


def _prepare_data(data: pd.DataFrame):
    """Prepare data for used in Shopper. Returns preprocessed
    data variables.
    """
    # Prices
    prices = data['price']
    # Order
    order = data.groupby(['user_id', 'session_id'])['item_id']\
                .cumcount()
    # Scaling factor
    sf = (order.apply(lambda x: 1 / x if x > 0 else 0)
               .to_numpy(dtype='float32'))
    # Observations index
    obs_idx = (preprocessing.LabelEncoder()
                            .fit_transform(data.index)
                            .astype('int32'))
    # Items
    items_idx = (preprocessing.LabelEncoder()
                              .fit_transform(data['item_id'])
                              .astype('int32'))
    # Users
    users_idx = (preprocessing.LabelEncoder()
                              .fit_transform(data['user_id'])
                              .astype('int32'))
    # Labels
    labels = (preprocessing.LabelEncoder()
                           .fit_transform(data['item_id'])
                           .astype('int32'))
    return {'prices': prices,
            'order': order,
            'sf': sf,
            'obs_idx': obs_idx,
            'items_idx': items_idx,
            'users_idx': users_idx,
            'labels': labels}


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
                 price_dtype: str = 'float32',
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
        """Initializes Shopper instance.

        Args:
            data (Pandas DataFrame): 
                Observed trips data (number of trips by 4).
                DataFrame with columns: user_id, item_id, session_id, 
                and price.

            K (int): 
                Number of latent factors for alpha_c, rho_c, and theta_u;
                defaults to 50.

            price_dim (int): 
                Number of latent factors for price vectors gamma_u and beta_c;
                defaults to 10.

            price_dtype (str): 
                The datatype used for prices; defaults to float32.

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

        # Number of items
        C = data['item_id'].nunique()
        # Number of users
        U = data['user_id'].nunique()

        # Get preprocessed data variables
        data_vars = _prepare_data(data)

        logging.info('Building the Shopper model...')
        with pm.Model() as shopper:
            # Data
            prices = pm.Data('prices', data_vars['prices'])
            order = pm.Data('order', data_vars['order'])
            sf = pm.Data('sf', data_vars['sf'])
            obs_idx = pm.Data('obs_idx', data_vars['obs_idx'])
            items_idx = pm.Data('items_idx', data_vars['items_idx'])
            users_idx = pm.Data('users_idx', data_vars['users_idx'])
            labels = pm.Data('labels', data_vars['labels'])

            # Latent variables
            # per item interaction coefficients
            rho_c = pm.Normal('rho_c',
                              mu=0,
                              sigma=rho_var,
                              shape=(K, C),
                              dtype='float32')
            # per item attributes
            alpha_c = pm.Normal('alpha_c',
                                mu=0,
                                sigma=alpha_var,
                                shape=(K, C),
                                dtype='float32')
            # per user preferences
            theta_u = pm.Normal('theta_u',
                                mu=0,
                                sigma=theta_var,
                                shape=(K, U),
                                dtype='float32')
            # per item popularity
            lambda_c = pm.Normal('lambda_c',
                                 mu=0,
                                 sigma=lambda_var,
                                 shape=C,
                                 dtype='float32')
            # per user price sensitivities
            gamma_u = pm.Gamma('gamma_u',
                               beta=gamma_rate,
                               alpha=gamma_shape,
                               shape=(price_dim, U),
                               dtype='float32')
            # per item price sensitivities
            beta_c = pm.Gamma('beta_c',
                              beta=beta_rate,
                              alpha=beta_shape,
                              shape=(price_dim, C),
                              dtype='float32')

            # Baseline utility per basket per item
            # Item popularity + Consumer Preferences - Price Effects
            # Note: variation comes from customer index and item prices
            psi_tc = pm.Deterministic(
                'psi_tc',
                lambda_c +
                pm.math.dot(theta_u[:, users_idx].T, alpha_c) -
                pm.math.dot(np.log(prices).astype(price_dtype),
                            pm.math.dot(gamma_u[:, users_idx].T, beta_c))
            )
            logging.info('psi_tc shape: {}'.format(psi_tc.tag.test_value.shape))

            # sum^{i-1}_j [alpha_{y_tj}]
            def basket_items_attr(omega_prev, idx, alpha_c, order):
                # If first item in basket
                if tt.eq(order[idx], 0):
                    # No price-attributes interaction effects
                    omega_ti = tt.zeros(K)
                else:
                    omega_ti = omega_prev + alpha_c[:, idx-1]
                return omega_ti

            # omega_ti initial value
            omega_0 = tt.zeros(K)
            omega_ti = tt.vector('omega_ti')
            omega_ti, updates = theano.scan(fn=basket_items_attr,
                                            outputs_info=omega_0,
                                            non_sequences=[obs_idx,
                                                           alpha_c,
                                                           order],
                                            n_steps=obs_idx.shape[0])
            # Mean utility per basket per item
            Psi_tci = pm.Deterministic(
                'Psi_tci',
                psi_tc + pm.math.dot(
                    sf[obs_idx],
                    pm.math.dot(omega_ti[obs_idx-1, :], rho_c))
            )
            logging.info('Psi_tci shape: {}'.format(
                Psi_tci.tag.test_value.shape))
            # Softmax likelihood p(y_ti = c | y_t0, y_t1, ..., y_ti-1)
            p = pm.Deterministic(
                'p',
                tt.nnet.softmax(Psi_tci[items_idx])
            )
            logging.info('p shape: {}'.format(p.tag.test_value.shape))
            y = pm.Categorical('y', p=p, observed=labels)
            logging.info('y shape: {}'.format(y.tag.test_value.shape))

        logging.info("Done building the Shopper model.")
        # Set shopper to model attribute
        self.model = shopper

    def fit(self,
            N,
            method='ADVI',
            step=None,
            diff='relative',
            return_inferencedata=True,
            random_seed=42,
            **kwargs):
        """Estimate parameters using Bayesian inference.
        Returns ShopperResults instance.

        Args:
            N (int): 
                Number of draws (MCMC) or iterations (ADVI).

            method (str): 
              - MCMC -- Monte Carlo Markov Chains
              - ADVI -- Automatic Differentiation Variational Inference

            diff (str): 
                Requires method to be ADVI. The difference type used
                to check convergence in the mean of the ADVI approximation

            step (function or iterable of functions):
                Requires method to be MCMC.
                A step function or collection of functions;
                defaults None (which uses the NUTS step method).

            return_inferencedata (bool): 
                Requires method to be MCMC.
                If True, returns arviz.InferenceData object.
                Otherwise, returns MultiTrace.InferenceData object.
                Defaults to True.

            random_seed (int): 
                Random seed; defaults to 42.
        """
        model = self.model
        with model:
            if method == 'ADVI':
                callback = CheckParametersConvergence(diff=diff)
                res = pm.fit(n=N,
                             method='advi',
                             callbacks=[callback],
                             random_seed=random_seed,
                             **kwargs)
            else:
                res = pm.sample(draws=N,
                                step=step,
                                return_inferencedata=True,
                                random_seed=random_seed,
                                **kwargs)
        return ShopperResults(model, res)


class ShopperResults:
    """Results class for a fitted Shopper model.

    Attributes:
        model (PyMC3 Model): 
            Shopper model.

        res (PyMC3 results instance): 
            If MCMC, then requires arviz.InferenceData or
            MultiTrace.InferenceData. Else if ADVI, then
            requires pymc3.variational.opvi.Approximation.
    """
    def __init__(self, model, res):
        self.model = model
        self.res = res

    def summary(self, **kwargs):
        """Returns text-based output of common posterior statistics.

        Requires 'draws' (sample size to be drawn from posterior distribution)
        to be set in kwargs if model was fitted with ADVI.
        """
        res = self.res
        if 'variational' in str(type(res)):
            logging.info('Sampling from posterior distribution...')
            trace = res.sample(draws=kwargs['draws'])
            logging.info('Sampling complete.')
            logging.info('Computing posterior statistics...')
            summary = az.summary(trace, kind='stats')
        else:
            summary = az.summary(res)
        return summary

    def trace_plot(self, **kwargs):
        """Returns the trace plot.

        Requires 'draws' (sample size to be drawn from posterior distribution)
        to be set in kwargs if model was fitted with ADVI.
        """
        res = self.res
        if 'variational' in str(type(res)):
            logging.info('Sampling from posterior distribution...')
            trace = res.sample(draws=kwargs['draws'])
            logging.info('Sampling complete.')
            plot = az.plot_trace(trace)
        else:
            plot = az.plot_trace(res)
        return plot

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

    def elbo_plot(self):
        """Returns trace plot of ADVI's objective function (ELBO).

        Requires the Shopper model to be fitted with ADVI.
        """
        fig = plt.figure()
        plt.plot(self.res.hist)
        plt.ylabel('ELBO')
        plt.xlabel('n iterations')
        return fig

    def predict(self, data, random_seed=42, **kwargs):
        """Returns predicted probabilities of outcomes for samples in X.
        """
        model = copy.deepcopy(self.model)
        res = self.res
        data_vars = _prepare_data(data)
        data_vars.pop('labels')  # Remove labels
        with model:
            # Pass new values to model
            pm.set_data(data_vars)
            # Use the updated values and
            # predict outcomes and probabilities
            if 'variational' in str(type(res)):
                logging.info('Sampling from posterior distribution...')
                trace = res.sample(draws=kwargs['draws'])
                logging.info('Sampling complete.')
                posterior_predictive = pm.sample_posterior_predictive(
                    trace,
                    random_seed=random_seed
                )
            else:
                posterior_predictive = pm.sample_posterior_predictive(
                    res,
                    random_seed=random_seed
                )
        return posterior_predictive

    def score(self, data):
        """Returns the mean accuracy on the given test data and labels.
        """
        pass


if __name__ == "__main__":
    pass
